#%% imports & check

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import gc
import os
import json
import math
import time as _time
import random
import itertools
import traceback
import tempfile
import datetime
from tqdm import tqdm
from collections import defaultdict
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use('Agg')  # non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login
from typing import Dict, List, Tuple, Optional

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU count:", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")


_hf_token = os.environ.get("HF_TOKEN")
if _hf_token:
    login(token=_hf_token)
    print("Logged in via HF_TOKEN environment variable")
else:
    print("WARNING: HF_TOKEN not set. Access to gated models (e.g. Llama-3) will fail.")
    print("  Set it with:  export HF_TOKEN=<your_huggingface_token>")


#%% loading models
model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"

# Load model in bfloat16 (NO quantization — full-precision weights for clean IG gradients)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model.eval()
print(f"Model loaded: {model_name}")
print(f"Device map: {model.hf_device_map if hasattr(model, 'hf_device_map') else 'N/A'}")

model.enable_input_require_grads()

#%% Integradients on up_proj
# ============================================================================
# [SECTION 1] IntegratedGradientsTracker — per-layer IG on up_proj
# ============================================================================

class IntegratedGradientsTracker:
    """
    Per-layer Integrated Gradients on up_proj output.

    (a) Tensor being attributed: up_proj output (config: input_ff_attr = "mlp.up_proj").
    (b) Cost: m × num_layers forward passes per prompt. With m=16 and 32 layers
        this is 512 forward passes per prompt.
    (c) Citation: NeuronLLM paper Eq. 1 (arXiv:2601.04548).

    Key invariant (per-layer isolation):
        For each layer l independently:
          1. Capture baseline w_hat^l via read-only forward pass
          2. For k = 1..m: patch ONLY layer l with (k/m)*w_hat^l, forward, grad
          3. Restore layer l; move to layer l+1
        All other layers run at natural activations throughout.

    IG formula (adapted from Eq. 1):
        IG(w_i^l) = (w_hat_i^l / m) * sum_{k=1}^{m} dF((k/m) * w_hat_i^l) / dw_i^l
        where alpha_k = k/m for k=1..m (right Riemann sum, excludes alpha=0)
    """

    def __init__(self, model, m=16):
        self.model = model
        self.m = m
        self.activations = {}
        self.num_layers = len(model.model.layers)
        self._original_up_projs: Dict[int, torch.nn.Module] = {}
        print(f"IntegratedGradientsTracker: {self.num_layers} layers, m={m} steps")
        print(f"  Mode: PER-LAYER on up_proj (matching supplementary code)")
        print(f"  {m}x{self.num_layers}={m*self.num_layers} forward passes per prompt")

    def _capture_baseline(self, input_ids, layer_idx, mask_idx=-1):
        """Capture baseline up_proj output at [batch, mask_idx, :] for one layer."""
        baseline_acts = None

        def hook(module, input, output):
            nonlocal baseline_acts
            baseline_acts = output[:, mask_idx, :].detach().clone()

        handle = self.model.model.layers[layer_idx].mlp.up_proj.register_forward_hook(hook)
        with torch.no_grad():
            self.model(input_ids)
        handle.remove()
        return baseline_acts

    def compute_integrated_gradients(self, input_ids, target_fn):
        """
        Compute IG for all neurons, one layer at a time.

        For each layer l:
          1. Capture baseline up_proj output w_hat^l at last token
          2. For k=1..m:
             a. Replace up_proj with _TempPatch injecting (k/m)*w_hat^l
             b. Forward pass (only layer l is modified)
             c. torch.autograd.grad(target, scaled_acts)
          3. IG^l = (w_hat^l / m) * sum(grads)

        Returns: Dict[layer_name -> IG tensor of shape [intermediate_size]]
        """
        mask_idx = -1
        integrated_grads = {}

        for layer_idx in range(self.num_layers):
            layer_name = f"layer_{layer_idx}"

            # Step 1: Capture baseline
            baseline_acts = self._capture_baseline(input_ids, layer_idx, mask_idx)

            grad_sum = torch.zeros(baseline_acts.shape[-1], dtype=torch.float32,
                                   device=baseline_acts.device)
            zero_grad_count = 0

            # Step 2: m interpolation steps — right Riemann sum, alpha_k = k/m for k=1..m
            scale_factors = torch.arange(1, self.m + 1, device=baseline_acts.device).float() / self.m
            for k in range(self.m):
                alpha = scale_factors[k].item()
                scaled_acts = (baseline_acts.float() * alpha).requires_grad_(True)

                original_up_proj = self.model.model.layers[layer_idx].mlp.up_proj

                patch = _TempPatch(original_up_proj, mask_idx, scaled_acts)
                self.model.model.layers[layer_idx].mlp.up_proj = patch

                outputs = self.model(input_ids)
                target = target_fn(outputs.logits)

                grad = torch.autograd.grad(
                    outputs=target, inputs=scaled_acts,
                    create_graph=False, retain_graph=False, allow_unused=True
                )

                if grad[0] is not None:
                    grad_sum += grad[0].detach().float().squeeze(0)
                else:
                    zero_grad_count += 1

                # Restore original up_proj
                self.model.model.layers[layer_idx].mlp.up_proj = original_up_proj
                del outputs, target, scaled_acts, patch, grad

            if zero_grad_count > 0:
                print(f"  WARNING: {zero_grad_count}/{self.m} steps had None grad for {layer_name}")

            # Step 3: IG = (w_hat / m) * sum(grads)
            baseline_squeezed = baseline_acts.squeeze(0).float()
            ig = baseline_squeezed * grad_sum / self.m
            integrated_grads[layer_name] = ig
            self.activations[layer_name] = baseline_acts

            # Diagnostic every 8th layer + last
            if layer_idx % 8 == 0 or layer_idx == self.num_layers - 1:
                ig_flat = ig.flatten()
                print(f"  {layer_name}: base_norm={baseline_squeezed.norm():.4f}, "
                      f"grad_norm={grad_sum.norm():.6f}, IG_abs_mean={ig_flat.abs().mean():.6f}, "
                      f"pos={int((ig_flat > 0).sum())}, neg={int((ig_flat < 0).sum())}")

            del baseline_acts, grad_sum, baseline_squeezed
            gc.collect()
            torch.cuda.empty_cache()

        return integrated_grads

    def clear(self):
        self.activations.clear()

    def remove_hooks(self):
        """Restore any patched up_proj modules and clear state."""
        for layer_idx in range(self.num_layers):
            current = self.model.model.layers[layer_idx].mlp.up_proj
            if hasattr(current, 'ff'):
                self.model.model.layers[layer_idx].mlp.up_proj = current.ff
        self.activations.clear()


# ============================================================================
# [SECTION 2] FixedNeuronTracker (legacy) — single-step gradient
# ============================================================================

class FixedNeuronTracker:
    """Legacy single-step gradient tracker for backward compatibility."""

    def __init__(self, model):
        self.model = model
        self.activations = {}
        self.gradients = {}
        self.handles = []
        self._original_forwards = {}
        self.num_layers = len(model.model.layers)

        print(f"[Legacy] Set up single-step gradient hooks for {self.num_layers} layers")

        for layer_idx in range(self.num_layers):
            self._hook_layer(layer_idx)

    def _hook_layer(self, layer_idx):
        """Hook up_proj output (pre-gating) to match IntegratedGradientsTracker's
        attribution site. Both trackers now attribute on the same tensor:
        up_proj(x), NOT SiLU(gate_proj(x)) * up_proj(x)."""
        layer = self.model.model.layers[layer_idx]
        layer_name = f"layer_{layer_idx}"
        self._original_forwards[layer_name] = layer.mlp.forward

        def modified_forward(hidden_states):
            up = layer.mlp.up_proj(hidden_states)

            # Capture up_proj output (pre-gating), matching IG tracker
            self.activations[layer_name] = up[:, -1, :].detach().clone()

            if up.requires_grad:
                def grad_hook(grad):
                    if grad is not None:
                        self.gradients[layer_name] = grad[:, -1, :].detach().clone()

                handle = up.register_hook(grad_hook)
                self.handles.append(handle)

            gate = layer.mlp.gate_proj(hidden_states)
            intermediate = F.silu(gate) * up
            output = layer.mlp.down_proj(intermediate)
            return output

        layer.mlp.forward = modified_forward

    def get_attributions(self):
        """Compute attribution scores: activation x gradient."""
        attributions = {}

        for layer_name in self.activations:
            if layer_name in self.gradients:
                act = self.activations[layer_name].float()
                grad = self.gradients[layer_name].float()

                if grad.abs().sum() < 1e-10:
                    print(f"Warning: Zero gradient for {layer_name}")
                    continue

                attributions[layer_name] = act * grad

        return attributions

    def clear(self):
        self.activations.clear()
        self.gradients.clear()

    def remove_hooks(self):
        """Restore all original MLP forward methods."""
        for layer_idx in range(self.num_layers):
            layer_name = f"layer_{layer_idx}"
            if layer_name in self._original_forwards:
                layer = self.model.model.layers[layer_idx]
                layer.mlp.forward = self._original_forwards[layer_name]
        self._original_forwards.clear()
        self.activations.clear()
        self.gradients.clear()


# ============================================================================
# [SECTION 3] KnowledgeNeuronFinder — per-layer IG + logit margin F
# ============================================================================

class KnowledgeNeuronFinder:
    """
    Finds knowledge neurons using per-layer Integrated Gradients on up_proj.

    Key features:
    - Per-layer IG on up_proj (NeuronLLM paper Eq. 1)
    - Logit margin target function: F(x) = Logit(correct) - Logit(incorrect)
    - Direct token IDs (space-prefixed, cached) — no combined token maps
    - Adaptive thresholding for coarse neuron selection
    """

    def __init__(self, model, tokenizer, m=16, use_ig=True):
        self.model = model
        self.tokenizer = tokenizer
        self.device = model.device
        self.m = m
        self.use_ig = use_ig
        self._token_id_cache: Dict[str, int] = {}

        if use_ig:
            self.tracker = IntegratedGradientsTracker(model, m=m)
        else:
            self.tracker = FixedNeuronTracker(model)
            print("[Warning] Using legacy single-step gradient instead of Integrated Gradients")

        # Build combined option token IDs (matching NeuronLLM supplementary code).
        # Each option letter maps to ALL its token IDs (bare + space-prefixed),
        # ensuring IG attribution sees the same tokens that evaluation uses.
        self.option_token_map = self._build_option_token_map()

    def _build_option_token_map(self):
        """
        Build combined token ID sets for each option letter.

        Matches supplementary code's COMBINED_OPTION_TOKENS approach:
        For each option letter, collect ALL token IDs that represent it
        (bare "A" and space-prefixed " A"), then combine via logsumexp
        during IG computation. This eliminates the mismatch between
        what IG attributes and what evaluation measures.
        """
        option_letters = ["A", "B"]
        token_map = {}

        for letter in option_letters:
            ids = set()
            bare_ids = self.tokenizer.encode(letter, add_special_tokens=False)
            if bare_ids:
                ids.add(bare_ids[0])
            space_ids = self.tokenizer.encode(" " + letter, add_special_tokens=False)
            if space_ids:
                ids.add(space_ids[-1])
            token_map[letter] = sorted(ids)
            decoded = [(tid, self.tokenizer.decode([tid])) for tid in token_map[letter]]
            print(f"  Option '{letter}' combined token IDs: {decoded}")

        return token_map

    def _get_robust_token_id(self, token):
        """
        Robustly get token ID for a given option letter.

        CRITICAL: For Llama 3, the model generates SPACE-PREFIXED tokens
        after "Answer:" — i.e., " A" (ID 362) not "A" (ID 32).

        Results are cached to avoid repeated tokenizer calls.
        """
        if token in self._token_id_cache:
            return self._token_id_cache[token]

        # ALWAYS try space-prefixed version first
        ids_space = self.tokenizer.encode(" " + token, add_special_tokens=False)
        if ids_space:
            token_id = ids_space[-1]
            decoded = self.tokenizer.decode([token_id])
            print(f"  Token '{token}' -> ID {token_id} (decoded: '{decoded}')")
            self._token_id_cache[token] = token_id
            return token_id

        # Fallback to bare token
        ids_bare = self.tokenizer.encode(token, add_special_tokens=False)
        if ids_bare:
            token_id = ids_bare[0]
            decoded = self.tokenizer.decode([token_id])
            print(f"  Token '{token}' (bare) -> ID {token_id} (decoded: '{decoded}')")
            self._token_id_cache[token] = token_id
            return token_id

        raise ValueError(f"Cannot tokenize: {token}")

    def _create_target_function(self, combined_option_ids, correct_idx):
        """
        Create target function using Logit Margin for binary forced-choice tasks.

        F(x) = logsumexp(correct_logits) - logsumexp(incorrect_logits)

        Uses combined token IDs (multiple tokenizations per letter) with
        logsumexp aggregation, matching the evaluation function
        get_combined_option_probs(). This eliminates the mismatch between
        "what IG attributes" and "what evaluation measures".

        Advantages over restricted softmax:
        - No gradient saturation: dP/dz = P(1-P) vanishes when P->0 or P->1,
          but logit margin has constant gradient, so strongly decisive neurons
          are never under-attributed.
        - Symmetric attribution: a neuron gets equal credit for increasing the
          correct logit and for decreasing the incorrect logit.

        Interpretation:
        - Positive IG -> neuron INCREASES margin -> GOOD (facilitator)
        - Negative IG -> neuron DECREASES margin -> BAD (inhibitor)

        Args:
            combined_option_ids: List of lists, where combined_option_ids[i]
                                 is a list of token IDs for option i
            correct_idx: Index of correct option in combined_option_ids (0 or 1)

        Returns:
            Function that takes logits and returns logit margin scalar
        """
        incorrect_idx = 1 - correct_idx

        def target_fn(logits):
            last_logits = logits[:, -1, :]  # [batch, vocab_size]
            # logsumexp over combined token IDs for each option
            correct_token_ids = combined_option_ids[correct_idx]
            incorrect_token_ids = combined_option_ids[incorrect_idx]
            logit_correct = torch.logsumexp(last_logits[:, correct_token_ids], dim=-1)
            logit_incorrect = torch.logsumexp(last_logits[:, incorrect_token_ids], dim=-1)
            return (logit_correct - logit_incorrect).mean()

        return target_fn

    def get_coarse_neurons(self, prompt, ground_truth_token, option_tokens, z=5000):
        """
        Get coarse neurons for a single prompt using per-layer IG.

        With logit margin as target:
        - Positive IG -> neuron INCREASES margin -> GOOD
        - Negative IG -> neuron DECREASES margin -> BAD
        """
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        ).to(self.device)

        # Use combined token IDs (multiple tokenizations per letter)
        combined_option_ids = [self.option_token_map[opt] for opt in option_tokens]
        correct_idx = option_tokens.index(ground_truth_token)

        if self.use_ig:
            target_fn = self._create_target_function(combined_option_ids, correct_idx)
            attributions = self.tracker.compute_integrated_gradients(
                inputs.input_ids,
                target_fn
            )
        else:
            # Legacy: single-step gradient with SAME logit-margin target as IG.
            # Uses combined token IDs + logsumexp so ablation isolates only
            # IG (m steps) vs single-step (1 step), not the target function.
            self.model.zero_grad()
            outputs = self.model(**inputs)
            logits = outputs.logits[0, -1, :]
            incorrect_idx = 1 - correct_idx
            logit_c = torch.logsumexp(logits[combined_option_ids[correct_idx]], dim=0)
            logit_i = torch.logsumexp(logits[combined_option_ids[incorrect_idx]], dim=0)
            margin = logit_c - logit_i
            margin.backward()
            attributions = self.tracker.get_attributions()

        coarse_good = {}
        coarse_bad = {}

        for layer_name, attr in attributions.items():
            attr_flat = attr.flatten() if attr.dim() > 1 else attr
            num_neurons = attr_flat.shape[0]
            actual_z = min(z, num_neurons)

            top_values, top_indices = torch.topk(attr_flat, actual_z, largest=True)
            good_indices = top_indices[top_values > 0].tolist()

            bottom_values, bottom_indices = torch.topk(attr_flat, actual_z, largest=False)
            bad_indices = bottom_indices[bottom_values < 0].tolist()

            if good_indices:
                coarse_good[layer_name] = good_indices
            if bad_indices:
                coarse_bad[layer_name] = bad_indices

        # Cleanup
        self.tracker.clear()
        del inputs
        if not self.use_ig:
            del outputs, logits, margin
        torch.cuda.empty_cache()

        return {'good': coarse_good, 'bad': coarse_bad}

    def compute_example_scores(self, prompts, correct_answers, option_tokens):
        """
        Compute ES_e (Example Score) by summing IG across all proxy questions.

        Implements Equation 4 from NeuronLLM paper:
        ES_e(w) = sum_{t=1}^{3} IG(w, p_t)

        Sums IG across proxy questions FIRST, then selects top-z neurons from
        the summed ES — NOT selecting from each separately.
        """
        # Use combined token IDs (multiple tokenizations per letter)
        combined_option_ids = [self.option_token_map[opt] for opt in option_tokens]

        es_scores = {}

        for pq_idx, (prompt, correct_answer) in enumerate(zip(prompts, correct_answers)):
            correct_idx = option_tokens.index(correct_answer)

            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=2048
            ).to(self.device)

            if self.use_ig:
                target_fn = self._create_target_function(combined_option_ids, correct_idx)
                attributions = self.tracker.compute_integrated_gradients(
                    inputs.input_ids,
                    target_fn
                )
            else:
                # Legacy: single-step with SAME logit-margin target as IG
                self.model.zero_grad()
                outputs = self.model(**inputs)
                logits = outputs.logits[0, -1, :]
                incorrect_idx = 1 - correct_idx
                logit_c = torch.logsumexp(logits[combined_option_ids[correct_idx]], dim=0)
                logit_i = torch.logsumexp(logits[combined_option_ids[incorrect_idx]], dim=0)
                margin = logit_c - logit_i
                margin.backward()
                attributions = self.tracker.get_attributions()

            for layer_name, attr in attributions.items():
                attr_flat = attr.flatten() if attr.dim() > 1 else attr
                attr_cpu = attr_flat.cpu().float()

                if layer_name not in es_scores:
                    es_scores[layer_name] = torch.zeros_like(attr_cpu)

                es_scores[layer_name] = es_scores[layer_name] + attr_cpu

            self.tracker.clear()
            del inputs, attributions
            if not self.use_ig:
                del outputs, margin

        return es_scores

    def get_refined_neurons_from_es(self, es_scores, z=5000):
        """Select top-z good and bottom-z bad neurons from ES scores."""
        coarse_good = {}
        coarse_bad = {}

        for layer_name, es in es_scores.items():
            es_flat = es.flatten() if es.dim() > 1 else es
            num_neurons = es_flat.shape[0]
            actual_z = min(z, num_neurons)

            top_values, top_indices = torch.topk(es_flat, actual_z, largest=True)
            good_indices = top_indices[top_values > 0].tolist()

            bottom_values, bottom_indices = torch.topk(es_flat, actual_z, largest=False)
            bad_indices = bottom_indices[bottom_values < 0].tolist()

            if good_indices:
                coarse_good[layer_name] = good_indices
            if bad_indices:
                coarse_bad[layer_name] = bad_indices

        return {'good': coarse_good, 'bad': coarse_bad, 'es_scores': es_scores}

    def get_refined_neurons(
        self,
        prompts,
        correct_answers,
        option_tokens,
        sharing_percentage=0.5,  # DEPRECATED - kept for API compatibility
        z=5000
    ):
        """
        Get refined neurons using corrected NeuronLLM methodology (Eq. 4-5).

        1. Compute ES_e by SUMMING IG across all proxy questions FIRST (Eq. 4)
        2. Select top-z/bottom-z neurons from the SUMMED ES
        """
        print(f"\n Computing ES_e across {len(prompts)} proxy questions (Eq. 4)...")
        print(f"   Count-based selection: z={z}")
        print(f"   Using Integrated Gradients: {self.use_ig} (m={self.m} steps)")
        print(f"   F: Logit Margin (correct - incorrect)")
        print(f"   NOTE: Summing IG first, then selecting top-z (corrected methodology)")

        es_scores = self.compute_example_scores(prompts, correct_answers, option_tokens)
        result = self.get_refined_neurons_from_es(es_scores, z=z)

        total_good = sum(len(v) for v in result['good'].values())
        total_bad = sum(len(v) for v in result['bad'].values())

        print(f"\nES-based Selection Results:")
        print(f"  Total G_j (good) neurons: {total_good}")
        print(f"  Total B_j (bad) neurons: {total_bad}")

        return result


# ============================================================================
# [SECTION 4] identify_good_bad_neurons
# ============================================================================

def identify_good_bad_neurons(
    global_scores,
    top_k=100,
    percentile_threshold=None
):
    """
    CNI: Final step to identify task-level good and bad neurons.

    Following NeuronLLM (Section 3.4):
    - G_T = top K neurons by ACE score (good neurons)
    - B_T = bottom K neurons by ACE score (bad neurons)
    """
    print(f"\nCNI: Final Neuron Selection (K={top_k})")

    all_neurons = []

    for layer_name, score_tensor in global_scores.items():
        scores_np = score_tensor.cpu().numpy().flatten()

        for idx, score in enumerate(scores_np):
            if abs(score) > 1e-10:
                all_neurons.append({
                    "layer": layer_name,
                    "neuron_idx": int(idx),
                    "score": float(score)
                })

    print(f"  Total scored neurons: {len(all_neurons)}")

    good_neurons = sorted(all_neurons, key=lambda x: x['score'], reverse=True)[:top_k]
    bad_neurons = sorted(all_neurons, key=lambda x: x['score'])[:top_k]

    good_neurons = [n for n in good_neurons if n['score'] > 0]
    bad_neurons = [n for n in bad_neurons if n['score'] < 0]

    good_avg = np.mean([n['score'] for n in good_neurons]) if good_neurons else 0
    bad_avg = np.mean([n['score'] for n in bad_neurons]) if bad_neurons else 0

    print(f"\nCNI Results:")
    print(f"  Good neurons (G_T): {len(good_neurons)} (avg ACE score: {good_avg:.6f})")
    print(f"  Bad neurons (B_T):  {len(bad_neurons)} (avg ACE score: {bad_avg:.6f})")

    good_layers = defaultdict(int)
    bad_layers = defaultdict(int)
    for n in good_neurons:
        good_layers[n['layer']] += 1
    for n in bad_neurons:
        bad_layers[n['layer']] += 1

    print(f"\n  Layer distribution:")
    all_layers = sorted(set(good_layers.keys()) | set(bad_layers.keys()),
                       key=lambda x: int(x.split('_')[1]))
    for layer in all_layers:
        print(f"    {layer}: {good_layers.get(layer, 0)} good, {bad_layers.get(layer, 0)} bad")

    if good_neurons:
        good_scores = [n['score'] for n in good_neurons]
        print(f"\n  Good neuron score range: [{min(good_scores):.6f}, {max(good_scores):.6f}]")
    if bad_neurons:
        bad_scores = [n['score'] for n in bad_neurons]
        print(f"  Bad neuron score range: [{min(bad_scores):.6f}, {max(bad_scores):.6f}]")

    return good_neurons, bad_neurons


# ============================================================================
# [SECTION 5] Intervention infrastructure — _TempPatch, UpProjPatch
# ============================================================================

class _TempPatch(torch.nn.Module):
    """Replaces up_proj output at mask_idx with a scaled activation.

    Used by IntegratedGradientsTracker during IG interpolation steps.
    """
    def __init__(self, ff_layer, midx, replacement):
        super().__init__()
        self.ff = ff_layer
        self.midx = midx
        self.replacement = replacement

    def forward(self, x):
        out = self.ff(x)
        new_out = out.clone()
        new_out[:, self.midx, :] = self.replacement.to(dtype=out.dtype, device=out.device)
        return new_out


class UpProjPatch(torch.nn.Module):
    """
    Wraps up_proj to apply neuron-level interventions at the last token position.

    Matches the supplementary code's Patch class (_patch.py) which wraps the
    ff_layer (up_proj) and modifies its output. Interventions are applied at
    the SAME representation where Integrated Gradients were computed
    (up_proj output), NOT at the post-gating intermediate (SiLU(gate)*up).

    Modes:
        - 'scale': Scale specific neurons by given factors at mask_idx position
    """
    def __init__(self, ff_layer, mask_idx, good_indices=None, bad_indices=None,
                 good_scale=1.0, bad_scale=1.0):
        super().__init__()
        self.ff = ff_layer
        self.mask_idx = mask_idx
        self.good_indices = good_indices or []
        self.bad_indices = bad_indices or []
        self.good_scale = good_scale
        self.bad_scale = bad_scale

    def forward(self, x):
        out = self.ff(x)
        if self.good_indices and self.good_scale != 1.0:
            out[:, self.mask_idx, self.good_indices] *= self.good_scale
        if self.bad_indices and self.bad_scale != 1.0:
            out[:, self.mask_idx, self.bad_indices] *= self.bad_scale
        return out


def patch_up_proj_layers(model, good_map, bad_map, good_scale=1.0, bad_scale=1.0, mask_idx=-1):
    """
    Patch up_proj modules for neuron intervention.

    Returns list of (layer_idx, original_up_proj) for later unpatching.
    """
    patched = []
    all_layers = set(good_map.keys()) | set(bad_map.keys())

    for layer_idx in all_layers:
        layer = model.model.layers[layer_idx]
        original_up_proj = layer.mlp.up_proj

        g_indices = good_map.get(layer_idx, [])
        b_indices = bad_map.get(layer_idx, [])

        patch = UpProjPatch(
            original_up_proj,
            mask_idx=mask_idx,
            good_indices=g_indices,
            bad_indices=b_indices,
            good_scale=good_scale,
            bad_scale=bad_scale
        )
        layer.mlp.up_proj = patch
        patched.append((layer_idx, original_up_proj))

    return patched


def unpatch_up_proj_layers(model, patched):
    """Restore original up_proj modules after intervention."""
    for layer_idx, original_up_proj in patched:
        model.model.layers[layer_idx].mlp.up_proj = original_up_proj


# ============================================================================
# [SECTION 6] Evaluation helpers
# ============================================================================

def get_combined_option_probs(logits, tokenizer):
    """
    Get option probabilities using combined token IDs.
    Uses logsumexp to combine multiple tokenizations of the same letter,
    then softmax over combined option logits.

    Returns dict: {'A': prob, 'B': prob}
    """
    option_letters = ['A', 'B']
    token_map = {}
    for letter in option_letters:
        ids = set()
        bare = tokenizer.encode(letter, add_special_tokens=False)
        if bare:
            ids.add(bare[0])
        space = tokenizer.encode(" " + letter, add_special_tokens=False)
        if space:
            ids.add(space[-1])
        token_map[letter] = sorted(ids)

    all_ids = [tid for letter in option_letters for tid in token_map[letter]]
    all_logits = logits[all_ids]

    option_logits = []
    offset = 0
    for letter in option_letters:
        n = len(token_map[letter])
        option_logits.append(torch.logsumexp(all_logits[offset:offset+n], dim=0))
        offset += n

    option_logits_tensor = torch.stack(option_logits)
    probs = F.softmax(option_logits_tensor, dim=-1)
    return {letter: probs[i].item() for i, letter in enumerate(option_letters)}


def predict_from_logits(logits, tokenizer):
    """
    Predict answer by argmax over combined option token logits.

    Uses logit-based evaluation consistent with the IG attribution regime.
    """
    probs = get_combined_option_probs(logits, tokenizer)
    predicted = max(probs, key=probs.get)
    return predicted, probs


def wilson_ci(correct: int, total: int, z: float = 1.96) -> tuple:
    """
    Wilson score 95% confidence interval for a proportion.
    Returns (lower, upper) as floats in [0, 1].
    """
    if total == 0:
        return (0.0, 0.0)
    p = correct / total
    denom = 1 + z**2 / total
    centre = (p + z**2 / (2 * total)) / denom
    spread = (z * (p * (1 - p) / total + z**2 / (4 * total**2)) ** 0.5) / denom
    return (max(0.0, centre - spread), min(1.0, centre + spread))


def mcnemar_test(results_a: list, results_b: list) -> float:
    """
    McNemar's test for paired binary classifiers.
    results_a[i] and results_b[i] are booleans (True = correct).
    Returns p-value (two-sided). Uses continuity correction.
    """
    b = sum(1 for a, b_ in zip(results_a, results_b) if a and not b_)
    c = sum(1 for a, b_ in zip(results_a, results_b) if not a and b_)
    if b + c == 0:
        return 1.0
    chi2_stat = (abs(b - c) - 1) ** 2 / (b + c)
    p = math.erfc(math.sqrt(chi2_stat / 2))
    try:
        from scipy.stats import chi2 as chi2_dist
        p = 1 - chi2_dist.cdf(chi2_stat, df=1)
    except ImportError:
        pass
    return p


def compute_per_class_metrics(per_sample_results: list) -> dict:
    """
    Compute precision, recall, F1, and support for each class (A=benign, B=malware).
    Binary classification: positive class = B (malware).
    """
    y_true, y_pred = [], []
    for r in per_sample_results:
        expected = r.get("expected", "")
        predicted = r.get("predicted", None)
        if expected not in ("A", "B"):
            continue
        y_true.append(1 if expected == "B" else 0)
        y_pred.append(1 if predicted == "B" else 0)

    if not y_true:
        return {}

    classes = [0, 1]
    class_names = {0: "benign (A)", 1: "malware (B)"}
    report: Dict[str, dict] = {}

    for cls in classes:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == cls and p == cls)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != cls and p == cls)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == cls and p != cls)
        support = sum(1 for t in y_true if t == cls)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) > 0 else 0.0)

        report[class_names[cls]] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": support,
        }

    macro_p = float(np.mean([v["precision"] for v in report.values()]))
    macro_r = float(np.mean([v["recall"] for v in report.values()]))
    macro_f = float(np.mean([v["f1"] for v in report.values()]))
    report["macro avg"] = {
        "precision": round(macro_p, 4),
        "recall": round(macro_r, 4),
        "f1": round(macro_f, 4),
        "support": len(y_true),
    }

    return report


# ============================================================================
# [SECTION 7] Prompt construction
# ============================================================================

_truncation_stats: Dict[str, int] = {
    "total_calls": 0,
    "truncated": 0,
    "total_original_chars": 0,
    "total_truncated_chars": 0,
}


def report_truncation_stats(tag: str = "") -> Dict[str, float]:
    """Print and return a summary of prompt-truncation statistics."""
    stats = _truncation_stats
    n = stats["total_calls"] or 1
    trunc_pct = 100.0 * stats["truncated"] / n
    mean_orig = stats["total_original_chars"] / n
    mean_removed = stats["total_truncated_chars"] / max(stats["truncated"], 1)

    label = f" ({tag})" if tag else ""
    print(f"\n--- Truncation statistics{label} ---")
    print(f"  Total prompts created : {stats['total_calls']}")
    print(f"  Truncated             : {stats['truncated']} ({trunc_pct:.1f}%)")
    print(f"  Mean original chars   : {mean_orig:.0f}")
    if stats["truncated"] > 0:
        print(f"  Mean chars removed    : {mean_removed:.0f} (among truncated samples)")
    print(f"  Threshold             : 2000 chars")

    return {
        "total_calls": stats["total_calls"],
        "truncated_count": stats["truncated"],
        "truncated_pct": round(trunc_pct, 2),
        "mean_original_chars": round(mean_orig, 1),
        "mean_chars_removed": round(mean_removed, 1),
    }


def create_malware_prompt(source_code: str, question_variant: int = 0) -> tuple:
    """
    Create a malware classification prompt with AQUA-style augmentation.

    Following NeuronLLM paper's AQUA module (Section 3.3):
    - Paper uses 3 proxy questions per example with shuffled options
    - For binary classification: 3 variants (option swap + phrasing variations)

    Args:
        source_code: The source code to classify
        question_variant: 0 = original order (A=Benign, B=Malware)
                         1 = swapped order (A=Malware, B=Benign)
                         2 = alternative phrasing with swapped order

    Returns:
        tuple: (formatted_prompt, answer_mapping, options_dict)
    """
    _truncation_stats["total_calls"] += 1
    original_len = len(source_code)
    _truncation_stats["total_original_chars"] += original_len

    max_code_length = 2000
    if len(source_code) > max_code_length:
        chars_removed = original_len - max_code_length
        _truncation_stats["truncated"] += 1
        _truncation_stats["total_truncated_chars"] += chars_removed
        source_code = source_code[:max_code_length] + "\n... [truncated]"

    if question_variant == 0:
        question = "Analyze the following source code and determine if it is benign (safe) or malicious (malware)."
        options = {"A": "Benign - The code appears to be safe and legitimate",
                   "B": "Malware - The code contains malicious functionality"}
        answer_mapping = {0: "A", 1: "B"}
    elif question_variant == 1:
        question = "Analyze the following source code and determine if it is benign (safe) or malicious (malware)."
        options = {"A": "Malware - The code contains malicious functionality",
                   "B": "Benign - The code appears to be safe and legitimate"}
        answer_mapping = {0: "B", 1: "A"}
    else:
        question = "Examine the code below and classify whether it poses a security threat."
        options = {"A": "Malicious - Contains harmful or suspicious functionality",
                   "B": "Safe - Appears to be legitimate, non-threatening code"}
        answer_mapping = {0: "B", 1: "A"}

    prompt = f"""{question}

Source Code:
```
{source_code}
```

Options:
A: {options['A']}
B: {options['B']}

Answer:"""

    return prompt, answer_mapping, options


# ============================================================================
# [SECTION 8] ACE scoring — logit margin + filter_ambiguous + train_indices
# ============================================================================

def calculate_ace_scores(
    model,
    tokenizer,
    dataset_path: str,
    max_samples: int = 100,
    z: int = 5000,
    sharing_percentage: float = 0.5,  # DEPRECATED
    num_proxy_questions: int = 3,
    m: int = 16,
    use_ig: bool = True,
    filter_ambiguous: bool = True
) -> Tuple[Dict[str, torch.Tensor], list]:
    """
    Calculate ACE (Additive Cross-Entropy) scores across dataset using per-layer IG.

    F function: Logit Margin — F(x) = Logit(correct) - Logit(incorrect)

    For each example e_j:
    1. Generate 3 proxy questions with shuffled options (AQUA, Section 3.3)
    2. Compute ES_e = SUM of IG across 3 proxy questions (Eq. 4)
    3. Select G_j (top-z good) and B_j (bottom-z bad) from SUMMED ES_e

    Across all examples:
    4. Track ambiguous neurons (appear in both G and B across examples)
    5. Compute ACE(w) = sum over examples of I[w in G_j union B_j] * ES_ej(w) (Eq. 5)
    6. Normalize by number of EXAMPLES

    Args:
        filter_ambiguous: If True (default, paper-correct), apply Eq. 5 mask
            that zeros out neurons appearing in both G and B across examples.

    Returns:
        Tuple of (global_scores dict, train_indices list)
    """
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Tag every item with its global index BEFORE any subsetting
    for i, item in enumerate(data):
        item['_global_idx'] = i

    label_counts = {0: 0, 1: 0}
    for item in data:
        label = item.get('label', 0)
        label_counts[label] += 1

    print(f"\nDataset class distribution:")
    print(f"  Benign (label=0): {label_counts[0]} samples")
    print(f"  Malware (label=1): {label_counts[1]} samples")
    print(f"  Ratio: {label_counts[0]}:{label_counts[1]}")

    # Stratified sampling
    if max_samples and max_samples < len(data):
        benign_samples = [item for item in data if item.get('label', 0) == 0]
        malware_samples = [item for item in data if item.get('label', 0) == 1]

        samples_per_class = max_samples // 2
        _class_rng = random.Random(42)
        _class_rng.shuffle(benign_samples)
        _class_rng.shuffle(malware_samples)
        benign_subset = benign_samples[:min(samples_per_class, len(benign_samples))]
        malware_subset = malware_samples[:min(samples_per_class, len(malware_samples))]

        data = benign_subset + malware_subset
        _rnd = random.Random(42)
        _rnd.shuffle(data)

        print(f"\nUsing stratified sampling:")
        print(f"  Selected {len(benign_subset)} benign + {len(malware_subset)} malware = {len(data)} total")

    print(f"\nStarting ACE scoring on {len(data)} examples...")
    print(f"  - Count-based selection: z={z}")
    print(f"  - Proxy questions per example: {num_proxy_questions}")
    print(f"  - Using Integrated Gradients: {use_ig} (m={m} steps)")
    print(f"  - F: Logit Margin (correct - incorrect)")
    print(f"  - Filter ambiguous neurons: {filter_ambiguous}")

    kn_finder = KnowledgeNeuronFinder(model, tokenizer, m=m, use_ig=use_ig)

    all_good_neurons_global = set()
    all_bad_neurons_global = set()

    example_results = []
    global_scores: Dict[str, torch.Tensor] = {}
    example_count = 0
    class_counts = {0: 0, 1: 0}

    print("\n=== PASS 1: Computing ES_e and selecting G_j/B_j per example ===")

    for item_idx, item in enumerate(tqdm(data, desc="Processing examples")):
        source_code = item.get('source_code', '')
        label = item.get('label', 0)
        class_counts[label] += 1

        if not source_code.strip():
            continue

        prompts = []
        correct_answers = []

        for pq_idx in range(num_proxy_questions):
            prompt, answer_mapping, options = create_malware_prompt(source_code, question_variant=pq_idx)
            correct_answer = answer_mapping[label]
            prompts.append(prompt)
            correct_answers.append(correct_answer)

        options_text = ["A", "B"]

        try:
            refined = kn_finder.get_refined_neurons(
                prompts,
                correct_answers,
                options_text,
                z=z
            )

            for layer_name, indices in refined['good'].items():
                for idx in indices:
                    all_good_neurons_global.add((layer_name, idx))

            for layer_name, indices in refined['bad'].items():
                for idx in indices:
                    all_bad_neurons_global.add((layer_name, idx))

            example_results.append({
                'refined': refined,
                'es_scores': refined['es_scores']
            })

            example_count += 1

            kn_finder.tracker.clear()
            torch.cuda.empty_cache()
            gc.collect()

        except Exception as e:
            torch.cuda.empty_cache()
            gc.collect()
            print(f"Error on example {item_idx}: {e}")
            traceback.print_exc()
            continue

    # Ambiguous neurons across examples
    ambiguous_neurons = all_good_neurons_global & all_bad_neurons_global
    print(f"\nProcessed class distribution:")
    print(f"  Benign (0): {class_counts[0]} examples")
    print(f"  Malware (1): {class_counts[1]} examples")
    print(f"\nAmbiguous neurons (in both G and B across examples): {len(ambiguous_neurons)}")
    if not filter_ambiguous:
        print("  -> NOT filtered (filter_ambiguous=False, binary classification mode)")
    else:
        print("  -> Will be filtered in PASS 2 (filter_ambiguous=True, paper Eq. 5)")

    print("\n=== PASS 2: Computing ACE scores (Eq. 5) ===")

    for ex_idx, ex_result in enumerate(tqdm(example_results, desc="Accumulating ACE")):
        refined = ex_result['refined']
        es_scores = ex_result['es_scores']

        all_refined_layers = set(refined['good'].keys()) | set(refined['bad'].keys())

        for layer_name in all_refined_layers:
            if layer_name not in es_scores:
                continue

            es = es_scores[layer_name]
            es_flat = es.flatten() if es.dim() > 1 else es

            if layer_name not in global_scores:
                global_scores[layer_name] = torch.zeros_like(es_flat)

            refined_indices = set(refined['good'].get(layer_name, []) +
                                 refined['bad'].get(layer_name, []))

            mask = torch.zeros_like(es_flat)
            for idx in refined_indices:
                if filter_ambiguous:
                    if (layer_name, idx) not in ambiguous_neurons:
                        if idx < mask.shape[0]:
                            mask[idx] = 1.0
                else:
                    if idx < mask.shape[0]:
                        mask[idx] = 1.0

            global_scores[layer_name] = global_scores[layer_name] + (es_flat * mask)

    print(f"\nProcessed {example_count} examples")

    if example_count > 0:
        for layer_name in global_scores:
            global_scores[layer_name] = global_scores[layer_name] / example_count

    train_indices = [item['_global_idx'] for item in data]
    return global_scores, train_indices


# ============================================================================
# [SECTION 9] Baselines — logit-based + TF-IDF
# ============================================================================

def test_baseline_accuracy(model, tokenizer, test_prompts, max_test=20):
    """
    Test baseline accuracy WITHOUT any neuron intervention.

    Uses logit-based prediction (argmax over combined option token logits),
    consistent with the IG attribution regime.
    """
    print("\n" + "="*60)
    print("BASELINE ACCURACY TEST (No Intervention)")
    print("="*60)

    correct = 0
    total = 0

    class_stats = {
        'A': {'correct': 0, 'total': 0, 'predicted_A': 0, 'predicted_B': 0},
        'B': {'correct': 0, 'total': 0, 'predicted_A': 0, 'predicted_B': 0}
    }

    predictions = {"A": 0, "B": 0, "other": 0}

    for i, sample in enumerate(test_prompts[:max_test]):
        prompt = sample['prompt']
        expected = sample['correct_answer_token'].strip().upper()[0]

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model(**inputs)
            last_logits = outputs.logits[0, -1, :]
            predicted, probs = predict_from_logits(last_logits, tokenizer)

        predictions[predicted] += 1

        class_stats[expected]['total'] += 1
        if predicted == 'A':
            class_stats[expected]['predicted_A'] += 1
        else:
            class_stats[expected]['predicted_B'] += 1

        if predicted == expected:
            correct += 1
            class_stats[expected]['correct'] += 1

        if i < 5:
            print(f"  Sample {i}: Expected={expected}, Predicted={predicted}, "
                  f"P(A)={probs['A']:.4f}, P(B)={probs['B']:.4f}")

        total += 1

    accuracy = correct / total if total > 0 else 0

    lo, hi = wilson_ci(correct, total)
    print(f"\nResults:")
    print(f"  Accuracy: {accuracy:.2%} ({correct}/{total})  95% CI [{lo:.2%}, {hi:.2%}]")
    print(f"  Predictions: A={predictions['A']}, B={predictions['B']}, Other={predictions['other']}")

    print(f"\nPer-class performance:")
    for cls in ['A', 'B']:
        stats = class_stats[cls]
        if stats['total'] > 0:
            cls_acc = stats['correct'] / stats['total']
            print(f"  Expected {cls}: {cls_acc:.1%} correct ({stats['correct']}/{stats['total']}) "
                  f"- Model predicted: A={stats['predicted_A']}, B={stats['predicted_B']}")

    if accuracy < 0.6:
        print("\nWARNING: Baseline accuracy is very low!")
        print("   Possible causes:")
        print("   1. Model doesn't understand the task format")
        print("   2. Dataset labels might be wrong/swapped")
        print("   3. Code samples are too complex for the model")

    return accuracy, predictions


def run_tfidf_baseline(
    dataset_path: str,
    train_indices: list,
    test_indices: list,
    output_dir: str,
) -> dict:
    """
    Train TF-IDF + Logistic Regression on the same training split used for
    neuron identification, and evaluate on the same test split.
    """
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score, classification_report
    except ImportError:
        print("scikit-learn not installed — skipping TF-IDF baseline.")
        return {}

    with open(dataset_path, 'r', encoding='utf-8') as f:
        all_data = json.load(f)

    train_set = [all_data[i] for i in train_indices if i < len(all_data)]
    test_set = [all_data[i] for i in test_indices if i < len(all_data)]

    if not train_set or not test_set:
        print("TF-IDF baseline: empty train or test set, skipping.")
        return {}

    X_train = [s['source_code'] for s in train_set]
    y_train = [s['label'] for s in train_set]
    X_test = [s['source_code'] for s in test_set]
    y_test = [s['label'] for s in test_set]

    vec = TfidfVectorizer(
        max_features=20_000,
        ngram_range=(1, 2),
        sublinear_tf=True,
        analyzer="word",
    )
    X_tr_vec = vec.fit_transform(X_train)
    X_te_vec = vec.transform(X_test)

    lr = LogisticRegression(solver="liblinear", C=1.0, max_iter=500, random_state=42)
    lr.fit(X_tr_vec, y_train)
    y_pred = lr.predict(X_te_vec)

    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred,
                                   target_names=["benign", "malware"],
                                   output_dict=True)

    correct_count = int(round(acc * len(y_test)))
    lo, hi = wilson_ci(correct_count, len(y_test))

    print(f"\n{'='*60}")
    print("TF-IDF + LOGISTIC REGRESSION BASELINE")
    print(f"{'='*60}")
    print(f"  Train size : {len(X_train)} ({sum(y_train)} malware, {len(y_train)-sum(y_train)} benign)")
    print(f"  Test size  : {len(X_test)}  ({sum(y_test)} malware, {len(y_test)-sum(y_test)} benign)")
    print(f"  Accuracy   : {acc:.2%}  95% CI [{lo:.2%}, {hi:.2%}]")
    print(classification_report(y_test, y_pred, target_names=["benign", "malware"]))

    result = {
        "accuracy": acc,
        "accuracy_ci": (lo, hi),
        "classification_report": report,
        "train_size": len(X_train),
        "test_size": len(X_test),
    }

    tfidf_file = os.path.join(output_dir, "tfidf_baseline.json")
    with open(tfidf_file, 'w') as f:
        json.dump(result, f, indent=4)
    print(f"  Saved to: {tfidf_file}")

    return result


# ============================================================================
# [SECTION 10] Validation — UpProjPatch + logit-based eval
# ============================================================================

def diagnose_probability_changes(model, tokenizer, test_samples, good_map, bad_map, num_samples=3):
    """
    Diagnostic function to check how interventions affect option probabilities.
    Uses up_proj patching + get_combined_option_probs.
    """
    print(f"\n  === PROBABILITY DIAGNOSIS (first {min(num_samples, len(test_samples))} samples) ===")

    def get_probs_with_intervention(prompt, good_scale, bad_scale):
        patched = patch_up_proj_layers(model, good_map, bad_map,
                                        good_scale=good_scale, bad_scale=bad_scale, mask_idx=-1)

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model(**inputs)
            last_logits = outputs.logits[0, -1, :]
            probs = get_combined_option_probs(last_logits, tokenizer)
            p_a = probs['A']
            p_b = probs['B']

        unpatch_up_proj_layers(model, patched)

        return p_a, p_b

    for i, sample in enumerate(test_samples[:num_samples]):
        prompt = sample['prompt']
        expected = sample['correct_answer_token'].strip().upper()[0]

        print(f"\n  Sample {i+1}: Expected={expected}")

        p_a_base, p_b_base = get_probs_with_intervention(prompt, 1.0, 1.0)
        print(f"    Baseline:     P(A)={p_a_base:.4f}, P(B)={p_b_base:.4f}")

        p_a_good, p_b_good = get_probs_with_intervention(prompt, 2.0, 1.0)
        delta_a_good = p_a_good - p_a_base
        delta_b_good = p_b_good - p_b_base
        print(f"    Good x2:      P(A)={p_a_good:.4f}, P(B)={p_b_good:.4f} "
              f"(dA={delta_a_good:+.4f}, dB={delta_b_good:+.4f})")

        p_a_bad, p_b_bad = get_probs_with_intervention(prompt, 1.0, 0.0)
        delta_a_bad = p_a_bad - p_a_base
        delta_b_bad = p_b_bad - p_b_base
        print(f"    Bad x0:       P(A)={p_a_bad:.4f}, P(B)={p_b_bad:.4f} "
              f"(dA={delta_a_bad:+.4f}, dB={delta_b_bad:+.4f})")

        p_a_enh, p_b_enh = get_probs_with_intervention(prompt, 2.0, 0.0)
        delta_a_enh = p_a_enh - p_a_base
        delta_b_enh = p_b_enh - p_b_base
        print(f"    Enhancer:     P(A)={p_a_enh:.4f}, P(B)={p_b_enh:.4f} "
              f"(dA={delta_a_enh:+.4f}, dB={delta_b_enh:+.4f})")

        p_a_deg, p_b_deg = get_probs_with_intervention(prompt, 0.0, 2.0)
        delta_a_deg = p_a_deg - p_a_base
        delta_b_deg = p_b_deg - p_b_base
        print(f"    Degrader:     P(A)={p_a_deg:.4f}, P(B)={p_b_deg:.4f} "
              f"(dA={delta_a_deg:+.4f}, dB={delta_b_deg:+.4f})")

        if expected == 'A':
            expected_up = 'P(A)'
            enh_delta = delta_a_enh
            deg_delta = delta_a_deg
        else:
            expected_up = 'P(B)'
            enh_delta = delta_b_enh
            deg_delta = delta_b_deg

        enh_correct = enh_delta > 0
        deg_correct = deg_delta < 0

        status_enh = "OK" if enh_correct else "FAIL"
        status_deg = "OK" if deg_correct else "FAIL"
        print(f"    Expectation: Enhancer should increase {expected_up}: {status_enh}, "
              f"Degrader should decrease {expected_up}: {status_deg}")


def validate_neurons(
    model,
    tokenizer,
    test_prompts,
    identified_neurons,
    max_test=50
):
    """
    Validate using Joint Intervention (Enhancer Strategy).

    Uses UpProjPatch + logit-based prediction.
    Returns dict with suppress_good / suppress_bad keys.
    test_joint_intervention returns 3-tuple (acc, confusion, per_sample_results).
    """
    print(f"\nVALIDATION: Testing {len(test_prompts[:max_test])} prompts...")

    good_neurons = identified_neurons['good_neurons']
    bad_neurons = identified_neurons['bad_neurons']

    print(f"  Good neurons: {len(good_neurons)}")
    print(f"  Bad neurons: {len(bad_neurons)}")

    def group_by_layer(neurons):
        groups = defaultdict(list)
        for n in neurons:
            layer_idx = int(n['layer'].split('_')[1])
            groups[layer_idx].append(n['neuron_idx'])
        return groups

    good_map = group_by_layer(good_neurons)
    bad_map = group_by_layer(bad_neurons)

    print(f"  Layers with good neurons: {sorted(good_map.keys())}")
    print(f"  Layers with bad neurons: {sorted(bad_map.keys())}")

    def test_joint_intervention(prompts, good_scale=1.0, bad_scale=1.0, verbose=False):
        """Run intervention using UpProjPatch + logit-based prediction.
        Returns (accuracy, confusion_matrix, per_sample_results).
        """
        patched = patch_up_proj_layers(model, good_map, bad_map,
                                        good_scale=good_scale, bad_scale=bad_scale, mask_idx=-1)

        correct = 0
        total = 0
        confusion = np.zeros((2, 3), dtype=int)
        label_to_row = {'A': 0, 'B': 1}
        per_sample_results = []

        for sample in tqdm(prompts[:max_test], desc="Evaluating", leave=False):
            prompt = sample['prompt']
            expected = sample['correct_answer_token'].strip().upper()[0]

            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

            with torch.no_grad():
                outputs = model(**inputs)
                last_logits = outputs.logits[0, -1, :]
                predicted, probs = predict_from_logits(last_logits, tokenizer)

            if predicted == expected:
                cat = 0
            elif predicted in ['A', 'B'] and predicted != expected:
                cat = 1
            else:
                cat = 2

            row = label_to_row.get(expected, None)
            if row is not None:
                confusion[row, cat] += 1

            per_sample_results.append({
                'index': total,
                'expected': expected,
                'predicted': predicted,
                'correct': (cat == 0),
                'prob_A': probs['A'],
                'prob_B': probs['B'],
            })

            if cat == 0:
                correct += 1
            total += 1

        unpatch_up_proj_layers(model, patched)

        accuracy = correct / total if total > 0 else 0
        return accuracy, confusion, per_sample_results

    print("\n1. Baseline (No Intervention):")
    baseline_acc, baseline_conf, baseline_per_sample = test_joint_intervention(
        test_prompts, good_scale=1.0, bad_scale=1.0)
    print(f"   Accuracy: {baseline_acc:.2%}")

    print("\n   DIAGNOSTIC: Checking probability changes on first 3 samples...")
    diagnose_probability_changes(model, tokenizer, test_prompts[:3], good_map, bad_map)

    print("\n2. Suppress Good Neurons (Good x0.0, Bad x1.0):")
    suppress_good_acc, suppress_good_conf, suppress_good_per_sample = test_joint_intervention(
        test_prompts, good_scale=0.0, bad_scale=1.0)
    print(f"   Accuracy: {suppress_good_acc:.2%} (d {suppress_good_acc - baseline_acc:+.2%})")

    print("\n3. Suppress Bad Neurons (Good x1.0, Bad x0.0):")
    suppress_bad_acc, suppress_bad_conf, suppress_bad_per_sample = test_joint_intervention(
        test_prompts, good_scale=1.0, bad_scale=0.0)
    print(f"   Accuracy: {suppress_bad_acc:.2%} (d {suppress_bad_acc - baseline_acc:+.2%})")

    print("\n4. Enhancer (Good x2.0 + Bad x0.0):")
    enhancer_acc, enhancer_conf, enhancer_per_sample = test_joint_intervention(
        test_prompts, good_scale=2.0, bad_scale=0.0)
    print(f"   Accuracy: {enhancer_acc:.2%} (d {enhancer_acc - baseline_acc:+.2%})")

    print("\n5. Degrader (Good x0.0 + Bad x2.0):")
    degrader_acc, degrader_conf, degrader_per_sample = test_joint_intervention(
        test_prompts, good_scale=0.0, bad_scale=2.0)
    print(f"   Accuracy: {degrader_acc:.2%} (d {degrader_acc - baseline_acc:+.2%})")

    # Samples fixed by enhancer
    fixed_samples = []
    for b_res, e_res in zip(baseline_per_sample, enhancer_per_sample):
        if not b_res['correct'] and e_res['correct']:
            fixed_samples.append({
                'index': b_res['index'],
                'expected': b_res['expected'],
                'baseline_predicted': b_res['predicted'],
                'enhancer_predicted': e_res['predicted'],
            })

    print(f"\n   Samples fixed by enhancer: {len(fixed_samples)} (baseline wrong -> enhancer correct)")

    fixed_file = os.path.join(OUTPUT_DIR, "baseline_wrong_enhancer_correct.json")
    with open(fixed_file, 'w') as f:
        json.dump(fixed_samples, f, indent=4)
    print(f"   Saved to: {fixed_file}")

    print("\n" + "="*60)
    if enhancer_acc > baseline_acc:
        print("PASS: Enhancer Strategy successfully improved performance.")
    else:
        print("FAIL: Enhancer Strategy failed to improve performance.")

    print(f"Effectiveness Gap (Enhancer - Degrader): {enhancer_acc - degrader_acc:.2%}")
    print("="*60)

    def print_confusion(name, conf):
        total = conf.sum()
        print(f"\nConfusion Matrix ({name}) - counts (rows: Expected A, Expected B; cols: Correct, Opposite, Other):")
        print(conf)
        if total > 0:
            print(f"Normalized (by row):")
            with np.errstate(divide='ignore', invalid='ignore'):
                row_norm = conf.astype(float) / conf.sum(axis=1, keepdims=True)
                row_norm = np.nan_to_num(row_norm)
            print(np.round(row_norm, 3))

    print_confusion('Baseline', baseline_conf)
    print_confusion('Enhancer', enhancer_conf)
    print_confusion('Degrader', degrader_conf)

    # Per-class metrics
    print("\n" + "="*60)
    print("PER-CLASS METRICS (Precision / Recall / F1)")
    print("="*60)

    _per_sample_map = {
        "baseline": baseline_per_sample,
        "suppress_good": suppress_good_per_sample,
        "suppress_bad": suppress_bad_per_sample,
        "enhancer": enhancer_per_sample,
        "degrader": degrader_per_sample,
    }

    per_class_all: Dict[str, dict] = {}
    for condition_label, samples in _per_sample_map.items():
        metrics = compute_per_class_metrics(samples)
        per_class_all[condition_label] = metrics
        print(f"\n  {condition_label}:")
        print(f"    {'Class':<18} {'Precision':>10} {'Recall':>8} {'F1':>8} {'Support':>9}")
        print(f"    {'-'*18} {'-'*10} {'-'*8} {'-'*8} {'-'*9}")
        for cls_name, vals in metrics.items():
            print(f"    {cls_name:<18} {vals['precision']:>10.3f} {vals['recall']:>8.3f} "
                  f"{vals['f1']:>8.3f} {vals['support']:>9}")

    # Wilson CIs for each condition
    total_samples = len(test_prompts[:max_test])
    results_map = [
        ("baseline", baseline_acc, baseline_conf),
        ("suppress_good", suppress_good_acc, suppress_good_conf),
        ("suppress_bad", suppress_bad_acc, suppress_bad_conf),
        ("enhancer", enhancer_acc, enhancer_conf),
        ("degrader", degrader_acc, degrader_conf),
    ]

    print("\n" + "="*60)
    print("ACCURACY WITH 95% WILSON CONFIDENCE INTERVALS")
    print("="*60)
    for label, acc, conf in results_map:
        correct_count = int(round(acc * total_samples))
        lo, hi = wilson_ci(correct_count, total_samples)
        print(f"  {label:20s}: {acc:.2%}  95% CI [{lo:.2%}, {hi:.2%}]")

    # McNemar test: baseline vs enhancer
    baseline_correct = [r['correct'] for r in baseline_per_sample]
    enhancer_correct = [r['correct'] for r in enhancer_per_sample]
    p_value = mcnemar_test(baseline_correct, enhancer_correct)
    print(f"\nMcNemar test (baseline vs enhancer): p = {p_value:.4f}")
    if p_value < 0.05:
        print("  -> Statistically significant improvement (p < 0.05)")
    else:
        print("  -> NOT statistically significant (p >= 0.05)")

    return {
        "baseline": baseline_acc,
        "baseline_ci": wilson_ci(int(round(baseline_acc * total_samples)), total_samples),
        "suppress_good": suppress_good_acc,
        "suppress_good_ci": wilson_ci(int(round(suppress_good_acc * total_samples)), total_samples),
        "suppress_bad": suppress_bad_acc,
        "suppress_bad_ci": wilson_ci(int(round(suppress_bad_acc * total_samples)), total_samples),
        "enhancer": enhancer_acc,
        "enhancer_ci": wilson_ci(int(round(enhancer_acc * total_samples)), total_samples),
        "degrader": degrader_acc,
        "degrader_ci": wilson_ci(int(round(degrader_acc * total_samples)), total_samples),
        "mcnemar_baseline_vs_enhancer": p_value,
        "confusion_matrices": {
            "baseline": baseline_conf.tolist(),
            "suppress_good": suppress_good_conf.tolist(),
            "suppress_bad": suppress_bad_conf.tolist(),
            "enhancer": enhancer_conf.tolist(),
            "degrader": degrader_conf.tolist()
        },
        "per_class_metrics": per_class_all,
    }


# ============================================================================
# [SECTION 11] Ablations — stability, single intervention, sweep
# ============================================================================

def compute_neuron_stability(
    model,
    tokenizer,
    dataset_path: str,
    n_runs: int = 5,
    max_samples: int = 10,
    top_k: int = 100,
    z: int = 5000,
    m: int = 16,
    use_ig: bool = True,
    seeds: list = None
) -> dict:
    """
    Run ACE scoring n_runs times with different random seeds/subsets.
    Report Jaccard similarity of identified good and bad neuron sets.
    """
    if seeds is None:
        seeds = list(range(n_runs))

    good_sets = []
    bad_sets = []

    with open(dataset_path, 'r', encoding='utf-8') as f:
        all_data = json.load(f)

    benign_pool = [x for x in all_data if x.get('label', 0) == 0 and x.get('source_code', '').strip()]
    malware_pool = [x for x in all_data if x.get('label', 0) == 1 and x.get('source_code', '').strip()]

    for run_idx, seed in enumerate(seeds):
        print(f"\n--- Stability run {run_idx+1}/{n_runs} (seed={seed}) ---")
        rng = random.Random(seed)

        spc = max_samples // 2
        subset_benign = rng.sample(benign_pool, min(spc, len(benign_pool)))
        subset_malware = rng.sample(malware_pool, min(spc, len(malware_pool)))
        subset = subset_benign + subset_malware

        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
                json.dump(subset, tmp)
                tmp_path = tmp.name

            result = calculate_ace_scores(
                model, tokenizer, tmp_path,
                max_samples=max_samples, z=z, m=m, use_ig=use_ig
            )
            scores = result[0] if isinstance(result, tuple) else result

            good_neurons, bad_neurons = identify_good_bad_neurons(scores, top_k=top_k)
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)

        good_sets.append(frozenset((n['layer'], n['neuron_idx']) for n in good_neurons))
        bad_sets.append(frozenset((n['layer'], n['neuron_idx']) for n in bad_neurons))

    def _jaccard(a, b):
        if not a and not b:
            return 1.0
        inter = len(a & b)
        union = len(a | b)
        return inter / union if union > 0 else 0.0

    pairs = list(itertools.combinations(range(n_runs), 2))
    good_j = [_jaccard(good_sets[i], good_sets[j]) for i, j in pairs]
    bad_j = [_jaccard(bad_sets[i], bad_sets[j]) for i, j in pairs]

    result = {
        'good_jaccard_mean': float(np.mean(good_j)) if good_j else 0.0,
        'good_jaccard_std': float(np.std(good_j)) if good_j else 0.0,
        'bad_jaccard_mean': float(np.mean(bad_j)) if bad_j else 0.0,
        'bad_jaccard_std': float(np.std(bad_j)) if bad_j else 0.0,
        'good_sets': [list(s) for s in good_sets],
        'bad_sets': [list(s) for s in bad_sets],
        'pairwise_good_jaccard': good_j,
        'pairwise_bad_jaccard': bad_j,
        'n_runs': n_runs,
        'seeds': seeds,
    }

    print(f"\nNeuron Stability over {n_runs} runs:")
    print(f"  Good neuron Jaccard: {result['good_jaccard_mean']:.3f} +/- {result['good_jaccard_std']:.3f}")
    print(f"  Bad  neuron Jaccard: {result['bad_jaccard_mean']:.3f} +/- {result['bad_jaccard_std']:.3f}")
    print("  (1.0 = perfectly stable, 0.0 = no overlap)")

    return result


def _run_single_intervention(
    model,
    tokenizer,
    test_prompts: list,
    identified_neurons: dict,
    good_scale: float,
    bad_scale: float,
    max_test: int = 200,
):
    """
    Run a single intervention with arbitrary scale factors using UpProjPatch +
    logit-based evaluation.

    Returns (accuracy, confusion_matrix, per_sample_results).
    """
    good_neurons = identified_neurons["good_neurons"]
    bad_neurons = identified_neurons["bad_neurons"]

    good_map: Dict[int, list] = defaultdict(list)
    bad_map: Dict[int, list] = defaultdict(list)
    for n in good_neurons:
        good_map[int(n["layer"].split("_")[1])].append(n["neuron_idx"])
    for n in bad_neurons:
        bad_map[int(n["layer"].split("_")[1])].append(n["neuron_idx"])

    patched = patch_up_proj_layers(model, good_map, bad_map,
                                    good_scale=good_scale, bad_scale=bad_scale, mask_idx=-1)

    correct = 0
    total = 0
    confusion = np.zeros((2, 3), dtype=int)
    label_to_row = {"A": 0, "B": 1}
    per_sample_results = []

    for sample in tqdm(test_prompts[:max_test], desc=f"scale g*{good_scale} b*{bad_scale}", leave=False):
        prompt = sample["prompt"]
        expected = sample["correct_answer_token"].strip().upper()[0]
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model(**inputs)
            last_logits = outputs.logits[0, -1, :]
            predicted, probs = predict_from_logits(last_logits, tokenizer)

        cat = 0 if predicted == expected else (1 if predicted in ["A", "B"] else 2)
        row = label_to_row.get(expected)
        if row is not None:
            confusion[row, cat] += 1
        per_sample_results.append({
            "expected": expected,
            "predicted": predicted,
            "correct": (cat == 0),
        })
        if cat == 0:
            correct += 1
        total += 1

    unpatch_up_proj_layers(model, patched)

    accuracy = correct / total if total > 0 else 0.0
    return accuracy, confusion, per_sample_results


def run_hyperparameter_sweep(
    model,
    tokenizer,
    dataset_path: str,
    test_prompts: list,
    output_dir: str,
    base_ig_steps: int = 16,
    base_top_k: int = 100,
    base_max_samples: int = 100,
):
    """
    Ablation sweep over key hyperparameters.
    Each dimension is varied independently while all others stay at base values.

    Dimensions:
      - tr   : training set size {10, 50, 100, 200}
      - m    : IG integration steps {8, 16, 32}
      - K    : neurons selected {50, 100, 200}
      - nq   : proxy questions per example {1, 3}
    """
    sweep_dir = os.path.join(output_dir, "ablation_sweep")
    os.makedirs(sweep_dir, exist_ok=True)

    sweep_configs = []

    for tr in [10, 50, 100, 200]:
        sweep_configs.append({
            "dim": "training_size", "value": tr,
            "label": f"tr{tr}",
            "max_samples": tr,
            "m": base_ig_steps,
            "top_k": base_top_k,
            "num_proxy": 3,
        })

    for m_val in [8, 16, 32]:
        if m_val == base_ig_steps:
            continue
        sweep_configs.append({
            "dim": "ig_steps", "value": m_val,
            "label": f"m{m_val}",
            "max_samples": base_max_samples,
            "m": m_val,
            "top_k": base_top_k,
            "num_proxy": 3,
        })

    for k in [50, 100, 200]:
        if k == base_top_k:
            continue
        sweep_configs.append({
            "dim": "neuron_k", "value": k,
            "label": f"K{k}",
            "max_samples": base_max_samples,
            "m": base_ig_steps,
            "top_k": k,
            "num_proxy": 3,
        })

    for nq in [1, 3]:
        if nq == 3:
            continue
        sweep_configs.append({
            "dim": "proxy_questions", "value": nq,
            "label": f"nq{nq}",
            "max_samples": base_max_samples,
            "m": base_ig_steps,
            "top_k": base_top_k,
            "num_proxy": nq,
        })

    all_sweep_results = []

    for cfg in sweep_configs:
        label = cfg["label"]
        print(f"\n{'='*60}")
        print(f"SWEEP: {cfg['dim']} = {cfg['value']}  ({label})")
        print(f"{'='*60}")

        cfg_output_dir = os.path.join(sweep_dir, label)
        os.makedirs(cfg_output_dir, exist_ok=True)

        try:
            sweep_scores, sweep_train_idx = calculate_ace_scores(
                model, tokenizer, dataset_path,
                max_samples=cfg["max_samples"],
                z=5000,
                m=cfg["m"],
                use_ig=True,
                num_proxy_questions=cfg["num_proxy"],
            )

            sweep_good, sweep_bad = identify_good_bad_neurons(
                sweep_scores, top_k=cfg["top_k"]
            )
            sweep_identified = {
                "good_neurons": sweep_good,
                "bad_neurons": sweep_bad,
            }

            sweep_val = validate_neurons(
                model, tokenizer, test_prompts, sweep_identified,
                max_test=len(test_prompts)
            )

            sweep_val["sweep_config"] = cfg
            with open(os.path.join(cfg_output_dir, "sweep_result.json"), "w") as f:
                json.dump(sweep_val, f, indent=4)

            lo, hi = sweep_val.get("enhancer_ci", (0.0, 0.0))
            result_row = {
                "dim": cfg["dim"],
                "value": cfg["value"],
                "label": label,
                "baseline": sweep_val["baseline"],
                "enhancer": sweep_val["enhancer"],
                "enhancer_ci_lo": lo,
                "enhancer_ci_hi": hi,
                "degrader": sweep_val["degrader"],
            }
            all_sweep_results.append(result_row)
            print(f"  baseline={sweep_val['baseline']:.2%}  enhancer={sweep_val['enhancer']:.2%}  "
                  f"CI=[{lo:.2%}, {hi:.2%}]")

        except Exception as e:
            print(f"  ERROR in sweep config {label}: {e}")
            traceback.print_exc()
            all_sweep_results.append({"label": label, "error": str(e)})

    print(f"\n{'='*70}")
    print("HYPERPARAMETER SWEEP SUMMARY")
    print(f"{'='*70}")
    print(f"  {'Config':<20} {'Baseline':>10} {'Enhancer':>10} {'CI Lo':>8} {'CI Hi':>8}")
    print(f"  {'-'*20} {'-'*10} {'-'*10} {'-'*8} {'-'*8}")
    for row in all_sweep_results:
        if "error" in row:
            print(f"  {row['label']:<20}  ERROR: {row['error']}")
        else:
            print(f"  {row['label']:<20} {row['baseline']:>9.1%} {row['enhancer']:>9.1%} "
                  f"{row['enhancer_ci_lo']:>7.1%} {row['enhancer_ci_hi']:>7.1%}")

    sweep_summary_file = os.path.join(sweep_dir, "sweep_summary.json")
    with open(sweep_summary_file, "w") as f:
        json.dump(all_sweep_results, f, indent=4)
    print(f"\nSaved sweep summary to: {sweep_summary_file}")

    return all_sweep_results


# ============================================================================
# [SECTION 12] Main pipeline
# ============================================================================
DATASET_PATH = "merged_dataset.json"
OUTPUT_DIR = "nr/outputs"

NC_STEPS = 10  # m integration steps
USE_IG = True

os.makedirs(OUTPUT_DIR, exist_ok=True)

if not os.path.exists(DATASET_PATH):
    print(f"Dataset not found: {DATASET_PATH}")
    print("Please ensure the dataset exists at the specified path.")
    exit(1)

_pipeline_start = _time.time()

max_samples_used = 20

# Metadata
_eval_metadata = {
    "model_name": model_name,
    "quantization": "none (full bfloat16)",
    "ig_steps": NC_STEPS,
    "use_integrated_gradients": USE_IG,
    "method": "per_layer_ig_on_up_proj",
    "target_function": "logit_margin_logsumexp_combined_tokens",
    "max_code_chars_in_prompt": 2000,
    "max_training_samples": max_samples_used,
    "max_tokenizer_length": 2048,
    "top_k_neurons": 100,
    "z_coarse_selection": 5000,
    "num_proxy_questions": 3,
    "scale_good": 2.0,
    "scale_bad": 0.0,
    "gpu_count": torch.cuda.device_count(),
    "gpu_names": [torch.cuda.get_device_name(i)
                  for i in range(torch.cuda.device_count())]
                 if torch.cuda.is_available() else [],
    "gpu_vram_gb": [round(torch.cuda.get_device_properties(i).total_memory / 1e9, 2)
                    for i in range(torch.cuda.device_count())]
                   if torch.cuda.is_available() else [],
    "dataset_path": DATASET_PATH,
    "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
    "ig_attribution_target": "up_proj_output",
    "ig_layer_isolation": "per_layer",
    "softmax_mode": "logit_margin",
    "evaluation_mode": "logit_argmax",
    "filter_ambiguous_neurons": True,
    "intervention_site": "up_proj_output",
}

# ── STEP 1: ACE scoring ──────────────────────────────────────────────────────
print("\n" + "="*70)
print("STEP 1: ACE SCORING — PER-LAYER IG ON up_proj + LOGIT MARGIN F")
print("="*70)

scores, train_indices = calculate_ace_scores(
    model,
    tokenizer,
    DATASET_PATH,
    max_samples=max_samples_used,
    z=5000,
    sharing_percentage=0.5,
    m=NC_STEPS,
    use_ig=USE_IG,
    filter_ambiguous=True,
)

# ── STEP 1b: Truncation report ───────────────────────────────────────────────
trunc_summary = report_truncation_stats(tag="after ACE scoring")
_eval_metadata["truncation_stats_after_ace"] = trunc_summary

# ── STEP 2: CNI neuron identification ────────────────────────────────────────
print("\n" + "="*70)
print("STEP 2: CNI - CONTRASTIVE NEURON IDENTIFICATION")
print("="*70)

good_neurons, bad_neurons = identify_good_bad_neurons(
    scores,
    top_k=100
)

identified_neurons = {
    "good_neurons": good_neurons,
    "bad_neurons": bad_neurons
}

output_file = os.path.join(OUTPUT_DIR, "identified_neurons.json")
with open(output_file, 'w') as f:
    json.dump(identified_neurons, f, indent=4)
print(f"\nSaved neurons to: {output_file}")

# ── STEP 2b: Neuron stability ─────────────────────────────────────────────────
print("\n" + "="*70)
print("STEP 2b: NEURON STABILITY ANALYSIS (5 random subsets)")
print("="*70)
stability_results = compute_neuron_stability(
    model, tokenizer, DATASET_PATH,
    n_runs=5, max_samples=max_samples_used,
    top_k=100, z=5000, m=NC_STEPS, use_ig=USE_IG
)
stability_file = os.path.join(OUTPUT_DIR, "neuron_stability.json")
with open(stability_file, 'w') as f:
    json.dump(stability_results, f, indent=4)
print(f"Saved stability results to: {stability_file}")

# ── STEP 3: Validation ──────────────────────────────────────────────────────
print("\n" + "="*70)
print("STEP 3: VALIDATION")
print("="*70)

with open(DATASET_PATH) as f:
    all_data = json.load(f)

# Build held-out test set: exclude any sample used in training
train_index_set = set(train_indices)

for i, item in enumerate(all_data):
    item['_global_idx'] = i

test_benign = [item for item in all_data
               if item.get('label', 0) == 0
               and item.get('source_code', '').strip()
               and item.get('_global_idx') not in train_index_set]

test_malware = [item for item in all_data
                if item.get('label', 0) == 1
                and item.get('source_code', '').strip()
                and item.get('_global_idx') not in train_index_set]

print(f"\nHeld-out test pool (training samples excluded): "
      f"{len(test_benign)} benign, {len(test_malware)} malware")

samples_per_class = min(len(test_benign), len(test_malware))
balanced_test_data = test_benign[:samples_per_class] + test_malware[:samples_per_class]

_rnd2 = random.Random(42)
_rnd2.shuffle(balanced_test_data)

print(f"Using {samples_per_class} samples per class = {len(balanced_test_data)} total balanced test samples")

test_prompts = []
for item in balanced_test_data:
    source_code = item.get('source_code', '')
    label = item.get('label', 0)

    prompt, answer_mapping, options = create_malware_prompt(source_code, question_variant=0)
    correct_answer = answer_mapping[label]

    test_prompts.append({
        'prompt': prompt,
        'correct_answer_token': correct_answer,
        'source_code': source_code,
        'label': label,
        '_global_idx': item.get('_global_idx'),
    })

test_labels = {}
for sample in test_prompts:
    expected = sample['correct_answer_token'].strip().upper()[0]
    test_labels[expected] = test_labels.get(expected, 0) + 1

print("\nTest set distribution:")
print(f"  Expected A (benign): {test_labels.get('A', 0)}")
print(f"  Expected B (malware): {test_labels.get('B', 0)}")

# ── STEP 3a: Baseline accuracy ───────────────────────────────────────────────
print("\n" + "="*70)
print("STEP 3a: BASELINE TEST (before neuron intervention)")
print("="*70)
baseline_acc, pred_dist = test_baseline_accuracy(model, tokenizer, test_prompts, max_test=len(test_prompts))

# ── STEP 3a-ii: TF-IDF baseline ──────────────────────────────────────────────
print("\n" + "="*70)
print("STEP 3a-ii: TF-IDF + LR BASELINE (same train/test split)")
print("="*70)
tfidf_results = run_tfidf_baseline(
    dataset_path=DATASET_PATH,
    train_indices=train_indices,
    test_indices=[item['_global_idx'] for item in balanced_test_data],
    output_dir=OUTPUT_DIR,
)

# ── STEP 3b: Neuron intervention validation ──────────────────────────────────
print("\n" + "="*70)
print("STEP 3b: NEURON INTERVENTION VALIDATION")
print("="*70)

validation_results = validate_neurons(
    model,
    tokenizer,
    test_prompts,
    identified_neurons,
    max_test=len(test_prompts)
)

validation_file = os.path.join(OUTPUT_DIR, "validation_results.json")
with open(validation_file, 'w') as f:
    json.dump(validation_results, f, indent=4)
print(f"\nSaved validation results to: {validation_file}")

# Report truncation stats after validation
trunc_summary_final = report_truncation_stats(tag="after validation")
_eval_metadata["truncation_stats_final"] = trunc_summary_final

# ── STEP 4: Ablation — IG vs. single-step ────────────────────────────────────
ABLATION_METHODS = [
    {"use_ig": True,  "label": "IntegratedGradients_m16"},
    {"use_ig": False, "label": "SingleStepGradient"},
]

all_ablation_results = {}

for method_cfg in ABLATION_METHODS:
    method_label = method_cfg["label"]
    print(f"\n{'#'*70}")
    print(f"# ABLATION: {method_label}")
    print(f"{'#'*70}")

    method_output_dir = os.path.join(OUTPUT_DIR, method_label)
    os.makedirs(method_output_dir, exist_ok=True)

    abl_scores, abl_train_indices = calculate_ace_scores(
        model, tokenizer, DATASET_PATH,
        max_samples=max_samples_used, z=5000,
        m=NC_STEPS, use_ig=method_cfg["use_ig"]
    )

    abl_good_neurons, abl_bad_neurons = identify_good_bad_neurons(abl_scores, top_k=100)

    abl_identified = {"good_neurons": abl_good_neurons, "bad_neurons": abl_bad_neurons}
    with open(os.path.join(method_output_dir, "identified_neurons.json"), 'w') as f:
        json.dump(abl_identified, f, indent=4)

    abl_val_results = validate_neurons(
        model, tokenizer, test_prompts, abl_identified,
        max_test=len(test_prompts)
    )

    abl_val_results['method'] = method_label
    with open(os.path.join(method_output_dir, "validation_results.json"), 'w') as f:
        json.dump(abl_val_results, f, indent=4)

    all_ablation_results[method_label] = abl_val_results
    print(f"\n{method_label} -- Enhancer accuracy: {abl_val_results['enhancer']:.2%}")

print("\n" + "="*70)
print("ABLATION SUMMARY: IG vs Single-Step Gradient")
print("="*70)
for label, res in all_ablation_results.items():
    lo, hi = res.get('enhancer_ci', (0, 0))
    print(f"  {label:35s}: baseline={res['baseline']:.2%}  enhancer={res['enhancer']:.2%}  "
          f"CI=[{lo:.2%}, {hi:.2%}]")

ablation_file = os.path.join(OUTPUT_DIR, "ablation_ig_vs_singlestep.json")
with open(ablation_file, 'w') as f:
    json.dump(all_ablation_results, f, indent=4)
print(f"\nSaved ablation results to: {ablation_file}")

# ── STEP 5: Ablation — scale factors ────────────────────────────────────────
print("\n" + "="*70)
print("ABLATION 2: SCALE-FACTOR SENSITIVITY")
print("="*70)

ABLATION_SCALE_FACTORS = [
    {"good_scale": 1.5, "bad_scale": 0.0, "label": "enhancer_g1.5_b0.0"},
    {"good_scale": 2.0, "bad_scale": 0.0, "label": "enhancer_g2.0_b0.0"},
    {"good_scale": 3.0, "bad_scale": 0.0, "label": "enhancer_g3.0_b0.0"},
    {"good_scale": 5.0, "bad_scale": 0.0, "label": "enhancer_g5.0_b0.0"},
    {"good_scale": 0.0, "bad_scale": 1.5, "label": "degrader_g0.0_b1.5"},
    {"good_scale": 0.0, "bad_scale": 2.0, "label": "degrader_g0.0_b2.0"},
    {"good_scale": 0.0, "bad_scale": 3.0, "label": "degrader_g0.0_b3.0"},
]

scale_ablation_results = {}

for sf_cfg in ABLATION_SCALE_FACTORS:
    sf_label = sf_cfg["label"]
    print(f"\n  Scale ablation: {sf_label} ...")
    sf_acc, sf_conf, sf_per_sample = _run_single_intervention(
        model, tokenizer, test_prompts,
        identified_neurons,
        good_scale=sf_cfg["good_scale"],
        bad_scale=sf_cfg["bad_scale"],
        max_test=len(test_prompts)
    )
    n_total = len(test_prompts)
    sf_lo, sf_hi = wilson_ci(int(round(sf_acc * n_total)), n_total)
    scale_ablation_results[sf_label] = {
        "good_scale": sf_cfg["good_scale"],
        "bad_scale": sf_cfg["bad_scale"],
        "accuracy": sf_acc,
        "ci_95": [sf_lo, sf_hi],
    }
    print(f"    Accuracy: {sf_acc:.2%}  CI [{sf_lo:.2%}, {sf_hi:.2%}]")

print("\n" + "="*70)
print("SCALE-FACTOR ABLATION SUMMARY")
print("="*70)
for lbl, r in scale_ablation_results.items():
    lo, hi = r["ci_95"]
    print(f"  {lbl:30s}: {r['accuracy']:.2%}  CI [{lo:.2%}, {hi:.2%}]")

scale_abl_file = os.path.join(OUTPUT_DIR, "ablation_scale_factors.json")
with open(scale_abl_file, 'w') as f:
    json.dump(scale_ablation_results, f, indent=4)
print(f"\nSaved scale-factor ablation to: {scale_abl_file}")

# ── STEP 5b: Ablation — ambiguous filter on/off ──────────────────────────────
print("\n" + "="*70)
print("ABLATION 3: AMBIGUOUS NEURON FILTER (ON vs OFF)")
print("="*70)

ambiguous_ablation_results = {}

for filter_flag in [True, False]:
    filter_label = f"filter_ambiguous_{filter_flag}"
    print(f"\n  {filter_label} ...")

    amb_scores, amb_train_idx = calculate_ace_scores(
        model, tokenizer, DATASET_PATH,
        max_samples=max_samples_used, z=5000,
        m=NC_STEPS, use_ig=USE_IG,
        filter_ambiguous=filter_flag,
    )

    amb_good, amb_bad = identify_good_bad_neurons(amb_scores, top_k=100)
    amb_identified = {"good_neurons": amb_good, "bad_neurons": amb_bad}

    amb_val = validate_neurons(
        model, tokenizer, test_prompts, amb_identified,
        max_test=len(test_prompts)
    )

    ambiguous_ablation_results[filter_label] = amb_val
    print(f"  {filter_label}: baseline={amb_val['baseline']:.2%}  enhancer={amb_val['enhancer']:.2%}")

amb_abl_file = os.path.join(OUTPUT_DIR, "ablation_ambiguous_filter.json")
with open(amb_abl_file, 'w') as f:
    json.dump(ambiguous_ablation_results, f, indent=4)
print(f"\nSaved ambiguous-filter ablation to: {amb_abl_file}")

# ── STEP 6: Hyperparameter sweep ─────────────────────────────────────────────
print("\n" + "="*70)
print("STEP 6: HYPERPARAMETER SWEEP")
print("="*70)
sweep_results = run_hyperparameter_sweep(
    model, tokenizer, DATASET_PATH, test_prompts,
    output_dir=OUTPUT_DIR,
    base_ig_steps=NC_STEPS,
    base_top_k=100,
    base_max_samples=max_samples_used,
)

# ── STEP 7: Save metadata ─────────────────────────────────────────────────────
_eval_metadata["wall_clock_seconds"] = round(_time.time() - _pipeline_start, 1)
if torch.cuda.is_available():
    _eval_metadata["peak_vram_gb"] = [
        round(torch.cuda.max_memory_allocated(i) / 1e9, 2)
        for i in range(torch.cuda.device_count())
    ]

meta_file = os.path.join(OUTPUT_DIR, "eval_metadata.json")
with open(meta_file, 'w') as f:
    json.dump(_eval_metadata, f, indent=4)
print(f"\nSaved evaluation metadata to: {meta_file}")

print("\n" + "="*70)
print("PIPELINE COMPLETE!")
print("="*70)
