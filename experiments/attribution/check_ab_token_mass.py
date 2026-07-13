"""Check whether the model naturally assigns next-token mass to A/B answers.

This script reproduces the train/evaluation split used by
``nr_model_ig_upproj_logit_margin.py`` but does not run attribution or neuron
interventions. On the held-out evaluation split it reports:

1. Forced-choice A/B accuracy (the experiment's existing metric).
2. Full-vocabulary probability mass assigned to ``A``, `` A``, ``B``, `` B``.
3. How often the actual top next token is one of those accepted A/B tokens.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"
DEFAULT_DATASET = Path(__file__).resolve().parent / "merged_dataset.json"
SEED = 42
MAX_CODE_LENGTH = 2000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument(
        "--attribution-samples",
        type=int,
        default=20,
        help="Number reserved by the attribution split (same default as the experiment).",
    )
    parser.add_argument(
        "--max-eval-samples",
        type=int,
        default=None,
        help="Optional limit after constructing the same balanced evaluation split.",
    )
    parser.add_argument(
        "--show-samples",
        type=int,
        default=10,
        help="Number of per-sample diagnostic rows to print.",
    )
    parser.add_argument(
        "--show-other-tokens",
        type=int,
        default=5,
        help="Number of highest-probability non-A/B tokens shown per displayed sample.",
    )
    return parser.parse_args()


def create_malware_prompt(source_code: str) -> tuple[str, dict[int, str]]:
    """Reproduce question variant 0 used for held-out evaluation."""
    if len(source_code) > MAX_CODE_LENGTH:
        source_code = source_code[:MAX_CODE_LENGTH] + "\n... [truncated]"

    question = (
        "Analyze the following source code and determine if it is benign "
        "(safe) or malicious (malware)."
    )
    options = {
        "A": "Benign - The code appears to be safe and legitimate",
        "B": "Malware - The code contains malicious functionality",
    }
    answer_mapping = {0: "A", 1: "B"}

    prompt = f"""{question}

Source Code:
```
{source_code}
```

Options:
A: {options['A']}
B: {options['B']}

Answer:"""
    return prompt, answer_mapping


def reproduce_split(
    dataset: list[dict], attribution_samples: int
) -> tuple[list[int], list[dict]]:
    """Reproduce the attribution indices and balanced held-out evaluation data."""
    indexed_data = []
    for global_idx, original_item in enumerate(dataset):
        item = dict(original_item)
        item["_global_idx"] = global_idx
        indexed_data.append(item)

    attribution_data = indexed_data
    if attribution_samples and attribution_samples < len(indexed_data):
        benign = [item for item in indexed_data if item.get("label", 0) == 0]
        malware = [item for item in indexed_data if item.get("label", 0) == 1]

        samples_per_class = attribution_samples // 2
        class_rng = random.Random(SEED)
        class_rng.shuffle(benign)
        class_rng.shuffle(malware)

        attribution_data = (
            benign[: min(samples_per_class, len(benign))]
            + malware[: min(samples_per_class, len(malware))]
        )
        split_rng = random.Random(SEED)
        split_rng.shuffle(attribution_data)

    attribution_indices = [item["_global_idx"] for item in attribution_data]
    attribution_index_set = set(attribution_indices)

    test_benign = [
        item
        for item in indexed_data
        if item.get("label", 0) == 0
        and item.get("source_code", "").strip()
        and item["_global_idx"] not in attribution_index_set
    ]
    test_malware = [
        item
        for item in indexed_data
        if item.get("label", 0) == 1
        and item.get("source_code", "").strip()
        and item["_global_idx"] not in attribution_index_set
    ]

    samples_per_class = min(len(test_benign), len(test_malware))
    evaluation_data = (
        test_benign[:samples_per_class] + test_malware[:samples_per_class]
    )
    evaluation_rng = random.Random(SEED)
    evaluation_rng.shuffle(evaluation_data)

    return attribution_indices, evaluation_data


def build_option_token_map(tokenizer) -> dict[str, list[int]]:
    """Reproduce the experiment's bare + space-prefixed option mapping."""
    token_map: dict[str, list[int]] = {}
    for letter in ["A", "B"]:
        ids = set()
        bare_ids = tokenizer.encode(letter, add_special_tokens=False)
        space_ids = tokenizer.encode(" " + letter, add_special_tokens=False)

        if bare_ids:
            ids.add(bare_ids[0])
        if space_ids:
            ids.add(space_ids[-1])

        token_map[letter] = sorted(ids)
        print(
            f"{letter}: ids={token_map[letter]}, "
            f"decoded={[tokenizer.decode([token_id]) for token_id in token_map[letter]]}"
        )

        if len(bare_ids) != 1 or len(space_ids) != 1:
            print(
                f"WARNING: {letter} has a multi-token variant: "
                f"bare={bare_ids}, space-prefixed={space_ids}. "
                "The mapping above follows the original experiment exactly."
            )

    return token_map


def evaluate(
    model,
    tokenizer,
    evaluation_data: list[dict],
    show_samples: int,
    show_other_tokens: int,
) -> None:
    token_map = build_option_token_map(tokenizer)
    accepted_ids = sorted(set(token_map["A"] + token_map["B"]))

    correct = 0
    actual_top_is_accepted = 0
    actual_top_is_correct = 0
    option_masses: list[float] = []
    largest_other_probs: list[float] = []

    for sample_idx, item in enumerate(tqdm(evaluation_data, desc="Evaluating")):
        prompt, answer_mapping = create_malware_prompt(item["source_code"])
        expected = answer_mapping[item.get("label", 0)]
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            logits = model(**inputs).logits[0, -1, :].float()

        full_probs = F.softmax(logits, dim=-1)
        option_mass = full_probs[accepted_ids].sum().item()
        option_masses.append(option_mass)

        other_probs = full_probs.clone()
        other_probs[accepted_ids] = 0.0
        largest_other_prob = other_probs.max().item()
        largest_other_probs.append(largest_other_prob)

        option_lse = {
            letter: torch.logsumexp(logits[token_ids], dim=0).item()
            for letter, token_ids in token_map.items()
        }
        predicted = max(option_lse, key=option_lse.get)
        is_correct = predicted == expected
        correct += int(is_correct)

        actual_top_id = int(torch.argmax(logits).item())
        top_is_accepted = actual_top_id in accepted_ids
        actual_top_is_accepted += int(top_is_accepted)
        actual_top_is_correct += int(actual_top_id in token_map[expected])

        if sample_idx < show_samples:
            print(
                f"\n[{sample_idx}] dataset_idx={item['_global_idx']} "
                f"expected={expected} forced={predicted} correct={is_correct}\n"
                f"    LSE(A)={option_lse['A']:.6f} LSE(B)={option_lse['B']:.6f}\n"
                f"    A/B full-vocab mass={option_mass:.8f} "
                f"other mass={1.0 - option_mass:.8f}\n"
                f"    actual top id={actual_top_id} "
                f"token={tokenizer.decode([actual_top_id])!r} "
                f"accepted_A/B={top_is_accepted}"
            )

            if show_other_tokens > 0:
                count = min(show_other_tokens, other_probs.numel())
                top_other_probs, top_other_ids = torch.topk(other_probs, k=count)
                cumulative_top_other_mass = top_other_probs.sum().item()

                print(
                    f"    top {count} other tokens "
                    f"(cumulative mass={cumulative_top_other_mass:.8f}):"
                )
                for rank, (token_id, probability) in enumerate(
                    zip(top_other_ids.tolist(), top_other_probs.tolist()), start=1
                ):
                    print(
                        f"      {rank}. id={token_id:<7} "
                        f"token={tokenizer.decode([token_id])!r:<16} "
                        f"prob={probability:.8f} "
                        f"logit={logits[token_id].item():.6f}"
                    )

    total = len(evaluation_data)
    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print(f"Evaluation samples:                 {total}")
    print(f"Forced-choice A/B accuracy:        {correct / total:.2%} ({correct}/{total})")
    print(
        "Actual top token is accepted A/B: "
        f"{actual_top_is_accepted / total:.2%} ({actual_top_is_accepted}/{total})"
    )
    print(
        "Actual top token is correct A/B:  "
        f"{actual_top_is_correct / total:.2%} ({actual_top_is_correct}/{total})"
    )
    print(f"Mean full-vocab A/B mass:           {statistics.fmean(option_masses):.8f}")
    print(f"Median full-vocab A/B mass:         {statistics.median(option_masses):.8f}")
    print(f"Minimum full-vocab A/B mass:        {min(option_masses):.8f}")
    print(f"Maximum full-vocab A/B mass:        {max(option_masses):.8f}")
    print(
        f"Mean total other-token mass:        "
        f"{statistics.fmean(1.0 - mass for mass in option_masses):.8f}"
    )
    print(
        f"Mean largest single other token:    "
        f"{statistics.fmean(largest_other_probs):.8f}"
    )
    print(
        f"Maximum single other token:         "
        f"{max(largest_other_probs):.8f}"
    )
    for threshold in [0.50, 0.90, 0.95, 0.99]:
        count = sum(mass >= threshold for mass in option_masses)
        print(
            f"Samples with A/B mass >= {threshold:.2f}:    "
            f"{count / total:.2%} ({count}/{total})"
        )


def main() -> None:
    args = parse_args()
    with args.dataset.open("r", encoding="utf-8") as file:
        dataset = json.load(file)

    attribution_indices, evaluation_data = reproduce_split(
        dataset, args.attribution_samples
    )
    if args.max_eval_samples is not None:
        evaluation_data = evaluation_data[: args.max_eval_samples]

    if not evaluation_data:
        raise RuntimeError("The reproduced evaluation split is empty.")

    print(f"Dataset: {args.dataset}")
    print(
        f"Attribution split: {len(attribution_indices)} samples "
        f"(indices={attribution_indices})"
    )
    print(f"Balanced held-out evaluation split: {len(evaluation_data)} samples")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map="auto" if torch.cuda.is_available() else None,
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    model.eval()

    print(f"Model: {args.model}")
    print(f"Tokenizer: {type(tokenizer).__name__}")
    evaluate(
        model,
        tokenizer,
        evaluation_data,
        args.show_samples,
        args.show_other_tokens,
    )


if __name__ == "__main__":
    main()
