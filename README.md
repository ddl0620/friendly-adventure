# Replication Package for "Which Neurons Detect Malicious Code? A Probing Study of LLM Security Knowledge"

## Repository Structure

```text
package/
|- data_collection/        # builds the 3,000-record dataset (see its README)
|  |- fetchpypi.py          # download benign sdists from live PyPI
|  |- collect_benign.py     # BenignSet/ -> benign.json
|  |- collect_malicious.py  # pypi_malregistry/ -> malicious.json
|  |- sample_data.py        # length-matched balanced merge
|  \- merged_dataset.json   # dataset used in the paper (~39 MB)
\- experiments/            # attribution + figures (see its README)
   |- attribution/          # IG and NC pipelines
   |- results/              # precomputed outputs: 6 runs (2 methods x 3 models)
   |- plots/                # Figure scripts
   \- figures/              # the generated figures
```

## Dataset

`data_collection/merged_dataset.json` — 3,000 records, each with `package_name`, `version`, `filename`, `source_code`, and `label` (`0` = benign, `1` = malicious).

- Balanced: 1,500 benign + 1,500 malicious.
- Length-matched so the probe can't exploit trivial code-length cues.
- Malicious samples are real packages from [PyPI Malregistry](https://github.com/lxyeternal/pypi_malregistry).

## Models

| Short name | Hugging Face ID |
| --- | --- |
| Llama   | `meta-llama/Meta-Llama-3.1-8B-Instruct` |
| Mistral | `mistralai/Mistral-7B-Instruct-v0.3` |
| Qwen    | `Qwen/Qwen2.5-7B-Instruct` |

All three are gated — set `HF_TOKEN` to a token with access before re-running attribution.

## Quick Start

### Reproduce the figures (no GPU)

Uses the committed outputs in `experiments/results/`.

```bash
pip install matplotlib numpy
cd experiments/plots
python plot_intervention_effects.py
python plot_stability_jaccard_violin.py
python plot_stability_cosine_violin.py
python plot_layer_distribution_grid.py
```

### Re-run attribution (GPU)

```bash
cd experiments
pip install -r requirements.txt
export HF_TOKEN=hf_...
# edit `model_name` near the top of the script to one of the 3 models
python attribution/nr_model_ig_upproj_logit_margin.py     # IG
python attribution/nr_model_conductance_logit_margin.py   # NC
```

Each run writes `identified_neurons.json` (ranked neurons), `validation_results.json` (intervention F1 + CIs, McNemar tests), `neuron_stability.json`, plus ablation files and `eval_metadata.json`. Headline runs used a single NVIDIA A40 (48 GB), bfloat16.

To rebuild the dataset from scratch, see [data_collection/README.md](data_collection/README.md). Full run order is in [experiments/README.md](experiments/README.md).

## Ethics and Safe Handling

**This dataset contains real, functional malware** (credential stealers, droppers, obfuscated payload loaders), released **only** for defensive research and reproducing this study. Please never to author or distribute malware.

Samples are inert as plain text and do not run when read or parsed, but become dangerous if reconstructed. So:

- **Never** install, build, import, or execute any package or snippet from these records, and do not fetch the payloads/URLs they reference (endpoints may still be live).
- Inspect samples only in an isolated, network-restricted environment with no access to credentials or production systems.

Malicious samples come from [PyPI Malregistry](https://github.com/lxyeternal/pypi_malregistry) and are redistributed under its terms for research. Please responsibly and lawfully handle per your institution's policies.
