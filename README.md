# Replication Package

Code and data for our study on neuron attribution in LLMs for malicious PyPI package detection.

## Contents

- [`data_collection/`](data_collection/) — pipeline that builds `merged_dataset.json` (3,000 records: 1,500 benign + 1,500 malicious). See [data_collection/README.md](data_collection/README.md).
- [`experiments/`](experiments/) — neuron attribution (IG & NC $\times$ 3 models) and the scripts that reproduce the paper's figures. See [experiments/README.md](experiments/README.md).


Re-running attribution (GPU) downloads the models from Hugging Face, which are gated — set `HF_TOKEN` to a token with access:

```bash
export HF_TOKEN=hf_...
```

Each subfolder's README has the full run order and details.
