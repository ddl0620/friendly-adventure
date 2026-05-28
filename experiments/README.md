# Experiments

Neuron attribution (2 methods × 3 models) plus the plotting scripts that
reproduce the 4 result figures in the paper.

- **Methods:** IG (Integrated Gradients), NC (Neuron Conductance)
- **Models:** Llama-3.1-8B-Instruct, Mistral-7B-Instruct-v0.3, Qwen2.5-7B-Instruct

## Reproduce the figures (no GPU)

Uses the precomputed outputs in `results/`.

```bash
pip install matplotlib numpy
cd plots
python plot_intervention_effects.py        # -> figures/intervention_effects_f1_delta.png
python plot_stability_jaccard_violin.py     # -> figures/stability_merged.png
python plot_stability_cosine_violin.py      # -> figures/stability_cosine_merged.png
python plot_layer_distribution_grid.py      # -> figures/layer_distribution_grid.png
```

## Re-run attribution (GPU)

```bash
pip install -r requirements.txt
# 1. build merged_dataset.json (see ../data_collection) and place it in the run dir
# 2. edit `model_name` near the top of the script to one of the 3 models
python attribution/nr_model_ig_upproj_logit_margin.py     # IG  method
python attribution/nr_model_conductance_logit_margin.py   # NC  method
# -> writes nr/outputs/{identified_neurons,neuron_stability,validation_results}.json
```

## Layout

- `attribution/` — 2 attribution scripts (model selected via `model_name`)
- `results/` — precomputed outputs per model × method (6 runs)
- `plots/` — 4 figure scripts
- `figures/` — generated figures
