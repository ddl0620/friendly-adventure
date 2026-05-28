"""
Per-layer distribution of the top good (blue) vs bad (orange) neurons,
as a 3 (models) x 2 (methods) grid. Reproduces the layer_distribution_grid
figure in the paper.

Reads identified_neurons.json from ../results and writes
../figures/layer_distribution_grid.png
"""

import json
import os

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(BASE, "..", "results")
OUT_DIR = os.path.join(BASE, "..", "figures")

# IG neurons live under IntegratedGradients_m16/, NC at the outputs root.
OUTPUTS = "orkspace/nr/outputs"
cfgs = {
    "Llama":   {"L": 32,
                "IG": f"{RESULTS}/ig_llama3.1-ig/{OUTPUTS}/IntegratedGradients_m16/identified_neurons.json",
                "NC": f"{RESULTS}/nc_llama/{OUTPUTS}/identified_neurons.json"},
    "Mistral": {"L": 32,
                "IG": f"{RESULTS}/ig_mistralv0.3-7b-ig/{OUTPUTS}/IntegratedGradients_m16/identified_neurons.json",
                "NC": f"{RESULTS}/nc_mistral/{OUTPUTS}/identified_neurons.json"},
    "Qwen":    {"L": 28,
                "IG": f"{RESULTS}/ig_qwen2.5-ig/{OUTPUTS}/IntegratedGradients_m16/identified_neurons.json",
                "NC": f"{RESULTS}/nc_qwen/{OUTPUTS}/identified_neurons.json"},
}

C_GOOD = "#3B7DD8"
C_BAD = "#E5754F"


def main():
    fig, axes = plt.subplots(3, 2, figsize=(8.5, 5.2), sharex='row', sharey='all')

    for row, (model, info) in enumerate(cfgs.items()):
        L = info["L"]
        for col, method in enumerate(["IG", "NC"]):
            with open(info[method]) as f:
                d = json.load(f)
            good = [int(n["layer"].split("_")[1]) for n in d["good_neurons"]]
            bad = [int(n["layer"].split("_")[1]) for n in d["bad_neurons"]]
            bins = np.arange(0, L + 1) - 0.5
            ax = axes[row, col]
            ax.hist([good, bad], bins=bins, label=["good", "bad"],
                    color=[C_GOOD, C_BAD], edgecolor="white", linewidth=0.3,
                    stacked=False)
            ax.set_xlim(-0.5, L - 0.5)
            ax.tick_params(axis="both", labelsize=8)
            ax.grid(axis="y", linewidth=0.3, alpha=0.4)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            if row == 0:
                ax.set_title(method, fontsize=10, weight="bold", pad=4)
            if col == 0:
                ax.set_ylabel(f"{model}\n({L}L)", fontsize=9.5, weight="bold",
                              rotation=0, ha="right", va="center", labelpad=22)
            if row == 2:
                ax.set_xlabel("layer index", fontsize=9)

    fig.legend(handles=[
        mpatches.Patch(facecolor=C_GOOD, edgecolor="white", label="good neurons"),
        mpatches.Patch(facecolor=C_BAD, edgecolor="white", label="bad neurons"),
    ], loc="upper center", ncol=2, fontsize=9.5, frameon=False,
        bbox_to_anchor=(0.5, 1.02))

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, "layer_distribution_grid.png")
    plt.savefig(out, dpi=200, bbox_inches='tight')
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
