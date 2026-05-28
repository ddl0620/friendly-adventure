"""
Intervention-effects bar chart: per-class F1 delta vs baseline.

Single combined figure: 3 columns (LLaMA, Mistral, Qwen) x 2 rows (IG, NC).
Each panel shows the 4 interventions (Suppress Good, Suppress Bad, Enhancer,
Degrader) on the x-axis with two bars per intervention: Delta-F1 for benign
and Delta-F1 for malware vs the baseline of that (model, method).

Designed for academic publication: muted palette, light grid, no background
tints. Baseline values are intentionally left out of the chart titles and
should be reported in the figure caption.
"""

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).parent
RESULTS = BASE_DIR / ".." / "results"

RESULT_DIRS = {
    "LLaMA 3.1":    {"IG": "ig_llama3.1-ig",       "NC": "nc_llama"},
    "Qwen 2.5":     {"IG": "ig_qwen2.5-ig",        "NC": "nc_qwen"},
    "Mistral v0.3": {"IG": "ig_mistralv0.3-7b-ig", "NC": "nc_mistral"},
}

JSON_REL = "orkspace/nr/outputs/validation_results.json"

MODELS = ["LLaMA 3.1", "Mistral v0.3", "Qwen 2.5"]
METHODS = [("IG", "IG"), ("NC", "NC")]

INTERVENTIONS = ["suppress_good", "suppress_bad", "enhancer", "degrader"]
INTERVENTION_LABELS = ["Supp.\nGood", "Supp.\nBad", "Enh.", "Deg."]

CLASSES = [
    ("benign (A)", "Benign", "#4C72B0"),
    ("malware (B)", "Malware", "#C44E52"),
]

METRIC_KEY = "f1"
METRIC_LABEL = "F1"


def load_per_class(folder):
    with open(RESULTS / folder / JSON_REL) as f:
        data = json.load(f)
    return data["per_class_metrics"]


def deltas_for_class(pcm, cls_key):
    base = pcm.get("baseline", {}).get(cls_key, {}).get(METRIC_KEY, 0.0)
    return [pcm.get(cond, {}).get(cls_key, {}).get(METRIC_KEY, 0.0) - base
            for cond in INTERVENTIONS]


YLIMS = {
    "LLaMA 3.1":    (-1.00, 0.25),
    "Mistral v0.3": (-0.25, 0.60),
    "Qwen 2.5":     (-0.25, 0.25),
}
YTICKS = {
    "LLaMA 3.1":    [-1.0, -0.75, -0.5, -0.25, 0.0, 0.25],
    "Mistral v0.3": [-0.25, 0.0, 0.25, 0.5],
    "Qwen 2.5":     [-0.25, 0.0, 0.25],
}


def plot_panel(ax, pcm, model_name, method_title=None, show_xlabels=True):
    n = len(INTERVENTIONS)
    bar_width = 0.36
    pos = np.arange(n)
    ylim_lo, ylim_hi = YLIMS[model_name]
    span = ylim_hi - ylim_lo

    for c_idx, (cls_key, cls_label, color) in enumerate(CLASSES):
        deltas = deltas_for_class(pcm, cls_key)
        offset = (c_idx - 0.5) * bar_width
        ax.bar(
            pos + offset, deltas,
            width=bar_width, color=color, label=cls_label,
            edgecolor="none", zorder=3,
        )
        for x, d in zip(pos + offset, deltas):
            va = "bottom" if d >= 0 else "top"
            pad = 0.035 * span / 1.86  # scale label offset to each panel's span
            pad = pad if d >= 0 else -pad
            txt = f"{d:+.2f}" if abs(d) >= 0.005 else "0"
            ax.text(
                x, d + pad, txt,
                ha="center", va=va, fontsize=6, color="#222222",
                zorder=4,
            )

    ax.axhline(y=0, color="#222222", linewidth=1.0, zorder=2)
    ax.set_xticks(pos)
    if show_xlabels:
        ax.set_xticklabels(INTERVENTION_LABELS, fontsize=11)
    else:
        ax.set_xticklabels([])
    ax.set_ylim(ylim_lo, ylim_hi)
    ax.set_yticks(YTICKS[model_name])
    ax.tick_params(axis="y", labelsize=8)
    ax.tick_params(axis="x", length=0)
    ax.grid(axis="y", color="#dddddd", linewidth=0.7, zorder=1)
    ax.set_axisbelow(True)

    if method_title:
        ax.set_title(method_title, fontsize=14, fontweight="bold", pad=8)

    for spine_name in ("top", "right"):
        ax.spines[spine_name].set_visible(False)
    for spine_name in ("left", "bottom"):
        ax.spines[spine_name].set_color("#888888")
        ax.spines[spine_name].set_linewidth(0.9)


def main():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.edgecolor": "#888888",
        "axes.labelcolor": "#222222",
        "xtick.color": "#222222",
        "ytick.color": "#222222",
    })

    # Per-model y-spans, used as height_ratios so the data-to-pixel ratio is
    # identical across panels and bar thicknesses match visually. The y-range
    # for Mistral and Qwen is trimmed to the smallest interval that still
    # contains every bar, removing dead whitespace without rescaling anything.
    spans = [YLIMS[m][1] - YLIMS[m][0] for m in MODELS]
    total_span = sum(spans)
    base_height = 8.5  # original figure height for full -1.08..0.78 range x3
    new_height = base_height * total_span / (3 * spans[0])

    fig, axes = plt.subplots(
        len(MODELS), len(METHODS),
        figsize=(7.5, new_height),
        sharey='row',
        gridspec_kw={'height_ratios': spans},
    )

    n_rows = len(MODELS)
    for r, model in enumerate(MODELS):
        for c, (method_key, method_label) in enumerate(METHODS):
            ax = axes[r, c]
            folder = RESULT_DIRS[model][method_key]
            pcm = load_per_class(folder)
            title = method_label if r == 0 else None
            plot_panel(
                ax, pcm, model,
                method_title=title,
                show_xlabels=(r == n_rows - 1),
            )

    for r, model in enumerate(MODELS):
        axes[r, 0].set_ylabel(
            f"$\\Delta$F1\n{model}",
            fontsize=13, fontweight="bold", labelpad=8,
        )

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="lower center", ncol=2,
        bbox_to_anchor=(0.5, -0.01), fontsize=13, frameon=False,
        handlelength=1.3, handletextpad=0.6, columnspacing=2.0,
    )

    fig.tight_layout(rect=(0, 0.03, 1, 1.0), h_pad=1.5, w_pad=1.5)

    out_dir = BASE_DIR / ".." / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"intervention_effects_{METRIC_KEY}_delta.png"
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
