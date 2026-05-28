#!/usr/bin/env python3
"""Parallel of stability_merged.png but for layer-distribution cosine similarity.

Same layout (3 horizontal panels, 4 violins each: IG-Good, IG-Bad, NC-Good,
NC-Bad) and same palette as plot_stability_violin.py, so the two figures sit
side-by-side in the paper.
"""

import itertools
import json
import os
import re
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 16,
    'axes.labelsize': 18,
    'axes.titlesize': 20,
    'legend.fontsize': 14,
    'xtick.labelsize': 15,
    'ytick.labelsize': 15,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'axes.linewidth': 0.5,
    'lines.linewidth': 1.0,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linewidth': 0.4,
})

BASE = os.path.dirname(os.path.abspath(__file__))
EXTRACT_ROOT = os.path.join(BASE, '..', 'results')
OUT_ROOT = os.path.join(BASE, '..', 'figures')

CONFIGS = {
    'llama':   {'IG': 'ig_llama3.1-ig',      'NC': 'nc_llama',   'title': 'Llama',   'n_layers': 32},
    'mistral': {'IG': 'ig_mistralv0.3-7b-ig','NC': 'nc_mistral', 'title': 'Mistral', 'n_layers': 32},
    'qwen':    {'IG': 'ig_qwen2.5-ig',       'NC': 'nc_qwen',    'title': 'Qwen',    'n_layers': 28},
}

PALETTE = ['#E89BA6', '#9FD7A7', '#9CC5E2', '#C9B6E0']
LABELS = ['IG·Good', 'IG·Bad', 'NC·Good', 'NC·Bad']


def load_stability(folder):
    path = os.path.join(
        EXTRACT_ROOT, folder, 'orkspace', 'nr', 'outputs', 'neuron_stability.json'
    )
    with open(path) as f:
        return json.load(f)


def layer_idx(layer_str):
    m = re.search(r'(\d+)', layer_str)
    return int(m.group(1)) if m else -1


def layer_hist(run_set, n_layers):
    c = Counter(layer_idx(l) for (l, _) in run_set)
    return np.array([c.get(i, 0) for i in range(n_layers)], dtype=float)


def cosine_pairs(sets, n_layers):
    hists = [layer_hist(s, n_layers) for s in sets]
    pairs = list(itertools.combinations(range(len(hists)), 2))
    out = []
    for i, j in pairs:
        a, b = hists[i], hists[j]
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        out.append(float(a @ b / (na * nb)) if na and nb else 0.0)
    return out


def plot_panel(ax, data, title, show_ylabel):
    positions = list(range(1, len(data) + 1))
    parts = ax.violinplot(
        data, positions=positions, widths=0.85,
        showmeans=False, showmedians=False, showextrema=False,
    )
    for body, c in zip(parts['bodies'], PALETTE):
        body.set_facecolor(c)
        body.set_edgecolor('black')
        body.set_alpha(0.8)
        body.set_linewidth(0.5)

    ax.boxplot(
        data, positions=positions, widths=0.18,
        patch_artist=True, showfliers=False,
        boxprops=dict(facecolor='white', edgecolor='black', linewidth=0.6),
        medianprops=dict(color='black', linewidth=1.0),
        whiskerprops=dict(color='black', linewidth=0.6),
        capprops=dict(color='black', linewidth=0.6),
    )

    ax.set_xticks(positions)
    ax.set_xticklabels(LABELS)
    ax.set_xlim(0.4, len(data) + 0.6)
    ax.set_ylim(0.2, 1.0)
    ax.set_title(title, fontsize=22, fontweight='bold')
    if show_ylabel:
        ax.set_ylabel('Layer-dist. cosine')
    ax.grid(False)


def main():
    fig, axes = plt.subplots(1, len(CONFIGS), figsize=(15.0, 3.5), sharey=True)
    for i, (short, cfg) in enumerate(CONFIGS.items()):
        data = load_stability(cfg['IG'])
        data_nc = load_stability(cfg['NC'])
        n_layers = cfg['n_layers']
        panel_data = [
            cosine_pairs(data['good_sets'], n_layers),
            cosine_pairs(data['bad_sets'], n_layers),
            cosine_pairs(data_nc['good_sets'], n_layers),
            cosine_pairs(data_nc['bad_sets'], n_layers),
        ]
        plot_panel(axes[i], panel_data, cfg['title'], show_ylabel=(i == 0))

    fig.tight_layout()
    os.makedirs(OUT_ROOT, exist_ok=True)
    out_path = os.path.join(OUT_ROOT, 'stability_cosine_merged.png')
    fig.savefig(out_path)
    plt.close(fig)
    print(f'-> {out_path}')


if __name__ == '__main__':
    main()
