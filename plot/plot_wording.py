"""Experiment 2: fixed instruction-wording sensitivity on the shared baseline cases."""
from __future__ import annotations
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from plot.style import (
    ROOT, MODELS, MODEL_NAMES, CONDS, SHORT, LABELS, COLORS, VERSIONS,
    VERSION_NAMES, INK, plt, matplotlib, np, FancyBboxPatch, FancyArrowPatch,
    Patch, load, digest, require, records, style, paper_style, save, parse_args,
)
from atomic_memory import paths
from plot.plot_baseline import check_inputs

def wording_sensitivity(formal, robust, args):
    lookup = {(r["model"], r["condition"], r["instruction_wording"]): r for r in robust["wording_condition_summary"]}
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.85), sharey=True)
    fig.subplots_adjust(left=.24, right=.91, top=.85, bottom=.23, wspace=.2)
    rows = []
    for mi, (model, ax) in enumerate(zip(MODELS, axs)):
        mat = np.array([[100 * lookup[model, c, v]["e2e_rate"] for v in VERSIONS] for c in CONDS])
        im = ax.imshow(mat, cmap="Blues", vmin=0, vmax=100, aspect="auto")
        ax.set_title(MODEL_NAMES[mi], fontweight="bold", pad=9)
        ax.set_xticks(range(3), ["Base", "I1", "I2"])
        ax.set_yticks(range(5), LABELS)
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        for ci, c in enumerate(CONDS):
            for vi, v in enumerate(VERSIONS):
                val = mat[ci, vi]
                ax.text(vi, ci, f"{val:.1f}", ha="center", va="center", color="white" if val >= 60 else INK, fontsize=8)
                rows.append({"model": model, "condition": c, "wording": v, "success_rate_percent": val, "n_targets": 40, "case_clusters": 20})
    cax = fig.add_axes([.935, .235, .016, .60])
    cb = fig.colorbar(im, cax=cax, ticks=[0, 50, 100])
    cb.ax.tick_params(labelsize=7, length=2)
    fig.text(.59, .055, "End-to-end success (%); I1/I2 change instructions on the same cases", ha="center", fontsize=7.5)
    save(fig, "wording_sensitivity", args, rows)


def main():
    args=parse_args(__file__,output_subdir='exp2_wording')
    formal=load(paths.BASELINE/'formal_analysis/formal_analysis.json')
    robust=load(paths.WORDING/'robustness_analysis/robustness_analysis.json')
    check_inputs(formal,robust,paths.OUTPUT_ROOT)
    style()
    wording_sensitivity(formal,robust,args)


if __name__=='__main__':main()
