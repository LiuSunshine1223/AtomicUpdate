"""Experiment 1 plots; the existing D-E interval comes from the paired wording analysis."""
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

FIGURES=['framework', 'main_results', 'paired_contrasts', 'failure_profiles', 'inference_overhead']

def check_inputs(formal, robust, results_dir=None):
    require(formal["audit"]["formal_cases"] == 20, "Unexpected number of case clusters.")
    require(formal["audit"]["formal_rows"] == 600, "Unexpected baseline row count.")
    require(len(formal["condition_summary"]) == 15, "Incomplete baseline matrix.")
    require(len(robust["wording_condition_summary"]) == 45, "Incomplete wording matrix.")
    lookup = {(r["model"], r["condition"]): r for r in formal["condition_summary"]}
    wlookup = {(r["model"], r["condition"], r["instruction_wording"]): r for r in robust["wording_condition_summary"]}
    for model in MODELS:
        for condition in CONDS:
            r = lookup[model, condition]
            require(abs(r["e2e_rate"] * 40 - r["e2e_success_count"]) < 1e-7, "Baseline rate/count mismatch.")
            require(r["e2e_success_count"] == wlookup[model, condition, "baseline"]["e2e_success_count"], "Baseline snapshots disagree.")
            for variant in VERSIONS:
                w = wlookup[model, condition, variant]
                require(abs(w["e2e_rate"] * 40 - w["e2e_success_count"]) < 1e-7, "Wording rate/count mismatch.")
    if results_dir is None:
        return {"mode": "analysis_summaries", "baseline_rows": 600, "additional_wording_rows": 1200, "case_clusters": 20}
    results_dir = Path(results_dir)
    b = results_dir / "exp1_baseline"
    r = results_dir / "exp2_wording"
    require(digest(b / "formal_results.jsonl") == formal["source_hashes"]["formal_results_sha256"], "Baseline raw-data hash mismatch.")
    rawb = records(b / "formal_results.jsonl")
    rawr = records(r / "robustness_results.jsonl")
    require(len(rawb) == 600 and len(rawr) == 1200, "Unexpected raw-data size.")
    for row in rawb:
        row["instruction_variant"] = "baseline"
    raw = rawb + rawr
    keys = {(x["model"], x["case_id"], x["update_id"], x["condition"], x["instruction_variant"]) for x in raw}
    require(len(keys) == 1800, "Missing or duplicate run keys.")
    require(len({x["case_id"] for x in raw}) == 20, "Runs do not share the same 20 cases.")
    for (model, condition, variant), row in wlookup.items():
        rr = [x for x in raw if (x["model"], x["condition"], x["instruction_variant"]) == (model, condition, variant)]
        require(len(rr) == 40, "Incomplete paired cell.")
        require(sum(x["evaluation"]["end_to_end_success"] for x in rr) == row["e2e_success_count"], "Raw outcomes disagree with summary.")
    for row in formal["failure_reason_summary"]:
        rr = [x for x in rawb if x["model"] == row["model"] and x["condition"] == row["condition"]]
        require(sum(row["failure_reason"] in x["failure_reasons"] for x in rr) == row["rows_with_reason"], "Failure count mismatch.")
    for row in formal["cost_summary"]:
        rr = [x for x in rawb if x["model"] == row["model"] and x["condition"] == row["condition"]]
        for source, target in [("input_tokens", "mean_input_tokens"), ("output_tokens", "mean_output_tokens")]:
            require(abs(np.mean([x["usage"][source] for x in rr]) - row[target]) < 1e-7, "Token summary mismatch.")
    return {"mode": "raw_and_summary_verified", "baseline_rows": 600, "additional_wording_rows": 1200, "unique_run_keys": 1800, "case_clusters": 20}

def framework(formal, robust, args):
    fig, ax = plt.subplots(figsize=(7.2, 3.55))
    fig.subplots_adjust(0, 0, 1, 1)
    ax.set(xlim=(0, 100), ylim=(0, 53))
    ax.axis("off")
    def box(x,y,w,h,color="#F3F6F8"):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.02,rounding_size=.7",facecolor=color,edgecolor="#CDD6DD",lw=.7))
    def txt(x,y,label,size=8,weight="normal",ha="left"):
        ax.text(x,y,label,fontsize=size,fontweight=weight,ha=ha,va="center",linespacing=1.4)
    def arrow(a,b):
        ax.add_patch(FancyArrowPatch(a,b,arrowstyle="-|>",mutation_scale=10,color="#586C78",lw=1))
    box(0,25,48,27); box(52,25,48,27)
    txt(2,49,"1  Freeze matched representations",8.8,"bold")
    txt(2,45,"Six facts; same sentence order",7.5)
    for g,y in [(1,40),(2,34),(3,28)]:
        txt(2,y,f"g = {g}",7.5,"bold")
        n=6//g; width=34/n
        for j in range(n):
            x=12+j*width
            box(x,y-2.1,width-.8,4.2,"white")
            txt(x+(width-.8)/2,y," | ".join(f"f{k+1}" for k in range(j*g,(j+1)*g)),7,ha="center")
    txt(54,49,"2  Reveal target, then edit a fresh clone",8.6,"bold")
    txt(76,41,"Sample two targets after grouping freeze\nEach starts from the same initial state",7.8,ha="center")
    txt(76,31,"Full memory + update evidence visible\nNo gold unit ID or edit span supplied",7.7,ha="center")
    arrow((48,39),(52,39))
    box(0,0,61,21); box(65,0,35,21)
    txt(2,18,"3  Two separately analyzed case batches",8.8,"bold")
    txt(2,12.5,"Baseline: g = 1/2 x Global/Select + g = 2 Patch",7.6)
    txt(2,8.5,"20 cases; 600 calls + 1,200 instruction-paraphrase calls",7.1)
    txt(2,3.5,"New batch: g = 1/2/3 x Select/Patch; 20 cases, 720 calls",7.1)
    txt(67,18,"4  Execute and evaluate",8.7,"bold")
    txt(67,12,"Apply proposed edit; no gold repair\nSelect/Patch: copy untouched text",7.5)
    txt(67,4.5,"Strict success + failure components\nTokens and generation latency",7.5)
    arrow((76,25),(76,23)); ax.plot([30,76],[23,23],color="#586C78",lw=1)
    arrow((30,23),(30,21)); arrow((61,10),(65,10))
    save(fig,"framework",args)

def main_results(formal, robust, args):
    lookup = {(r["model"], r["condition"]): r for r in formal["condition_summary"]}
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.65), sharex=True, sharey=True)
    fig.subplots_adjust(left=.245, right=.985, top=.85, bottom=.22, wspace=.18)
    rows = []
    for mi, (model, ax) in enumerate(zip(MODELS, axs)):
        ax.set_title(MODEL_NAMES[mi], fontweight="bold", pad=9)
        ax.set(xlim=(0, 106), ylim=(4.5, -.5), xticks=[0, 25, 50, 75, 100])
        ax.set_yticks(range(5), LABELS)
        ax.tick_params(axis="y", length=0)
        ax.grid(axis="x", zorder=0)
        ax.spines["left"].set_visible(False)
        for ci, condition in enumerate(CONDS):
            r = lookup[model, condition]
            rate = 100 * r["e2e_rate"]
            ax.plot([0, rate], [ci, ci], lw=1.4, color=COLORS[ci], alpha=.45)
            ax.scatter(rate, ci, color=COLORS[ci], s=29, zorder=3)
            ax.text(rate - 4, ci - .15, f"{r['e2e_success_count']}/40", ha="right", va="bottom", fontsize=7)
            rows.append({"model": model, "condition": condition, "successes": r["e2e_success_count"], "n_targets": 40, "case_clusters": 20, "rate_percent": rate})
    fig.supxlabel("Observed end-to-end success (%)", y=.035, fontsize=8)
    save(fig, "main_results", args, rows)

def paired_contrasts(formal, robust, args):
    lookup = {(r["model"], r["contrast_id"]): r for r in formal["contrast_summary"]}
    interaction = {r["model"]: r for r in formal["interaction_summary"]}
    contrasts = [("C_minus_A", "C - A: Atomic vs Compound / Global"),
                 ("D_minus_B", "D - B: Atomic vs Compound / Select"),
                 ("B_minus_A", "B - A: Select vs Global / Compound"),
                 ("D_minus_C", "D - C: Select vs Global / Atomic"),
                 ("E_minus_B", "E - B: Patch vs Select / Compound"),
                 ("D_minus_E", "D - E: Atomic Select vs Patch"),
                 ("interaction", "Interaction: (D - B) - (C - A)")]
    de = {(r["model"], r["instruction_wording"]): r for r in robust["atomic_select_vs_exact_patch"]}
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 3.55), sharex=True, sharey=True)
    fig.subplots_adjust(left=.365, right=.985, top=.9, bottom=.16, wspace=.14)
    rows = []
    for mi, (model, ax) in enumerate(zip(MODELS, axs)):
        ax.set_title(MODEL_NAMES[mi], fontweight="bold", fontsize=8.5)
        ax.axvline(0, lw=.8, linestyle="--", color="#8899A4", zorder=0)
        ax.set(xlim=(-40, 50), ylim=(6.5, -.5), xticks=[-40, -20, 0, 20, 40])
        ax.set_yticks(range(7), [c[1] for c in contrasts])
        ax.tick_params(axis="y", length=0, labelsize=7.3)
        ax.spines["left"].set_visible(False)
        ax.grid(axis="x", alpha=.45)
        for yi, (key, _) in enumerate(contrasts):
            if key == "interaction":
                r = interaction[model]
                effect = r["effect_e2e_rate_difference_in_differences"]
                lo, hi = r["cluster_bootstrap_ci_low"], r["cluster_bootstrap_ci_high"]
            elif key == "D_minus_E":
                r = de[model, "baseline"]
                effect = r["e2e_rate_difference"]
                lo, hi = r["case_cluster_ci_low"], r["case_cluster_ci_high"]
            else:
                r = lookup[model, key]
                effect = r["effect_e2e_rate_difference"]
                lo, hi = r["cluster_bootstrap_ci_low"], r["cluster_bootstrap_ci_high"]
            effect, lo, hi = 100 * effect, 100 * lo, 100 * hi
            require(-40 <= lo <= effect <= hi <= 50, "Contrast interval would be clipped; expand shared x limits.")
            color = "#009E73" if key == "interaction" else "#0072B2"
            ax.errorbar(effect, yi, xerr=[[effect-lo], [hi-effect]], fmt="D" if key == "interaction" else "o",
                        color=color, capsize=2.5, ms=4, elinewidth=1.15, zorder=3)
            rows.append({"model": model, "contrast": key, "effect_pp": effect, "ci_low_pp": lo, "ci_high_pp": hi, "case_clusters": 20})
    fig.supxlabel("Paired difference in success (percentage points); 95% case-cluster bootstrap CI", y=.025, fontsize=7.6)
    save(fig, "paired_contrasts", args, rows)

def failure_profiles(formal, robust, args):
    reasons = ["locator_failure", "execution_failure", "memory_contract_failure", "structure_failure",
               "target_new_missing", "stale_old_retained", "protected_fact_loss", "unsupported_change"]
    columns = ["Wrong\nunit", "Execution", "Grammar", "Structure", "New\nmissing", "Old still\ncurrent", "Protected\nlost", "Unsupported"]
    lookup = {(r["model"], r["condition"], r["failure_reason"]): r["rows_with_reason"] for r in formal["failure_reason_summary"]}
    cmap = plt.colormaps["Oranges"].copy()
    cmap.set_bad("#E8EDF0")
    fig, axs = plt.subplots(3, 1, figsize=(7.2, 4.75), sharex=True)
    fig.subplots_adjust(left=.265, right=.91, top=.94, bottom=.145, hspace=.5)
    rows = []
    for mi, (model, ax) in enumerate(zip(MODELS, axs)):
        mat = np.array([[lookup[model, c, reason] for reason in reasons] for c in CONDS], dtype=float)
        mat[[0, 2], 0] = np.nan
        im = ax.imshow(mat, cmap=cmap, vmin=0, vmax=40, aspect="auto")
        ax.set_title(MODEL_NAMES[mi], loc="left", fontweight="bold", pad=5)
        ax.set_yticks(range(5), LABELS, fontsize=7)
        ax.set_xticks(range(8), columns, fontsize=7)
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        for ci, c in enumerate(CONDS):
            for ri, reason in enumerate(reasons):
                value = mat[ci, ri]
                ax.text(ri, ci, "N/A" if np.isnan(value) else str(int(value)), ha="center", va="center", fontsize=7,
                        color="white" if value >= 25 else INK)
                rows.append({"model": model, "condition": c, "failure_reason": reason, "count": "NA" if np.isnan(value) else int(value), "n_targets": 40, "labels_overlap": True})
    cax = fig.add_axes([.935, .24, .016, .55])
    cb = fig.colorbar(im, cax=cax, ticks=[0, 10, 20, 30, 40])
    cb.set_label("Failed calls / 40", fontsize=7.5)
    fig.text(.54, .022, "Labels overlap; counts must not be summed. N/A = no explicit locator action.", ha="center", fontsize=7.3)
    save(fig, "failure_profiles", args, rows)

def inference_overhead(formal, robust, args):
    lookup = {(r["model"], r["condition"]): r for r in formal["cost_summary"]}
    fig, axs = plt.subplots(2, 3, figsize=(7.2, 3.8), sharex=True, sharey="row")
    fig.subplots_adjust(left=.09, right=.985, top=.855, bottom=.14, wspace=.2, hspace=.25)
    rows = []
    max_total = max(r["mean_input_tokens"] + r["mean_output_tokens"] for r in formal["cost_summary"])
    max_latency = max(r["mean_latency_seconds_excluding_first_model_call"] for r in formal["cost_summary"])
    for mi, model in enumerate(MODELS):
        inp = np.array([lookup[model, c]["mean_input_tokens"] for c in CONDS])
        out = np.array([lookup[model, c]["mean_output_tokens"] for c in CONDS])
        sec = [lookup[model, c]["mean_latency_seconds_excluding_first_model_call"] for c in CONDS]
        top, bottom = axs[:, mi]
        top.set_title(MODEL_NAMES[mi], fontweight="bold", pad=8)
        top.bar(range(5), inp, color="#B9D9EA", width=.62, zorder=3)
        top.bar(range(5), out, bottom=inp, color="#0072B2", width=.62, zorder=3)
        top.set_ylim(0, max_total * 1.20)
        bottom.bar(range(5), sec, color="#78B5A3", width=.62, zorder=3)
        bottom.set_ylim(0, max_latency * 1.25)
        for ci, c in enumerate(CONDS):
            top.text(ci, inp[ci]+out[ci]+max_total*.025, f"{inp[ci]+out[ci]:.0f}", ha="center", fontsize=6.7)
            bottom.text(ci, sec[ci]+max_latency*.035, f"{sec[ci]:.2f}", ha="center", fontsize=6.7)
            rows.append({"model": model, "condition": c, "input_tokens": inp[ci], "output_tokens": out[ci], "total_tokens": inp[ci]+out[ci], "mean_warm_seconds": sec[ci]})
        for ax in (top, bottom):
            ax.set_xticks(range(5), SHORT)
            ax.grid(axis="y", zorder=0)
            ax.tick_params(axis="x", length=0)
        if mi == 0:
            top.set_ylabel("Mean tokens / call")
            bottom.set_ylabel("Warm latency (s)")
    fig.legend(handles=[Patch(facecolor="#B9D9EA", label="Input tokens"), Patch(facecolor="#0072B2", label="Output tokens")],
               loc="upper center", bbox_to_anchor=(.52, 1.0), ncol=2, frameon=False)
    fig.text(.54, .02, "A-E: same configurations; latency is descriptive, from one backend run.", ha="center", fontsize=7.3)
    save(fig, "inference_overhead", args, rows)


def main():
    args=parse_args(__file__,FIGURES,output_subdir='exp1_baseline')
    formal=load(paths.BASELINE/'formal_analysis/formal_analysis.json')
    robust=load(paths.WORDING/'robustness_analysis/robustness_analysis.json')
    check_inputs(formal,robust,paths.OUTPUT_ROOT)
    style()
    for name in dict.fromkeys(args.only):globals()[name](formal,robust,args)


if __name__=='__main__':main()
