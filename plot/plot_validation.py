"""Experiment 5: verified endpoint plots in analysis and manuscript layouts."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from plot.style import style, plt, save, parse_args
from atomic_memory import validation as v
from matplotlib.text import Text

FIGURES = ["analysis", "manuscript"]

LABELS = ["Clean", "Surface\nrealization", "Current-value\ncontradiction",
          "Queried-attribute\ndeletion", "Cross-subject\nmisbinding", "Other-subject\ndamage"]


def make_figure(data, order=None):
    lookup = {r["variant"]: r for r in data["arm_summary"]}
    v.require(set(lookup) == set(v.VARIANTS) and data["num_cases"] == v.N_CASES,
              "Incomplete fixed experiment-5 arms")
    rows = [lookup[arm] for arm in v.VARIANTS]
    style()
    heights = [1.25, 1] if order is None else [1.25, 1, 1]
    fig, axes = plt.subplots(len(heights), 1,
                             figsize=(7.05, 4.85 if order is None else 6.8),
                             layout="constrained", gridspec_kw={"height_ratios": heights})
    x = list(range(len(rows)))
    ax = axes[0]
    for offset, color, key, label, hatch in ((-.16, "#275C8C", "strict_rate_pp", "Strict state check (Y)", "///"),
                                    (.16, "#B85623", "exact_match_rate_pp", "Profile exact match (Z)", "..")):
        ax.bar([i + offset for i in x], [r[key] for r in rows], width=.29, color=color,
               hatch=hatch, edgecolor="#22313D", linewidth=.35, label=label)
        for i, row in enumerate(rows):
            if row[key] == 0:
                ax.plot(i + offset, 0, marker="_", markersize=7, color=color,
                        markeredgewidth=1.7, clip_on=False, zorder=4)
                ax.annotate("0", (i + offset, 0), xytext=(0, 4), textcoords="offset points",
                            ha="center", va="bottom", fontsize=6.5, color=color)
    z = [r["exact_match_rate_pp"] for r in rows]
    ax.errorbar([i + .16 for i in x], z,
        yerr=[[max(0, val - r["exact_match_ci_low_pp"]) for val, r in zip(z, rows)],
              [max(0, r["exact_match_ci_high_pp"] - val) for val, r in zip(z, rows)]],
        fmt="none", color="#22313D", capsize=2, linewidth=.8)
    ax.set(ylim=(0, 111), ylabel="Pass rate (%)", xticks=x, xticklabels=LABELS)
    ax.set_title("(a) Endpoints on paired constructed states (50 cases per arm)", fontweight="bold")
    ax.legend(loc="upper right", ncol=2, fontsize=7, frameon=False)
    ax.grid(axis="y", alpha=.3)
    ax.set_axisbelow(True)
    ax = axes[1]
    # Whole-answer support: UNKNOWN/invalid is undetermined, never a supported answer.
    bottom = [0] * len(rows)
    for key, label, color, hatch in (("supported_answer", "Supported", "#2A7886", "///"),
                              ("unsupported_concrete_answer", "Unsupported concrete value", "#D88A00", "xx"),
                              ("support_undetermined", "Abstention / invalid", "#A9AFB5", "..")):
        counts = [r[key] for r in rows]
        ax.bar(x, counts, bottom=bottom, width=.62, color=color, hatch=hatch,
               edgecolor="#22313D", linewidth=.35, label=label)
        for i, count in enumerate(counts):
            if count:
                center = bottom[i] + count / 2
                if count <= 3:
                    # Small stacked segments need separate labels and leader lines.
                    left = key == "unsupported_concrete_answer"
                    dx = -.43 if left else .43
                    y = center - 1.8 if left and center > 40 else center + 2.5
                    ax.annotate(str(count), xy=(i, center), xytext=(i + dx, y),
                                ha="right" if left else "left", va="center", fontsize=7,
                                color="#22313D", arrowprops=dict(arrowstyle="-", color=color,
                                linewidth=.8, shrinkA=1, shrinkB=1))
                else:
                    ax.text(i, center, str(count), ha="center", va="center", fontsize=7,
                            color="white" if key == "supported_answer" else "#22313D")
        bottom = [start + count for start, count in zip(bottom, counts)]
    v.require(bottom == [v.N_CASES] * len(rows), "Missing answer-support outcomes")
    ax.set(ylim=(0, 64), ylabel="Logical answers", xticks=x, xticklabels=LABELS)
    ax.set_title("(b) Answer support from constructed current assertions", fontweight="bold")
    ax.legend(loc="upper center", ncol=3, fontsize=6.7, frameon=False)
    ax.grid(axis="y", alpha=.3)
    ax.set_axisbelow(True)
    if order is not None:
        ax = axes[2]
        n = order["paired_cases"]
        counts = [order["new_old"]["exact_match"], order["old_new"]["exact_match"]]
        ax.bar([0, 1], [100 * c / n for c in counts], width=.5, color=["#275C8C", "#B85623"])
        for i, count in enumerate(counts):
            if count == 0:
                ax.plot(i, 0, marker="_", markersize=9, color="#22313D", clip_on=False)
            ax.annotate("{}/{}".format(count, n), (i, 100 * count / n),
                        xytext=(0, 4), textcoords="offset points", ha="center", fontsize=8)
        t = order["exact_match_transitions"]
        lines = ["Paired exact-match transitions (n={})".format(n),
                 "Pass to pass: {}    Pass to fail: {}".format(t["1->1"], t["1->0"]),
                 "Fail to pass: {}    Fail to fail: {}".format(t["0->1"], t["0->0"]),
                 "Changed three-field answers: {}/{}".format(order["changed_answers"], n),
                 "Old->New minus New->Old: {:+g} pp".format(order["profile_difference_pp"]),
                 "95% paired interval: [{:g}, {:g}] pp".format(
                     order["profile_difference_ci_low_pp"], order["profile_difference_ci_high_pp"])]
        ax.text(1.6, 101, "\n".join(lines), ha="left", va="top", fontsize=7.3, linespacing=1.6)
        ax.set(ylim=(0, 115), xlim=(-.55, 3.95), ylabel="Exact-match rate (%)", xticks=[0, 1],
               xticklabels=["New -> Old\n(original responses)", "Old -> New\n(adjacent swap)"])
        ax.set_title("(c) Exploratory adjacent swap on the same 50 contradictory states", fontweight="bold")
        ax.grid(axis="y", alpha=.3)
        ax.set_axisbelow(True)
    return fig


def make_manuscript_figure(data, order):
    v.require(order is not None, "The manuscript layout requires completed adjacent-swap records")
    fig = make_figure(data, order)
    before = [[float(p.get_height()) for p in ax.patches] for ax in fig.axes]

    # Adjust the existing artists so both layouts retain the same data bars.
    fig.set_size_inches(5.15, 5.95)
    for text in fig.findobj(match=Text):
        text.set_fontsize(8.75)
    labels = ["Clean", "Surface\nwording", "Current-value\nconflict",
              "Queried\nattribute\ndeleted", "Subject\nmisbinding", "Other-subject\ndamage"]
    a, b, c = fig.axes
    for ax in (a, b):
        ax.set_xticks(range(6), labels=labels)
    a.set_title("(a) Strict state checks and profile matches", fontsize=9.5, fontweight="bold")
    a.set_ylim(0, 126)
    a.set_yticks([0, 20, 40, 60, 80, 100])
    a.legend(loc="upper center", ncol=2, fontsize=8.75, frameon=False,
             columnspacing=1.3, handlelength=1.4)
    b.set_title("(b) Answer support by current assertions", fontsize=9.5, fontweight="bold")
    b.set_ylim(0, 70)
    b.legend(handles=b.get_legend_handles_labels()[0],
             labels=["Supported", "Unsupported", "Abstention"], loc="upper center",
             ncol=3, fontsize=8.75, frameon=False, columnspacing=1.1, handlelength=1.3)

    # Detailed paired transitions remain in the manuscript and supplement table.
    for text in list(c.texts):
        if text.get_text().startswith("Paired exact-match transitions"):
            text.remove()
    c.set_xlim(-.55, 1.55)
    c.set_title("(c) Adjacent swap in contradictory states", fontsize=9.5, fontweight="bold")
    c.set_xticks([0, 1], labels=["New -> Old\n(original responses)", "Old -> New\n(adjacent swap)"])
    for ax in fig.axes:
        ax.tick_params(axis="both", labelsize=8.75)
        ax.yaxis.label.set_fontsize(9)
    for text in c.texts:
        text.set_fontsize(9.5)
    fig.canvas.draw()
    after = [[float(p.get_height()) for p in ax.patches] for ax in fig.axes]
    v.require(after == before, "Manuscript typesetting changed data bars")
    return fig


def load_plot_data():
    """Load the frozen analyses and verify their records before either layout."""
    v.require((v.OUT / "analysis.json").is_file(), "Run analyze_validation.py first")
    data = v.load(v.OUT / "analysis.json")
    v.verify(check_code=False)
    for name, sha in data["source_hashes"].items():
        v.require(v.digest(v.OUT / name) == sha, "Plot source hash mismatch: " + name)
    order = None
    order_dir = v.OUT / "order_check"
    if (order_dir / "analysis.json").is_file():
        from scripts.exp5_validation import order_check as o
        freeze, plan = o.verify(check_code=False)
        order = v.load(order_dir / "analysis.json")
        for name, sha in order["source_hashes"].items():
            v.require(v.digest(order_dir / name) == sha, "Order-plot source hash mismatch: " + name)
        cache = v.validate_raw(v.records(order_dir / "responses.jsonl"), plan,
                               freeze["model_snapshot"], complete=True)
        added = v.score(plan, cache)
        v.require(added == v.records(order_dir / "reader_results.jsonl"), "Order-plot scores differ")
        _, _, original, _ = o.parent_records(check_code=False)
        for key, value in o.summarize(list(original.values()), added).items():
            v.require(order[key] == value, "Order-plot statistic differs: " + key)
    return data, order


def main():
    args = parse_args(__file__, FIGURES, output_subdir='exp5_validation')
    data, order = load_plot_data()
    v.require(not args.output_dir.resolve().is_relative_to(v.OUT.resolve()),
              "Figure output must be outside experiment inputs/results")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected = set(args.only)
    v.require("manuscript" not in selected or order is not None,
              "The manuscript layout requires completed adjacent-swap records")
    if "analysis" in selected:
        save(make_figure(data, order), "endpoint_validation", args,
             copy_to_paper="manuscript" not in selected)
    if "manuscript" in selected:
        save(make_manuscript_figure(data, order), "endpoint_validation_manuscript", args,
             paper_name="endpoint_validation",
             metadata={"Creator": "plot_validation.py with manuscript-only typesetting",
                       "CreationDate": None, "ModDate": None})
    print("Adjacent-swap panel: " + ("50 completed pairs" if order is not None else "pending added responses"))


if __name__ == "__main__":
    main()
