"""Experiment 4: state-check framework and profile endpoint contrasts."""
from __future__ import annotations
from pathlib import Path
import re
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from plot.style import (
    ROOT, MODELS, MODEL_NAMES, CONDS, SHORT, LABELS, COLORS, VERSIONS,
    VERSION_NAMES, INK, plt, matplotlib, np, FancyBboxPatch, FancyArrowPatch,
    Patch, load, digest, require, records, style, paper_style, save, parse_args,
)
from atomic_memory import paths

CONDS=['L1_select', 'L1_patch', 'L3_select', 'L3_patch']
FIGURES=['framework', 'example', 'contrasts']

def check_inputs(analysis, source_dir, verify_raw):
    require(analysis["cluster_unit"] == "case", "Expected case-cluster intervals.")
    require(analysis["bootstrap_replicates"] == 20000, "Unexpected bootstrap count.")
    cells = {(r["model"], r["condition"]): r for r in analysis["condition_summary"]}
    require(set(cells) == {(m,c) for m in MODELS for c in CONDS}, "Incomplete cell matrix.")
    for row in cells.values():
        require(row["n_cases"] == 50 and row["n_rows"] == 100, "Unexpected cell denominator.")
        for endpoint in ["strict", "task"]:
            require(abs(row[f"{endpoint}_success_rate"]*100-row[f"{endpoint}_success_count"]) < 1e-8,
                    "Rate and count disagree.")
    contrasts = {(r["model"], r["contrast_id"]): r for r in analysis["contrast_summary"]}
    require(set(contrasts) == {(m,k) for m in MODELS for k in ["G_Patch","F_Patch","G_Select","F_Select"]},
            "Incomplete contrast matrix.")
    for (model, name), row in contrasts.items():
        endpoint = "strict" if name.startswith("G_") else "task"
        interface = name.split("_")[1].lower()
        difference = cells[model,"L1_"+interface][endpoint+"_success_count"] - cells[model,"L3_"+interface][endpoint+"_success_count"]
        require(abs(difference-row["estimate_percentage_points"]) < 1e-8, "Contrast and cells disagree.")
        require(row["n_case_clusters"] == 50, "Incorrect contrast sample size.")
    if verify_raw:
        for name, expected in analysis["source_hashes"].items():
            require(digest(source_dir/name) == expected, f"Frozen source hash mismatch: {name}")
        updaters = records(source_dir/"updater_results.jsonl")
        readers = records(source_dir/"actual_reader_results.jsonl")
        lookup = {r["request_id"]: r for r in updaters}
        require(len(lookup) == len(updaters) == len(readers) == 1200, "Unexpected row count.")
        require(len({(r['updater_model'],r['case_id'],r['update_id'],r['condition']) for r in readers}) == 1200,
                "Duplicate reader identity.")
        for r in readers:
            u = lookup[r["source_updater_request_id"]]
            require(r["strict_state_success"] == u["evaluation"]["end_to_end_success"], "Strict score changed.")
            require(r["functional_task_pass"] == (r["parsed_response"] == r["oracle_expected_values"]), "Profile score mismatch.")
            require((r['updater_model'],r['case_id'],r['update_id'],r['condition']) ==
                    (u['model'],u['case_id'],u['update_id'],u['condition']), "Incorrect updater/reader join.")
        for (model, condition), row in cells.items():
            group = [r for r in readers if (r["updater_model"],r["condition"]) == (model,condition)]
            require(len(group) == 100 and len({r["case_id"] for r in group}) == 50, "Cell coverage mismatch.")
            require(sum(r["strict_state_success"] for r in group) == row["strict_success_count"], "Strict count mismatch.")
            require(sum(r["functional_task_pass"] for r in group) == row["task_success_count"], "Profile count mismatch.")
    return contrasts

def framework(args):
    fig, ax = plt.subplots(figsize=(5.15, 3.2))
    ax.set(xlim=(0, 10), ylim=(0, 6))
    ax.axis('off')

    def box(x, y, w, h, title, body, color):
        ax.add_patch(FancyBboxPatch((x, y), w, h,
            boxstyle='round,pad=0.03,rounding_size=0.08',
            facecolor=color, edgecolor='#8B9AA4', linewidth=.8))
        ax.text(x+w/2, y+h*.78, title, ha='center', va='center',
                fontsize=9, fontweight='bold', color=INK)
        ax.text(x+w/2, y+h*.36, body, ha='center', va='center',
                fontsize=8.5, color=INK, linespacing=1.35)

    def arrow(start, end):
        ax.annotate('', xy=end, xytext=start,
                    arrowprops=dict(arrowstyle='->', lw=1, color='#53616B'))

    box(.10, 3.70, 3.0, 2.0, 'Matched six facts',
        'g1: 6 units\ng3: 2 units\nFrozen before target', '#EDF2F6')
    box(3.50, 3.70, 3.0, 2.0, 'Updater + executor',
        'Select: rewrite unit\nPatch: replace span\nEvidence supplied', '#EDF2F6')
    box(6.90, 3.70, 3.0, 2.0, 'Result',
        'One final\nmemory state', '#E7F0E8')
    arrow((3.13, 4.70), (3.46, 4.70))
    arrow((6.53, 4.70), (6.86, 4.70))
    box(.10, .35, 4.55, 2.0, 'Strict state check (Y)',
        'All six facts\nGrammar + structure', '#EAF0F7')
    box(5.35, .35, 4.55, 2.0, 'Reader + exact match (Z)',
        'Target subject: 3 attributes\nNo evidence or answer candidates', '#FBEEE5')
    ax.plot([8.4, 8.4, 2.375], [3.66, 2.85, 2.85], color='#53616B', lw=1)
    arrow((2.375, 2.85), (2.375, 2.39))
    arrow((7.625, 2.85), (7.625, 2.39))
    ax.text(5, 3.10, 'Evaluate the same final state', ha='center', fontsize=8.5, color=INK)
    fig.subplots_adjust(left=.01, right=.99, top=.99, bottom=.01)
    save(fig, 'profile_framework', args)

def worked_example(args):
    """Display four logged executions of one update, without constructing results."""
    from textwrap import fill
    case, update, model = 'layer2_formal_case_0036', 'u01', MODELS[0]
    updaters = records(paths.PROFILE/'updater_results.jsonl')
    readers = {r['source_updater_request_id']: r for r in
               records(paths.PROFILE/'actual_reader_results.jsonl')}
    chosen = {r['condition']: r for r in updaters if
              (r['case_id'], r['update_id'], r['model']) == (case, update, model)}
    require(set(chosen) == set(CONDS), 'Worked example needs all four matched executions')
    base = chosen['L1_select']
    facts = [u['text'] for u in base['initial_memory']]
    for row in chosen.values():
        sentences = [s.strip()+'.' for u in row['initial_memory'] for s in
                     u['text'].split('.') if s.strip()]
        require(sentences == facts, 'Example initial content or sentence order differs')
        require(row['evidence_text'] == base['evidence_text'], 'Example update differs')
        reader = readers[row['request_id']]
        require(reader['parsed_response'] == reader['oracle_expected_values'] and
                reader['functional_task_pass'], 'Expected the four logged exact matches')
        changed = [u for u, before in zip(row['executed_public_memory'], row['initial_memory'])
                   if u['text'] != before['text']]
        require(len(changed) == 1 and changed[0]['memory_id'] == row['parsed_action']['memory_id'],
                'Example changed more than the reported unit')

    fig, ax = plt.subplots(figsize=(5.15, 5.8))
    ax.set(xlim=(0, 100), ylim=(0, 112))
    ax.axis('off')
    def text(x, y, value, size=8.5, weight='normal', **kw):
        return ax.text(x, y, value, fontsize=size, fontweight=weight,
                       color=INK, ha='left', va='top', linespacing=1.25, **kw)
    def box(y, height):
        ax.add_patch(FancyBboxPatch((.5, y), 99, height,
                     boxstyle='round,pad=0.02,rounding_size=.6',
                     facecolor='white', edgecolor='#ABB4BA', lw=.7))
    text(1, 110, 'Llama 3.1 updater and reader | case 0036 / u01', 9, 'bold')
    text(1, 105, base['evidence_text'], 9)
    box(83, 17)
    text(2, 99, 'Initial current facts (same sentence order)', 9, 'bold')
    for i, sentence in enumerate(facts):
        match = re.fullmatch(r"(.+)'s current (.+) is (.+)\.", sentence)
        require(match is not None, 'Non-canonical initial example fact')
        subject, attribute, value = match.groups()
        text(2 if i < 3 else 52, 94 - 4.2*(i % 3),
             '{}  {} / {}: {}'.format(i+1, subject, attribute, value), 8.25)
    text(1, 78, 'g=1:  m00 [1]   m01 [2]   m02 [3]   m03 [4]   m04 [5]   m05 [6]', 8.25)
    text(1, 73, 'g=3:  m00 [1, 2, 3]   m01 [4, 5, 6]', 8.5)
    text(1, 67, 'Config. / action', 8.5, 'bold')
    text(26, 67, 'Edited unit (other units copied)', 8.5, 'bold')
    text(89, 67, 'Scores', 8.5, 'bold')
    plotted = []
    for condition, top, bottom in [('L1_select', 63, 53), ('L1_patch', 51, 41),
                                   ('L3_select', 39, 24), ('L3_patch', 22, 7)]:
        row = chosen[condition]
        reader = readers[row['request_id']]
        action = row['parsed_action']
        unit = next(u for u in row['executed_public_memory'] if
                    u['memory_id'] == action['memory_id'])
        box(bottom, top-bottom)
        heading = 'g={} / {}'.format(row['unit_load'], row['interface'].title())
        text(2, top-1, heading, 9, 'bold')
        operation = 'Rewrite '+action['memory_id'] if row['interface']=='select' else 'Replace span\n'+action['memory_id']
        text(2, top-4.8, operation, 8.25)
        sentences = [s.strip()+'.' for s in unit['text'].split('.') if s.strip()]
        text(26, top-1, '\n'.join(fill(s, width=48) for s in sentences), 8.5)
        text(90, top-1, 'Y = '+str(int(row['strict_state_success'])), 9, 'bold')
        text(90, top-6, 'Z = '+str(int(reader['functional_task_pass'])), 9, 'bold')
        plotted.append({'condition': condition, 'updater_request_id': row['request_id'],
                        'action': action, 'edited_unit': unit, 'Y': row['strict_state_success'],
                        'Z': reader['functional_task_pass'], 'profile': reader['parsed_response']})
    profile = readers[base['request_id']]['parsed_response']
    text(1, 5, 'All four answers for Emma: '+', '.join('{}={}'.format(k, v) for k, v in profile.items()), 8.25)
    fig.subplots_adjust(left=.01, right=.99, top=.99, bottom=.01)
    save(fig, 'profile_example', args, plotted)


def contrasts(rows, args):
    fig, axes = plt.subplots(3, 1, figsize=(5.15, 5.0), sharex=True,
                              layout='constrained')
    plotted = []
    for ax, model, title in zip(axes, MODELS, MODEL_NAMES):
        for j, interface in enumerate(['Patch', 'Select']):
            for metric, offset, color, marker, label in [
                ('G', .14, '#275C8C', 'o', 'Strict state'),
                ('F', -.14, '#B85623', 's', 'Profile exact match')]:
                r = rows[model, metric+'_'+interface]
                value, low, high = [r[k] for k in [
                    'estimate_percentage_points', 'cluster_bootstrap_ci_low_pp',
                    'cluster_bootstrap_ci_high_pp']]
                ax.errorbar(value, 1-j+offset, xerr=[[value-low], [high-value]],
                            fmt=marker, color=color, capsize=3, markersize=4.5,
                            label=label if j == 0 else None)
                plotted.append(dict(model=model, interface=interface, endpoint=metric,
                    estimate_pp=value, ci_low_pp=low, ci_high_pp=high,
                    case_clusters=r['n_case_clusters'], role=r['role']))
        ax.set_title(title, loc='left', fontweight='bold', fontsize=9.5)
        ax.set_yticks([0, 1], ['Select', 'Patch'])
        ax.set(xlim=(-3, 49), ylim=(-.55, 1.55), xticks=[0, 10, 20, 30, 40])
        ax.axvline(0, color='#777777', lw=.8, ls=':')
        ax.grid(axis='x', alpha=.3)
    axes[0].legend(loc='lower left', bbox_to_anchor=(0, 1.28), ncol=2,
                   frameon=False, fontsize=8.5, borderaxespad=0, handlelength=1.5)
    axes[-1].set_xlabel('g1 − g3 (percentage points)')
    save(fig, 'profile_contrasts', args, plotted)


def main():
    args=parse_args(__file__,FIGURES,output_subdir='exp4_profile')
    data=load(paths.PROFILE/'formal_analysis.json')
    rows=check_inputs(data,paths.PROFILE,verify_raw=True)
    paper_style()
    for name in dict.fromkeys(args.only):
        if name=='framework':framework(args)
        elif name=='example':worked_example(args)
        else:contrasts(rows,args)


if __name__=='__main__':main()
