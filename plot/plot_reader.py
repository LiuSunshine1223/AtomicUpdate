"""Experiment 4 follow-up: qualified-reader checks and order sensitivity."""
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

def robustness(data, args):
    fig, axes = plt.subplots(2, 1, figsize=(5.15, 5.7),
        gridspec_kw={'height_ratios': [1.7, 1]}, layout='constrained')
    ax = axes[0]
    plotted = []
    for i, (model, interface) in enumerate((m, j) for m in MODELS for j in ['patch', 'select']):
        for reader, offset, color, marker, label in [
            (MODELS[0], .14, '#B85623', 's', 'Llama reader'),
            (MODELS[1], -.14, '#009E73', 'D', 'Qwen reader')]:
            r = next(x for x in data['grouping_contrasts'] if
                (x['reader_model'], x['updater_model'], x['interface']) == (reader, model, interface))
            value = r['estimate_pp']
            ax.errorbar(value, 5-i+offset,
                xerr=[[max(0, value-r['ci_low_pp'])], [max(0, r['ci_high_pp']-value)]],
                fmt=marker, color=color, capsize=3, markersize=4.5,
                label=label if i == 0 else None)
            plotted.append(dict(panel='grouping', reader=reader, updater=model,
                interface=interface, estimate=value, low=r['ci_low_pp'], high=r['ci_high_pp']))
    ax.set_yticks(range(6), ['Gemma / Select', 'Gemma / Patch', 'Qwen / Select',
                           'Qwen / Patch', 'Llama / Select', 'Llama / Patch'])
    ax.set(xlabel='g1 − g3 exact match (percentage points)', xlim=(-3, 40),
           xticks=[0, 10, 20, 30, 40], ylim=(-.6, 5.65))
    ax.set_title('(a) Grouping contrasts', loc='left', fontweight='bold')
    ax.axvline(0, color='#777777', lw=.8, ls=':')
    ax.grid(axis='x', alpha=.3)
    ax.legend(loc='lower right', frameon=False, fontsize=8.5)
    ax = axes[1]
    transitions = [('1->1', 'Pass / pass', '#2A7886', '///'), ('1->0', 'Pass / fail', '#D88A00', '\\\\'),
                   ('0->1', 'Fail / pass', '#8D63A8', 'xx'), ('0->0', 'Fail / fail', '#A9AFB5', '..')]
    for y, reader in [(1, MODELS[0]), (0, MODELS[1])]:
        r = next(x for x in data['order_probes'] if x['reader_model'] == reader
                 and x['state'] == 'actual' and x['stratum'] == 'all')
        left = 0
        for key, legend, color, hatch in transitions:
            count = r['transitions'].get(key, 0)
            ax.barh(y, count, left=left, height=.48, color=color, hatch=hatch,
                    edgecolor=INK, linewidth=.35,
                    label=legend if y == 1 else None)
            if count:
                ax.text(left+count/2, y, str(count), ha='center', va='center', fontsize=9,
                        color='white' if key in ['1->1', '0->1'] else INK)
            left += count
            plotted.append(dict(panel='order', reader=reader, transition=key, count=count))
        require(left == 30, 'Missing contradiction identities')
    ax.set(xlim=(0, 30), xticks=[0, 10, 20, 30], ylim=(-.55, 2.2),
           xlabel='Logical contradiction records (n = 30)')
    ax.set_yticks([0, 1], ['Qwen', 'Llama'])
    ax.set_title('(b) Original / reversed order', loc='left', fontweight='bold')
    ax.legend(loc='upper center', ncol=2, frameon=False, fontsize=8.5)
    fields = list(dict.fromkeys(key for row in plotted for key in row))
    save(fig, 'reader_robustness', args,
         [{key: row.get(key, '') for key in fields} for row in plotted])


def main():
    args=parse_args(__file__,output_subdir='exp4_profile/followup')
    data=load(paths.FOLLOWUP/'analysis.json')
    for filename,field in [('freeze_manifest.json','freeze_manifest_sha256'),
                           ('reader_results.jsonl','reader_results_sha256')]:
        require(digest(paths.FOLLOWUP/filename)==data[field],'Frozen follow-up hash mismatch: '+filename)
    paper_style()
    robustness(data,args)


if __name__=='__main__':main()
