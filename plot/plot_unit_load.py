"""Experiment 3: grouping-size results and updater-token context."""
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

def check_unit_inputs(unit, results_dir=None):
    require(unit["validation"]["row_count"]==720 and unit["validation"]["case_count"]==20,"Unexpected extension dimensions.")
    require(len(unit["condition_summary"])==18,"Incomplete extension matrix.")
    for r in unit["condition_summary"]:
        require(r["n_targets"]==40 and abs(r["success_rate"]*40-r["success_count"])<1e-7,"Extension rate/count mismatch.")
    if results_dir:
        root=Path(results_dir)/"exp3_unit_load"
        manifest=load(root/"unit_load_analysis/analysis_manifest.json")
        require(digest(root/"unit_load_results.jsonl")==manifest["results_sha256"],"Extension raw hash mismatch.")
        raw=records(root/"unit_load_results.jsonl")
        require(len(raw)==len({(r["model"],r["case_id"],r["update_id"],r["condition"]) for r in raw})==720,"Duplicate/missing extension keys.")
        for r in unit["condition_summary"]:
            rr=[x for x in raw if x["model"]==r["model"] and x["condition"]==r["condition"]]
            require(len(rr)==40 and sum(x["evaluation"]["end_to_end_success"] for x in rr)==r["success_count"],"Extension raw outcomes differ.")
        for r in unit["failure_reasons"]:
            require(sum(r["failure_reason"] in x["failure_reasons"] for x in raw if x["model"]==r["model"] and x["condition"]==r["condition"])==r["row_count"],"Extension failure mismatch.")
        for r in unit["costs"]:
            rr=[x for x in raw if x["model"]==r["model"] and x["condition"]==r["condition"]]
            for k in ["input_tokens","output_tokens"]:
                require(abs(np.mean([x["usage"][k] for x in rr])-r["mean_"+k])<1e-7,"Extension token mismatch.")
    return {"rows":720,"case_clusters":20,"pool_with_baseline":False}

def unit_load_results(unit, args):
    lookup={(r["model"],r["condition"]):r for r in unit["condition_summary"]}
    costs={(r["model"],r["condition"]):r for r in unit["costs"]}
    fig, axs=plt.subplots(2,3,figsize=(7.2,3.55),sharex=True,sharey="row")
    fig.subplots_adjust(left=.09,right=.985,top=.86,bottom=.14,wspace=.17,hspace=.28)
    rows=[]
    for mi,model in enumerate(MODELS):
        top,bottom=axs[:,mi]
        top.set_title(MODEL_NAMES[mi],fontweight="bold",pad=7)
        for ji,(interface,color,marker) in enumerate([("select","#0072B2","o"),("patch","#D55E00","s")]):
            gs=np.array([1,2,3]); xx=gs+(-.035 if ji==0 else .035)
            rr=[lookup[model,f"L{g}_{interface}"] for g in gs]
            cc=[costs[model,f"L{g}_{interface}"] for g in gs]
            top.plot(xx,[r["success_rate"]*100 for r in rr],color=color,marker=marker,ms=4,lw=1.15,label=interface.title())
            bottom.plot(xx,[r["mean_total_tokens"] for r in cc],color=color,marker=marker,ms=4,lw=1.15)
            for g,x,r,c in zip(gs,xx,rr,cc):
                other = lookup[model, f"L{g}_{'patch' if interface=='select' else 'select'}"]["success_rate"]
                above = r["success_rate"] > other or (r["success_rate"] == other and ji == 0)
                top.annotate(f"{r['success_count']}/40",(x,r["success_rate"]*100),xytext=(0,7 if above else -11),textcoords="offset points",ha="center",fontsize=6.8,color=color)
                rows.append({"model":model,"grouping_size":int(g),"interface":interface,"successes":r["success_count"],"targets":40,"clusters":20,"rate_percent":r["success_rate"]*100,"input_tokens":c["mean_input_tokens"],"output_tokens":c["mean_output_tokens"],"total_tokens":c["mean_total_tokens"],"warm_seconds":c["mean_warm_latency_seconds"]})
        top.set(ylim=(52,112),yticks=[60,80,100])
        bottom.set(ylim=(0,630),yticks=[0,200,400,600],xlabel="Facts per unit (g)")
        for ax in (top,bottom):
            ax.set(xlim=(.75,3.25),xticks=[1,2,3]); ax.grid(axis="y",alpha=.7)
        if mi==0:
            top.set_ylabel("Strict success (%)")
            bottom.set_ylabel("Mean tokens / call")
    handles,labels=axs[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc="upper center",bbox_to_anchor=(.53,1.01),ncol=2,frameon=False)
    save(fig,"unit_load_results",args,rows)


def main():
    args=parse_args(__file__,output_subdir='exp3_unit_load')
    data=load(paths.UNIT_LOAD/'unit_load_analysis/unit_load_analysis.json')
    check_unit_inputs(data,paths.OUTPUT_ROOT)
    style()
    unit_load_results(data,args)


if __name__=='__main__':main()
