"""Shared appearance, PDF output and read-only file helpers for experiment plots."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Patch
    import numpy as np
except ImportError as exc:
    raise SystemExit("Install plotting dependencies: python -m pip install -r requirements.txt") from exc
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from atomic_memory import paths
MODELS = ["llama3.1:8b-instruct-q4_K_M", "qwen2.5:7b-instruct-q4_K_M", "gemma3:12b-it-q4_K_M"]
MODEL_NAMES = ["Llama 3.1 8B", "Qwen2.5 7B", "Gemma 3 12B"]
CONDS = ["A_compound_global", "B_compound_select", "C_atomic_global", "D_atomic_select", "E_compound_exact_patch"]
SHORT = list("ABCDE")
LABELS = ["A  Compound / Global", "B  Compound / Select", "C  Atomic / Global", "D  Atomic / Select", "E  Compound / Patch"]
COLORS = ["#6B7280", "#D88A00", "#0072B2", "#009E73", "#9B5A93"]
VERSIONS = ["baseline", "instruction_paraphrase_1", "instruction_paraphrase_2"]
VERSION_NAMES = ["Baseline", "Paraphrase 1", "Paraphrase 2"]
INK = "#22313D"

def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def require(condition, message):
    if not condition:
        raise ValueError(message)

def records(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]

def style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 9,
        "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
        "legend.fontsize": 7.5, "text.color": INK, "axes.labelcolor": INK,
        "xtick.color": INK, "ytick.color": INK, "axes.edgecolor": "#ABB4BA",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": .65, "grid.color": "#DCE2E6", "grid.linewidth": .5,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "savefig.facecolor": "white", "figure.facecolor": "white",
        "axes.unicode_minus": False,
    })


def paper_style():
    style()
    matplotlib.rcParams.update({'font.size': 9, 'axes.titlesize': 9.5,
        'axes.labelsize': 9, 'xtick.labelsize': 9, 'ytick.labelsize': 9, 'legend.fontsize': 8.5})


def parse_args(source, figures=None, *, output_subdir):
    parser=argparse.ArgumentParser(description=Path(source).stem)
    parser.add_argument('--output-dir',type=Path,default=ROOT/'figure'/output_subdir,
                        help='Override this experiment\'s figure directory.')
    parser.add_argument('--paper-root',type=Path,help='Optional manuscript root; copy only generated PDFs.')
    if figures:
        parser.add_argument('--only',nargs='+',choices=figures,default=figures)
    args=parser.parse_args()
    require(not args.output_dir.resolve().is_relative_to(paths.OUTPUT_ROOT.resolve()),
            'Figure output must be outside frozen inputs and results')
    if args.paper_root:
        require((args.paper_root/'main.tex').is_file(),'Missing manuscript main.tex')
    args.output_dir.mkdir(parents=True,exist_ok=True)
    args.dpi=300
    args.creator=Path(source).name
    return args


def save(fig,name,args,rows=None,*,paper_name=None,copy_to_paper=True,metadata=None):
    # Statistical rows stay in outputs; figure/ receives PDFs only.
    path=args.output_dir/(name+'.pdf')
    pdf_metadata={'Creator':args.creator}
    if metadata:
        pdf_metadata.update(metadata)
    fig.savefig(path,dpi=args.dpi,bbox_inches='tight',pad_inches=.05,
                metadata=pdf_metadata)
    plt.close(fig)
    if args.paper_root and copy_to_paper:
        target=args.paper_root/'figures'/((paper_name or name)+'.pdf')
        target.parent.mkdir(parents=True,exist_ok=True)
        if path.resolve()!=target.resolve():shutil.copyfile(path,target)
        require(digest(path)==digest(target),'Manuscript figure copy differs')
    print('Generated: '+str(path))
