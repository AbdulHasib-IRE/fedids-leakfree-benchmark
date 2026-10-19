import itertools, json, sys
from pathlib import Path
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / 'outputs' / 'figures'
OUT.mkdir(parents=True, exist_ok=True)
DPI, W, H = 1000, 3.6, 2.55                        # 3.6 in x 1000 dpi = 3600 px (>= 3543 px single column)
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 7.5, 'axes.labelsize': 7.5,
                     'xtick.labelsize': 6.8, 'ytick.labelsize': 6.8, 'legend.fontsize': 6.3,
                     'axes.linewidth': 0.6, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
                     'axes.spines.top': False, 'axes.spines.right': False, 'savefig.dpi': DPI})
INK, MUTED, GRID = '#1f1f1f', '#5f5f5f', '#e3e3e3'

# ---------- fixed encoding: each rule always has the same colour AND marker AND line style (Okabe-Ito) ----------
METHODS = ['fedavg', 'fedprox', 'scaffold', 'feddyn', 'fedadapt', 'fedadapt_pure']
NAME = {'fedavg': 'FedAvg', 'fedprox': 'FedProx', 'scaffold': 'SCAFFOLD', 'feddyn': 'FedDyn',
        'fedadapt': 'FedAdapt-F', 'fedadapt_pure': 'FedAdapt-pure'}
COL = {'fedavg': '#0072B2', 'fedprox': '#E69F00', 'scaffold': '#009E73', 'feddyn': '#CC79A7',
       'fedadapt': '#D55E00', 'fedadapt_pure': '#56B4E9'}
MRK = {'fedavg': 'o', 'fedprox': 's', 'scaffold': '^', 'feddyn': 'D', 'fedadapt': 'v', 'fedadapt_pure': 'X'}
LS = {'fedavg': '-', 'fedprox': '--', 'scaffold': '-.', 'feddyn': ':', 'fedadapt': '-', 'fedadapt_pure': '--'}
SET_LABEL = {'random_dirichlet0.1': 'Dir. α=0.1', 'random_dirichlet0.5': 'Dir. α=0.5', 'random_iid': 'IID',
             'random_ip': 'Source IP', 'time_dirichlet0.1': 'Dir. α=0.1', 'time_ip': 'Source IP'}
CLS_SHORT = ['Benign', 'Brute\nForce', 'DDoS', 'DoS', 'Mirai', 'Recon', 'Spoof-\ning', 'Web-\nBased']
CLS = ['Benign', 'Brute Force', 'DDOS', 'DOS', 'Mirai', 'Recon', 'Spoofing', 'Web-Based']

# ---------- overlap / clipping checker ----------
def check(fig, name, data_axes=()):
    fig.canvas.draw(); r = fig.canvas.get_renderer(); fb = fig.bbox
    problems = []
    undrawn = set()                                                          # tick labels outside the view are never drawn
    for a in fig.axes:
        for axis in (a.xaxis, a.yaxis):
            drawn = set() if not a.axison else ({id(tk.label1) for tk in axis._update_ticks()} | {id(tk.label2) for tk in axis._update_ticks()})
            for tk in axis.get_major_ticks() + axis.get_minor_ticks():
                for lb in (tk.label1, tk.label2):
                    if id(lb) not in drawn: undrawn.add(id(lb))
    texts = [t for t in fig.findobj(matplotlib.text.Text) if t.get_visible() and t.get_text().strip() and id(t) not in undrawn]
    boxes = [(t, t.get_window_extent(r)) for t in texts]
    for t, b in boxes:                                                       # nothing clipped
        if b.x0 < fb.x0 - 0.5 or b.y0 < fb.y0 - 0.5 or b.x1 > fb.x1 + 0.5 or b.y1 > fb.y1 + 0.5:
            problems.append(f'clipped text "{t.get_text()}"')
    for (t1, b1), (t2, b2) in itertools.combinations(boxes, 2):           # text vs text
        if b1.overlaps(b2):
            inter = (min(b1.x1, b2.x1) - max(b1.x0, b2.x0)) * (min(b1.y1, b2.y1) - max(b1.y0, b2.y0))
            if inter > 1.0:
                problems.append(f'text overlap "{t1.get_text()}" / "{t2.get_text()}"')
    for ax in data_axes:                                                     # legend vs data
        leg = ax.get_legend()
        if leg is None: continue
        lb = leg.get_window_extent(r)
        pts = []
        for ln in ax.get_lines():
            if ln.get_label().startswith('_ref'): continue
            pts.append(ax.transData.transform(np.column_stack(ln.get_data())))
        for coll in ax.collections:
            off = coll.get_offsets()
            if len(off): pts.append(ax.transData.transform(off))
        for pa in ax.patches:                                                # bars vs legend
            pb = pa.get_window_extent(r)
            if pb.width > 0.5 and pb.height > 0.5 and pb.overlaps(lb):
                inter = (min(pb.x1, lb.x1) - max(pb.x0, lb.x0)) * (min(pb.y1, lb.y1) - max(pb.y0, lb.y0))
                if inter > 1.0: problems.append('legend covers bars'); break
        for P in pts:
            P = P[np.isfinite(P).all(1)]
            if np.any((P[:, 0] > lb.x0) & (P[:, 0] < lb.x1) & (P[:, 1] > lb.y0) & (P[:, 1] < lb.y1)):
                problems.append('legend covers data'); break
    return problems

REPORT = {}
def save(fig, name, data_axes=()):
    p = check(fig, name, data_axes)
    fig.savefig(OUT / f'{name}.png', dpi=DPI, bbox_inches=None, facecolor='white')
    plt.close(fig)
    REPORT[name] = p
    print(f'{name}.png  {"PASS" if not p else "FAIL: " + "; ".join(sorted(set(p)))}')

def ax_style(ax, ygrid=True):
    if ygrid: ax.yaxis.grid(True, color=GRID, lw=0.5); ax.set_axisbelow(True)
    ax.tick_params(colors=INK, length=2.5)

