"""Regenerate every figure of the FedAdapt benchmark paper from the exported CSVs.

Usage: python3 make_figures.py <export_dir> <out_dir>
Every number drawn comes from the CSV/JSON files in <export_dir>; nothing is typed in.
Each panel is one PNG at 1000 dpi, no titles inside the image. An automatic overlap
checker runs on every panel and the script exits non-zero if any panel fails.
"""
import json, sys, itertools
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from scipy import stats

EX, OUT = Path(sys.argv[1]), Path(sys.argv[2]); OUT.mkdir(parents=True, exist_ok=True)
EX2 = Path(sys.argv[1]).parent / 'export2b'
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

# ---------- load ----------
runs = pd.read_csv(EX / 'grid_runs.csv'); runs = runs[~runs.diverged]
hist = pd.read_csv(EX / 'grid_histories_long.csv')
TB = pd.read_csv(EX / 'table_B_vs_fedavg.csv')
cen = pd.read_csv(EX / 'central_summary.csv')
C1 = pd.read_csv(EX / 'table_C1_windowing_pitfall.csv')
lr = pd.read_csv(EX / 'lr_control.csv')
ws = pd.read_csv(EX / 'fedadapt_weight_shift.csv')
assert len(runs) == 360, len(runs)

# ======================= Fig. 1: study overview =======================
def fig1():
    split = json.load(open(EX / 'splits_summary.json'))
    n = {s: {k: sum(v.values()) for k, v in split[s]['counts'].items()} for s in ('random', 'time')}
    drop = split['split_time_dropped_twins']
    fig = plt.figure(figsize=(W, 4.4)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis('off')
    def box(x, y, w, h, txt, fc='#f4f4f2', ec='#8a8a86', bold=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.4,rounding_size=1.5', fc=fc, ec=ec, lw=0.6))
        ax.text(x + w / 2, y + h / 2, txt, ha='center', va='center', fontsize=6.3, color=INK,
                fontweight='bold' if bold else 'normal', linespacing=1.25)
    def arrow(x0, y0, x1, y1):
        ax.annotate('', xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle='-|>', lw=0.6, color='#5f5f5f', mutation_scale=6))
    box(20, 90, 60, 7.5, 'CIC IoT-DIAD 2024 flows (8 classes)', bold=True)
    arrow(50, 89.3, 28, 81.5); arrow(50, 89.3, 72, 81.5)
    box(3, 68, 44, 13, f'Random stratified split\n(duplicate-safe)\n{n["random"]["train"]:,} / {n["random"]["val"]:,} / {n["random"]["test"]:,}')
    box(53, 68, 44, 13, f'Per-class time-block split\n(train twins removed)\n{n["time"]["train"]:,} / {n["time"]["val"]:,} / {n["time"]["test"]:,}')
    arrow(25, 67.3, 45, 61.5); arrow(75, 67.3, 55, 61.5)
    box(20, 53.5, 60, 7.5, f'Train-only cleaning and scaling, {split["random"]["n_features"]} features')
    arrow(50, 52.8, 50, 47.3)
    box(8, 36, 84, 11, '10 clients: Dirichlet α = 0.1, α = 0.5, IID, source IP\n(time split: α = 0.1 and source IP)')
    arrow(50, 35.3, 50, 29.8)
    box(4, 17, 92, 12.8, 'FA-MLP, 100 rounds, 10 seeds per cell\nFedAvg | FedProx | SCAFFOLD | FedDyn | FedAdapt-F | FedAdapt-pure\nAdded later: FedAdapt-inv | FedEntropy-F | FedNova', fc='#e8f0f8', ec='#0072B2')
    arrow(50, 16.3, 50, 13.3)
    box(4, 1, 92, 11.5, 'Test macro-F1 (mean of rounds 80/90/100)\npaired tests vs FedAvg with Holm correction\nwindowing audit: label-grouped, time-ordered, random-order', fc='#f4f4f2')
    save(fig, 'fig1_overview')

# ======================= Fig. 2: windowing pitfall (3 seeds, A4 multi-seed audit) =======================
def fig2():
    MS = json.load(open(EX2 / 'phase0_p0_3c_multiseed.json'))
    R = pd.DataFrame(MS['runs'])
    names = [('TACNet centralised', 'TACNet'), ('FedProx a=0.1', 'FedProx (old)'), ('FedAvg a=0.1', 'FedAvg (old)')]
    lgb = {'LightGBM probabilities averaged over the window': 'LightGBM, window avg.', 'LightGBM single flow (no window)': 'LightGBM, single flow'}
    cols = [('label_grouped', 'label_grouped_windows', 'Label-grouped', '#7a7a7a', ''),
            ('time_ordered', 'time_ordered_windows_dayfirst', 'Time-ordered', '#0072B2', '////'),
            ('random_order', 'random_order_windows', 'Random-order', '#E69F00', '')]
    d = C1.set_index('model')
    labels = [n[1] for n in names] + list(lgb.values())
    fig, ax = plt.subplots(figsize=(W, H), layout='constrained'); ax_style(ax)
    y = np.arange(len(labels)); h = 0.26
    for j, (c, c1, lab, colr, hatch) in enumerate(cols):
        g = R.groupby('model')[c]
        assert all(g.count()[m] == 3 for m, _ in names)
        mu = [g.mean()[m] for m, _ in names] + [d.loc[k, c1] for k in lgb]
        sd = [g.std(ddof=1)[m] for m, _ in names] + [0, 0]
        ax.barh(y + (j - 1) * h, mu, height=h * 0.88, color=colr, hatch=hatch, edgecolor='white', lw=0.3, label=lab,
                xerr=[s if s > 0 else np.nan for s in sd], error_kw=dict(lw=0.6, capsize=1.5, capthick=0.6, ecolor=INK))
    ax.set_yticks(y); ax.set_yticklabels(labels); ax.invert_yaxis()
    ax.set_xlim(0, 1.0); ax.set_xlabel('Test macro-F1 (mean ± SD, 3 seeds)')
    ax.xaxis.grid(True, color=GRID, lw=0.5); ax.yaxis.grid(False)
    fig.legend(loc='outside upper center', ncol=3, frameon=False, handlelength=1.2, columnspacing=1.0)
    save(fig, 'fig2a_windowing_f1', [ax])

    fig, ax = plt.subplots(figsize=(W, H), layout='constrained'); ax_style(ax)
    v = [MS['window_purity'][c] for c, *_ in cols]
    bars = ax.bar(range(3), v, width=0.55, color=[c[3] for c in cols], hatch=None, edgecolor='white')
    for b, c in zip(bars, cols): b.set_hatch(c[4])
    for i, val in enumerate(v): ax.text(i, val + 0.02, f'{val:.3f}', ha='center', va='bottom', fontsize=6.5, color=INK)
    ax.set_xticks(range(3)); ax.set_xticklabels([c[2] for c in cols]); ax.set_ylim(0, 1.12)
    ax.set_ylabel('Share of single-class windows')
    save(fig, 'fig2b_window_purity')

# ======================= Fig. 3: final macro-F1 by setting and rule =======================
def fig3():
    A = runs.groupby(['setting', 'method']).last3.agg(['mean', 'std']).reset_index()
    for name, sets, split in (('fig3a_random_split', ['random_dirichlet0.1', 'random_dirichlet0.5', 'random_ip', 'random_iid'], 'random'),
                              ('fig3b_time_split', ['time_dirichlet0.1', 'time_ip'], 'time')):
        fig, ax = plt.subplots(figsize=(W, H), layout='constrained'); ax_style(ax)
        off = np.linspace(-0.3, 0.3, len(METHODS))
        for k, m in enumerate(METHODS):
            g = A[A.method == m].set_index('setting').loc[sets]
            ax.errorbar(np.arange(len(sets)) + off[k], g['mean'], yerr=g['std'], fmt=MRK[m], color=COL[m], ms=3.6,
                        mec='white', mew=0.3, elinewidth=0.7, capsize=1.2, label=NAME[m])
        cref = cen[(cen.split == split) & (cen.model == 'FA-MLP')].test_macro_f1.mean()
        ax.axhline(cref, color=MUTED, lw=0.7, ls=(0, (4, 2)), label='_ref')
        ax.text(len(sets) - 0.5, cref + 0.004, f'Centralised FA-MLP {cref:.3f}', ha='right', va='bottom', fontsize=6.2, color=MUTED)
        ax.set_xticks(range(len(sets))); ax.set_xticklabels([SET_LABEL[s] for s in sets]); ax.set_xlim(-0.5, len(sets) - 0.5)
        ax.set_ylabel('Test macro-F1 (mean ± SD, 10 seeds)')
        lo = (A[A.setting.isin(sets)]['mean'] - A[A.setting.isin(sets)]['std']).min()
        ax.set_ylim(np.floor((lo - 0.02) * 20) / 20, cref + 0.10 if split == 'random' else cref + 0.12)
        ax.legend(loc='upper left', ncol=3, frameon=False, columnspacing=0.8, handletextpad=0.3, borderaxespad=0.2)
        save(fig, name, [ax])

# ======================= Fig. 4: convergence =======================
def fig4():
    for name, st in (('fig4a_conv_random_a01', 'random_dirichlet0.1'), ('fig4b_conv_time_ip', 'time_ip')):
        fig, ax = plt.subplots(figsize=(W, H), layout='constrained'); ax_style(ax)
        h = hist[hist.setting == st]
        for m in METHODS:
            g = h[h.method == m].groupby('round').test_macro_f1.agg(['mean', 'std'])
            ax.fill_between(g.index, g['mean'] - g['std'], g['mean'] + g['std'], color=COL[m], alpha=0.10, lw=0)
            ax.plot(g.index, g['mean'], color=COL[m], ls=LS[m], lw=1.0, marker=MRK[m], ms=2.6, mec='white', mew=0.25, label=NAME[m])
        ax.set_xlabel('Communication round'); ax.set_ylabel('Test macro-F1 (mean ± SD, 10 seeds)')
        ax.set_xlim(0, 102); ax.set_xticks([1, 20, 40, 60, 80, 100])
        top = (h.groupby(['method', 'round']).test_macro_f1.mean()).max()
        ax.set_ylim(0, top + 0.16)
        ax.legend(loc='upper left', ncol=3, frameon=False, columnspacing=0.8, handletextpad=0.3, borderaxespad=0.2, handlelength=1.8)
        save(fig, name, [ax])

# ======================= Fig. 5: paired difference vs FedAvg (95% CI) =======================
def fig5():
    for name, sets in (('fig5a_diff_random', ['random_dirichlet0.1', 'random_dirichlet0.5', 'random_ip', 'random_iid']),
                       ('fig5b_diff_time', ['time_dirichlet0.1', 'time_ip'])):
        fig, ax = plt.subplots(figsize=(W, H), layout='constrained'); ax_style(ax, ygrid=False); ax.xaxis.grid(False)
        rules = METHODS[1:]; off = np.linspace(-0.3, 0.3, len(rules))
        for k, m in enumerate(rules):
            g = TB[TB.method == m].set_index('setting').loc[sets]
            x = np.arange(len(sets)) + off[k]
            ax.errorbar(x, g.diff_mean * 100, yerr=[(g.diff_mean - g.ci_lo) * 100, (g.ci_hi - g.diff_mean) * 100], fmt='none',
                        ecolor=COL[m], elinewidth=0.7, capsize=1.2)
            sig = g.sig_holm.values.astype(bool)
            ax.scatter(x[sig], (g.diff_mean * 100).values[sig], marker=MRK[m], s=13, color=COL[m], edgecolor='white', lw=0.3, zorder=3)
            ax.scatter(x[~sig], (g.diff_mean * 100).values[~sig], marker=MRK[m], s=13, facecolor='white', edgecolor=COL[m], lw=0.7, zorder=3)
        ax.axhline(0, color=MUTED, lw=0.7, label='_ref')
        ax.set_xticks(range(len(sets))); ax.set_xticklabels([SET_LABEL[s] for s in sets]); ax.set_xlim(-0.5, len(sets) - 0.5)
        ax.set_ylabel('Δ macro-F1 vs FedAvg (pp, 95% CI)')
        sub = TB[TB.setting.isin(sets)]
        ymin, ymax = sub.ci_lo.min() * 100, sub.ci_hi.max() * 100
        ax.set_ylim(np.floor(ymin) - 1, np.ceil(ymax) + 6.5)
        from matplotlib.lines import Line2D
        hd = [Line2D([], [], ls='none', marker=MRK[m], color=COL[m], mec=COL[m], ms=3.6) for m in rules]
        lb = [NAME[m] for m in rules]
        hd += [Line2D([], [], ls='none', marker='o', mfc=MUTED, mec=MUTED, ms=3.6), Line2D([], [], ls='none', marker='o', mfc='white', mec=MUTED, ms=3.6)]
        lb += ['Holm p < 0.05', 'not significant']
        ax.legend(hd, lb, loc='upper left', ncol=3, frameon=False, columnspacing=0.8, handletextpad=0.2, borderaxespad=0.2)
        save(fig, name, [ax])

# ======================= Fig. 7: per-class F1 (final round) =======================
def fig6():
    for name, st in (('fig7a_perclass_random_a01', 'random_dirichlet0.1'), ('fig7b_perclass_time_a01', 'time_dirichlet0.1')):
        M = runs[runs.setting == st].groupby('method')[[f'pc_{c}' for c in CLS]].mean().reindex(METHODS)
        fig, ax = plt.subplots(figsize=(W, H))
        im = ax.imshow(M.values, cmap='Blues', vmin=0, vmax=1, aspect='auto')
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                v = M.values[i, j]
                ax.text(j, i, f'{v:.2f}', ha='center', va='center', fontsize=5.6, color='white' if v > 0.6 else INK)
        ax.set_xticks(range(len(CLS))); ax.set_xticklabels(CLS_SHORT, fontsize=5.9, linespacing=1.0)
        ax.set_yticks(range(len(METHODS))); ax.set_yticklabels([NAME[m] for m in METHODS])
        for s in ax.spines.values(): s.set_visible(False)
        ax.tick_params(length=0)
        cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02); cb.ax.tick_params(labelsize=6, length=2); cb.outline.set_linewidth(0.4)
        cb.set_label('Test F1 (final round, mean of 10 seeds)', fontsize=6.3)
        fig.subplots_adjust(left=0.22, right=0.87, bottom=0.13, top=0.98)
        save(fig, name)

# ======================= Fig. 6: learning-rate control and FedAdapt-F weight shift =======================
def fig7():
    sets = ['random_dirichlet0.5', 'random_ip', 'random_iid']
    arms = [('fedavg_lr0.3', 'FedAvg, lr 0.3', COL['fedavg'], ''), ('fedavg_lr0.03', 'FedAvg, lr 0.03', '#9ecae9', '////'),
            ('feddyn_lr0.03', 'FedDyn, lr 0.03', COL['feddyn'], '')]
    fig, ax = plt.subplots(figsize=(W, H), layout='constrained'); ax_style(ax)
    g = lr.groupby('setting')[[a[0] for a in arms]].agg(['mean', 'std'])
    w = 0.25
    for j, (c, lab, colr, hatch) in enumerate(arms):
        x = np.arange(len(sets)) + (j - 1) * w
        ax.bar(x, g.loc[sets, (c, 'mean')], width=w * 0.9, yerr=g.loc[sets, (c, 'std')], color=colr, hatch=hatch,
               edgecolor='white', lw=0.3, error_kw=dict(elinewidth=0.6, capsize=1.2, ecolor=INK), label=lab)
    ax.set_xticks(range(len(sets))); ax.set_xticklabels([SET_LABEL[s] for s in sets])
    ax.set_ylim(0.40, 0.62); ax.set_ylabel('Test macro-F1 (mean ± SD, 10 seeds)')
    ax.legend(loc='upper left', ncol=3, frameon=False, columnspacing=0.8, handlelength=1.2, borderaxespad=0.2)
    save(fig, 'fig6a_lr_control', [ax])

    order = ['random_iid', 'random_dirichlet0.5', 'random_ip', 'random_dirichlet0.1', 'time_dirichlet0.1', 'time_ip']
    lab = {'random_iid': 'R: IID', 'random_dirichlet0.5': 'R: α=0.5', 'random_ip': 'R: IP', 'random_dirichlet0.1': 'R: α=0.1',
           'time_dirichlet0.1': 'T: α=0.1', 'time_ip': 'T: IP'}
    fig, ax = plt.subplots(figsize=(W, H), layout='constrained'); ax_style(ax)
    rng = np.random.default_rng(0)
    for i, st in enumerate(order):
        v = ws[ws.setting == st].L1_dist_fedadapt_vs_fedavg_weights.values
        ax.scatter(i + rng.uniform(-0.12, 0.12, len(v)), v, s=7, marker=MRK['fedadapt'], facecolor='white', edgecolor=COL['fedadapt'], lw=0.6, zorder=2)
        ax.plot([i - 0.22, i + 0.22], [v.mean()] * 2, color=INK, lw=1.0, zorder=3)
    ax.set_xticks(range(len(order))); ax.set_xticklabels([lab[s] for s in order])
    ax.set_ylabel('L1 distance, FedAdapt-F vs FedAvg weights'); ax.set_ylim(0, None)
    save(fig, 'fig6b_weight_shift')

# ======================= Partition heterogeneity (Phase 2d export) =======================
EX4 = Path(sys.argv[1]).parent / 'export2d'
OKABE8 = ['#000000', '#E69F00', '#56B4E9', '#009E73', '#F0E442', '#0072B2', '#D55E00', '#CC79A7']
CLS_NAME = ['Benign', 'Brute Force', 'DDoS', 'DoS', 'Mirai', 'Recon', 'Spoofing', 'Web-Based']
def fig_partitions():
    CC = pd.read_csv(EX4 / 'partition_client_class_counts.csv'); SU = pd.read_csv(EX4 / 'partition_client_summary.csv')
    d = CC[(CC.setting == 'random_dirichlet0.1') & (CC.seed == 42)].sort_values('client')
    M = d[CLS].values.astype(float); n = M.sum(1); F = M / n[:, None]
    assert len(d) == 10 and (n == d.n.values).all()
    fig, ax = plt.subplots(figsize=(W, H + 0.35), layout='constrained'); ax_style(ax, ygrid=False)
    left = np.zeros(len(d)); y = np.arange(len(d))
    for j in range(8):
        ax.barh(y, F[:, j], left=left, height=0.72, color=OKABE8[j], edgecolor='white', lw=0.3, label=CLS_NAME[j]); left += F[:, j]
    ax.set_yticks(y); ax.set_yticklabels([f'C{k + 1} (n={int(v):,})' for k, v in zip(d.client, n)]); ax.invert_yaxis()
    ax.set_xlim(0, 1); ax.set_xlabel('Share of client flows')
    ax.xaxis.grid(True, color=GRID, lw=0.5); ax.set_axisbelow(True)
    fig.legend(loc='outside upper center', ncol=4, frameon=False, handlelength=1.0, columnspacing=0.8)
    save(fig, 'figP_a_client_composition')

    order = ['random_iid', 'random_ip', 'random_dirichlet0.5', 'random_dirichlet0.1', 'time_ip', 'time_dirichlet0.1']
    lab = {'random_iid': 'R: IID', 'random_ip': 'R: IP', 'random_dirichlet0.5': 'R: α=0.5', 'random_dirichlet0.1': 'R: α=0.1',
           'time_ip': 'T: IP', 'time_dirichlet0.1': 'T: α=0.1'}
    fig, ax = plt.subplots(figsize=(W, H + 0.35), layout='constrained'); ax_style(ax)
    rng = np.random.default_rng(0)
    for i, st in enumerate(order):
        v = SU[SU.setting == st].label_entropy.values; assert len(v) == 100
        ax.scatter(i + rng.uniform(-0.18, 0.18, len(v)), v, s=4, color='#0072B2', alpha=0.55, lw=0, zorder=2)
        ax.plot([i - 0.28, i + 0.28], [v.mean()] * 2, color=INK, lw=1.0, zorder=3)
    ax.axhline(np.log(8), color=MUTED, lw=0.7, ls='--', zorder=1, label='_ref')
    ax.text(len(order) - 0.5, np.log(8) + 0.04, 'ln 8', ha='right', va='bottom', fontsize=6.3, color=MUTED)
    ax.set_xticks(range(len(order))); ax.set_xticklabels([lab[s] for s in order])
    ax.set_ylim(0, 2.35); ax.set_ylabel('Client label entropy (nats)')
    save(fig, 'figP_b_label_entropy')

# ======================= Effect sizes: protocol vs rule =======================
def effect_rows():
    MS = pd.DataFrame(json.load(open(EX2 / 'phase0_p0_3c_multiseed.json'))['runs'])
    lift = (MS.label_grouped - MS.random_order) * 100
    r = pd.read_csv(EX / 'grid_runs.csv'); e = pd.read_csv(EX2 / 'grid_runs_ext.csv'); e = e[e.method != 'fedavg']
    allr = pd.concat([r[['setting', 'method', 'seed', 'last3']], e[['setting', 'method', 'seed', 'last3']]])
    P = allr.pivot_table(index=['method', 'seed'], columns='setting', values='last3')
    cost = []
    for s in ('dirichlet0.1', 'ip'):
        g = ((P['random_' + s] - P['time_' + s]).groupby('method').mean() * 100).dropna(); cost += g.tolist()
    cen = pd.read_csv(EX / 'central_summary.csv'); fa = cen[(cen.split == 'random') & (cen.model == 'FA-MLP')].test_macro_f1.mean()
    fav = r[(r.setting == 'random_dirichlet0.1') & (r.method == 'fedavg')].last3.mean()
    TB = pd.read_csv(EX / 'table_B_vs_fedavg.csv')
    pos = TB[(TB.p_holm < 0.05) & (TB.diff_mean > 0)].diff_mean * 100
    fa_row = TB[(TB.setting == 'random_dirichlet0.1') & (TB.method == 'fedadapt')].iloc[0]
    return [('Label-grouped windows\n(lift, 9 checkpoints)', lift.mean(), lift.min(), lift.max(), '#7a7a7a'),
            ('Federated vs centralised\n(FedAvg, α=0.1)', (fa - fav) * 100, None, None, '#7a7a7a'),
            ('Time-block split\n(cost, 17 rule–partition pairs)', float(np.mean(cost)), min(cost), max(cost), '#7a7a7a'),
            ('Best rule vs FedAvg\n(Holm-significant gains)', pos.max(), pos.min(), pos.max(), '#0072B2'),
            ('FedAdapt-F vs FedAvg\n(α=0.1, 95% CI)', fa_row.diff_mean * 100, fa_row.ci_lo * 100, fa_row.ci_hi * 100, '#0072B2')]

def draw_effects(ax, rows):
    for i, (lab, v, lo, hi, c) in enumerate(rows):
        if lo is not None: ax.plot([lo, hi], [i, i], color=c, lw=2.2, solid_capstyle='butt', zorder=2)
        ax.plot([v], [i], marker='o', ms=4.5, color=c, mec='white', mew=0.6, zorder=3)
        ax.text(max(v, hi if hi is not None else v) + 2.0, i, f'{v:.1f}', va='center', ha='left', fontsize=6.5, color=INK)
    ax.axvline(0, color=MUTED, lw=0.6)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows]); ax.invert_yaxis()
    ax.set_xlim(-5, 95); ax.set_xlabel('Effect on test macro-F1 (pp)')
    ax.xaxis.grid(True, color=GRID, lw=0.5); ax.yaxis.grid(False); ax.set_axisbelow(True)

def fig_effects():
    rows = effect_rows()
    for lab, v, lo, hi, c in rows: print(f'   effect: {lab.replace(chr(10), " ")}: {v:.2f} [{lo}, {hi}]')
    fig, ax = plt.subplots(figsize=(W, H + 0.2), layout='constrained'); ax_style(ax, ygrid=False)
    draw_effects(ax, rows); save(fig, 'fig8_effect_sizes')
    # graphical abstract: wide, 300 dpi, PDF + TIFF
    fig, ax = plt.subplots(figsize=(7.2, 3.0), layout='constrained'); ax_style(ax, ygrid=False)
    draw_effects(ax, rows)
    ax.set_title('Evaluation protocol vs aggregation rule in federated IoT intrusion detection (CIC IoT-DIAD 2024)', fontsize=7.5, color=INK, loc='left')
    p = check(fig, 'graphical_abstract'); REPORT['graphical_abstract'] = p
    fig.savefig(OUT / 'graphical_abstract.pdf', facecolor='white'); fig.savefig(OUT / 'graphical_abstract.tif', dpi=300, facecolor='white')
    plt.close(fig); print('graphical_abstract', 'PASS' if not p else p)

for f in (fig1, fig2, fig3, fig4, fig5, fig6, fig7, fig_partitions, fig_effects): f()
bad = {k: v for k, v in REPORT.items() if v}
print(f'\n{len(REPORT)} panels, {len(bad)} failing')
sys.exit(1 if bad else 0)
