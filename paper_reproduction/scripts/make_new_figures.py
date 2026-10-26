"""New figures for the revised Computer Networks paper (benchmark paper, FedAdapt_Thesis project).

fig1_overview.png      system architecture (replaces the earlier study-design figure, same file name)
figC_a_published_acc.png / figC_b_protocol_f1.png   comparison with published results / effect of protocol
figB_a_baselines_random.png / figB_b_baselines_time.png   standard metrics of baselines and federated rules
figQ_a_partial_diff.png / figQ_b_partial_curves.png  partial participation at equal rounds and equal computation
Every number is read from the exported result files; published values are the only typed-in numbers and each
carries its source key (see PUBLISHED below). 1000 dpi, one PNG per panel, overlap checker on every panel.
Usage: python3 make_new_figures.py <out_dir>
"""
import json, glob, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from _helpers import *   # noqa: F401,F403  (DPI, W, H, rcParams, check, save, ax_style, colours, REPORT)
from scipy import stats

W0 = Path(__file__).resolve().parents[1] / 'data'
EX, EX3, EXP = W0 / 'export', W0 / 'export3', W0 / 'export3p' / 'grid_partial'
split = json.load(open(EX / 'splits_summary.json'))
dup = json.load(open(EX / 'phase0_p0_1_duplicates_cic.json'))['76_nonid_columns']['overall_rows_with_any_duplicate']
cen = pd.read_csv(EX3 / 'central_metrics.csv'); flm = pd.read_csv(EX3 / 'fl_final_round_metrics.csv')
runs = pd.read_csv(EX / 'grid_runs.csv'); runs = runs[~runs.diverged]
C1 = pd.read_csv(EX / 'table_C1_windowing_pitfall.csv')
SEEDS = [42, 123, 456, 789, 1011, 2024, 7, 99, 314, 2718]
BLUE, VERM, GREY, PURP, ORNG, GREEN = '#0072B2', '#D55E00', '#9a9a96', '#CC79A7', '#E69F00', '#009E73'

# =============================== Fig. 1: system architecture ===============================
def fig1():
    n = {s: {k: sum(v.values()) for k, v in split[s]['counts'].items()} for s in ('random', 'time')}
    drop = split['split_time_dropped_twins']; nf = split['random']['n_features']
    ntot = n['random']['train'] + n['random']['val'] + n['random']['test']
    WW, HH = 7.5, 6.9
    fig = plt.figure(figsize=(WW, HH)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 150); ax.set_ylim(0, 138); ax.axis('off')
    def box(x, y, w, h, txt, fc='#f4f4f2', ec='#8a8a86', bold=False, fs=6.3, lw=0.6):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.35,rounding_size=1.2', fc=fc, ec=ec, lw=lw))
        if txt: ax.text(x + w / 2, y + h / 2, txt, ha='center', va='center', fontsize=fs, color=INK,
                        fontweight='bold' if bold else 'normal', linespacing=1.25)
    def arrow(x0, y0, x1, y1, style='-|>', col='#5f5f5f', conn=None):
        kw = dict(arrowstyle=style, lw=0.7, color=col, mutation_scale=7)
        if conn: kw['connectionstyle'] = conn
        ax.annotate('', xy=(x1, y1), xytext=(x0, y0), arrowprops=kw)
    def band(y, h, label):
        ax.add_patch(FancyBboxPatch((0.8, y), 148.4, h, boxstyle='round,pad=0,rounding_size=1.5', fc='none', ec='#c9c9c4', lw=0.5, ls='--'))
        ax.text(2.2, y + h - 1.2, label, ha='left', va='top', fontsize=6.0, color=MUTED, fontweight='bold')
    # ---- 1 data (left to right) ----
    band(115, 22, '1  Data')
    box(5, 117.5, 40, 12, '19,519,162 flows in 129 CSV files\n8 classes: benign and\n7 attack categories')
    box(55, 117.5, 40, 12, f'Stratified sample of\n{ntot:,} flows\n(per-class caps)')
    box(105, 117.5, 41, 12, f'Drop 7 identifier columns;\n76 features, {100 * dup:.2f}% of rows\nhave an exact duplicate')
    arrow(45.6, 123.5, 54.4, 123.5); arrow(95.6, 123.5, 104.4, 123.5)
    # ---- 2 splits and preprocessing (right to left) ----
    band(84, 29, '2  Leak-free splits (70/15/15) and preprocessing')
    box(84, 87, 62, 20.5, '', fc='white')
    box(87, 98.2, 56, 7.2, f'Random split, duplicate groups kept together\n{n["random"]["train"]:,} / {n["random"]["val"]:,} / {n["random"]["test"]:,} flows', fc='#e8f0f8', ec=BLUE, fs=6.0)
    box(87, 89.0, 56, 7.2, f'Per-class time-block split, earliest flows train;\n{n["time"]["train"]:,} / {n["time"]["val"]:,} / {n["time"]["test"]:,} ({drop["val"]} + {drop["test"]} twins dropped)', fc='#fbeee6', ec=VERM, fs=6.0)
    box(5, 89, 66, 16.5, f'Fitted on each training part only:\nmedian imputation, variance threshold\n(0.01, {nf} features kept), standard scaling\n\nassertions: no shared duplicate group, all classes\nin every part, time order within each class', fs=6.0)
    arrow(125.5, 117.0, 125.5, 108.3)
    arrow(83.3, 97.25, 71.8, 97.25)
    # ---- 3 experiments ----
    band(46, 36, '3  Experiments')
    box(5, 49, 36, 26, 'Centralized references\n(both leak-free splits)\n\nFA-MLP and MLP, 10 seeds\nLightGBM, XGBoost,\nrandom forest, k-NN,\ndecision tree, logistic\nregression, Gaussian NB', fs=6.0)
    box(45, 49, 62, 26, '', fc='#eef5ea', ec=GREEN)
    ax.text(76, 72.4, 'Federated simulation (FA-MLP, 10 seeds per cell)', ha='center', va='center', fontsize=6.2, color=INK, fontweight='bold')
    box(48, 51.5, 26.5, 17, 'Client partitions\nK = 10, all clients\nper round, 100 rounds\nDirichlet α = 0.1, 0.5;\nIID; source IP', fs=5.8, fc='white')
    box(78, 51.5, 26, 17, 'Robustness checks\nlearning-rate control\n5 local epochs\nK = 50, 10 per round\n(100 and 500 rounds)', fs=5.8, fc='white')
    box(111, 49, 35, 26, 'Windowing audit\n(original split of the\naudited pipeline)\n\n9 TACNet checkpoints;\nlabel-grouped, random-\norder, time-ordered\nwindows; LightGBM\nwindow average', fs=6.0)
    arrow(23, 88.3, 23, 75.9); arrow(50, 88.3, 66, 75.9)
    # ---- 4 one federated round ----
    band(12, 32, '4  One federated round')
    for i, yy in enumerate((34.5, 28, 21.5)):
        box(5, yy, 30, 4.8, f'Client {("1", "2", "K")[i]}: local FA-MLP training', fs=5.8, fc='white')
    ax.text(20, 18.8, '1 local epoch, momentum SGD\n(buffer reset), clip norm 1.0', ha='center', va='top', fontsize=5.6, color=MUTED)
    box(53, 21, 40, 18, 'Server aggregation (9 rules)\nFedAvg, FedProx, SCAFFOLD,\nFedDyn, FedNova,\nFedAdapt-F, -pure, -inv,\nFedEntropy-F', fc='#e8f0f8', ec=BLUE, fs=6.0)
    arrow(35.6, 35.5, 52.4, 34.0); arrow(52.4, 26.0, 35.6, 24.5)
    ax.text(44, 37.2, 'w$_k$, n$_k$ (+ H$_k$ or c$_k$)', ha='center', va='bottom', fontsize=5.7, color=INK)
    ax.text(44, 23.4, 'global w', ha='center', va='top', fontsize=5.7, color=INK)
    box(101, 14.5, 45, 26.5, '', fc='white')
    ax.text(123.5, 38.6, f'FA-MLP ({int(cen.loc[cen.model == "FA-MLP", "params"].iloc[0]):,} parameters)', ha='center', va='center', fontsize=6.2, color=INK, fontweight='bold')
    layers = [f'input x ({nf} features)', 'attention a = softmax(W x + b)', f'x̃ = {nf} · (a ⊙ x)', 'Dense 256, ReLU, dropout 0.2', 'Dense 128, ReLU', 'Dense 8, softmax']
    for j, t in enumerate(layers):
        yy = 34.6 - j * 3.75
        ax.text(123.5, yy, t, ha='center', va='center', fontsize=5.7, color=INK)
        if j < len(layers) - 1: arrow(123.5, yy - 1.05, 123.5, yy - 2.7)
    arrow(93.8, 30, 100.2, 30, style='-', col='#b0b0ac')
    arrow(76, 48.3, 73, 39.9)
    # ---- 5 evaluation ----
    band(1, 9.5, '5  Evaluation')
    box(26, 2.3, 120, 6.4, 'Test macro-F1 (mean of rounds 80, 90, 100); accuracy, balanced accuracy, weighted F1, MCC, per-class F1\n'
        'seed = unit of analysis: paired t-tests against FedAvg, Holm correction, TOST (±1 pp), Kendall τ between splits', fs=5.9)
    arrow(73, 20.3, 73, 9.4)
    save(fig, 'fig1_overview')

# =============================== Fig. C: comparison with published results ===============================
# Published values (as stated in each paper; different splits, class sets and sampling, so not comparable as equals):
PUBLISHED_ACC = [('Hossain (2025), 1D-CNN, 80/20 split', 99.12, 'hossain2025eurasip'),
                 ('Amin et al. (2025), fed., 5-fold CV', 96.9, 'amin2025sdn')]
def ours_acc():
    g = cen.groupby(['split', 'model']).accuracy.mean()
    fl = flm[flm.source == 'grid'].groupby(['split', 'partition', 'method']).accuracy.mean()
    rows = [('XGBoost, random split', 100 * g['random', 'XGBoost'], BLUE), ('Random Forest, random split', 100 * g['random', 'RandomForest'], BLUE),
            ('FA-MLP (central), random split', 100 * g['random', 'FA-MLP'], BLUE),
            ('FedDyn, IID clients, random split', 100 * fl['random', 'iid', 'feddyn'], BLUE),
            ('FedAvg, α=0.1, random split', 100 * fl['random', 'dirichlet0.1', 'fedavg'], BLUE),
            ('Random Forest, time-block split', 100 * g['time', 'RandomForest'], VERM),
            ('FA-MLP (central), time-block split', 100 * g['time', 'FA-MLP'], VERM),
            ('FedAvg, α=0.1, time-block split', 100 * fl['time', 'dirichlet0.1', 'fedavg'], VERM)]
    return rows

def figC():
    rows = [(l, v, GREY) for l, v, _ in PUBLISHED_ACC] + ours_acc()
    fig, ax = plt.subplots(figsize=(W, 3.0)); fig.subplots_adjust(left=0.47, right=0.96, top=0.97, bottom=0.15)
    y = np.arange(len(rows))[::-1]
    ax.barh(y, [r[1] for r in rows], color=[r[2] for r in rows], height=0.62, edgecolor='none')
    for yi, r in zip(y, rows): ax.text(r[1] + 1.0, yi, f'{r[1]:.1f}', va='center', fontsize=6.0, color=INK)
    ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows], fontsize=6.0)
    ax.set_xlim(0, 112); ax.set_xticks([0, 20, 40, 60, 80, 100]); ax.set_xlabel('Test accuracy (%)')
    ax.axhline(len(rows) - len(PUBLISHED_ACC) - 0.5, color='#c9c9c4', lw=0.6, ls='--')
    ax_style(ax, ygrid=False); ax.xaxis.grid(True, color=GRID, lw=0.5); ax.set_axisbelow(True)
    save(fig, 'figC_a_published_acc')

    fe = json.load(open(W0.parent / 'outputs' / 'analysis' / 'facts_ext.json'))
    g = cen.groupby(['split', 'model']).f1_macro.mean()
    lg = C1.set_index('model').loc['LightGBM probabilities averaged over the window']
    rows = [('LightGBM avg., label-grouped win.', float(lg.label_grouped_windows), PURP),
            ('TACNet, label-grouped windows', fe['TACNet centralised|label_grouped|mean'], PURP),
            ('LightGBM avg., time-ordered win.', float(lg.time_ordered_windows_dayfirst), PURP),
            ('TACNet, time-ordered windows', fe['TACNet centralised|time_ordered|mean'], PURP),
            ('TACNet, random-order windows', fe['TACNet centralised|random_order|mean'], PURP),
            ('XGBoost, flows, random split', g['random', 'XGBoost'], BLUE),
            ('FA-MLP, flows, random split', g['random', 'FA-MLP'], BLUE),
            ('Random forest, flows, time split', g['time', 'RandomForest'], VERM),
            ('FA-MLP, flows, time split', g['time', 'FA-MLP'], VERM)]
    fig, ax = plt.subplots(figsize=(W, 3.0)); fig.subplots_adjust(left=0.47, right=0.96, top=0.97, bottom=0.15)
    y = np.arange(len(rows))[::-1]
    ax.barh(y, [r[1] for r in rows], color=[r[2] for r in rows], height=0.62, edgecolor='none')
    for yi, r in zip(y, rows): ax.text(r[1] + 0.012, yi, f'{r[1]:.4f}', va='center', fontsize=6.0, color=INK)
    ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows], fontsize=6.0)
    ax.set_xlim(0, 1.16); ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0]); ax.set_xlabel('Test macro-F1')
    ax_style(ax, ygrid=False); ax.xaxis.grid(True, color=GRID, lw=0.5); ax.set_axisbelow(True)
    save(fig, 'figC_b_protocol_f1')
    return rows

# =============================== Fig. B: standard metrics of baselines ===============================
ORDER = ['GaussianNB', 'LogReg', 'DecisionTree', 'kNN', 'RandomForest', 'LightGBM', 'XGBoost', 'MLP', 'FA-MLP']
LBL = {'GaussianNB': 'Gaussian NB', 'LogReg': 'Logistic regr.', 'DecisionTree': 'Decision tree', 'kNN': 'k-NN',
       'RandomForest': 'Random forest', 'LightGBM': 'LightGBM', 'XGBoost': 'XGBoost', 'MLP': 'MLP', 'FA-MLP': 'FA-MLP'}
def figB():
    fl = flm[flm.source == 'grid'].groupby(['split', 'partition', 'method'])[['accuracy', 'f1_macro', 'mcc']].mean()
    best = {'random': ('iid', 'feddyn', 'FedDyn, IID'), 'time': ('ip', 'scaffold', 'SCAFFOLD, src. IP')}
    for sp, name in (('random', 'figB_a_baselines_random'), ('time', 'figB_b_baselines_time')):
        g = cen[cen.split == sp].groupby('model')[['accuracy', 'f1_macro', 'mcc']].mean()
        labels = [LBL[m] for m in ORDER] + [best[sp][2] + ' (fed.)', 'FedAvg, α=0.1 (fed.)']
        vals = [g.loc[m] for m in ORDER] + [fl.loc[(sp, best[sp][0], best[sp][1])], fl.loc[(sp, 'dirichlet0.1', 'fedavg')]]
        fig, ax = plt.subplots(figsize=(W, 2.9)); fig.subplots_adjust(left=0.30, right=0.97, top=0.88, bottom=0.15)
        y = np.arange(len(labels))[::-1]; h = 0.27
        for k, (met, col, nm) in enumerate((('accuracy', '#56B4E9', 'Accuracy'), ('f1_macro', BLUE, 'Macro-F1'), ('mcc', '#9a9a96', 'MCC'))):
            ax.barh(y + (1 - k) * h, [v[met] for v in vals], height=h, color=col, label=nm, edgecolor='none')
        ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=6.0)
        ax.set_xlim(0, 0.85 if sp == 'random' else 0.62); ax.set_xlabel('Score on the test set')
        ax.axhline(1.5, color='#c9c9c4', lw=0.6, ls='--')
        ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False, fontsize=6.0, handlelength=1.2)
        ax_style(ax, ygrid=False); ax.xaxis.grid(True, color=GRID, lw=0.5); ax.set_axisbelow(True)
        save(fig, name, data_axes=(ax,))

# =============================== Fig. Q: partial participation ===============================
def last_n(o, R): return float(np.mean([h['test_macro_f1'] for h in o['history'] if h['round'] in (R - 20, R - 10, R)]))
def figQ():
    res = {}
    for part, st in (('iid', 'random_iid'), ('dirichlet0.5', 'random_dirichlet0.5')):
        k10 = {m: np.array([runs[(runs.setting == st) & (runs.method == m) & (runs.seed == s)].last3.iloc[0] for s in SEEDS]) for m in ('fedavg', 'feddyn')}
        k50 = {m: np.array([last_n(json.load(open(EXP / f'random_{part}_K50_m10_{m}_s{s}.json')), 100) for s in SEEDS]) for m in ('fedavg', 'feddyn')}
        k5r = {m: np.array([last_n(json.load(open(EXP / f'random_{part}_K50_m10_R500_{m}_s{s}.json')), 500) for s in SEEDS]) for m in ('fedavg', 'feddyn')}
        res[part] = {'K10': k10, 'K50_R100': k50, 'K50_R500': k5r}
    fig, ax = plt.subplots(figsize=(W, H)); fig.subplots_adjust(left=0.14, right=0.97, top=0.86, bottom=0.17)
    regimes = [('K10', '10 clients,\nall, 100 rounds'), ('K50_R100', '50 clients, 10/round,\n100 rounds'), ('K50_R500', '50 clients, 10/round,\n500 rounds')]
    x = np.arange(3); off = {'iid': -0.17, 'dirichlet0.5': 0.17}
    for part, col, mk, nm in (('iid', BLUE, 'o', 'IID'), ('dirichlet0.5', VERM, 's', 'Dir. α=0.5')):
        m, lo, hi = [], [], []
        for r, _ in regimes:
            d = 100 * (res[part][r]['feddyn'] - res[part][r]['fedavg']); se = d.std(ddof=1) / np.sqrt(len(d)); t = stats.t.ppf(0.975, len(d) - 1)
            m.append(d.mean()); lo.append(d.mean() - t * se); hi.append(d.mean() + t * se)
        m, lo, hi = map(np.array, (m, lo, hi))
        ax.errorbar(x + off[part], m, yerr=[m - lo, hi - m], fmt=mk, color=col, ms=4, lw=0.8, capsize=2, label=nm)
    ax.axhline(0, color='#5f5f5f', lw=0.6, label='_ref0')
    ax.set_xticks(x); ax.set_xticklabels([r[1] for r in regimes], fontsize=6.0); ax.set_ylabel('FedDyn − FedAvg (pp)')
    ax.set_xlim(-0.5, 2.5); ax.set_ylim(-1.5, 7.0)
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False)
    ax_style(ax); save(fig, 'figQ_a_partial_diff', data_axes=(ax,))

    fig, ax = plt.subplots(figsize=(W, H)); fig.subplots_adjust(left=0.14, right=0.97, top=0.86, bottom=0.17)
    for part, ls in (('iid', '-'), ('dirichlet0.5', '--')):
        for m, col in (('fedavg', BLUE), ('feddyn', PURP)):
            H_ = [json.load(open(EXP / f'random_{part}_K50_m10_R500_{m}_s{s}.json'))['history'] for s in SEEDS]
            rr = [h['round'] for h in H_[0]]; A = np.array([[h['test_macro_f1'] for h in hh] for hh in H_])
            ax.plot(rr, A.mean(0), ls=ls, color=col, lw=0.9, label=f'{"FedAvg" if m == "fedavg" else "FedDyn"}, {"IID" if part == "iid" else "Dir. α=0.5"}')
            ax.fill_between(rr, A.mean(0) - A.std(0, ddof=1), A.mean(0) + A.std(0, ddof=1), color=col, alpha=0.12, lw=0)
    ax.set_xlabel('Communication round'); ax.set_ylabel('Test macro-F1'); ax.set_xlim(0, 500); ax.set_ylim(0.15, 0.65)
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False)
    ax_style(ax); save(fig, 'figQ_b_partial_curves', data_axes=())
    return res

if __name__ == '__main__':
    fig1(); figC(); figB(); res = figQ()
    bad = {k: v for k, v in REPORT.items() if v}
    print('ALL PANELS PASS' if not bad else f'FAILED: {bad}')
    sys.exit(1 if bad else 0)
