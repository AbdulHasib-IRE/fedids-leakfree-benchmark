"""Numbers for the new tables of the revised CN paper; writes cn2/analysis/facts_rev.json and prints LaTeX rows."""
import json, numpy as np, pandas as pd
from pathlib import Path
from scipy import stats
W0 = Path(__file__).resolve().parents[1] / 'data'; EXP = W0/'export3p'/'grid_partial'
cen = pd.read_csv(W0/'export3/central_metrics.csv'); flm = pd.read_csv(W0/'export3/fl_final_round_metrics.csv'); flm = flm[flm.source == 'grid']
runs = pd.read_csv(W0/'export/grid_runs.csv'); runs = runs[~runs.diverged]
SEEDS = [42, 123, 456, 789, 1011, 2024, 7, 99, 314, 2718]
M = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro', 'f1_weighted', 'mcc']
F = {}
LBL = {'GaussianNB': 'Gaussian NB', 'LogReg': 'Logistic regression', 'DecisionTree': 'Decision tree', 'kNN': '$k$-NN ($k=5$)',
       'RandomForest': 'Random forest', 'LightGBM': 'LightGBM', 'XGBoost': 'XGBoost', 'MLP': 'MLP', 'FA-MLP': 'FA-MLP'}
ORDER = ['GaussianNB', 'LogReg', 'DecisionTree', 'kNN', 'RandomForest', 'LightGBM', 'XGBoost', 'MLP', 'FA-MLP']
FLSEL = {'random': [('dirichlet0.1', 'fedavg', 'FedAvg, Dir.\\ $\\alpha=0.1$'), ('dirichlet0.5', 'feddyn', 'FedDyn, Dir.\\ $\\alpha=0.5$'),
                    ('iid', 'fedavg', 'FedAvg, IID'), ('iid', 'feddyn', 'FedDyn, IID')],
         'time': [('dirichlet0.1', 'fedavg', 'FedAvg, Dir.\\ $\\alpha=0.1$'), ('dirichlet0.1', 'fedprox', 'FedProx, Dir.\\ $\\alpha=0.1$'),
                  ('ip', 'fedavg', 'FedAvg, source IP'), ('ip', 'scaffold', 'SCAFFOLD, source IP')]}
def fmt(v): return f'{v:.4f}'
for sp in ('random', 'time'):
    print(f'% ---- {sp}')
    best = {m: max(cen[cen.split == sp].groupby('model')[m].mean().max(), flm[flm.split == sp].groupby(['partition', 'method'])[m].mean().max()) for m in M}
    for mod in ORDER:
        d = cen[(cen.split == sp) & (cen.model == mod)]
        vals = [d[m].mean() for m in M]
        F.update({f'cen|{sp}|{mod}|{m}': float(v) for m, v in zip(M, vals)})
        cells = [('\\textbf{%s}' if abs(v - best[m]) < 5e-5 else '%s') % fmt(v) for v, m in zip(vals, M)]
        print(f'{LBL[mod]}{"$^{\\ast}$" if len(d) > 1 else ""} & ' + ' & '.join(cells) + ' \\\\')
    print('\\midrule')
    for part, meth, lab in FLSEL[sp]:
        d = flm[(flm.split == sp) & (flm.partition == part) & (flm.method == meth)]; assert len(d) == 10
        vals = [d[m].mean() for m in M]; F.update({f'fl|{sp}|{part}|{meth}|{m}': float(v) for m, v in zip(M, vals)})
        cells = [('\\textbf{%s}' if abs(v - best[m]) < 5e-5 else '%s') % fmt(v) for v, m in zip(vals, M)]
        print(f'{lab}$^{{\\ast}}$ & ' + ' & '.join(cells) + ' \\\\')
# FL vs best centralised gap on accuracy
for sp in ('random', 'time'):
    g = cen[cen.split == sp].groupby('model')
    F[f'gap|{sp}|xgb_minus_famlp_f1'] = float(g.f1_macro.mean()['XGBoost'] - g.f1_macro.mean()['FA-MLP'])
# per-class F1 of the best tree models
for sp in ('random', 'time'):
    for mod in ('XGBoost', 'RandomForest', 'LightGBM', 'FA-MLP'):
        d = cen[(cen.split == sp) & (cen.model == mod)]
        for c in ('Brute Force', 'Web-Based', 'Mirai', 'Spoofing', 'Benign'): F[f'cen|{sp}|{mod}|f1_{c}'] = float(d[f'f1_{c}'].mean())
# ---- partial participation ----
def last_n(o, R): return float(np.mean([h['test_macro_f1'] for h in o['history'] if h['round'] in (R - 20, R - 10, R)]))
print('% ---- partial participation')
P = []
for part, st, lab in (('iid', 'random_iid', 'IID'), ('dirichlet0.5', 'random_dirichlet0.5', 'Dir.\\ $\\alpha=0.5$')):
    for reg, lab2 in (('K10', '10 / all / 100'), ('K50_R100', '50 / 10 / 100'), ('K50_R500', '50 / 10 / 500')):
        v = {}
        for m in ('fedavg', 'feddyn'):
            if reg == 'K10': v[m] = np.array([runs[(runs.setting == st) & (runs.method == m) & (runs.seed == s)].last3.iloc[0] for s in SEEDS])
            elif reg == 'K50_R100': v[m] = np.array([last_n(json.load(open(EXP/f'random_{part}_K50_m10_{m}_s{s}.json')), 100) for s in SEEDS])
            else: v[m] = np.array([last_n(json.load(open(EXP/f'random_{part}_K50_m10_R500_{m}_s{s}.json')), 500) for s in SEEDS])
        d = 100 * (v['feddyn'] - v['fedavg']); t = stats.t.ppf(.975, 9); se = d.std(ddof=1) / np.sqrt(10)
        p = stats.ttest_rel(v['feddyn'], v['fedavg']).pvalue
        P.append(dict(part=part, reg=reg, lab=lab, lab2=lab2, a=v['fedavg'].mean(), sa=v['fedavg'].std(ddof=1), b=v['feddyn'].mean(), sb=v['feddyn'].std(ddof=1),
                      d=d.mean(), lo=d.mean() - t * se, hi=d.mean() + t * se, w=int((d > 0).sum()), p=p))
ps = np.array([r['p'] for r in P]); o = np.argsort(ps); hp = np.empty(6); run = 0
for i, j in enumerate(o): run = max(run, min(1, (6 - i) * ps[j])); hp[j] = run
for r, h in zip(P, hp):
    r['holm'] = h; F.update({f'pp|{r["part"]}|{r["reg"]}|{k}': float(r[k]) for k in ('a', 'sa', 'b', 'sb', 'd', 'lo', 'hi', 'w', 'p', 'holm')})
    pv = '$<$0.0001' if h < 1e-4 else f'{h:.4f}'
    print(f'{r["lab"]} & {r["lab2"]} & {r["a"]:.4f} ({r["sa"]:.4f}) & {r["b"]:.4f} ({r["sb"]:.4f}) & {r["d"]:+.2f} & [{r["lo"]:.2f}, {r["hi"]:.2f}] & {r["w"]}/10 & {pv} \\\\')
# rise of FedAvg IID between round 100 and 500 in the 500-round runs
for part in ('iid', 'dirichlet0.5'):
    for m in ('fedavg', 'feddyn'):
        H = [json.load(open(EXP/f'random_{part}_K50_m10_R500_{m}_s{s}.json'))['history'] for s in SEEDS]
        a = lambda R: float(np.mean([[h['test_macro_f1'] for h in hh if h['round'] == R][0] for hh in H]))
        F[f'pp|{part}|R500|{m}|r100'] = a(100); F[f'pp|{part}|R500|{m}|r400'] = a(400); F[f'pp|{part}|R500|{m}|r500'] = a(500)
json.dump(F, open(W0.parent/'outputs'/'analysis'/'facts_rev.json', 'w'), indent=1)
for k in sorted(F):
    if k.startswith(('gap', 'pp|iid|R500', 'pp|dirichlet0.5|R500')) or 'f1_B' in k or 'f1_W' in k: print(k, round(F[k], 4))
