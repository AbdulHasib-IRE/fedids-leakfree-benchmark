# Derive every number quoted in the manuscript from the exported files -> facts.json (used by the Phase 7 trace check).
import json, sys, numpy as np, pandas as pd
from scipy import stats
EX = sys.argv[1]; OUT = sys.argv[2]
r = pd.read_csv(f'{EX}/grid_runs.csv'); ok = r[~r.diverged]
A = pd.read_csv(f'{EX}/table_A_setting_method.csv'); B = pd.read_csv(f'{EX}/table_B_vs_fedavg.csv')
cen = pd.read_csv(f'{EX}/central_summary.csv'); lr = pd.read_csv(f'{EX}/lr_control.csv')
H = pd.read_csv(f'{EX}/hypotheses.csv'); ws = pd.read_csv(f'{EX}/fedadapt_weight_shift.csv')
cl = pd.read_csv(f'{EX}/client_statistics.csv'); comm = pd.read_csv(f'{EX}/communication_cost.csv')
F = {}
F['n_runs'] = int(len(r)); F['n_diverged'] = int(r.diverged.sum())
F['n_runs_lt8_classes'] = int((ok.pred_classes < 8).sum())
piv = A.pivot(index='method', columns='setting', values='mean')
for a, b in (('random_dirichlet0.1', 'time_dirichlet0.1'), ('random_ip', 'time_ip')):
    t = stats.kendalltau(piv[a], piv[b]); F[f'kendall_{a}_vs_{b}'] = {'tau': t.statistic, 'p': t.pvalue}
for (sp, m), g in cen.groupby(['split', 'model']):
    F[f'central_{sp}_{m}'] = {'mean': g.test_macro_f1.mean(), 'sd': g.test_macro_f1.std(), 'n': int(len(g))}
CLS = ['Benign', 'Brute Force', 'DDOS', 'DOS', 'Mirai', 'Recon', 'Spoofing', 'Web-Based']
pc = ok.groupby(['setting', 'method'])[[f'pc_{c}' for c in CLS]].mean()
F['perclass_range'] = {c: [float(pc[f'pc_{c}'].min()), float(pc[f'pc_{c}'].max())] for c in CLS}
g = lr.groupby('setting')
for st, d in g:
    a1 = d['feddyn_lr0.03'] - d['fedavg_lr0.03']
    F[f'lrctrl_{st}'] = {'fedavg_lr0.03_mean': d['fedavg_lr0.03'].mean(), 'fedavg_lr0.03_sd': d['fedavg_lr0.03'].std(),
                         'feddyn_minus_fedavg_same_lr': a1.mean(), 'wins': int((a1 > 0).sum()),
                         'p_t': stats.ttest_rel(d['feddyn_lr0.03'], d['fedavg_lr0.03']).pvalue,
                         'fedavg_lr003_minus_lr03': (d['fedavg_lr0.03'] - d['fedavg_lr0.3']).mean()}
w = ws.groupby('setting').mean(numeric_only=True)
F['weight_shift'] = w[['L1_dist_fedadapt_vs_fedavg_weights', 'max_abs_weight_shift', 'entropy_min', 'entropy_max', 'corr_entropy_vs_size']].round(4).to_dict('index')
F['client_size_ratio'] = cl.groupby('setting').size_ratio.mean().round(1).to_dict()
F['client_min_max'] = cl.groupby('setting')[['min_size', 'max_size']].mean().round(1).to_dict('index')
F['comm_MB'] = comm.set_index('method').total_MB_100_rounds_10_clients_fp32.round(1).to_dict()
F['n_params'] = int(comm.params.iloc[0])
F['fedavg_gap_random_minus_time'] = {m: float(H[H.hypothesis.str.contains(f'\\({m},')].mean_diff.iloc[0]) for m in
                                     ['fedavg', 'fedprox', 'scaffold', 'feddyn', 'fedadapt', 'fedadapt_pure']}
# best / worst cells for "best/most/only" claims
F['best_cell'] = A.loc[A['mean'].idxmax(), ['setting', 'method', 'mean']].to_dict()
F['worst_cell'] = A.loc[A['mean'].idxmin(), ['setting', 'method', 'mean']].to_dict()
F['best_rule_per_setting'] = A.loc[A.groupby('setting')['mean'].idxmax(), ['setting', 'method', 'mean']].set_index('setting').to_dict('index')
F['holm_significant'] = B[B.sig_holm][['setting', 'method', 'diff_mean', 'wins', 'p_holm']].to_dict('records')
F['n_tests'] = int(len(B))
json.dump(F, open(OUT, 'w'), indent=1, default=float)
print(json.dumps({k: F[k] for k in ['n_runs', 'n_diverged', 'n_runs_lt8_classes', 'best_cell', 'worst_cell', 'best_rule_per_setting',
                                    'kendall_random_dirichlet0.1_vs_time_dirichlet0.1', 'kendall_random_ip_vs_time_ip', 'perclass_range',
                                    'client_min_max']}, indent=1, default=float))
for k in [k for k in F if k.startswith('lrctrl')]: print(k, {a: round(b, 4) for a, b in F[k].items()})
