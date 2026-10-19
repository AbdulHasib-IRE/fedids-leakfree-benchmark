# Windowing-audit means/SDs, lift range and extension-grid facts -> facts_ext.json.  Usage: python facts_ext.py <data_dir> <out.json>
import json, sys, numpy as np, pandas as pd
D, OUT = sys.argv[1], sys.argv[2]; F = {}
r = pd.DataFrame(json.load(open(f'{D}/export2b/phase0_p0_3c_multiseed.json'))['runs'])
for w in ('label_grouped', 'random_order', 'time_ordered'):
    for m, g in r.groupby('model'):
        F[f'{m}|{w}|mean'] = float(g[w].mean()); F[f'{m}|{w}|sd'] = float(g[w].std(ddof=1))
lift = 100 * (r.label_grouped - r.random_order); F['lift_min_pp'] = float(lift.min()); F['lift_max_pp'] = float(lift.max())
g = pd.read_csv(f'{D}/export/grid_runs.csv'); e = pd.read_csv(f'{D}/export2b/grid_runs_ext.csv')
allr = pd.concat([g, e[e.method != 'fedavg']]); ok = allr[~allr.diverged]
for m in ('fedavg', 'fedadapt_inv', 'fedentropy', 'fednova'):
    a = ok[(ok.setting == 'random_dirichlet0.1') & (ok.method == m)].set_index('seed').last3
    b = ok[(ok.setting == 'time_dirichlet0.1') & (ok.method == m)].set_index('seed').last3
    F[f'gap_rt_a01|{m}|pp'] = float(100 * (a - b).mean())
F['n_runs_total'] = int(len(allr)); F['ext_lt8'] = int((e[e.method != 'fedavg'].pred_classes < 8).sum())
e5 = pd.read_csv(f'{D}/export2c/local_epochs5.csv')
for m, g5 in e5.groupby('method'):
    for c in ('E1', 'E5'): F[f'epochs5|{m}|{c}|mean'] = float(g5[c].mean()); F[f'epochs5|{m}|{c}|sd'] = float(g5[c].std(ddof=1))
sp = json.load(open(f'{D}/export/splits_summary.json'))
for s_ in ('random', 'time'):
    for k, v in sp[s_]['counts'].items(): F[f'split_total|{s_}|{k}'] = int(sum(v.values()))
F['sample_total'] = int(sum(sum(v.values()) for v in sp['random']['counts'].values()))
c3 = pd.read_csv(f'{D}/export3/central_metrics.csv'); f3 = pd.read_csv(f'{D}/export3/fl_final_round_metrics.csv'); f3 = f3[f3.source == 'grid']
F['xgb_minus_best_fl_f1_r100_pp'] = float(100 * (c3[(c3.split == 'random') & (c3.model == 'XGBoost')].f1_macro.mean()
    - f3[f3.split == 'random'].groupby(['partition', 'method']).f1_macro.mean().max()))
F['ln8'] = float(np.log(8))
json.dump(F, open(OUT, 'w'), indent=1); print(len(F), 'facts ->', OUT)
