# TOST equivalence of FedAdapt-F / FedAdapt-pure vs FedAvg (paired over 10 seeds), margins 1 and 1.5 pp.
import sys, pandas as pd, numpy as np
from scipy import stats
r = pd.read_csv(sys.argv[1]); r = r[~r.diverged]
def tost(d, m):
    se = d.std(ddof=1) / np.sqrt(len(d))
    return max(stats.t.sf((d.mean() + m) / se, len(d) - 1), stats.t.cdf((d.mean() - m) / se, len(d) - 1))
rows = []
for st in sorted(r.setting.unique()):
    a = r[(r.setting == st) & (r.method == 'fedavg')].set_index('seed').last3
    for m in ('fedadapt', 'fedadapt_pure'):
        d = (r[(r.setting == st) & (r.method == m)].set_index('seed').last3 - a).dropna().values
        rows.append({'setting': st, 'method': m, 'n': len(d), 'diff_mean': d.mean(),
                     'tost_p_1pp': tost(d, 0.01), 'tost_p_1.5pp': tost(d, 0.015)})
out = pd.DataFrame(rows); out.to_csv(sys.argv[2], index=False); print(out.round(4).to_string(index=False))
