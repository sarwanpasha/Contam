"""stats_dose.py - pre-specified statistical analysis of the controlled
re-pretraining dose-response. Endpoints and tests were fixed in SESSION.md
before any run was analysed.

PRIMARY   full_asw_celltype (cell-type ASW, scIB-scaled) on the full benchmark
SECONDARY full_ari, full_nmi, full_graph_conn, full_isolated_asw
STRATA    full / seen / unseen
TESTS     one-sided exact permutation over seeds (alt: contaminated > clean)
          percentile bootstrap CI over seeds
          Holm-Bonferroni across the secondary panel, within stratum and K
          TOST (delta = 0.02 scIB units) for any non-rejection
          Spearman dose-response across all K with permutation p
"""
import os, sys, itertools
import numpy as np, pandas as pd
from scipy import stats

W = os.environ['WORK']
BENCH = sys.argv[1] if len(sys.argv) > 1 else 'pancreas_islet'
TAG = sys.argv[2] if len(sys.argv) > 2 else ''
# TAG is the RUN-NAME suffix and is used for the scIB INPUT path too, so it
# cannot double as an output disambiguator. Output names carried TAG only, which meant
# `stats_dose.py aida_v2` would have overwritten the pancreas primary tables 4/5/6.
# Fix: derive the output suffix from BENCH as well. Pancreas names are unchanged, so every
# existing reference still resolves; any other benchmark self-names and cannot collide.
SUF = TAG if BENCH == 'pancreas_islet' else f'_{BENCH}{TAG}'
DELTA = 0.02
NBOOT = 20000
RNG = np.random.default_rng(42)

METRICS = ['asw_celltype', 'ari', 'nmi', 'graph_conn', 'isolated_asw']
PRIMARY = 'asw_celltype'
STRATA = ['full', 'seen', 'unseen']


def perm_test(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    obs = a.mean() - b.mean()
    pool = np.concatenate([a, b]); n = len(a); N = len(pool)
    st = []
    for idx in itertools.combinations(range(N), n):
        m = np.zeros(N, bool); m[list(idx)] = True
        st.append(pool[m].mean() - pool[~m].mean())
    st = np.asarray(st)
    return float(obs), float(np.mean(st >= obs - 1e-12)), len(st)


def boot_ci(a, b, nboot=NBOOT, alpha=0.05):
    a = np.asarray(a, float); b = np.asarray(b, float)
    d = np.array([RNG.choice(a, len(a), True).mean() - RNG.choice(b, len(b), True).mean()
                  for _ in range(nboot)])
    lo, hi = np.percentile(d, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def tost(a, b, delta=DELTA):
    a = np.asarray(a, float); b = np.asarray(b, float)
    d = a.mean() - b.mean()
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    if not np.isfinite(se) or se == 0:
        return float(d), float('nan'), 0
    dof = len(a) + len(b) - 2
    p_lo = stats.t.sf((d + delta) / se, dof)
    p_hi = stats.t.cdf((d - delta) / se, dof)
    return float(d), float(max(p_lo, p_hi)), int(dof)


def holm(pvals):
    p = np.asarray(pvals, float)
    order = np.argsort(p); m = len(p)
    adj = np.empty(m); run = 0.0
    for i, idx in enumerate(order):
        run = max(run, (m - i) * p[idx])
        adj[idx] = min(run, 1.0)
    return adj


df = pd.read_csv(f'{W}/outputs/scib/{BENCH}{TAG}_scib.csv')
ks = sorted(df['k'].unique())
clean = df[df['k'] == 0]

rows = []
for stratum in STRATA:
    for metric in METRICS:
        col = f'{stratum}_{metric}'
        if col not in df.columns:
            continue
        base = clean[col].dropna().values
        if len(base) < 2:
            continue
        for k in [x for x in ks if x > 0]:
            arm = df[df['k'] == k][col].dropna().values
            if len(arm) < 2:
                continue
            obs, p, nperm = perm_test(arm, base)
            lo, hi = boot_ci(arm, base)
            d, p_tost, dof = tost(arm, base)
            rows.append(dict(stratum=stratum, metric=metric, k=int(k),
                             contam_frac=float(df[df['k'] == k]['contam_frac'].iloc[0]),
                             n_clean=len(base), n_contam=len(arm),
                             mean_clean=float(base.mean()), mean_contam=float(arm.mean()),
                             diff=obs, ci_lo=lo, ci_hi=hi,
                             perm_p=p, n_perm=nperm,
                             tost_p=p_tost, tost_delta=DELTA,
                             is_primary=bool(metric == PRIMARY and stratum == 'full')))

res = pd.DataFrame(rows)
res['holm_p'] = np.nan
for (s, k), g in res[~res['is_primary']].groupby(['stratum', 'k']):
    res.loc[g.index, 'holm_p'] = holm(g['perm_p'].values)

# ---- dose-response across all K (monotone trend) ----
dose = []
for stratum in STRATA:
    for metric in METRICS:
        col = f'{stratum}_{metric}'
        if col not in df.columns:
            continue
        sub = df[['k', col]].dropna()
        if sub[col].nunique() < 3 or sub['k'].nunique() < 3:
            continue
        rho = stats.spearmanr(sub['k'].values, sub[col].values).statistic
        null = np.array([stats.spearmanr(RNG.permutation(sub['k'].values),
                                         sub[col].values).statistic
                         for _ in range(10000)])
        p = float((np.sum(null >= rho) + 1) / (len(null) + 1))
        dose.append(dict(stratum=stratum, metric=metric, n_runs=len(sub),
                         spearman_rho=float(rho), perm_p_one_sided=p))
dose = pd.DataFrame(dose)

# ---- memorisation vs generalisation: seen minus unseen, within contaminated runs ----
gap = []
for metric in METRICS:
    cs, cu = f'seen_{metric}', f'unseen_{metric}'
    if cs not in df.columns or cu not in df.columns:
        continue
    sub = df[df['k'] > 0][['k', 'seed', cs, cu]].dropna()
    for k, g in sub.groupby('k'):
        d = (g[cs] - g[cu]).values
        if len(d) < 2:
            continue
        signs = np.array(list(itertools.product([1, -1], repeat=len(d))))
        null = (signs * d).mean(axis=1)
        gap.append(dict(metric=metric, k=int(k), n_seeds=len(d),
                        mean_seen=float(g[cs].mean()), mean_unseen=float(g[cu].mean()),
                        mean_gap=float(d.mean()),
                        signflip_p=float(np.mean(null >= d.mean() - 1e-12))))
gap = pd.DataFrame(gap)

out = f'{W}/outputs/analysis'
os.makedirs(out, exist_ok=True)
res.to_csv(f'{out}/table4_scib_contrasts{SUF}.csv', index=False)
dose.to_csv(f'{out}/table5_dose_response{SUF}.csv', index=False)
gap.to_csv(f'{out}/table6_seen_vs_unseen{SUF}.csv', index=False)

pd.set_option('display.width', 220)
print('=== PRIMARY ENDPOINT: full cell-type ASW, contaminated vs clean ===')
print(res[res['is_primary']][['k', 'contam_frac', 'mean_clean', 'mean_contam',
                              'diff', 'ci_lo', 'ci_hi', 'perm_p', 'tost_p']].to_string(index=False))
print()
print('=== SECONDARY PANEL (Holm-adjusted within stratum and K) ===')
print(res[~res['is_primary']][['stratum', 'metric', 'k', 'diff', 'ci_lo', 'ci_hi',
                               'perm_p', 'holm_p', 'tost_p']].to_string(index=False))
print()
print('=== DOSE-RESPONSE (Spearman vs K, one-sided permutation) ===')
print(dose.to_string(index=False))
print()
print('=== SEEN MINUS UNSEEN within contaminated runs ===')
print(gap.to_string(index=False))
print()
print('NOTE with 3 seeds per arm the smallest attainable exact permutation p is',
      round(1 / 20, 4), '(one-sided, 3 vs 3).')
print('WROTE', out + '/table4_scib_contrasts.csv, table5_dose_response.csv, table6_seen_vs_unseen{SUF}.csv')
