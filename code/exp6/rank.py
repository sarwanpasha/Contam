"""Rank each pretrained model among the integration baselines, one model at a time
(leave-one-in), with scIB aggregation: per-metric min-max scaling over the set,
bio = mean of bio metrics, batch = mean of batch metrics, total = 0.6 bio + 0.4 batch.
Also reports unscaled aggregates and seed-matched contaminated-minus-clean contrasts."""
import os, glob, itertools, numpy as np, pandas as pd
W = os.environ['WORK']; S = f'{W}/outputs/exp6/scores'
BASE = ['Unintegrated', 'ComBat', 'Harmony', 'Scanorama', 'scVI']
def load(b):
    rows, mtype = {}, None
    for f in sorted(glob.glob(f'{S}/{b}__*.csv')):
        d = pd.read_csv(f, index_col=0); key = os.path.basename(f)[len(b) + 2:-4]
        mtype = d.loc['Metric Type']; rows[key] = d.drop('Metric Type').iloc[0].astype(float)
    T = pd.DataFrame(rows).T
    bio = [c for c in T.columns if mtype[c] == 'Bio conservation']
    bat = [c for c in T.columns if mtype[c] == 'Batch correction']
    return T, bio, bat
def agg(T, bio, bat, scale):
    X = T[bio + bat].copy()
    if scale:
        rng = (X.max() - X.min()).replace(0, np.nan); X = ((X - X.min()) / rng).fillna(0.5)
    out = pd.DataFrame({'bio': X[bio].mean(axis=1), 'batch': X[bat].mean(axis=1)})
    out['total'] = 0.6 * out['bio'] + 0.4 * out['batch']; return out
def perm_p(x, y):
    z = np.r_[x, y]; n = len(x); obs = y.mean() - x.mean(); c = t = 0
    for idx in itertools.combinations(range(len(z)), n):
        m = np.zeros(len(z), bool); m[list(idx)] = True
        t += 1; c += abs(z[~m].mean() - z[m].mean()) >= abs(obs) - 1e-12
    return c / t
for b in ['scib_pancreas', 'scib_lung']:
    T, bio, bat = load(b)
    base = [m for m in BASE if m in T.index]; runs = [i for i in T.index if i.startswith('k')]
    if not runs: print(b, 'no runs scored'); continue
    R = []
    for r in runs:
        sub = T.loc[base + [r]]
        for scale in (True, False):
            A = agg(sub, bio, bat, scale); A['rank'] = A['total'].rank(ascending=False, method='min')
            k, s = r[1:].split('_s')
            R.append({'run': r, 'K': int(k), 'seed': int(s), 'scaled': scale, 'bio': A.loc[r, 'bio'],
                      'batch': A.loc[r, 'batch'], 'total': A.loc[r, 'total'], 'rank_of_%d' % len(sub): A.loc[r, 'rank'],
                      'n_baselines_below': int((A.loc[base, 'total'] < A.loc[r, 'total']).sum()),
                      **{c: T.loc[r, c] for c in bio + bat}})
    R = pd.DataFrame(R); R.to_csv(f'{W}/outputs/exp6/rank_{b}.csv', index=False)
    agg(T.loc[base], bio, bat, True).to_csv(f'{W}/outputs/exp6/baselines_{b}.csv')
    C = []
    for scale in (True, False):
        Q = R[R.scaled == scale]; x0 = Q[Q.K == 0].set_index('seed')
        for k in sorted(Q.K.unique()):
            if k == 0: continue
            xk = Q[Q.K == k].set_index('seed')
            for met in ['total', 'bio', 'batch'] + bio + bat:
                a0, a1 = x0[met].values, xk[met].values
                C.append({'scaled': scale, 'K': k, 'metric': met, 'clean': a0.mean(), 'contam': a1.mean(),
                          'diff': a1.mean() - a0.mean(), 'perm_p': perm_p(a0, a1),
                          'rank_clean': x0.filter(like='rank_of').iloc[:, 0].mean(),
                          'rank_contam': xk.filter(like='rank_of').iloc[:, 0].mean(),
                          'below_clean': x0['n_baselines_below'].mean(), 'below_contam': xk['n_baselines_below'].mean()})
    C = pd.DataFrame(C); C.to_csv(f'{W}/outputs/exp6/contrasts_{b}.csv', index=False)
    print('=====', b, 'baselines:', base); print(agg(T.loc[base], bio, bat, True).round(3))
    print(C[C.metric.isin(['total', 'bio', 'batch'])].round(4).to_string(index=False))
