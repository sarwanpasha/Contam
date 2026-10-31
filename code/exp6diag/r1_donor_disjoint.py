"""Same-tissue injection from another dataset, evaluated only on benchmark cells whose
donors do not occur in the injected dataset."""
import os, itertools, numpy as np, pandas as pd
from sklearn.metrics import silhouette_score
W = os.environ.get('WORK', '.')
z = np.load(W + '/data/benchmarks/pancreas_islet_tokens.npz', allow_pickle=True)
lab = z['cell_type'].astype(str); don = z['donor_id'].astype(str)
oth = set(np.load(W + '/data/benchmarks/pancreas_other_tokens.npz', allow_pickle=True)['donor_id'].astype(str))
mask = ~np.isin(don, list(oth)); print('cells kept', mask.sum(), 'donors kept', len(set(don[mask])), flush=True)


def asw(tag, k, s):
    E = np.load(W + f'/outputs/pretrain/pancreas_islet_k{k}_s{s}{tag}/benchmark_embedding.npz')['emb']
    return (silhouette_score(E[mask], lab[mask]) + 1) / 2


rows = []
for tag, k in [('_xtPOTHER_ord', 20000), ('_ord', 20000)]:
    a0 = np.array([asw(tag, 0, s) for s in range(5)]); a1 = np.array([asw(tag, k, s) for s in range(5)])
    d = a1.mean() - a0.mean(); zz = np.r_[a0, a1]; c = n = 0
    for idx in itertools.combinations(range(10), 5):
        m = np.zeros(10, bool); m[list(idx)] = True; n += 1; c += (zz[~m].mean() - zz[m].mean()) >= d - 1e-12
    rng = np.random.default_rng(0)
    bs = a1[rng.integers(0, 5, (20000, 5))].mean(1) - a0[rng.integers(0, 5, (20000, 5))].mean(1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    rows.append(dict(tag=tag, k=k, clean=a0.mean(), diff=d, lo=lo, hi=hi, p=c / n)); print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv(W + '/outputs/analysis/table_r1_donor_disjoint.csv', index=False)
