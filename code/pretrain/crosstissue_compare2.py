"""Three-way matched cross-tissue comparison, either direction.

  A  K=0                      N background, no injection
  B  K=20000 SAME tissue      N-K background + K cells of the evaluated benchmark
  C  K=20000 OTHER tissue     N-K background + K cells of a different tissue

All three evaluated on the SAME benchmark, identical corpus size, optimiser steps and
background prefix. If C ~= B the inflation is a generic "more data" effect; if C ~= A it is
tissue-specific domain shift, which is claim C4.

Usage: python code/pretrain/crosstissue_compare2.py <benchmark> <xt_tag> [n_seeds]
   e.g. ... pancreas_islet _xtAIDA 5      (immune cells injected, pancreas evaluated)
        ... aida_v2        _xtPANC 5      (pancreas cells injected, immune evaluated)
"""
import os, sys, itertools
import numpy as np, pandas as pd

W = os.environ['WORK']
BENCH = sys.argv[1]
XTAG = sys.argv[2]
NS = int(sys.argv[3]) if len(sys.argv) > 3 else 5
SEEDS = list(range(NS))
SUF = XTAG if BENCH == 'pancreas_islet' else f'_{BENCH}{XTAG}'
M = ['asw_celltype', 'ari', 'nmi', 'graph_conn', 'isolated_asw']

d = pd.read_csv(f'{W}/outputs/scib/{BENCH}_scib.csv')
x = pd.read_csv(f'{W}/outputs/scib/{BENCH}{XTAG}_scib.csv')
A = d[(d.k == 0) & (d.seed.isin(SEEDS))]
B = d[(d.k == 20000) & (d.seed.isin(SEEDS))]
C = x[x.seed.isin(SEEDS)]
print(f'{BENCH}{XTAG}:  A(k=0)={len(A)}  B(same tissue)={len(B)}  C(cross tissue)={len(C)}')
assert len(A) and len(B) and len(C), 'missing runs'

def perm(a, b):
    """one-sided exact permutation, H1: mean(a) > mean(b)"""
    a = np.asarray(a, float); b = np.asarray(b, float)
    obs = a.mean() - b.mean()
    pool = np.concatenate([a, b]); n = len(a); N = len(pool); st = []
    for idx in itertools.combinations(range(N), n):
        m = np.zeros(N, bool); m[list(idx)] = True
        st.append(pool[m].mean() - pool[~m].mean())
    st = np.asarray(st)
    return obs, float(np.mean(st >= obs - 1e-12)), len(st)

rows = []
for m in M:
    c = 'full_' + m
    a, b, cc = A[c].values, B[c].values, C[c].values
    if np.isnan(a).all() or np.isnan(b).all() or np.isnan(cc).all():
        continue
    dB, pB, nperm = perm(b, a)
    dC, pC, _ = perm(cc, a)
    rows.append(dict(benchmark=BENCH, injected=XTAG.lstrip('_'), metric=m, n_seeds=len(a),
                     n_perm=nperm, A_clean=a.mean(), B_same_tissue=b.mean(),
                     C_cross_tissue=cc.mean(), B_minus_A=dB, perm_p_B=pB,
                     C_minus_A=dC, perm_p_C=pC,
                     ratio_C_over_B=(dC / dB if dB != 0 else np.nan)))
r = pd.DataFrame(rows)
out = f'{W}/outputs/analysis/table8_crosstissue{SUF}.csv'
r.to_csv(out, index=False)
pd.set_option('display.width', 200)
print(r[['metric','A_clean','B_same_tissue','C_cross_tissue','B_minus_A','perm_p_B',
         'C_minus_A','perm_p_C']].to_string(index=False, float_format=lambda v: f'{v:.4f}'))
print('WROTE', out)
