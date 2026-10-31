"""Paired low-dose analysis: K=1000 vs K=0 matched by seed, tag _ld.
Exact paired (sign-flip) permutation test over 2^S sign assignments and a paired
bootstrap CI of the mean per-seed difference, for scaled cell-type ASW on all cells."""
import os, sys, glob, json, re, itertools, numpy as np, pandas as pd
from eval_scib import asw_label
W=os.environ['WORK']; TAG='_ld'
z=np.load(W+'/data/benchmarks/pancreas_islet_tokens.npz', allow_pickle=True); lab=z['cell_type'].astype(str)
def val(k,s):
    f=W+f'/outputs/pretrain/pancreas_islet_k{k}_s{s}{TAG}/benchmark_embedding.npz'
    return asw_label(np.load(f)['emb'], lab)
seeds=sorted(int(re.search(r'_s(\d+)_ld',p).group(1)) for p in glob.glob(W+f'/outputs/pretrain/pancreas_islet_k1000_s*{TAG}')
             if os.path.exists(p+'/benchmark_embedding.npz'))
seeds=[s for s in seeds if os.path.exists(W+f'/outputs/pretrain/pancreas_islet_k0_s{s}{TAG}/benchmark_embedding.npz')]
d=np.array([val(1000,s)-val(0,s) for s in seeds]); S=len(d); obs=d.mean()
signs=np.array(list(itertools.product([1,-1],repeat=S))) if S<=22 else np.random.default_rng(0).choice([1,-1],size=(1<<20,S))
null=(signs*np.abs(d)).mean(1); p=float((null>=obs-1e-12).mean())
rng=np.random.default_rng(42); bs=np.array([d[rng.integers(0,S,S)].mean() for _ in range(20000)])
lo,hi=np.percentile(bs,[2.5,97.5])
pd.DataFrame([{'metric':'asw_celltype','n_seeds':S,'mean_paired_diff':obs,'ci_lo':lo,'ci_hi':hi,'paired_perm_p':p,'exact':S<=22}]).to_csv(
    W+'/outputs/analysis/table_paired_lowdose.csv',index=False)
print('paired K=1000 vs K=0: n=%d diff=%.4f [%.4f,%.4f] p=%.4f exact=%s'%(S,obs,lo,hi,p,S<=22))
