"""Study-disjoint analysis for tag _studyA_ord: injected cells come only from study
3294d050 (Tritschler). Contrast contaminated vs clean scaled ASW on cells of the OTHER
study (37b21763, Fasolino), which share no donors with the injected cells."""
import os, glob, itertools, numpy as np, pandas as pd
from eval_scib import asw_label
W=os.environ['WORK']; TAG='_studyA_ord'; OTHER='37b21763-7f0f-41ae-9001-60bad6e2841d'
z=np.load(W+'/data/benchmarks/pancreas_islet_tokens.npz', allow_pickle=True)
lab=z['cell_type'].astype(str); src=z['source_dataset_id'].astype(str); mask=src==OTHER
def val(k,s):
    f=W+f'/outputs/pretrain/pancreas_islet_k{k}_s{s}{TAG}/benchmark_embedding.npz'
    E=np.load(f)['emb']; return asw_label(E[mask], lab[mask])
seeds=sorted(int(os.path.basename(p).split('_s')[1].split('_')[0]) for p in glob.glob(W+f'/outputs/pretrain/pancreas_islet_k10000_s*{TAG}')
             if os.path.exists(p+'/benchmark_embedding.npz'))
seeds=[s for s in seeds if os.path.exists(W+f'/outputs/pretrain/pancreas_islet_k0_s{s}{TAG}/benchmark_embedding.npz')]
c=np.array([val(10000,s) for s in seeds]); b=np.array([val(0,s) for s in seeds]); obs=c.mean()-b.mean()
pool=np.concatenate([c,b]); N=len(pool); n=len(c)
st=np.array([pool[list(i)].mean()-pool[[j for j in range(N) if j not in i]].mean() for i in itertools.combinations(range(N),n)])
p=float((st>=obs-1e-12).mean())
rng=np.random.default_rng(42); bs=np.array([rng.choice(c,n).mean()-rng.choice(b,n).mean() for _ in range(20000)])
lo,hi=np.percentile(bs,[2.5,97.5])
pd.DataFrame([{'stratum':'other_study_Fasolino','metric':'asw_celltype','n_seeds':n,'diff':obs,'ci_lo':lo,'ci_hi':hi,'perm_p':p}]).to_csv(
    W+'/outputs/analysis/table_study_disjoint.csv',index=False)
print('study-disjoint (Fasolino cells, inject Tritschler) K=10000: n=%d diff=%.4f [%.4f,%.4f] p=%.4f'%(n,obs,lo,hi,p))
