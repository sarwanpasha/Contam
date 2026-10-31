import os, glob, numpy as np, pandas as pd, sys
W=os.environ['WORK']; sys.path.insert(0,W+'/code/pretrain'); os.chdir(W+'/code/pretrain')
from sklearn.metrics import silhouette_score
z=np.load(W+'/data/benchmarks/aida_v2_tokens.npz',allow_pickle=True); lab=z['cell_type'].astype(str)
rows=[]
for p in sorted(glob.glob(W+'/outputs/pretrain/aida_v2_k*_gfAIDAonly/benchmark_embedding.npz'))+[W+'/outputs/pretrain/aida_v2_k0_s0_gfAIDApristine/benchmark_embedding.npz',W+'/outputs/pretrain/aida_v2_k0_s0_gfAIDA/benchmark_embedding.npz']:
    E=np.load(p)['emb']; a=(silhouette_score(E,lab)+1)/2
    rows.append(dict(run=os.path.basename(os.path.dirname(p)),asw=a)); print(rows[-1],flush=True)
pd.DataFrame(rows).to_csv(W+'/outputs/analysis/gf_aida_extra.csv',index=False)
