"""scIB-style evaluation of benchmark embeddings from matched pretraining runs.

PRE-SPECIFIED ENDPOINTS (fixed before any run was analysed):
  PRIMARY   cell-type ASW (bio-conservation), scaled (asw+1)/2 as in scIB.
  SECONDARY ARI and NMI of Leiden clustering vs cell type; graph connectivity;
            isolated-label ASW.
  STRATA    full benchmark; cells SEEN in pretraining; cells UNSEEN.
            The full-benchmark value is the one a user would report.
All metrics computed identically for every run; no per-run tuning.
"""
from __future__ import annotations
import os, sys, json, glob, re
import numpy as np, pandas as pd
import scanpy as sc, anndata as ad
from sklearn.metrics import silhouette_samples, adjusted_rand_score, normalized_mutual_info_score
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

W=os.environ['WORK']; LEIDEN_RES=1.0; NN=15; SEED=42

def asw_label(E, lab):
    if len(np.unique(lab))<2: return np.nan
    return float((silhouette_samples(E, lab).mean()+1)/2)

def isolated_asw(E, lab, batch):
    df=pd.DataFrame({'l':lab,'b':batch})
    nb=df.groupby('l')['b'].nunique()
    if len(nb)==0: return np.nan
    iso=nb[nb==nb.min()].index.tolist()
    vals=[]
    for l in iso:
        m=(lab==l)
        if m.sum()<10 or m.sum()==len(lab): continue
        s=silhouette_samples(E,(m).astype(int))
        vals.append((s[m].mean()+1)/2)
    return float(np.mean(vals)) if vals else np.nan

def graph_connectivity(A, lab):
    vals=[]
    for l in np.unique(lab):
        idx=np.where(lab==l)[0]
        if len(idx)<3: continue
        sub=A[idx][:,idx]
        n,c=connected_components(sub, directed=False)
        if n==0: continue
        vals.append(np.bincount(c).max()/len(idx))
    return float(np.mean(vals)) if vals else np.nan

def evaluate(E, lab, batch, tag):
    a=ad.AnnData(X=np.ascontiguousarray(E, dtype=np.float32))
    a.obs['ct']=pd.Categorical(lab); a.obs['batch']=pd.Categorical(batch)
    sc.pp.neighbors(a, n_neighbors=NN, use_rep='X', random_state=SEED)
    sc.tl.leiden(a, resolution=LEIDEN_RES, key_added='leiden',
                 random_state=SEED, flavor='igraph', n_iterations=2, directed=False)
    cl=a.obs['leiden'].values
    A=a.obsp['connectivities']
    return {f'{tag}_n':int(len(lab)),
            f'{tag}_asw_celltype':asw_label(E,lab),
            f'{tag}_ari':float(adjusted_rand_score(lab,cl)),
            f'{tag}_nmi':float(normalized_mutual_info_score(lab,cl)),
            f'{tag}_graph_conn':graph_connectivity(A,lab),
            f'{tag}_isolated_asw':isolated_asw(E,lab,batch),
            f'{tag}_n_leiden':int(len(np.unique(cl)))}

bench=sys.argv[1] if len(sys.argv)>1 else 'pancreas_islet'

TAG = sys.argv[2] if len(sys.argv) > 2 else ''
PAT = re.compile('^' + re.escape(bench) + r'_k\d+_s\d+' + re.escape(TAG) + '$')
z=np.load(W+f'/data/benchmarks/{bench}_tokens.npz', allow_pickle=True)
lab=z['cell_type'].astype(str); batch=z['donor_id'].astype(str) if 'donor_id' in z else np.array(['b0']*len(lab))
src=z['source_dataset_id'].astype(str) if 'source_dataset_id' in z else None

rows=[]
for d in sorted(glob.glob(W+f'/outputs/pretrain/{bench}_k*')):
    if not PAT.match(os.path.basename(d)): continue
    f=d+'/benchmark_embedding.npz'
    if not os.path.exists(f): continue
    meta=json.load(open(d+'/run.json'))
    zz=np.load(f); E=zz['emb']; seen=zz['seen']
    r={'run':os.path.basename(d),'k':meta['k'],'seed':meta['seed'],
       'n_pretrain':meta['n_pretrain'],'n_params':meta['n_params'],
       'capacity_per_cell':meta['capacity_per_cell'],
       'contam_frac':meta['k']/meta['n_pretrain']}
    r.update(evaluate(E,lab,batch,'full'))
    if seen.sum()>=50:  r.update(evaluate(E[seen], lab[seen], batch[seen], 'seen'))
    if (~seen).sum()>=50: r.update(evaluate(E[~seen], lab[~seen], batch[~seen], 'unseen'))
    rows.append(r); print('done',r['run'],'full_asw=%.4f ari=%.4f'%(r['full_asw_celltype'],r['full_ari']),flush=True)

if rows:
    df=pd.DataFrame(rows).sort_values(['k','seed'])
    os.makedirs(W+'/outputs/scib',exist_ok=True)
    df.to_csv(W+f'/outputs/scib/{bench}{TAG}_scib.csv',index=False)
    pd.set_option('display.width',250)
    print(df[['run','k','contam_frac','seed','full_asw_celltype','full_ari','full_nmi','full_graph_conn']].to_string(index=False))
else:
    print('no runs found')
