"""Diagnose the non-monotone scIB-lung ASW result.
(1) cross-embed: every scib_pancreas_e6 model embeds scib_lung and vice versa
    (the evaluated tissue was never seen, so any K effect is generic, not memorization);
(2) per-cell-type ASW for the lung family, K vs K=0;
(3) clean replicate: K=0 models of both families evaluated on the same benchmark."""
import os, sys, json, numpy as np, pandas as pd, torch
from sklearn.metrics import silhouette_samples
W=os.environ['WORK']; sys.path.insert(0, W+'/code')
from pretrain.model import SmallSCFM
PAD=0; dev=torch.device('cuda')
def sets(b):
    z=np.load(f'{W}/data/benchmarks/{b}_tokens.npz',allow_pickle=True); return z['tokens'],z['offsets'],z['cell_type'].astype(str)
def embed(mp, b):
    ck=torch.load(mp,map_location='cpu',weights_only=False); a=ck['args']
    m=SmallSCFM(d_model=a['d_model'],n_layers=a['n_layers'],dropout=a['dropout'],max_len=a['max_len']).to(dev)
    m.load_state_dict(ck['model']); m.eval(); t,o,_=B[b]; out=[]
    seqs=[np.asarray(t[o[i]:o[i+1]][:a['max_len']],dtype=np.int64) for i in range(len(o)-1)]
    with torch.no_grad():
        for s in range(0,len(seqs),64):
            q=seqs[s:s+64]; L=max(1,max(len(x) for x in q)); ids=np.full((len(q),L),PAD,dtype=np.int64)
            for r,x in enumerate(q): ids[r,:len(x)]=x
            with torch.autocast('cuda',dtype=torch.bfloat16):
                out.append(m.cell_embedding(torch.as_tensor(ids,device=dev)).float().cpu().numpy())
    return np.concatenate(out)
def asw(E, lab):
    s=(silhouette_samples(E,lab)+1)/2; return s.mean(), pd.Series(s).groupby(lab).mean()
B={b:sets(b) for b in ['scib_lung','scib_pancreas']}
rows=[]; pct={}
for fam in ['scib_pancreas','scib_lung']:
    for k in [0,1000,5000,10000]:
        for s in range(5):
            d=f'{W}/outputs/pretrain/{fam}_k{k}_s{s}_e6'
            for ev in ['scib_lung','scib_pancreas']:
                if ev==fam:
                    E=np.load(d+'/benchmark_embedding.npz')['emb']
                else:
                    E=embed(d+'/model.pt', ev)
                    np.save(f'{W}/outputs/exp6/diag/{ev}_by_{fam}_k{k}_s{s}.npy',E)
                a,pc=asw(E,B[ev][2]); rows.append(dict(family=fam,eval=ev,K=k,seed=s,asw=a))
                if ev=='scib_lung' and fam=='scib_lung': pct[(k,s)]=pc
                print(fam,ev,k,s,round(a,4),flush=True)
R=pd.DataFrame(rows); R.to_csv(f'{W}/outputs/exp6/diag/cross_asw.csv',index=False)
S=R.groupby(['eval','family','K']).asw.agg(['mean','std']).round(4); print(S.to_string())
P=pd.DataFrame({f'k{k}':pd.concat([pct[(k,s)] for s in range(5)],axis=1).mean(axis=1) for k in [0,1000,5000,10000]})
for k in [1000,5000,10000]: P[f'd{k}']=P[f'k{k}']-P['k0']
P['n']=pd.Series(B['scib_lung'][2]).value_counts()
P.sort_values('d1000').round(4).to_csv(f'{W}/outputs/exp6/diag/lung_percelltype.csv'); print(P.sort_values('d1000').round(3).to_string())
