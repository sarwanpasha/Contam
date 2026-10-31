import glob, os, numpy as np, pandas as pd
from scipy import stats
W=os.environ['WORK']; rng=np.random.default_rng(0); out=[]
for f in sorted(glob.glob(W+'/outputs/scib/*_scib.csv')):
    tag=os.path.basename(f)[:-9]; d=pd.read_csv(f)
    if 'k' not in d or 0 not in set(d.k): continue
    for met in ['full_asw_celltype','unseen_asw_celltype','full_nmi','full_ari','full_graph_conn','full_iso_asw']:
        if met not in d: continue
        a0=d[d.k==0][met].dropna().values
        for k in sorted(set(d.k)-{0}):
            a1=d[d.k==k][met].dropna().values
            if len(a0)<2 or len(a1)<2: continue
            diff=a1.mean()-a0.mean()
            bs=a1[rng.integers(0,len(a1),(20000,len(a1)))].mean(1)-a0[rng.integers(0,len(a0),(20000,len(a0)))].mean(1)
            lo,hi=np.percentile(bs,[2.5,97.5])
            v1,v0=a1.var(ddof=1)/len(a1),a0.var(ddof=1)/len(a0); se=np.sqrt(v1+v0)
            df=(v1+v0)**2/(v1**2/(len(a1)-1)+v0**2/(len(a0)-1)); t=stats.t.ppf(0.95,df)
            r=d[d.k==k].iloc[0]
            out.append(dict(tag=tag,metric=met,k=k,N=int(r.n_pretrain),params=int(r.n_params),n0=len(a0),n1=len(a1),clean=a0.mean(),diff=diff,lo=lo,hi=hi,dmin=max(abs(diff-t*se),abs(diff+t*se))))
pd.DataFrame(out).to_csv(W+'/outputs/analysis/summary_all.csv',index=False); print(len(out))
