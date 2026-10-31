"""Overlay measured contamination effects on the published scIB leaderboards.

Requires a clone of https://github.com/theislab/scib-reproducibility in the working
directory. The effect sizes below (mean, 95% CI lower, 95% CI upper) are taken from the
contrast tables written by code/pretrain/stats_dose.py. An effect is added to a method's
metric only if its interval excludes zero (deltas='sig'); deltas='point' adds every mean."""
import pandas as pd, numpy as np, itertools, json
df = pd.read_csv('scib-reproducibility/data/metrics.csv', index_col=0)
BIO   = ['NMI_cluster/label','ARI_cluster/label','ASW_label','isolated_label_F1','isolated_label_silhouette','cLISI','cell_cycle_conservation','hvg_overlap','trajectory']
BATCH = ['ASW_label/batch','PCR_batch','graph_conn','kBET','iLISI']
COL = {'asw':'ASW_label','nmi':'NMI_cluster/label','ari':'ARI_cluster/label','gc':'graph_conn','iso':'isolated_label_silhouette'}
SC = {'pancreas': {
 'Scratch K=1,000':  dict(asw=(.0060,.0022,.0101), nmi=(-.0046,-.0154,.0066), ari=(-.0158,-.0325,.0003), gc=(-.0006,-.0121,.0120), iso=(.0073,.0023,.0127)),
 'Scratch K=5,000':  dict(asw=(.0254,.0215,.0297), nmi=(.0236,.0097,.0383), ari=(.0009,-.0178,.0198), gc=(.0163,.0015,.0319), iso=(.0299,.0247,.0346)),
 'Scratch K=20,000': dict(asw=(.0347,.0314,.0384), nmi=(.0589,.0481,.0699), ari=(.0159,-.0018,.0328), gc=(.0304,.0173,.0444), iso=(.0585,.0555,.0613)),
 'Geneformer K=5,000':  dict(asw=(.0017,.0005,.0028), nmi=(.0043,-.0014,.0099), ari=(-.0112,-.0198,-.0026), gc=(-.0052,-.0157,.0049), iso=(.0014,-.0006,.0034)),
 'Geneformer K=10,000': dict(asw=(.0038,.0024,.0052), nmi=(.0087,.0028,.0142), ari=(-.0116,-.0189,-.0050), gc=(-.0052,-.0146,.0041), iso=(.0040,.0023,.0056))},
 'lung_atlas': {
 'Scratch K=1,000':  dict(asw=(.0007,-.0014,.0025), nmi=(.0002,-.0103,.0110), ari=(.0026,-.0122,.0174), gc=(-.0015,-.0159,.0119)),
 'Scratch K=5,000':  dict(asw=(.0046,.0024,.0069), nmi=(.0249,.0148,.0355), ari=(.0238,.0080,.0392), gc=(.0098,-.0006,.0198)),
 'Scratch K=20,000': dict(asw=(.0236,.0209,.0265), nmi=(.0872,.0783,.0965), ari=(.0815,.0690,.0934), gc=(.0511,.0380,.0644))}}
def deltas(sc, mode): return {COL[k]:(p if (mode=='point' or lo>0 or hi<0) else 0.0) for k,(p,lo,hi) in sc.items()}
def parse(i): p=i.strip('/').split('/'); return p[4].rsplit('_',1)[0]
def overall(tab):
    sc = tab.copy()
    for c in BIO+BATCH:
        if c in sc and sc[c].notna().sum()>1:
            lo,hi = sc[c].min(), sc[c].max(); sc[c] = (sc[c]-lo)/(hi-lo) if hi>lo else 0.5
    return 0.6*sc[[c for c in BIO if c in sc]].mean(axis=1)+0.4*sc[[c for c in BATCH if c in sc]].mean(axis=1)
rows=[]; detail={}
for task, scen in SC.items():
    sub = df[[i.startswith(f'/{task}/') for i in df.index]].copy(); sub['method']=[parse(i) for i in sub.index]
    sub['ov0']=overall(sub[BIO+BATCH]); best=sub.loc[sub.groupby('method')['ov0'].idxmax()].set_index('method')
    base=overall(best[BIO+BATCH]); r0=base.rank(ascending=False,method='min').astype(int)
    gap=float(np.median(-np.diff(np.sort(base.values)[::-1])))
    asw=np.sort(best['ASW_label'].dropna().values)[::-1]; n=len(asw)
    for name, sc in scen.items():
        for mode in ['sig','point']:
            d=deltas(sc,mode); mv=[]
            for m in best.index:
                pert=best[BIO+BATCH].copy()
                for c,v in d.items():
                    if pd.notna(pert.loc[m,c]): pert.loc[m,c]=min(1.0,max(0.0,pert.loc[m,c]+v))
                ov=overall(pert); r1=ov.rank(ascending=False,method='min').astype(int)
                mv.append(dict(method=m,base_rank=int(r0[m]),new_rank=int(r1[m]),climb=int(r0[m]-r1[m]),d_overall=float(ov[m]-base[m])))
            mv=pd.DataFrame(mv); da=d['ASW_label']
            pairs=sum(1 for i,j in itertools.combinations(range(n),2) if 0<(asw[i]-asw[j])<da) if da>0 else 0
            rows.append(dict(task=task,scenario=name,deltas=mode,n_methods=len(mv),median_gap=round(gap,4),
                median_gain=round(mv.d_overall.median(),4),gain_in_gaps=round(mv.d_overall.median()/gap,2),
                n_climb=int((mv.climb>0).sum()),max_climb=int(mv.climb.max()),n_climb_ge3=int((mv.climb>=3).sum()),
                any_new_first=bool(((mv.new_rank==1)&(mv.base_rank>1)).any()),asw_pairs_pct=round(100*pairs/(n*(n-1)/2),1)))
            detail[f'{task}|{name}|{mode}']=mv.sort_values('base_rank').to_dict('records')
R=pd.DataFrame(rows); pd.set_option('display.width',250)
print(R[R.deltas=='sig'].drop(columns=['deltas']).to_string(index=False))
print(R[R.deltas=='point'][['task','scenario','gain_in_gaps','n_climb','max_climb']].to_string(index=False))
R.to_csv('leaderboard_overlay_summary.csv',index=False); json.dump(detail,open('leaderboard_overlay_detail.json','w'),indent=1)
for k in ['pancreas|Scratch K=5,000|sig','pancreas|Scratch K=20,000|sig','lung_atlas|Scratch K=5,000|sig','lung_atlas|Scratch K=20,000|sig']:
    print(k,[(r['method'],r['base_rank'],r['new_rank']) for r in detail[k] if r['climb']>=2])
