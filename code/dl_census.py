import os, sys, numpy as np, pandas as pd, cellxgene_census as cc
W=os.environ['WORK']; D=W+'/data/benchmarks'; os.makedirs(D,exist_ok=True)
SEED=42
JOBS={
 'aida_v2':      [('c838aec3-03ef-4398-b882-0e3912abfff0', 50000)],
 'pancreas_islet':[('3294d050-6eeb-4a00-b24c-71aacc9b777f', 15000),
                   ('37b21763-7f0f-41ae-9001-60bad6e2841d', 15000)],
}
COLS=['soma_joinid','dataset_id','cell_type','donor_id','assay','disease','sex','tissue','tissue_general','suspension_type','development_stage']
name=sys.argv[1]
census=cc.open_soma(census_version='latest')
hs=census['census_data']['homo_sapiens']
parts=[]
for did,ncell in JOBS[name]:
    obs=hs.obs.read(value_filter=f"dataset_id == '{did}'", column_names=COLS).concat().to_pandas()
    print(did,'available cells:',len(obs), flush=True)
    rng=np.random.default_rng(SEED)
    idx=obs['soma_joinid'].values
    sel=rng.choice(idx, size=min(ncell,len(idx)), replace=False) if len(idx)>ncell else idx
    sel=np.sort(sel)
    a=cc.get_anndata(census, organism='Homo sapiens', X_name='raw', obs_coords=sel.tolist(), column_names={'obs':COLS})
    a.obs['source_dataset_id']=did
    print('  got',a.shape, flush=True)
    parts.append(a)
import anndata as ad
out = parts[0] if len(parts)==1 else ad.concat(parts, join='outer', label='part', index_unique='-')
out.var = parts[0].var.copy()
out.obs['benchmark']=name
print(name,'FINAL',out.shape)
print(out.obs['source_dataset_id'].value_counts().to_string())
out.write_h5ad(D+f'/{name}.h5ad')
print('wrote',D+f'/{name}.h5ad')
census.close()
