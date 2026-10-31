import os, csv, numpy as np, cellxgene_census as cc
W=os.environ['WORK']; D=W+'/data/benchmarks'; SEED=42
COLS=['soma_joinid','dataset_id','cell_type','donor_id','assay','disease','sex','tissue','tissue_general','suspension_type','development_stage']
did=[r['dataset_id'] for r in csv.DictReader(open(W+'/data/raw/census_datasets.csv')) if r['dataset_id'].startswith('31f657dc')][0]
census=cc.open_soma(census_version='latest'); hs=census['census_data']['homo_sapiens']
obs=hs.obs.read(value_filter=f"dataset_id == '{did}'", column_names=COLS).concat().to_pandas()
print('available', len(obs), flush=True)
rng=np.random.default_rng(SEED); idx=obs['soma_joinid'].values
sel=np.sort(rng.choice(idx, size=min(25000,len(idx)), replace=False))
a=cc.get_anndata(census, organism='Homo sapiens', X_name='raw', obs_coords=sel.tolist(), column_names={'obs':COLS})
a.obs['source_dataset_id']=did; a.obs['benchmark']='pancreas_other'
a.write_h5ad(D+'/pancreas_other.h5ad'); print('wrote', a.shape); census.close()
