"""E16 third tissue: HLCA core (canonical lung benchmark in the scFM literature).

Census dataset_id 066943a2-fdac-4b29-b348-40cede398e4e
"An integrated cell atlas of the human lung in health and disease (core)"
30,000 cells, seed 42 -- matched to pancreas_islet exactly.
"""
import os, numpy as np, cellxgene_census as cc
W=os.environ['WORK']; D=W+'/data/benchmarks'; os.makedirs(D, exist_ok=True)
SEED=42; N=30000
DID='066943a2-fdac-4b29-b348-40cede398e4e'
COLS=['soma_joinid','dataset_id','cell_type','donor_id','assay','disease','sex','tissue','tissue_general','suspension_type','development_stage']
census=cc.open_soma(census_version='latest')
hs=census['census_data']['homo_sapiens']
obs=hs.obs.read(value_filter=f"dataset_id == '{DID}'", column_names=COLS).concat().to_pandas()
print('available cells:', len(obs), flush=True)
print('cell types:', obs.cell_type.nunique(), ' donors:', obs.donor_id.nunique(), flush=True)
rng=np.random.default_rng(SEED)
idx=obs['soma_joinid'].values
sel=np.sort(rng.choice(idx, size=min(N,len(idx)), replace=False))
a=cc.get_anndata(census, organism='Homo sapiens', X_name='raw', obs_coords=sel.tolist(), column_names={'obs':COLS})
a.obs['source_dataset_id']=DID
a.obs['benchmark']='lung_hlca'
print('got', a.shape, flush=True)
print('cell types in sample:', a.obs.cell_type.nunique(), ' donors:', a.obs.donor_id.nunique(), flush=True)
print(a.obs.cell_type.value_counts().head(15).to_string(), flush=True)
a.write_h5ad(D+'/lung_hlca.h5ad')
print('wrote', D+'/lung_hlca.h5ad', flush=True)
census.close()
