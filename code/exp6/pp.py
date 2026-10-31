"""Shared preprocessing for baselines and scoring: log-normalize the stored
counts, 2,000 batch-aware HVGs, 50-component PCA (the unintegrated baseline)."""
import os, sys, numpy as np, scanpy as sc
W = os.environ['WORK']; b = sys.argv[1]
a = sc.read_h5ad(f'{W}/data/benchmarks/{b}.h5ad')
a.layers['counts'] = a.X.copy()
sc.pp.normalize_total(a, target_sum=1e4); sc.pp.log1p(a)
sc.pp.highly_variable_genes(a, n_top_genes=2000, batch_key='donor_id', subset=True)
sc.pp.pca(a, n_comps=50, random_state=0)
a.write_h5ad(f'{W}/outputs/exp6/{b}_pp.h5ad')
np.save(f'{W}/outputs/exp6/emb/{b}__Unintegrated.npy', a.obsm['X_pca'].astype(np.float32))
print('pp done', a.shape, flush=True)
