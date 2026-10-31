"""Full scIB metric suite (scib-metrics Benchmarker, unscaled) for ONE embedding."""
import os, sys, numpy as np, pandas as pd, anndata as ad
from scib_metrics.benchmark import Benchmarker, BioConservation, BatchCorrection
W = os.environ['WORK']; b, path, key = sys.argv[1:4]
nj = int(os.environ.get('SLURM_CPUS_PER_TASK', '8')) // 2
a = ad.read_h5ad(f'{W}/outputs/exp6/{b}_pp.h5ad')
E = np.load(path)['emb'] if path.endswith('.npz') else np.load(path)
assert E.shape[0] == a.n_obs, (E.shape, a.n_obs)
a.obsm[key] = np.asarray(E, dtype=np.float32)
bm = Benchmarker(a, batch_key='donor_id', label_key='cell_type', embedding_obsm_keys=[key],
                 pre_integrated_embedding_obsm_key='X_pca', bio_conservation_metrics=BioConservation(),
                 batch_correction_metrics=BatchCorrection(), n_jobs=max(1, nj))
bm.benchmark()
df = bm.get_results(min_max_scale=False)
df.to_csv(f'{W}/outputs/exp6/scores/{b}__{key}.csv'); print(df.T, flush=True)
