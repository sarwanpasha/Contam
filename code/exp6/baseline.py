"""One integration baseline on one scIB task; writes outputs/exp6/emb/{b}__{m}.npy"""
import os, sys, numpy as np, scanpy as sc
W = os.environ['WORK']; b, m = sys.argv[1:3]
a = sc.read_h5ad(f'{W}/outputs/exp6/{b}_pp.h5ad')
if m == 'ComBat':
    c = a.copy(); sc.pp.combat(c, key='donor_id'); sc.pp.pca(c, n_comps=50, random_state=0); E = c.obsm['X_pca']
elif m == 'Harmony':
    import harmonypy
    ho = harmonypy.run_harmony(a.obsm['X_pca'], a.obs, 'donor_id', random_state=0)
    Z = np.asarray(ho.Z_corr); E = Z.T if Z.shape[0] != a.n_obs else Z
elif m == 'Scanorama':
    import scanorama
    batches = a.obs['donor_id'].values; order = np.argsort(batches, kind='stable')
    parts = [a[np.where(batches == u)[0]].copy() for u in np.unique(batches)]
    idx = np.concatenate([np.where(batches == u)[0] for u in np.unique(batches)])
    scanorama.integrate_scanpy(parts, dimred=50, seed=0)
    Es = np.concatenate([p.obsm['X_scanorama'] for p in parts]); E = np.empty_like(Es); E[idx] = Es
elif m == 'scVI':
    import scvi, torch
    scvi.settings.seed = 0
    c = a.copy(); c.X = c.layers['counts']
    scvi.model.SCVI.setup_anndata(c, batch_key='donor_id')
    mod = scvi.model.SCVI(c, n_latent=30, n_layers=2, gene_likelihood='nb')
    mod.train(accelerator='gpu' if torch.cuda.is_available() else 'cpu')
    E = mod.get_latent_representation()
else:
    raise SystemExit('unknown method ' + m)
E = np.asarray(E, dtype=np.float32); assert E.shape[0] == a.n_obs and np.isfinite(E).all()
np.save(f'{W}/outputs/exp6/emb/{b}__{m}.npy', E); print(b, m, E.shape, flush=True)
