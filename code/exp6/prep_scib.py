"""Download one scIB integration task, check md5, store raw counts with
cell_type and donor_id (= scIB batch key) and tokenize with prep_benchmark.py."""
import os, sys, hashlib, subprocess, numpy as np, anndata as ad, scipy.sparse as sp
W = os.environ['WORK']
name, url, md5 = sys.argv[1:4]
raw = f'{W}/data/raw/scib/{name}.h5ad'
def h(p):
    m = hashlib.md5()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 22), b''): m.update(b)
    return m.hexdigest()
if not (os.path.exists(raw) and h(raw) == md5):
    subprocess.run(['curl', '-sSL', '--retry', '5', '-o', raw + '.part', url], check=True)
    os.replace(raw + '.part', raw)
got = h(raw); print('md5', got, 'expected', md5, flush=True)
assert got == md5, 'md5 mismatch'
a = ad.read_h5ad(raw)
print(a, flush=True)
lab = 'celltype' if 'celltype' in a.obs else 'cell_type'
bat = 'tech' if 'tech' in a.obs else 'batch'
X = a.layers['counts'] if 'counts' in a.layers else a.X
X = sp.csr_matrix(X).astype(np.float32)
assert X.min() >= 0
b = ad.AnnData(X=X, obs=a.obs[[lab, bat]].copy(), var=a.var[[]].copy())
b.obs.columns = ['cell_type', 'donor_id']
b.obs = b.obs.astype(str)
b.var_names = a.var_names.astype(str)
b.var_names_make_unique()
print(name, b.shape, 'cell types', b.obs.cell_type.nunique(), 'batches', b.obs.donor_id.nunique(), flush=True)
b.write_h5ad(f'{W}/data/benchmarks/{name}.h5ad')
subprocess.run([sys.executable, f'{W}/code/pretrain/prep_benchmark.py', name], check=True)
