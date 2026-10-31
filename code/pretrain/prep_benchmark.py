"""Tokenize a benchmark once and cache token sets + labels for pretraining/eval."""
import os, sys, numpy as np, anndata as ad
sys.path.insert(0, os.environ['WORK']+'/code')
from scontam2 import GeneformerVocab, tokenize
W=os.environ['WORK']; R=W+'/data/reference/geneformer/gene_dictionaries_30m/'
name=sys.argv[1] if len(sys.argv)>1 else 'pancreas_islet'
ID_COL={'pbmc3k':'gene_ids','aida_v2':'feature_id','pancreas_islet':'feature_id','lung_hlca':'feature_id','pancreas_other':'feature_id'}
vocab=GeneformerVocab(R+'gene_median_dictionary_gc30M.pkl',R+'token_dictionary_gc30M.pkl',R+'gene_name_id_dict_gc30M.pkl')
a=ad.read_h5ad(W+f'/data/benchmarks/{name}.h5ad')
idc=ID_COL.get(name); idc=idc if (idc and idc in a.var.columns) else None
sets,rep=tokenize(a,vocab,id_col=idc)
print(name,a.shape,rep,flush=True)
offs=np.zeros(len(sets)+1,dtype=np.int64)
for i,s in enumerate(sets): offs[i+1]=offs[i]+len(s)
toks=np.concatenate([np.asarray(s,dtype=np.uint16) for s in sets]) if len(sets) else np.array([],dtype=np.uint16)
out=W+f'/data/benchmarks/{name}_tokens.npz'
meta={}
for c in ['cell_type','donor_id','source_dataset_id','assay','disease']:
    if c in a.obs.columns: meta[c]=a.obs[c].astype(str).values
np.savez(out, tokens=toks, offsets=offs,
         n_genes=np.asarray((a.X>0).sum(1)).ravel(),
         lib_size=np.asarray(a.X.sum(1)).ravel(), **meta)
L=np.diff(offs)
print('saved',out,'n=%d median_len=%.0f mean_len=%.0f'%(len(sets),np.median(L),L.mean()))

# --- fail fast on a benchmark that cannot be evaluated ---------------
# eval_scib.py does z['cell_type'] unconditionally and isolated_asw needs a donor/batch
# key. If a new benchmark (lung_hlca was the first) loses either column upstream, the
# old behaviour was to sail through prep, burn 30 GPU pretraining runs, and only die in
# eval. Check it here, in the 2-minute prep job, instead.
_chk = np.load(out, allow_pickle=True)
_problems = []
if 'cell_type' not in _chk:
    _problems.append("obs['cell_type'] missing -> eval_scib.py will KeyError on every run")
else:
    _ct = _chk['cell_type']
    if len(_ct) != len(sets):
        _problems.append(f'cell_type length {len(_ct)} != n_cells {len(sets)}')
    elif len(np.unique(_ct)) < 2:
        _problems.append(f'cell_type has {len(np.unique(_ct))} distinct value(s); '
                         'ASW/ARI/NMI are undefined')
if 'donor_id' not in _chk:
    print('WARN: no donor_id; isolated_asw will fall back to a single batch b0', flush=True)
if len(sets) == 0:
    _problems.append('zero tokenized cells')
else:
    _empty = int((np.diff(offs) == 0).sum())
    if _empty:
        print(f'WARN: {_empty}/{len(sets)} cells tokenized to zero genes', flush=True)
if _problems:
    raise SystemExit('PREP FAILED for %s:\n  - %s' % (name, '\n  - '.join(_problems)))
print('prep validated: n=%d cell_types=%d donors=%s'
      % (len(sets), len(np.unique(_chk['cell_type'])),
         len(np.unique(_chk['donor_id'])) if 'donor_id' in _chk else 'n/a'), flush=True)
