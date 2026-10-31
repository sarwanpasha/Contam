"""E20 prep: fetch Geneformer-V1-10M weights, verify vocab, build post-cutoff benchmark."""
import os, pickle, json, numpy as np
from huggingface_hub import hf_hub_download
W = os.environ['WORK']
D = f'{W}/data/reference/geneformer'
POST = '37b21763-7f0f-41ae-9001-60bad6e2841d'

for fn in ['config.json', 'model.safetensors']:
    p = hf_hub_download('ctheodoris/Geneformer', f'Geneformer-V1-10M/{fn}', local_dir=D)
    print('got', p, os.path.getsize(p), flush=True)
cfg = json.load(open(f'{D}/Geneformer-V1-10M/config.json'))
print('config', {k: cfg.get(k) for k in ['architectures', 'hidden_size', 'num_hidden_layers', 'vocab_size', 'pad_token_id', 'max_position_embeddings']})

td = pickle.load(open(f'{D}/gene_dictionaries_30m/token_dictionary_gc30M.pkl', 'rb'))
print('vocab len', len(td), 'specials', {k: v for k, v in td.items() if str(k).startswith('<')}, 'max id', max(td.values()))
assert td.get('<pad>') == 0 and td.get('<mask>') == 1 and len(td) == cfg['vocab_size']

z = np.load(f'{W}/data/benchmarks/pancreas_islet_tokens.npz', allow_pickle=True)
keys = list(z.keys()); print('bench keys', keys)
toks, offs = z['tokens'], z['offsets']
n = len(offs) - 1
src = z['source_dataset_id'].astype(str)
print('sources', dict(zip(*np.unique(src, return_counts=True))))
idx = np.where(src == POST)[0]
print('post cells', len(idx))
assert len(idx) == 15000
assert toks.min() >= 2 and toks.max() < cfg['vocab_size'], (toks.min(), toks.max())
lens = np.diff(offs)
new_t = np.concatenate([toks[offs[i]:offs[i + 1]] for i in idx]).astype(toks.dtype)
new_o = np.zeros(len(idx) + 1, dtype=np.int64); new_o[1:] = np.cumsum(lens[idx])
out = {'tokens': new_t, 'offsets': new_o}
for k in keys:
    if k in ('tokens', 'offsets'): continue
    a = z[k]
    out[k] = a[idx] if (a.ndim >= 1 and len(a) == n) else a
np.savez(f'{W}/data/benchmarks/pancreas_post_tokens.npz', **out)
ct = out['cell_type'].astype(str)
print('saved pancreas_post n', len(idx), 'median_len', float(np.median(lens[idx])), 'cell types', dict(zip(*np.unique(ct, return_counts=True))))
print('donors', len(np.unique(out['donor_id'].astype(str))))

r = np.load(f'{W}/data/reference/refset_1000000.npz', allow_pickle=True)
print('refset keys', list(r.keys()), 'n', len(r['offsets']) - 1, 'tok range', int(r['tokens'][:10_000_000].min()), int(r['tokens'][:10_000_000].max()))
print('PREP_OK')
