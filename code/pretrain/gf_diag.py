"""E20 sanity: does the released checkpoint 'understand' our token encoding?
Compares MLM loss (fixed mask) on refset (Genecorpus) and benchmark cells for
pristine vs random-init model, and true rank order vs shuffled order."""
import os, numpy as np, torch
from transformers import BertForMaskedLM, BertConfig
W = os.environ['WORK']; CK = f'{W}/data/reference/geneformer/Geneformer-V1-10M'
dev = 'cuda'
def load(p):
    z = np.load(p, allow_pickle=True); return z['tokens'], z['offsets']
def cells(t, o, idx): return [np.asarray(t[o[i]:o[i+1]][:2048], np.int64) for i in idx]
def loss(model, seqs, shuffle=False, seed=0):
    rng = np.random.default_rng(seed); tot = []
    with torch.no_grad():
        for b in range(0, len(seqs), 16):
            ss = [rng.permutation(s) if shuffle else s for s in seqs[b:b+16]]
            L = max(len(s) for s in ss); ids = np.zeros((len(ss), L), np.int64)
            for i, s in enumerate(ss): ids[i, :len(s)] = s
            att = ids != 0; msk = (rng.random(ids.shape) < 0.15) & att
            mi = ids.copy(); mi[msk] = 1
            lab = np.where(msk, ids, -100)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                out = model(input_ids=torch.tensor(mi, device=dev), attention_mask=torch.tensor(att.astype(np.int64), device=dev), labels=torch.tensor(lab, device=dev))
            tot.append(out.loss.item())
    return float(np.mean(tot))
rt, ro = load(os.environ.get('REFSET', f'{W}/data/reference/refset_1000000.npz')); print('refset', os.environ.get('REFSET','default'))
bt, bo = load(f'{W}/data/benchmarks/pancreas_post_tokens.npz')
pt_, po_ = load(f'{W}/data/benchmarks/pancreas_islet_tokens.npz')
rng = np.random.default_rng(1)
sets = {'refset': cells(rt, ro, rng.choice(len(ro)-1, 320, replace=False)),
        'post': cells(bt, bo, rng.choice(len(bo)-1, 320, replace=False)),
        'pre_baron': cells(pt_, po_, rng.choice(15000, 320, replace=False))}
print('median lens', {k: int(np.median([len(s) for s in v])) for k, v in sets.items()})
pri = BertForMaskedLM.from_pretrained(CK, attn_implementation='sdpa').to(dev).eval()
torch.manual_seed(0)
rnd = BertForMaskedLM(BertConfig.from_pretrained(CK)).to(dev).eval()
for name, s in sets.items():
    print(f'DIAG {name}: pristine {loss(pri, s):.3f} | pristine-shuffled {loss(pri, s, True):.3f} | random-init {loss(rnd, s):.3f}', flush=True)
print('ln(V)=', np.log(25426))
