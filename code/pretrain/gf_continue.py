"""E20: matched-pair contamination via CONTINUED pretraining of released Geneformer-V1-10M.

Corpus construction mirrors train.py exactly (same RNG streams, same background
for every K within a seed; only the last K slots differ). Benchmark cells come
from a dataset published after Genecorpus-30M was assembled, so the pristine
checkpoint provably never saw them.
--eval_only evaluates the pristine checkpoint without training.
"""
import os, json, time, argparse, numpy as np, torch
from transformers import BertForMaskedLM, get_linear_schedule_with_warmup

PAD, MASK = 0, 1
GC30M_CELLS = 27_406_216
W = os.environ['WORK']
CKPT = f'{W}/data/reference/geneformer/Geneformer-V1-10M'

ap = argparse.ArgumentParser()
ap.add_argument('--benchmark', default='pancreas_post')
ap.add_argument('--k', type=int, default=0)
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--n_pretrain', type=int, default=200_000)
ap.add_argument('--epochs', type=float, default=2.0)
ap.add_argument('--batch', type=int, default=12)
ap.add_argument('--lr', type=float, default=1e-4)
ap.add_argument('--warmup_frac', type=float, default=0.06)
ap.add_argument('--wd', type=float, default=0.001)
ap.add_argument('--max_len', type=int, default=2048)
ap.add_argument('--mask_ratio', type=float, default=0.15)
ap.add_argument('--max_steps', type=int, default=0, help='smoke test cap')
ap.add_argument('--eval_only', action='store_true')
ap.add_argument('--tag', default='_gfV1')
a = ap.parse_args()

run = f'{a.benchmark}_k{a.k}_s{a.seed}{a.tag}'
out = f'{W}/outputs/pretrain/{run}'; os.makedirs(out, exist_ok=True)
dev = 'cuda'
torch.manual_seed(a.seed)
print('RUN', run, vars(a), flush=True)


def load_sets(p):
    z = np.load(p, allow_pickle=True); return z['tokens'], z['offsets']


def pack(tokens, offsets, idx, max_len):
    return [np.asarray(tokens[offsets[i]:offsets[i + 1]][:max_len], dtype=np.int64) for i in idx]


def make_batch(seqs, rng, mask_ratio, vocab):
    """Standard BERT 80/10/10 masking, as in Geneformer's original pretraining."""
    L = max(len(s) for s in seqs)
    ids = np.full((len(seqs), L), PAD, np.int64)
    for i, s in enumerate(seqs): ids[i, :len(s)] = s
    att = (ids != PAD)
    sel = (rng.random(ids.shape) < mask_ratio) & att
    labels = np.where(sel, ids, -100)
    r = rng.random(ids.shape)
    ids = ids.copy()
    ids[sel & (r < 0.8)] = MASK
    rnd = sel & (r >= 0.8) & (r < 0.9)
    ids[rnd] = rng.integers(2, vocab, size=int(rnd.sum()))
    t = lambda x: torch.from_numpy(x).to(dev)
    return t(ids), t(att.astype(np.int64)), t(labels)


model = BertForMaskedLM.from_pretrained(CKPT, attn_implementation='sdpa').to(dev)
V = model.config.vocab_size
n_params = sum(p.numel() for p in model.parameters())
bt, bo = load_sets(f'{W}/data/benchmarks/{a.benchmark}_tokens.npz')
n_inject = len(bo) - 1
steps = 0; minutes = 0.0; loss_hist = []
injected = np.array([], dtype=np.int64)

if not a.eval_only:
    REF = os.environ.get('REFSET', f'{W}/data/reference/refset_1000000.npz')
    pt, po = load_sets(REF); n_pool = len(po) - 1
    assert a.k <= n_inject and a.n_pretrain <= n_pool and a.k <= a.n_pretrain  # REV_OK
    sel_rng = np.random.default_rng(10_000 + a.seed)
    pool_perm = sel_rng.permutation(n_pool)
    bench_perm = sel_rng.permutation(n_inject)
    background = pool_perm[:a.n_pretrain - a.k]
    injected = bench_perm[:a.k]
    corpus = pack(pt, po, background, a.max_len) + pack(bt, bo, injected, a.max_len)
    del pt, po
    order_rng = np.random.default_rng(20_000 + a.seed)
    mask_rng = np.random.default_rng(30_000 + a.seed)
    nb = len(corpus) // a.batch
    total = int(nb * a.epochs) if not a.max_steps else a.max_steps
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    sch = get_linear_schedule_with_warmup(opt, int(a.warmup_frac * total), total)
    model.train(); t0 = time.time(); run_loss = 0.0
    while steps < total:
        order = order_rng.permutation(len(corpus))
        for b in range(nb):
            if steps >= total: break
            ids, att, lab = make_batch([corpus[j] for j in order[b * a.batch:(b + 1) * a.batch]], mask_rng, a.mask_ratio, V)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                loss = model(input_ids=ids, attention_mask=att, labels=lab).loss
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sch.step(); steps += 1; run_loss += loss.item()
            if steps % 500 == 0 or steps == total:
                el = time.time() - t0
                loss_hist.append([steps, run_loss / (500 if steps % 500 == 0 else steps % 500)]); run_loss = 0.0
                print(f'step {steps}/{total} loss {loss_hist[-1][1]:.4f} {el/60:.1f}min eta {el/steps*(total-steps)/60:.1f}min', flush=True)
    minutes = (time.time() - t0) / 60
    torch.save(model.state_dict(), f'{out}/model.pt')

# ---- evaluation on the full benchmark: embeddings + per-cell MLM loss (fixed mask) ----
model.eval()
cells = pack(bt, bo, np.arange(n_inject), a.max_len)
emb_pen, emb_last, mlm = [], [], []
eval_rng = np.random.default_rng(777)  # identical mask for every model/seed
with torch.no_grad():
    for b in range(0, n_inject, 32):
        seqs = cells[b:b + 32]
        L = max(len(s) for s in seqs)
        ids = np.full((len(seqs), L), PAD, np.int64)
        for i, s in enumerate(seqs): ids[i, :len(s)] = s
        att = ids != PAD
        msk = (eval_rng.random(ids.shape) < a.mask_ratio) & att
        with torch.autocast('cuda', dtype=torch.bfloat16):
            o = model.bert(input_ids=torch.from_numpy(ids).to(dev), attention_mask=torch.from_numpy(att.astype(np.int64)).to(dev), output_hidden_states=True)
            m = torch.from_numpy(att).to(dev).unsqueeze(-1).float()
            for hs, dst in ((o.hidden_states[-2], emb_pen), (o.hidden_states[-1], emb_last)):
                dst.append(((hs.float() * m).sum(1) / m.sum(1)).cpu().numpy())
            mids = ids.copy(); mids[msk] = MASK
            logits = model(input_ids=torch.from_numpy(mids).to(dev), attention_mask=torch.from_numpy(att.astype(np.int64)).to(dev)).logits.float()
        lab = torch.from_numpy(np.where(msk, ids, -100)).to(dev)
        ce = torch.nn.functional.cross_entropy(logits.transpose(1, 2), lab, ignore_index=-100, reduction='none')
        cnt = (lab != -100).sum(1).clamp(min=1)
        mlm.append((ce.sum(1) / cnt).cpu().numpy())
seen = np.zeros(n_inject, bool); seen[injected] = True
np.savez(f'{out}/benchmark_embedding.npz', emb=np.concatenate(emb_pen), emb_last=np.concatenate(emb_last),
         mlm_loss=np.concatenate(mlm), seen=seen, injected=injected)
ml = np.concatenate(mlm)
check = {'mlm_seen': float(ml[seen].mean()) if seen.any() else None, 'mlm_unseen': float(ml[~seen].mean())}
json.dump(dict(run=run, k=a.k, seed=a.seed, n_pretrain=(GC30M_CELLS if a.eval_only else a.n_pretrain), n_params=n_params,
               capacity_per_cell=n_params / GC30M_CELLS,
               capacity_per_cell_continued=(None if a.eval_only else n_params / a.n_pretrain),
               steps=steps, minutes=minutes, loss_hist=loss_hist, manipulation_check=check, args=vars(a)),
          open(f'{out}/run.json', 'w'), indent=1)
print('DONE', run, 'steps', steps, 'min %.1f' % minutes, check, flush=True)
