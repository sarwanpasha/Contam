"""Controlled re-pretraining: matched scFM pairs differing ONLY in whether the
target benchmark is present in the pretraining corpus.

Matching guarantees
-------------------
* Both arms train on exactly N cells and the SAME N-K background cells, in the
  same order, from the same shuffled pool.  Only the final K slots differ:
  background cells (clean arm) vs target-benchmark cells (contaminated arm).
* Identical architecture, initialisation seed, optimiser, schedule, dropout,
  weight decay, batch size, epochs and masking seed.  The ONLY free variable is
  the contamination count K.
* Because the two corpora are the same size, the batch boundaries and the number
  of optimiser steps are identical by construction.
"""
from __future__ import annotations
import os, sys, json, time, argparse
import numpy as np, torch
sys.path.insert(0, os.environ['WORK']+'/code')
from pretrain.model import SmallSCFM, n_params

W=os.environ['WORK']
PAD, MASK = 0, 1

def load_sets(npz):
    z=np.load(npz, allow_pickle=True); return z['tokens'], z['offsets']

def pack(tokens, offsets, idx, max_len):
    out=[]
    for i in idx:
        s=tokens[offsets[i]:offsets[i+1]]
        out.append(np.asarray(s[:max_len], dtype=np.int64))
    return out

def make_batch(seqs, max_len, rng, mask_ratio, device):
    L=max(1, min(max_len, max(len(s) for s in seqs)))
    ids=np.full((len(seqs), L), PAD, dtype=np.int64)
    for r,s in enumerate(seqs): ids[r,:len(s)]=s[:L]
    ids=torch.as_tensor(ids, device=device)
    labels=torch.full_like(ids, -100)
    real=ids.ne(PAD)
    prob=torch.as_tensor(rng.random(ids.shape), device=device, dtype=torch.float32)
    sel=real & (prob < mask_ratio)
    labels[sel]=ids[sel]
    ids=ids.masked_fill(sel, MASK)
    return ids, labels

p=argparse.ArgumentParser()
p.add_argument('--k', type=int, required=True, help='number of benchmark cells injected')
p.add_argument('--seed', type=int, default=0)
p.add_argument('--n-pretrain', type=int, default=200000)
p.add_argument('--benchmark', default='pancreas_islet')
p.add_argument('--epochs', type=int, default=3)
p.add_argument('--batch-size', type=int, default=32)
p.add_argument('--max-len', type=int, default=512)
p.add_argument('--lr', type=float, default=3e-4)
p.add_argument('--weight-decay', type=float, default=0.01)
p.add_argument('--dropout', type=float, default=0.1)
p.add_argument('--mask-ratio', type=float, default=0.15)
p.add_argument('--d-model', type=int, default=256)
p.add_argument('--n-layers', type=int, default=6)
p.add_argument('--tag', default='')
# cross-tissue control: inject cells from a DIFFERENT benchmark than the one evaluated.
# Empty means inject the evaluated benchmark itself (the standard contaminated arm).
p.add_argument('--inject-benchmark', default='')
p.add_argument('--max-steps', type=int, default=0)
p.add_argument('--inject-source', default='')
a=p.parse_args()

run=f'{a.benchmark}_k{a.k}_s{a.seed}{a.tag}'
out=W+f'/outputs/pretrain/{run}'; os.makedirs(out, exist_ok=True)
dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(a.seed); np.random.seed(a.seed)

pt,po=load_sets(os.environ.get('REFSET', W+'/data/reference/refset_1000000.npz'))
bt,bo=load_sets(W+f'/data/benchmarks/{a.benchmark}_tokens.npz')
INJ=a.inject_benchmark or a.benchmark
it,io=(bt,bo) if INJ==a.benchmark else load_sets(W+f'/data/benchmarks/{INJ}_tokens.npz')
n_pool=len(po)-1; n_bench=len(bo)-1; n_inject=len(io)-1
assert a.k<=n_inject and a.n_pretrain<=n_pool

# --- matched corpus construction (identical across arms except the last K slots)
sel_rng=np.random.default_rng(10_000+a.seed)
pool_perm=sel_rng.permutation(n_pool)
bench_perm=sel_rng.permutation(n_inject)
if a.inject_source:
    _isrc=np.load(W+f'/data/benchmarks/{INJ}_tokens.npz',allow_pickle=True)['source_dataset_id'].astype(str)
    bench_perm=bench_perm[np.isin(bench_perm, np.where(_isrc==a.inject_source)[0])]
    assert len(bench_perm)>=a.k, (len(bench_perm), a.k)
background=pool_perm[:a.n_pretrain-a.k]
injected=bench_perm[:a.k]
corpus=pack(pt,po,background,a.max_len)+pack(it,io,injected,a.max_len)
print(f'run={run} corpus={len(corpus)} background={len(background)} injected={len(injected)}', flush=True)

model=SmallSCFM(d_model=a.d_model, n_layers=a.n_layers, dropout=a.dropout, max_len=a.max_len).to(dev)
np_=n_params(model)
opt=torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
steps=(len(corpus)//a.batch_size)*a.epochs
if a.max_steps: steps=min(steps, a.max_steps)
sched=torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=max(steps,1), pct_start=0.1)
scaler=torch.amp.GradScaler('cuda', enabled=(dev.type=='cuda'))
print(f'params={np_/1e6:.2f}M capacity_per_cell={np_/len(corpus):.2f} steps={steps}', flush=True)

order_rng=np.random.default_rng(20_000+a.seed)
mask_rng=np.random.default_rng(30_000+a.seed)
t0=time.time(); step=0; hist=[]
model.train()
for ep in range(a.epochs):
    if a.max_steps and step>=steps: break
    order=order_rng.permutation(len(corpus))
    for s in range(0, len(order)-a.batch_size+1, a.batch_size):
        seqs=[corpus[j] for j in order[s:s+a.batch_size]]
        ids,labels=make_batch(seqs, a.max_len, mask_rng, a.mask_ratio, dev)
        with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=(dev.type=='cuda')):
            _,loss=model(ids, labels)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt); scaler.update(); sched.step(); step+=1
        if a.max_steps and step>=steps: break
        if step%500==0:
            hist.append({'step':step,'loss':float(loss.item())})
            print(f'  ep{ep} step {step}/{steps} loss {loss.item():.4f} {time.time()-t0:.0f}s', flush=True)
torch.save({'model':model.state_dict(),'args':vars(a),'n_params':np_}, out+'/model.pt')

# --- extract embeddings for the FULL benchmark under this model
model.eval()
embs=[]; bs=64
allseq=pack(bt,bo,np.arange(n_bench),a.max_len)
with torch.no_grad():
    for s in range(0,len(allseq),bs):
        seqs=allseq[s:s+bs]
        L=max(1,max(len(x) for x in seqs))
        ids=np.full((len(seqs),L),PAD,dtype=np.int64)
        for r,x in enumerate(seqs): ids[r,:len(x)]=x
        with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=(dev.type=='cuda')):
            e=model.cell_embedding(torch.as_tensor(ids,device=dev))
        embs.append(e.float().cpu().numpy())
E=np.concatenate(embs)
seen=np.zeros(n_bench,dtype=bool)
if INJ==a.benchmark: seen[injected]=True  # cross-tissue arm: no evaluated cell was seen
np.savez(out+'/benchmark_embedding.npz', emb=E, seen=seen, injected=injected)
json.dump({'run':run,'k':a.k,'seed':a.seed,'n_pretrain':a.n_pretrain,'n_params':np_,
           'capacity_per_cell':np_/len(corpus),'steps':steps,'minutes':(time.time()-t0)/60,
           'loss_hist':hist,'args':vars(a)}, open(out+'/run.json','w'), indent=1)
print('WROTE',out,'in %.1f min'%((time.time()-t0)/60), flush=True)
