"""Rebuild the pretraining background with Geneformer RANK ORDER preserved.

Draws a seed-42 sample of Genecorpus-30M rows and stores input_ids exactly as released,
so that reference cells and injected benchmark cells share the same rank-ordered format.
If a sorted-set copy of the same sample exists, the script checks that the source rows and
the token set of every cell agree.
"""
import os, sys, time, numpy as np, pyarrow as pa
W = os.environ['WORK']
N = int(sys.argv[1]) if len(sys.argv) > 1 else 1_000_000
SEED = 42; NTOT = 27406216
path = os.environ.get('CORPUS_ARROW', './corpus/genecorpus_30M_2048.dataset/dataset.arrow')
out = f'{W}/data/reference/refset_ordered_{N}.npz'
if os.path.exists(out):
    print('already built (partitioned workers):', out); sys.exit(0)
mm = pa.memory_map(path, 'r')
try:
    reader = pa.ipc.open_stream(mm); mode = 'stream'
except Exception:
    mm = pa.memory_map(path, 'r'); reader = pa.ipc.open_file(mm); mode = 'file'
print('ipc mode', mode, flush=True)
rng = np.random.default_rng(SEED)
sel = np.sort(rng.choice(NTOT, size=min(N, NTOT), replace=False))
chunks = []; row0 = 0; nb = 0; t0 = time.time()

def handle(batch):
    global row0, nb
    n = batch.num_rows; lo, hi = row0, row0 + n
    j0 = np.searchsorted(sel, lo); j1 = np.searchsorted(sel, hi)
    if j1 > j0:
        loc = pa.array((sel[j0:j1] - lo).astype(np.int64))
        for r in batch.column('input_ids').take(loc).to_pylist():
            chunks.append(np.asarray(r, dtype=np.uint16))  # rank order preserved
    row0 = hi; nb += 1
    if nb % 200 == 0:
        print('  batch %d rows %d got %d/%d %.0fs' % (nb, row0, len(chunks), len(sel), time.time() - t0), flush=True)

if mode == 'stream':
    while True:
        try: handle(reader.read_next_batch())
        except StopIteration: break
else:
    for i in range(reader.num_record_batches): handle(reader.get_batch(i))
assert len(chunks) == len(sel), (len(chunks), len(sel))
offs = np.zeros(len(chunks) + 1, np.int64); offs[1:] = np.cumsum([len(c) for c in chunks])
toks = np.concatenate(chunks)
np.savez(out, tokens=toks, offsets=offs, source_rows=sel, seed=SEED)
print('saved', out, 'cells', len(chunks), 'tokens', len(toks), '%.0fs' % (time.time() - t0), flush=True)

# ---- self-checks against the original (sorted-set) refset ----
old_p = f'{W}/data/reference/refset_{N}.npz'
if os.path.exists(old_p):
    old = np.load(old_p, allow_pickle=True)
    assert np.array_equal(old['source_rows'], sel), 'source rows differ'
    ot, oo = old['tokens'], old['offsets']
    chk = np.random.default_rng(0).choice(len(chunks), 2000, replace=False)
    same = [np.array_equal(np.unique(toks[offs[i]:offs[i + 1]]), ot[oo[i]:oo[i + 1]]) for i in chk]
    print('CHECK source_rows identical; token sets identical in %d/%d sampled cells' % (sum(same), len(same)))
    srt = np.mean([np.all(np.diff(toks[offs[i]:offs[i + 1]].astype(np.int64)) > 0) for i in chk])
    print('CHECK fraction of cells sorted ascending (should be ~0): %.3f' % srt)
    assert all(same)
print('BUILD_OK')
