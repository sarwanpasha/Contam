"""Parallel/partitioned version of build_refset_ordered.py.
Part i of n collects the seed-42 sampled Genecorpus rows in global row range
[i*NTOT/n, (i+1)*NTOT/n) with RANK ORDER preserved. Batches outside the range are
skipped without touching their data (memory-mapped, zero-copy)."""
import os, sys, time, numpy as np, pyarrow as pa
W = os.environ['WORK']; PART, NP = int(sys.argv[1]), int(sys.argv[2]); N = int(os.environ.get('REFN', 1_000_000))  # REFN-aware
SEED = 42; NTOT = 27406216
path = './corpus/genecorpus_30M_2048.dataset/dataset.arrow'
PD = f'{W}/data/reference/refset_ordered_parts' + ('' if N == 1_000_000 else f'_{N}')
lo_r, hi_r = PART * NTOT // NP, (PART + 1) * NTOT // NP
if os.path.exists(f'{PD}/part{PART}of{NP}.npz'): print('skip existing'); sys.exit(0)
rng = np.random.default_rng(SEED)
sel = np.sort(rng.choice(NTOT, size=N, replace=False))
mysel = sel[(sel >= lo_r) & (sel < hi_r)]
print('part', PART, NP, 'rows', lo_r, hi_r, 'want', len(mysel), flush=True)
mm = pa.memory_map(path, 'r')
try: reader = pa.ipc.open_stream(mm); mode = 'stream'
except Exception: mm = pa.memory_map(path, 'r'); reader = pa.ipc.open_file(mm); mode = 'file'
print('mode', mode, flush=True)
def batches():
    if mode == 'stream':
        while True:
            try: yield reader.read_next_batch()
            except StopIteration: return
    else:
        for i in range(reader.num_record_batches): yield reader.get_batch(i)
chunks, rows = [], []; row0 = 0; t0 = time.time(); nb = 0
for batch in batches():
    n = batch.num_rows; lo, hi = row0, row0 + n; row0 = hi; nb += 1
    if hi <= lo_r: continue
    if lo >= hi_r: break
    j0 = np.searchsorted(mysel, lo); j1 = np.searchsorted(mysel, hi)
    if j1 > j0:
        loc = pa.array((mysel[j0:j1] - lo).astype(np.int64))
        for r in batch.column('input_ids').take(loc).to_pylist():
            chunks.append(np.asarray(r, dtype=np.uint16))
        rows.extend(mysel[j0:j1].tolist())
    if nb % 200 == 0: print('  batch %d row %d got %d/%d %.0fs' % (nb, row0, len(chunks), len(mysel), time.time() - t0), flush=True)
assert len(chunks) == len(mysel), (len(chunks), len(mysel))
offs = np.zeros(len(chunks) + 1, np.int64); offs[1:] = np.cumsum([len(c) for c in chunks])
os.makedirs(PD, exist_ok=True); TMP = f'{PD}/tmp_part{PART}of{NP}_{os.getpid()}.npz'
np.savez(TMP, tokens=np.concatenate(chunks), offsets=offs, rows=np.array(rows))
os.replace(TMP, f'{PD}/part{PART}of{NP}.npz')
print('PART_OK', PART, len(chunks), '%.0fs' % (time.time() - t0), flush=True)
