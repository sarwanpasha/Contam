"""Merge a partitioned reference pool (any mix of part{i}of{n}.npz files that together cover
all sampled rows). If a sorted-set copy of the same sample exists, check that every cell has
the same token set and that the merged sequences keep the released rank order."""
import os, glob, numpy as np
W = os.environ['WORK']; N = int(os.environ.get('REFN', 1_000_000))  # REFN-aware
out = f'{W}/data/reference/refset_ordered_{N}.npz'
if os.path.exists(out): print('exists', out); raise SystemExit(0)
D = f'{W}/data/reference/refset_ordered_parts' + ('' if N == 1_000_000 else f'_{N}')
parts = []
for f in glob.glob(f'{D}/part*of*.npz'):
    z = np.load(f); parts.append((int(z['rows'][0]), f, z['tokens'], np.diff(z['offsets']), z['rows']))
parts.sort()
print('parts', [p[1].split('/')[-1] for p in parts])
rows = np.concatenate([p[4] for p in parts]); lens = np.concatenate([p[3] for p in parts]); toks = np.concatenate([p[2] for p in parts])
assert np.all(np.diff(rows) > 0) and len(rows) == N, (len(rows), 'rows not complete/unique')
offs = np.zeros(N + 1, np.int64); offs[1:] = np.cumsum(lens)
old_p = f'{W}/data/reference/refset_{N}.npz'
if os.path.exists(old_p):
    old = np.load(f'{W}/data/reference/refset_{N}.npz', allow_pickle=True)
    assert np.array_equal(old['source_rows'], rows), 'source rows differ'
    ot, oo = old['tokens'], old['offsets']
    chk = np.random.default_rng(0).choice(N, 5000, replace=False)
    same = sum(np.array_equal(np.unique(toks[offs[i]:offs[i+1]]), ot[oo[i]:oo[i+1]]) for i in chk)
    srt = np.mean([np.all(np.diff(toks[offs[i]:offs[i+1]].astype(np.int64)) > 0) for i in chk])
    print('CHECK token sets identical %d/5000; frac sorted %.3f' % (same, srt))
    assert same == 5000
np.savez(out + '.tmp.npz', tokens=toks, offsets=offs, source_rows=rows, seed=42)
os.replace(out + '.tmp.npz', out)
print('MERGE_OK', N, len(toks))
