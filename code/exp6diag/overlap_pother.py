"""Overlap between the pancreas benchmark and the other pancreas dataset used for injection:
identical token sequences, shared donor identifiers and source datasets."""
import os, hashlib, numpy as np
W = os.environ.get('WORK', '.')


def load(b):
    z = np.load(f'{W}/data/benchmarks/{b}_tokens.npz', allow_pickle=True); t, o = z['tokens'], z['offsets']
    h = set(hashlib.md5(t[o[i]:o[i + 1]].tobytes()).hexdigest() for i in range(len(o) - 1))
    src = set(z['source_dataset_id'].astype(str)) if 'source_dataset_id' in z else set()
    return h, set(z['donor_id'].astype(str)), src


a = load('pancreas_islet'); b = load('pancreas_other')
print('identical token sequences', len(a[0] & b[0]), 'of', len(b[0]))
print('shared donor ids', len(a[1] & b[1]), sorted(a[1] & b[1]))
print('datasets', a[2], b[2])
