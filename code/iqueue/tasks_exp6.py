"""Task list for the rebuilt-leaderboard experiment on the scIB pancreas and lung tasks.

Writes code/iqueue/tasks_e6.txt. Each line is name|prerequisites|done_file|est_min|command
and is executed by code/iqueue/iworker.py (point a worker at it with IW_TASKS)."""
import os
W = os.environ.get('WORK', '.')
R1 = f'{W}/data/reference/refset_ordered_1000000.npz'
STD = '--n-pretrain 200000 --epochs 2 --d-model 256 --n-layers 6'
PY6 = 'export OMP_NUM_THREADS=8 NUMBA_NUM_THREADS=8 JAX_PLATFORMS=cpu; envs/exp6/bin/python'
ENVOK = 'envs/exp6/E6_OK'
TASKS = {'scib_pancreas': ('https://ndownloader.figshare.com/files/24539828', '5aa155652b998c90395a0c9bd731b4b3'),
         'scib_lung': ('https://ndownloader.figshare.com/files/24539942', 'aac0832c3e3413ae279343f5de02a2c6')}
L = ['e6_env||envs/exp6/E6_OK|60|bash code/exp6/env_build.sh > logs/e6_env_build.log 2>&1']
scores = []
for b, (url, md5) in TASKS.items():
    tok = f'data/benchmarks/{b}_tokens.npz'; pp = f'outputs/exp6/{b}_pp.h5ad'
    L.append(f'e6_prep_{b}||{tok}|30|python code/exp6/prep_scib.py {b} {url} {md5}')
    L.append(f'e6_pp_{b}|{tok},{ENVOK}|{pp}|20|{PY6} code/exp6/pp.py {b}')
    embs = []
    for m in ['ComBat', 'Harmony', 'Scanorama', 'scVI']:
        L.append(f'e6_bl_{b}_{m}|{pp}|outputs/exp6/emb/{b}__{m}.npy|60|{PY6} code/exp6/baseline.py {b} {m}')
    for m in ['Unintegrated', 'ComBat', 'Harmony', 'Scanorama', 'scVI']:
        e = f'outputs/exp6/emb/{b}__{m}.npy'; sc = f'outputs/exp6/scores/{b}__{m}.csv'; scores.append(sc)
        L.append(f'e6_sc_{b}_{m}|{e},{pp}|{sc}|45|{PY6} code/exp6/score.py {b} {e} {m}')
    for s in range(5):
        for k in [0, 1000, 5000, 10000]:
            e = f'outputs/pretrain/{b}_k{k}_s{s}_e6/benchmark_embedding.npz'; embs.append(e)
            L.append(f'e6_t_{b}_k{k}_s{s}|{tok},{R1}|{e}|25|REFSET={R1} python code/pretrain/train.py --benchmark {b} --k {k} --seed {s} --tag _e6 {STD}')
            sc = f'outputs/exp6/scores/{b}__k{k}_s{s}.csv'; scores.append(sc)
            L.append(f'e6_sc_{b}_k{k}_s{s}|{e},{pp}|{sc}|45|{PY6} code/exp6/score.py {b} {e} k{k}_s{s}')
    L.append(f'e6_scib_{b}|{",".join(embs)}|outputs/scib/{b}_e6_scib.csv|30|python code/pretrain/eval_scib.py {b} _e6')
    L.append(f'e6_stats_{b}|outputs/scib/{b}_e6_scib.csv|outputs/analysis/table4_scib_contrasts_{b}_e6.csv|20|python code/pretrain/stats_dose.py {b} _e6')
L.append(f'e6_rank|{",".join(scores)}|outputs/exp6/contrasts_scib_lung.csv|15|{PY6} code/exp6/rank.py')
open(f'{W}/code/iqueue/tasks_e6.txt', 'w').write('# Experiment 6 rebuilt leaderboard\n' + '\n'.join(L) + '\n')
print('wrote', len(L), 'tasks;', sum('train.py' in x for x in L), 'training runs;', sum('score.py' in x for x in L), 'scoring tasks')

