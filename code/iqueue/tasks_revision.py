"""Task list for the control experiments: same-tissue injection from another study,
study-restricted injection, extra seeds at low exposure, a fixed-step control and
continued pretraining of Geneformer on the immune benchmark.

Appends to code/iqueue/tasks.txt (format name|prerequisites|done_file|est_min|command)."""
import os
W = os.environ.get('WORK', '.')
R1 = f'{W}/data/reference/refset_ordered_1000000.npz'

T = 'python code/pretrain/train.py'
STD = '--n-pretrain 200000 --epochs 2 --d-model 256 --n-layers 6'
def emb(b, k, s, tag): return f'outputs/pretrain/{b}_k{k}_s{s}{tag}/benchmark_embedding.npz'
L = ['# ---- REVISION experiments R1-R5 ----']
groups = {}
def run(name, b, k, s, tag, extra, est, dep='', ref=R1, grp=True):
    d = emb(b, k, s, tag)
    if grp: groups.setdefault((b, tag), []).append(d)
    pre = dep or 'data/reference/refset_ordered_1000000.npz'
    L.append(f'{name}|{pre}|{d}|{est}|REFSET={ref} {T} --benchmark {b} --k {k} --seed {s} --tag {tag} {extra}')

POTHER = 'data/benchmarks/pancreas_other_tokens.npz'
# R1 same-tissue different-study injection
for s in range(5):
    for k, inj in [(0, ''), (20000, '--inject-benchmark pancreas_other')]:
        run(f'r1_k{k}_s{s}', 'pancreas_islet', k, s, '_xtPOTHER_ord', f'{STD} {inj}', 25, dep=POTHER)
# R2 study-restricted injection (Tritschler only)
for s in range(5):
    for k, srcf in [(0, ''), (10000, '--inject-source 3294d050-6eeb-4a00-b24c-71aacc9b777f')]:
        run(f'r2_k{k}_s{s}', 'pancreas_islet', k, s, '_studyA_ord', f'{STD} {srcf}', 25)
# R4 fixed-step control
for s in range(5):
    for k in [0, 1000, 5000, 20000]:
        run(f'r4_k{k}_s{s}', 'pancreas_islet', k, s, '_bigNfs_ord',
            '--n-pretrain 1000000 --epochs 2 --d-model 256 --n-layers 6 --max-steps 12500', 60)
# generic scib + stats for R1, R2, R4
for (b, tag), ds in groups.items():
    suf = tag if b == 'pancreas_islet' else f'_{b}{tag}'
    sc = f'outputs/scib/{b}{tag}_scib.csv'
    L.append(f'scib_{b}{tag}|{",".join(ds)}|{sc}|30|python code/pretrain/eval_scib.py {b} {tag}')
    L.append(f'stats_{b}{tag}|{sc}|outputs/analysis/table4_scib_contrasts{suf}.csv|20|python code/pretrain/stats_dose.py {b} {tag}')
# R2 extra study-disjoint analysis
r2done = [emb('pancreas_islet', k, s, '_studyA_ord') for s in range(5) for k in (0, 10000)]
L.append(f'studydisjoint|{",".join(r2done)}|outputs/analysis/table_study_disjoint.csv|15|cd code/pretrain && python study_disjoint.py')
# R3 extra seeds at K in {0,1000}, tag _ld, with paired analysis (no generic stats)
ld = []
for s in range(20):
    for k in [0, 1000]:
        d = emb('pancreas_islet', k, s, '_ld'); ld.append(d)
        L.append(f'r3_k{k}_s{s}|data/reference/refset_ordered_1000000.npz|{d}|25|REFSET={R1} {T} --benchmark pancreas_islet --k {k} --seed {s} --tag _ld {STD}')
L.append(f'paired_ld|{",".join(ld)}|outputs/analysis/table_paired_lowdose.csv|15|cd code/pretrain && python paired_lowdose.py')
# R5 Geneformer on AIDA + benchmark-only
GF = 'python code/pretrain/gf_continue.py'; gdone = []
for s in range(5):
    for k in [0, 1000, 5000, 10000]:
        d = f'outputs/pretrain/aida_v2_k{k}_s{s}_gfAIDA/benchmark_embedding.npz'; gdone.append(d)
        L.append(f'r5_k{k}_s{s}|data/benchmarks/aida_v2_tokens.npz|{d}|180|REFSET={R1} {GF} --benchmark aida_v2 --k {k} --seed {s} --tag _gfAIDA')
for s in range(5):
    d = f'outputs/pretrain/aida_v2_k15000_s{s}_gfAIDAonly/benchmark_embedding.npz'; gdone.append(d)
    L.append(f'r5only_s{s}|data/benchmarks/aida_v2_tokens.npz|{d}|120|{GF} --benchmark aida_v2 --k 15000 --n_pretrain 15000 --seed {s} --tag _gfAIDAonly')
dp = 'outputs/pretrain/aida_v2_k0_s0_gfAIDApristine/benchmark_embedding.npz'
L.append(f'r5pris|data/benchmarks/aida_v2_tokens.npz|{dp}|20|{GF} --benchmark aida_v2 --k 0 --seed 0 --eval_only --tag _gfAIDApristine')
L.append(f'scib_aida_v2_gfAIDA|{",".join(gdone+[dp])}|outputs/scib/aida_v2_gfAIDA_scib.csv|40|python code/pretrain/eval_scib.py aida_v2 _gfAIDA')
L.append('stats_aida_v2_gfAIDA|outputs/scib/aida_v2_gfAIDA_scib.csv|outputs/analysis/table4_scib_contrasts_aida_v2_gfAIDA.csv|20|python code/pretrain/stats_dose.py aida_v2 _gfAIDA')

open(f'{W}/code/iqueue/tasks.txt', 'a').write('\n'.join(L) + '\n')
print('appended', len(L) - 1, 'tasks;', sum('train.py' in x for x in L), 'scratch runs;', sum('gf_continue' in x for x in L), 'gf runs')
