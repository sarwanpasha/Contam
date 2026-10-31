# Contam

Code for matched-pair contamination experiments with single-cell foundation models.
A clean and a contaminated model are pretrained on corpora that are identical except for a
block of `K` cells, which holds either reference cells or cells from the evaluation
benchmark. Training order and masking are seeded identically in both arms, so the
difference between the two models measures the effect of benchmark exposure.

## Layout

| Path | Contents |
| --- | --- |
| `code/scontam2/` | Geneformer rank-value tokenization (gene medians and token dictionary of Geneformer) |
| `code/dl_*.py` | Download of Genecorpus-30M, CZ CELLxGENE Census benchmarks and the Human Lung Cell Atlas subset |
| `code/build_refset_ordered*.py`, `code/merge_refset_ordered.py` | Reference pools sampled from Genecorpus-30M, kept in rank order |
| `code/pretrain/prep_benchmark.py` | Tokenizes a benchmark `.h5ad` file and stores labels |
| `code/pretrain/model.py`, `code/pretrain/train.py` | Transformer encoder and matched-pair pretraining from scratch |
| `code/pretrain/gf_prep.py`, `code/pretrain/gf_continue.py`, `code/pretrain/gf_diag.py` | Continued pretraining of the released Geneformer-V1 checkpoint |
| `code/pretrain/eval_scib.py` | Cell-type silhouette, Leiden ARI and NMI, graph connectivity and isolated-label silhouette on all, injected and unseen cells |
| `code/pretrain/stats_dose.py` | Exact permutation tests, bootstrap intervals, Holm correction and equivalence tests |
| `code/pretrain/paired_lowdose.py`, `code/pretrain/study_disjoint.py`, `code/pretrain/crosstissue_compare2.py` | Paired low-dose, study-disjoint and cross-tissue analyses |
| `code/exp6/` | Rescoring on the scIB pancreas and lung tasks with integration baselines and the scib-metrics suite |
| `code/exp6diag/` | Cross-embedding and per-cell-type diagnostics, summary tables |
| `code/leaderboard_overlay.py` | Overlay of measured effects on the published scIB leaderboards |
| `code/iqueue/` | A small file-based task queue (`iworker.py`) and the task-list generators |
| `slurm/iworker.sbatch` | Example Slurm script that runs one queue worker |

## Setup

```bash
git clone https://github.com/sarwanpasha/Contam.git && cd Contam
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
source env.sh          # sets WORK to the repository root
```

The rescoring experiment in `code/exp6/` uses a separate environment with `scib-metrics`,
`scvi-tools`, `harmonypy` and `scanorama`; `code/exp6/env_build.sh` creates it.

## Typical workflow

```bash
python code/dl_genecorpus.py                       # Genecorpus-30M and Geneformer dictionaries
python code/build_refset_ordered.py                # reference pool in rank order
python code/dl_census.py                           # benchmark datasets
python code/pretrain/prep_benchmark.py pancreas_islet

# one matched pair: clean (K = 0) and contaminated (K = 5000) arm, seed 0
REFSET=data/reference/refset_ordered_1000000.npz python code/pretrain/train.py \
    --benchmark pancreas_islet --k 0    --seed 0 --tag _ord --n-pretrain 200000 --epochs 2 --d-model 256 --n-layers 6
REFSET=data/reference/refset_ordered_1000000.npz python code/pretrain/train.py \
    --benchmark pancreas_islet --k 5000 --seed 0 --tag _ord --n-pretrain 200000 --epochs 2 --d-model 256 --n-layers 6

python code/pretrain/eval_scib.py pancreas_islet _ord
python code/pretrain/stats_dose.py pancreas_islet _ord
```

Larger sweeps are written as task lists (`code/iqueue/tasks_*.py`) and executed by one or
more `code/iqueue/iworker.py` processes, which claim tasks through lock files and start a
task once its prerequisite files exist.

## License

MIT, see `LICENSE`.
