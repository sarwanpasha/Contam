#!/bin/bash
# separate venv so the pinned training env (scontam312) is never touched
set -e
cd $WORK
module load cray-python/3.12.12 2>/dev/null
rm -rf envs/exp6
python3 -m venv envs/exp6
source envs/exp6/bin/activate
pip install -q --upgrade pip
pip install -q scib-metrics scvi-tools harmonypy scanorama scanpy anndata igraph leidenalg pynndescent
python -c "import scib_metrics, scvi, harmonypy, scanorama, scanpy, jax; print('scib_metrics', scib_metrics.__version__, 'scvi', scvi.__version__, 'jax', jax.__version__, jax.devices())"
pip freeze > envs/exp6_freeze.txt
touch envs/exp6/E6_OK
