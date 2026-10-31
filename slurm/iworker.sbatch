#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=48g
#SBATCH --output=./logs/%x_%j.out
#SBATCH --error=./logs/%x_%j.err
cd "${WORK:?set WORK to the repository root}"
source env.sh
echo "HOST=$(hostname) GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) START=$(date)"
python code/iqueue/iworker.py
