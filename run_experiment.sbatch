#!/bin/bash
#SBATCH -p Main
#SBATCH --job-name=causal_synth
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

echo "Job started on $(hostname) at $(date)"

source /u/rominadoz/demetra_env/bin/activate
cd /u/rominadoz/projects/Causal_Models_Synthesis/src

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export TORCH_NUM_THREADS=1
export TORCH_NUM_INTEROP_THREADS=1

ulimit -n 65535
python -u experiment_manager.py


echo "Job finished at $(date)"
