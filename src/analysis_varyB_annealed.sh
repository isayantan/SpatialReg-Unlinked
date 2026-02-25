#!/bin/bash
#SBATCH --job-name=run_analysis_varyB2
#SBATCH --partition=long
#SBATCH --output=out/analysis_%A_%a.out
#SBATCH --error=log/analysis_%A_%a.err
#SBATCH --array=0-2499         # 5 B values × 5 n_i values × 100 seeds = 2500 jobs
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G

set -euo pipefail

module load Python/3.11.5-GCCcore-13.2.0
source ~/.venvs/gp/bin/activate
which python

# Ensure log/output dirs exist
mkdir -p out log

# ---- Static params ----
B_vals=(49 81 100 121 144)
n_i_vals=(6 8 10 12 20)
phi=2.0

# Seeds per array sweep (1..100)
total_seed=100

# Derived sizes
total_B=${#B_vals[@]}
total_n_i=${#n_i_vals[@]}

# SLURM array index
idx=${SLURM_ARRAY_TASK_ID}

# Compute indices
b_idx=$(( idx / (total_n_i * total_seed) ))
n_i_idx=$(( (idx / total_seed) % total_n_i ))
seed_idx=$(( idx % total_seed ))

# Actual values
B=${B_vals[$b_idx]}
n_i=${n_i_vals[$n_i_idx]}
seed=$((seed_idx + 1))  # seeds 1..100

echo "Running with B=${B}, n_i=${n_i}, seed=${seed}, phi=${phi}"

# Run analysis (new CLI: B n_i seed [phi])
python analysis_varyB_annealed.py "${B}" "${n_i}" "${seed}" "${phi}"