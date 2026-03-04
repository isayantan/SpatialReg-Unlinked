#!/bin/bash
#SBATCH --job-name=run_analysis_varyB2
#SBATCH --partition=long
#SBATCH --output=out/analysis_%A_%a.out
#SBATCH --error=log/analysis_%A_%a.err
#SBATCH --array=0-799     # <= MaxArraySize (1001 tasks). %200 is optional.
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G

set -euo pipefail

module load Python/3.11.5-GCCcore-13.2.0
source ~/.venvs/gp/bin/activate
which python

mkdir -p out log

# ---- Static params ----
B_vals=(100 121)
n_i_vals=(6 8 10 12)
phi=2.0
total_seed=100

total_n_i=${#n_i_vals[@]}


idx=$SLURM_ARRAY_TASK_ID

# Safety: avoid running past total jobs
TOTAL_JOBS=$(( ${#B_vals[@]} * ${#n_i_vals[@]} * total_seed ))
if (( idx >= TOTAL_JOBS )); then
  echo "idx=${idx} >= TOTAL_JOBS=${TOTAL_JOBS}. Exiting."
  exit 0
fi

# Compute indices
b_idx=$(( idx / (total_n_i * total_seed) ))
n_i_idx=$(( (idx / total_seed) % total_n_i ))
seed_idx=$(( idx % total_seed ))

B=${B_vals[$b_idx]}
n_i=${n_i_vals[$n_i_idx]}
seed=$((seed_idx + 1))

echo "Running idx=${idx} with B=${B}, n_i=${n_i}, seed=${seed}, phi=${phi}"

python analysis_varyB.py "${B}" "${n_i}" "${seed}" "${phi}"