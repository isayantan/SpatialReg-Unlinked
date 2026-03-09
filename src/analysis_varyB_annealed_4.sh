#!/bin/bash
#SBATCH --job-name=run_analysis_varyB2_remaining
#SBATCH --partition=long
#SBATCH --output=out_annealed/analysis_%A_%a.out
#SBATCH --error=log_annealed/analysis_%A_%a.err
#SBATCH --array=0-199
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=20G

set -euo pipefail

module load Python/3.11.5-GCCcore-13.2.0
source ~/.venvs/gp/bin/activate
which python

mkdir -p out_annealed log_annealed

# ---- Remaining settings only ----
B_vals=(81 121)
n_i_vals=(20 12)
phi=2.0
total_seed=100
total_settings=${#B_vals[@]}

idx=$SLURM_ARRAY_TASK_ID

# Safety
TOTAL_JOBS=$(( total_settings * total_seed ))
if (( idx >= TOTAL_JOBS )); then
  echo "idx=${idx} >= TOTAL_JOBS=${TOTAL_JOBS}. Exiting."
  exit 0
fi

# Map array index -> (setting, seed)
setting_idx=$(( idx / total_seed ))
seed_idx=$(( idx % total_seed ))

B=${B_vals[$setting_idx]}
n_i=${n_i_vals[$setting_idx]}
seed=$((seed_idx + 1))

echo "Running idx=${idx} with B=${B}, n_i=${n_i}, seed=${seed}, phi=${phi}"

python analysis_varyB_annealed.py "${B}" "${n_i}" "${seed}" "${phi}"