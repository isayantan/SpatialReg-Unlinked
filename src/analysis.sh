#!/bin/bash
#SBATCH --job-name=run_analysis
#SBATCH --output=out/analysis_%A_%a.out
#SBATCH --error=log/analysis_%A_%a.err
#SBATCH --array=0-499 # 100 seeds * 5 n_i values = 500 total jobs
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ddey1@jhu.edu

module load python/3

# Define static parameter values
B_vals=(100)
n_i_vals=(4 6 8 10 20)

# Total combinations
total_B=${#B_vals[@]}
total_n_i=${#n_i_vals[@]}
total_seed=100  # seeds from 1 to 100

# SLURM index
idx=$SLURM_ARRAY_TASK_ID

# Compute indices
b_idx=$(( idx / (total_n_i * total_seed) ))
n_i_idx=$(( (idx / total_seed) % total_n_i ))
seed_idx=$(( idx % total_seed ))

# Get actual values
B=${B_vals[$b_idx]}
n_i=${n_i_vals[$n_i_idx]}
seed=$((seed_idx + 1))  # shift to 1–100 instead of 0–99

echo "Running with B=$B, n_i=$n_i, seed=$seed"

# Run your analysis
python analysis.py $B $n_i $seed
