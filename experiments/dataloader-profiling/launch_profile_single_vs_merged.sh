#!/bin/bash
#SBATCH -J single_vs_merged
#SBATCH --qos=acc_ehpc
#SBATCH --output=jobs-map/%j/%j.out
#SBATCH --error=jobs-map/%j/%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --account=ehpc536

set -eu

mkdir -p "${SLURM_SUBMIT_DIR}/jobs-map/${SLURM_JOB_ID}"
cd "${SLURM_SUBMIT_DIR}"

uv run python -m ai4land.utils.profile_single_vs_merged \
    > "${SLURM_SUBMIT_DIR}/jobs-map/${SLURM_JOB_ID}/single_vs_merged.csv"
