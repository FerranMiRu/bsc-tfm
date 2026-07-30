#!/bin/bash
#SBATCH -J load-timing
#SBATCH --exclusive
#SBATCH --qos=acc_ehpc
#SBATCH --output=jobs-map/%j/%j.out
#SBATCH --error=jobs-map/%j/%j.err
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --account=ehpc536

set -eu

echo "START TIME: $(date)"

source "${SLURM_SUBMIT_DIR}/scripts/utils.sh"
mkdir -p "${SLURM_SUBMIT_DIR}/jobs-map/${SLURM_JOB_ID}"
cd "${SLURM_SUBMIT_DIR}"

uv run python tests/profile_load_timing.py

echo "END TIME: $(date)"
