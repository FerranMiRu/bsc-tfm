#!/bin/bash
#SBATCH -J cache-miss
#SBATCH --exclusive
#SBATCH --qos=acc_ehpc
#SBATCH --output=jobs-map/%j/%j.out
#SBATCH --error=jobs-map/%j/%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --account=ehpc536

set -eu

echo "START TIME: $(date)"

source "${SLURM_SUBMIT_DIR}/scripts/utils.sh"
JOB_DIR="${SLURM_SUBMIT_DIR}/jobs-map/${SLURM_JOB_ID}"
mkdir -p "${JOB_DIR}"
cd "${SLURM_SUBMIT_DIR}"

uv run python tests/profile_cache_miss.py > "${JOB_DIR}/cache_miss.csv"

echo "--- analysis ---"
uv run python tests/analyze_cache_miss.py "${JOB_DIR}/cache_miss.csv"

echo "END TIME: $(date)"
