#!/bin/bash
#SBATCH -J chunksize-parity
#SBATCH --qos=acc_debug
#SBATCH --output=logs/tests/%j.out
#SBATCH --error=logs/tests/%j.err
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --account=ehpc536

set -eu

echo "START TIME: $(date)"

source "${SLURM_SUBMIT_DIR}/scripts/utils.sh"

mkdir -p "${SLURM_SUBMIT_DIR}/logs/tests"

cd "${SLURM_SUBMIT_DIR}"

uv run python -m tests.test_loader_parity_chunksize "$@"

echo "END TIME: $(date)"
