#!/bin/bash
#SBATCH -J profile_layers
#SBATCH --exclusive
#SBATCH --qos=acc_ehpc
#SBATCH --output=jobs-map/%j/%j.out
#SBATCH --error=jobs-map/%j/%j.err
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20

## EuroHPC allocation
#SBATCH --account=ehpc536
#
# Layered decomposition profile: localizes the ~250 ms of non-I/O cost in
# _get_luh2_data by timing 4 read layers back-to-back on the same chunk.
#
# See src/ai4land/utils/profile_layers.py for what each layer measures.
#
# Env knobs (override on the command line):
#   STORE=<path>     zarr store to profile (default: NON_preprocessed 66 G)
#   VAR=<name>       on-disk variable to use (default: auto-discover first 3-dim float32 var)
#   N_SAMPLES=<int>  number of chunk samples (default: 50)
#   SEED=<int>       RNG seed (default: 42)
#
# This script reads the SAME zarr store the diagnostic profiler jobs target,
# so submit only when no other job is hitting that store (AGENTS.md rule).

set -eu

echo "START TIME: $(date)"
echo "HOST: $(hostname)"

source "${SLURM_SUBMIT_DIR}/scripts/utils.sh"

JOB_OUT_DIR="${SLURM_SUBMIT_DIR}/jobs-map/${SLURM_JOB_ID}"
mkdir -p "${JOB_OUT_DIR}"
export AI4LAND_LOG_DIR="${JOB_OUT_DIR}"

cd "${SLURM_SUBMIT_DIR}"

echo
echo "=== Running profile_layers ==="
echo "STORE=${STORE:-<default>} VAR=${VAR:-<auto>} N_SAMPLES=${N_SAMPLES:-50} SEED=${SEED:-42}"
echo

srun --wait=60 --kill-on-bad-exit=1 --jobid "${SLURM_JOB_ID}" \
    bash -c "uv run python -m ai4land.utils.profile_layers"

echo "END TIME: $(date)"
