#!/bin/bash
#SBATCH -J chunk_sizes
#SBATCH --qos=acc_ehpc
#SBATCH --output=jobs-map/%j/%j.out
#SBATCH --error=jobs-map/%j/%j.err
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --account=ehpc536

set -eu

mkdir -p "${SLURM_SUBMIT_DIR}/jobs-map/${SLURM_JOB_ID}"
cd "${SLURM_SUBMIT_DIR}"

uv run python -m ai4land.utils.profile_chunk_sizes \
    > "${SLURM_SUBMIT_DIR}/jobs-map/${SLURM_JOB_ID}/chunk_sizes.csv"
