#!/bin/bash
#SBATCH -J file_count
#SBATCH --qos=acc_ehpc
#SBATCH --output=jobs-map/%j/%j.out
#SBATCH --error=jobs-map/%j/%j.err
#SBATCH --time=01:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --account=ehpc536

set -eu

trap 'rm -rf /gpfs/scratch/bsc32/bsc096444/ai4land-tfm/tmp/file_count_test' EXIT

mkdir -p "${SLURM_SUBMIT_DIR}/jobs-map/${SLURM_JOB_ID}"
cd "${SLURM_SUBMIT_DIR}"

uv run python -m ai4land.utils.profile_file_count \
    > "${SLURM_SUBMIT_DIR}/jobs-map/${SLURM_JOB_ID}/file_count.csv"
