#!/bin/bash
#SBATCH -J synth_vs_real
#SBATCH --qos=acc_ehpc
#SBATCH --output=jobs-map/%j/%j.out
#SBATCH --error=jobs-map/%j/%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --account=ehpc536

set -eu

trap 'rm -rf /gpfs/scratch/bsc32/bsc096444/ai4land-tfm/tmp/synth_vs_real' EXIT

mkdir -p "${SLURM_SUBMIT_DIR}/jobs-map/${SLURM_JOB_ID}"
cd "${SLURM_SUBMIT_DIR}"

uv run python -m ai4land.utils.profile_synth_vs_real \
    > "${SLURM_SUBMIT_DIR}/jobs-map/${SLURM_JOB_ID}/synth_vs_real.csv"
