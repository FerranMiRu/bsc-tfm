#!/bin/bash
#SBATCH -J profile_loader_nsys
#SBATCH --exclusive
#SBATCH --qos=acc_ehpc
#SBATCH --output=jobs-map/%j/%j.out
#SBATCH --error=jobs-map/%j/%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20

## EuroHPC allocation
#SBATCH --account=ehpc536

echo "START TIME: $(date)"

module load cuda/12.8

source "${SLURM_SUBMIT_DIR}/scripts/utils.sh"

export PYTHONWARNINGS="ignore::FutureWarning"
export HYDRA_FULL_ERROR=1

# --- Path Configuration ---
CONFIG_NAME="${1:-profiling-singlezarr}"
NUM_BATCHES="${2:-20}"
WORKERS_SWEEP="${3:-[12]}"

# --- Per-job output directory ---
JOB_OUT_DIR="${SLURM_SUBMIT_DIR}/jobs-map/${SLURM_JOB_ID}"
mkdir -p "${JOB_OUT_DIR}"
export AI4LAND_LOG_DIR="${JOB_OUT_DIR}"
NSYS_OUT="${JOB_OUT_DIR}/nsys_${SLURM_JOB_ID}"

cp "${SLURM_SUBMIT_DIR}/inputs/${CONFIG_NAME}.yaml" "${JOB_OUT_DIR}/"

cd "${SLURM_SUBMIT_DIR}"

# Keep num_getitem=5 so we get a single-thread baseline in the trace as well as the
# multi-worker phase 3. That lets us compare sem_wait dominance with and without IPC:
# if it's library-internal (Blosc, GPFS daemon) it shows up in both; if it's only
# multiprocessing queue traffic, only phase 3 shows it.
# osrt trace captures pthread/futex/sem_wait events; process-tree follows DataLoader workers.
LAUNCH_CMD="nsys profile \
    -o ${NSYS_OUT} \
    -t osrt \
    --sample=process-tree \
    --cpuctxsw=process-tree \
    --trace-fork-before-exec=true \
    --stats=true \
    --force-overwrite=true \
    uv run python -m ai4land.utils.profile_dataloader \
        --config-name ${CONFIG_NAME} \
        system.slurm_job_id=${SLURM_JOB_ID} \
        ++profile.num_getitem=5 \
        ++profile.num_batches=${NUM_BATCHES} \
        ++profile.batch_timeout_s=60 \
        ++profile.workers_sweep=${WORKERS_SWEEP}"

echo "Config: ${CONFIG_NAME} | num_batches=${NUM_BATCHES} | workers_sweep=${WORKERS_SWEEP}"
echo "Launch command: $LAUNCH_CMD"

srun --wait=60 --kill-on-bad-exit=1 --jobid "${SLURM_JOB_ID}" bash -c "$LAUNCH_CMD"

echo "nsys trace written to: ${NSYS_OUT}.nsys-rep"
echo "END TIME: $(date)"
