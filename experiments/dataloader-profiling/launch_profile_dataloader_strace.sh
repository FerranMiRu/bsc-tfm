#!/bin/bash
#SBATCH -J profile_loader_strace
#SBATCH --exclusive
#SBATCH --qos=acc_ehpc
#SBATCH --output=jobs-map/%j/%j.out
#SBATCH --error=jobs-map/%j/%j.err
#SBATCH --time=01:30:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20

## EuroHPC allocation
#SBATCH --account=ehpc536

echo "START TIME: $(date)"

source "${SLURM_SUBMIT_DIR}/scripts/utils.sh"

export PYTHONWARNINGS="ignore::FutureWarning"
export HYDRA_FULL_ERROR=1

# --- Path Configuration ---
CONFIG_NAME="${1:-profiling-singlezarr}"
NUM_GETITEM="${2:-5}"
NUM_BATCHES="${3:-30}"
SPLIT="${4:-train}"
BATCH_TIMEOUT_S="${5:-30}"
WORKERS_SWEEP="${6:-[12]}"
LUH2_THREADS="${7:-1}"

# --- Per-job output directory ---
JOB_OUT_DIR="${SLURM_SUBMIT_DIR}/jobs-map/${SLURM_JOB_ID}"
mkdir -p "${JOB_OUT_DIR}"
export AI4LAND_LOG_DIR="${JOB_OUT_DIR}"

cp "${SLURM_SUBMIT_DIR}/inputs/${CONFIG_NAME}.yaml" "${JOB_OUT_DIR}/"

cd "${SLURM_SUBMIT_DIR}"

STRACE_OUT="${JOB_OUT_DIR}/strace_${SLURM_JOB_ID}.txt"

LAUNCH_CMD="strace -c -f \
    -e trace=read,pread64,openat,close,futex,sched_yield,mmap,munmap \
    -o ${STRACE_OUT} \
    -- uv run python -m ai4land.utils.profile_dataloader \
    --config-name ${CONFIG_NAME} \
    system.slurm_job_id=${SLURM_JOB_ID} \
    ++profile.num_getitem=${NUM_GETITEM} \
    ++profile.num_batches=${NUM_BATCHES} \
    ++profile.split=${SPLIT} \
    ++profile.batch_timeout_s=${BATCH_TIMEOUT_S} \
    ++profile.workers_sweep=${WORKERS_SWEEP} \
    ++profile.num_coldwarm=0 \
    ++data.luh2_load_threads=${LUH2_THREADS}"

echo "Config: ${CONFIG_NAME} | num_getitem=${NUM_GETITEM} | num_batches=${NUM_BATCHES} | split=${SPLIT} | batch_timeout=${BATCH_TIMEOUT_S}s | workers_sweep=${WORKERS_SWEEP} | luh2_threads=${LUH2_THREADS} | strace_out=${STRACE_OUT}"
echo "Launch command: $LAUNCH_CMD"

srun --wait=60 --kill-on-bad-exit=1 --jobid "${SLURM_JOB_ID}" bash -c "$LAUNCH_CMD"

echo "END TIME: $(date)"
