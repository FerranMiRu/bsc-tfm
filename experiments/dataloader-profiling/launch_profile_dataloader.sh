#!/bin/bash
#SBATCH -J profile_loader
#SBATCH --exclusive
#SBATCH --qos=acc_ehpc
#SBATCH --output=jobs-map/%j/%j.out
#SBATCH --error=jobs-map/%j/%j.err
#SBATCH --time=04:00:00
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
NUM_GETITEM="${2:-50}"
NUM_BATCHES="${3:-80}"
SPLIT="${4:-train}"
BATCH_TIMEOUT_S="${5:-30}"
WORKERS_SWEEP="${6:-[0,1,4,12]}"
LUH2_THREADS="${7:-1}"
NUM_COLDWARM="${8:-50}"

# --- Per-job output directory ---
JOB_OUT_DIR="${SLURM_SUBMIT_DIR}/jobs-map/${SLURM_JOB_ID}"
mkdir -p "${JOB_OUT_DIR}"
export AI4LAND_LOG_DIR="${JOB_OUT_DIR}"

cp "${SLURM_SUBMIT_DIR}/inputs/${CONFIG_NAME}.yaml" "${JOB_OUT_DIR}/"

cd "${SLURM_SUBMIT_DIR}"

# --- NVMe staging (opt-in via NVME_STAGE=1) ---
NVME_STAGE="${NVME_STAGE:-0}"
SOURCE_STORE="/gpfs/scratch/ehpc736/data/AI4LAND_NON_preprocessed-data-2000-2015.zarr"

STORE_OVERRIDE=""
if [ "$NVME_STAGE" = "1" ]; then
    if [ -z "${TMPDIR:-}" ]; then
        echo "ERROR: NVME_STAGE=1 but TMPDIR is unset"
        exit 1
    fi
    NVME_STORE="${TMPDIR}/$(basename ${SOURCE_STORE})"
    echo "NVMe staging:"
    echo "  source: ${SOURCE_STORE}"
    echo "  target: ${NVME_STORE}"
    echo "  $(date +%H:%M:%S) starting rsync..."
    time rsync -a "${SOURCE_STORE}/" "${NVME_STORE}/"
    echo "  $(date +%H:%M:%S) rsync complete. Staged size: $(du -sh ${NVME_STORE} | cut -f1)"
    STORE_OVERRIDE="data.stores=${NVME_STORE}"
fi

LAUNCH_CMD="uv run python -m ai4land.utils.profile_dataloader \
    --config-name ${CONFIG_NAME} \
    system.slurm_job_id=${SLURM_JOB_ID} \
    ++profile.num_getitem=${NUM_GETITEM} \
    ++profile.num_batches=${NUM_BATCHES} \
    ++profile.split=${SPLIT} \
    ++profile.batch_timeout_s=${BATCH_TIMEOUT_S} \
    ++profile.workers_sweep=${WORKERS_SWEEP} \
    ++profile.num_coldwarm=${NUM_COLDWARM} \
    ++data.luh2_load_threads=${LUH2_THREADS} \
    ${STORE_OVERRIDE}"

echo "Config: ${CONFIG_NAME} | num_getitem=${NUM_GETITEM} | num_batches=${NUM_BATCHES} | split=${SPLIT} | batch_timeout=${BATCH_TIMEOUT_S}s | workers_sweep=${WORKERS_SWEEP} | luh2_threads=${LUH2_THREADS} | num_coldwarm=${NUM_COLDWARM} | nvme_stage=${NVME_STAGE}"
echo "Launch command: $LAUNCH_CMD"

srun --wait=60 --kill-on-bad-exit=1 --jobid "${SLURM_JOB_ID}" bash -c "$LAUNCH_CMD"

echo "END TIME: $(date)"
