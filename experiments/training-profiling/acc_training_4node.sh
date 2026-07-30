#!/bin/bash
#SBATCH -J training_4node
#SBATCH --exclusive
#SBATCH --qos=acc_ehpc # acc_ehpc, acc_debug, acc_bsces, acc_bsccs
#SBATCH --output=logs/training/%j/%j.out
#SBATCH --error=logs/training/%j/%j.err
#SBATCH --time=72:00:00
#SBATCH --nodes=4
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=80

# --- Email Notifications ---
##SBATCH --mail-type=end
##SBATCH --mail-user=oscar.molina@bsc.es

## EuroHPC allocation
#SBATCH --account=ehpc819 # ehpc249, bsc70, bsc32

echo "START TIME: $(date)"

source "${SLURM_SUBMIT_DIR}/scripts/utils.sh"

export PYTHONWARNINGS="ignore::FutureWarning"
export HYDRA_FULL_ERROR=1
export LOGLEVEL=WARN

# Cache & Offline Mode
export HF_HOME="/gpfs/scratch/ehpc606/ELLIOT/huggingface_cache"
export TORCH_HOME="/gpfs/scratch/ehpc606/ELLIOT/torch_cache"
export HF_HUB_OFFLINE=1

# --- NCCL Configuration ---
# IB disabled matches the proven 2-node bench recipe (Run 35, 88-92% scaling).
export NCCL_IB_DISABLE=1
# accelerate launch (unlike torchrun) does not set this; without it each of the ~52 processes
# defaults OMP to the 80-core count -> oversubscription on the CPU-bound dataloader.
export OMP_NUM_THREADS=1

# --- Distributed Training Configuration ---
export SLURM_CPU_BIND=none
export SRUN_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}
export GPUS_PER_NODE=4
export NNODES=${SLURM_NNODES}

# export CUDA_LAUNCH_BLOCKING=1 # For debugging assertions

export MASTER_ADDR=$(scontrol show hostnames ${SLURM_JOB_NODELIST} | head -n 1)
export MASTER_PORT=29500
echo "Master: ${MASTER_ADDR}:${MASTER_PORT}"

# --- Path Configuration ---
TRAIN_SCRIPT="${SLURM_SUBMIT_DIR}/src/ai4land/training/train.py"
CONFIG_NAME="${1:-unified}" # Defaults to "unified", pass "fsdp" as first arg to override

# --- Per-job output directory (slurm .out/.err, unet log, config snapshot) ---
JOB_OUT_DIR="${SLURM_SUBMIT_DIR}/logs/training/${SLURM_JOB_ID}"
mkdir -p "${JOB_OUT_DIR}"
export AI4LAND_LOG_DIR="${JOB_OUT_DIR}"

cp "${SLURM_SUBMIT_DIR}/inputs/${CONFIG_NAME}.yaml" "${JOB_OUT_DIR}/"

# --- Model Checkpoint ---
# Pass the path to a folder or specific .pth file to resume training from that point
CKPT=""
MODEL_TYPE=""

# --- Launch Training with Accelerate ---
# We use bash -c to defer evaluation of SLURM_PROCID to the compute node
LAUNCH_CMD="uv run accelerate launch \
    --multi_gpu \
    --num_processes \$(( ${NNODES} * ${GPUS_PER_NODE} )) \
    --num_machines ${NNODES} \
    --machine_rank \${SLURM_PROCID} \
    --main_process_ip ${MASTER_ADDR} \
    --main_process_port ${MASTER_PORT} \
    --mixed_precision bf16 \
    --dynamo_backend no \
    ${TRAIN_SCRIPT} \
    --config-name ${CONFIG_NAME} \
    system.slurm_job_id=${SLURM_JOB_ID} \
    training.resume.path=${CKPT}"

# --mixed_precision fp16 \

# Modify model type to match checkpoint if passed
if [ -n "$CKPT" ]; then
    LAUNCH_CMD="${LAUNCH_CMD} \
    model.type=${MODEL_TYPE}"
fi

echo "Launch command: $LAUNCH_CMD"

SRUN_ARGS="\
    --wait=60 \
    --kill-on-bad-exit=1\
"

# Change into the repository root so that relative imports and hydra config paths work seamlessly
cd ${SLURM_SUBMIT_DIR}

srun ${SRUN_ARGS} --jobid ${SLURM_JOB_ID} bash -c "$LAUNCH_CMD"

echo "END TIME: $(date)"
