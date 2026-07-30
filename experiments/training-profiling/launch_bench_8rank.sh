#!/bin/bash
#SBATCH -J bench_8rank
#SBATCH --exclusive
#SBATCH --qos=acc_ehpc
#SBATCH --output=jobs-map/%j/%j.out
#SBATCH --error=jobs-map/%j/%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=2
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=80
#SBATCH --account=ehpc536

set -eu

source "${SLURM_SUBMIT_DIR}/scripts/utils.sh"

export PYTHONWARNINGS="ignore::FutureWarning"
export HYDRA_FULL_ERROR=1
export LOGLEVEL=WARN
export HF_HOME="/gpfs/scratch/ehpc606/ELLIOT/huggingface_cache"
export TORCH_HOME="/gpfs/scratch/ehpc606/ELLIOT/torch_cache"
export HF_HUB_OFFLINE=1
export NCCL_IB_DISABLE=1
export SLURM_CPU_BIND=none
export SRUN_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}
export GPUS_PER_NODE=4
export NNODES=${SLURM_NNODES}

export MASTER_ADDR=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)
export MASTER_PORT=29500

CONFIG_NAME="${1:-debug-single}"
shift || true
HYDRA_OVERRIDES="$@"

TRAIN_SCRIPT="${SLURM_SUBMIT_DIR}/src/ai4land/training/train.py"
JOB_OUT_DIR="${SLURM_SUBMIT_DIR}/jobs-map/${SLURM_JOB_ID}"
mkdir -p "${JOB_OUT_DIR}"
export AI4LAND_LOG_DIR="${JOB_OUT_DIR}"

cp "${SLURM_SUBMIT_DIR}/inputs/${CONFIG_NAME}.yaml" "${JOB_OUT_DIR}/"

cd "${SLURM_SUBMIT_DIR}"

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
    training.resume.path= \
    ${HYDRA_OVERRIDES}"

srun --wait=60 --kill-on-bad-exit=1 --jobid "${SLURM_JOB_ID}" bash -c "$LAUNCH_CMD"
