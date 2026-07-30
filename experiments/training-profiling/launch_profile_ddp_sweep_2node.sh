#!/bin/bash
#SBATCH -J ddp_sweep_2node
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
export NCCL_IB_DISABLE=1
export SLURM_CPU_BIND=none
export SRUN_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}
export GPUS_PER_NODE=4
export NNODES=${SLURM_NNODES}

export MASTER_ADDR=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)
export MASTER_PORT=29501

mkdir -p "${SLURM_SUBMIT_DIR}/jobs-map/${SLURM_JOB_ID}"
cd "${SLURM_SUBMIT_DIR}"

SCRIPT="${SLURM_SUBMIT_DIR}/src/ai4land/utils/profile_ddp_sweep.py"

LAUNCH_CMD="uv run accelerate launch \
    --multi_gpu \
    --num_processes \$(( ${NNODES} * ${GPUS_PER_NODE} )) \
    --num_machines ${NNODES} \
    --machine_rank \${SLURM_PROCID} \
    --main_process_ip ${MASTER_ADDR} \
    --main_process_port ${MASTER_PORT} \
    ${SCRIPT}"

srun --wait=60 --kill-on-bad-exit=1 --jobid "${SLURM_JOB_ID}" bash -c "$LAUNCH_CMD"
