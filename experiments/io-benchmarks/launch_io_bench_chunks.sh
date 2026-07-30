#!/bin/bash
#SBATCH -J io_bench_chunks
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
# Disk -> RAM benchmark on real zarr chunk files.
#
# Picks N random chunk files from one LUH2-style variable in the production
# zarr store, then times each read individually (`cat $chunk > /dev/null`).
# Reports per-chunk latency stats and effective MB/s.
#
# Companion to launch_io_bench.sh (synthetic 4 GiB /dev/zero sequential read):
# this one matches the workload more closely (small files, random access,
# real Blosc-compressed bytes).
#
# Caveats:
# - No O_DIRECT (small files + O_DIRECT alignment is fiddly). To get cold
#   reads we rely on picking N *different* random chunks per job. Re-submit-
#   ting back-to-back may show warmer numbers if chunks remain in page cache.
# - This script reads the SAME store the diagnostic profiler jobs target,
#   so submit only when no other job is hitting that store (AGENTS.md rule).
# - Measures open+read+close, NOT decompress. The dataloader does
#   chunk_read + Blosc_decode + xarray_wrap; this only catches step 1.

set -eu

echo "START TIME: $(date)"
echo "HOST: $(hostname)"

source "${SLURM_SUBMIT_DIR}/scripts/utils.sh"

JOB_OUT_DIR="${SLURM_SUBMIT_DIR}/jobs-map/${SLURM_JOB_ID}"
mkdir -p "${JOB_OUT_DIR}"

# --- Inputs (env-overridable) ---
STORE="${STORE:-/gpfs/scratch/ehpc736/data/AI4LAND_NON_preprocessed-data-2000-2015.zarr}"
CHUNK_VAR="${CHUNK_VAR:-}"
N_CHUNKS="${N_CHUNKS:-100}"

echo "Store:      ${STORE}"
echo "Variable:   ${CHUNK_VAR:-<auto-discover>}"
echo "N_CHUNKS:   ${N_CHUNKS}"

if [ ! -d "${STORE}" ]; then
    echo "ERROR: store directory ${STORE} not found"
    exit 1
fi

# --- Resolve variable: explicit, or auto-discover first var with 3-dim chunks ---
if [ -n "${CHUNK_VAR}" ]; then
    CHUNK_DIR="${STORE}/${CHUNK_VAR}"
    if [ ! -d "${CHUNK_DIR}" ]; then
        echo "WARNING: ${CHUNK_DIR} not found; auto-discovering instead"
        CHUNK_VAR=""
    fi
fi

if [ -z "${CHUNK_VAR}" ]; then
    echo
    echo "=== Auto-discovering a variable with 3-dim chunk files ==="
    for candidate_dir in "${STORE}"/*/; do
        candidate=$(basename "${candidate_dir}")
        first_chunk=$(find "${candidate_dir}" -maxdepth 1 -type f -regextype posix-extended \
            -regex '.*/[0-9]+\.[0-9]+\.[0-9]+' -print -quit 2>/dev/null)
        if [ -n "${first_chunk}" ]; then
            CHUNK_VAR="${candidate}"
            echo "Found: ${CHUNK_VAR}"
            break
        fi
    done
fi

if [ -z "${CHUNK_VAR}" ]; then
    echo "ERROR: no variable with 3-dim chunk files found under ${STORE}"
    echo "Top-level entries at ${STORE}/:"
    ls "${STORE}/" | head -30
    exit 1
fi

CHUNK_DIR="${STORE}/${CHUNK_VAR}"

# --- Select N random chunk files ---
echo
echo "=== Selecting ${N_CHUNKS} random chunks from ${CHUNK_DIR}/ ==="
CHUNK_LIST="${JOB_OUT_DIR}/chunk_list.txt"
find "${CHUNK_DIR}" -maxdepth 1 -type f -regextype posix-extended \
    -regex '.*/[0-9]+\.[0-9]+\.[0-9]+' | shuf | head -${N_CHUNKS} > "${CHUNK_LIST}"
N_FOUND=$(wc -l < "${CHUNK_LIST}")
echo "Selected ${N_FOUND} chunks"

if [ "${N_FOUND}" -lt 10 ]; then
    echo "ERROR: only ${N_FOUND} chunks found"
    exit 1
fi

# --- Read each chunk; record per-chunk wall ms and file size ---
echo
echo "=== Reading chunks (cat > /dev/null, ms wall-clock per read) ==="

LOG="${JOB_OUT_DIR}/chunk_reads.txt"
: > "${LOG}"

while IFS= read -r chunk; do
    bytes=$(stat -c %s "${chunk}")
    start_ns=$(date +%s%N)
    cat "${chunk}" > /dev/null
    end_ns=$(date +%s%N)
    elapsed_ms=$(( (end_ns - start_ns) / 1000000 ))
    printf "%5d %8d %s\n" "${elapsed_ms}" "${bytes}" "$(basename "${chunk}")" >> "${LOG}"
done < "${CHUNK_LIST}"

# --- Summary stats ---
SORTED="${LOG}.sorted"
awk '{print $1}' "${LOG}" | sort -n > "${SORTED}"

n=$(wc -l < "${SORTED}")
mean_ms=$(awk '{s+=$1} END {printf "%.1f", s/NR}' "${SORTED}")
min_ms=$(head -1 "${SORTED}")
max_ms=$(tail -1 "${SORTED}")
p50_idx=$(( n / 2 ));        [ ${p50_idx} -lt 1 ] && p50_idx=1
p95_idx=$(( n * 95 / 100 )); [ ${p95_idx} -lt 1 ] && p95_idx=1
p99_idx=$(( n * 99 / 100 )); [ ${p99_idx} -lt 1 ] && p99_idx=1
p50_ms=$(sed -n "${p50_idx}p" "${SORTED}")
p95_ms=$(sed -n "${p95_idx}p" "${SORTED}")
p99_ms=$(sed -n "${p99_idx}p" "${SORTED}")
total_bytes=$(awk '{s+=$2} END {print s}' "${LOG}")
total_ms=$(awk '{s+=$1} END {print s}' "${LOG}")
mean_bytes=$(awk '{s+=$2} END {printf "%.0f", s/NR}' "${LOG}")
mbps=$(awk -v b="${total_bytes}" -v t="${total_ms}" 'BEGIN {
    if (t > 0) printf "%.1f", (b / 1048576.0) / (t / 1000.0); else print "N/A"
}')

echo
echo "=== Per-chunk read summary (variable=${CHUNK_VAR}, n=${n}) ==="
printf "%-22s %s\n" "Mean:"             "${mean_ms} ms"
printf "%-22s %s\n" "Median (p50):"     "${p50_ms} ms"
printf "%-22s %s\n" "p95:"              "${p95_ms} ms"
printf "%-22s %s\n" "p99:"              "${p99_ms} ms"
printf "%-22s %s\n" "Min:"              "${min_ms} ms"
printf "%-22s %s\n" "Max:"              "${max_ms} ms"
printf "%-22s %s\n" "Mean chunk size:"  "${mean_bytes} bytes"
printf "%-22s %s\n" "Total bytes:"      "${total_bytes}"
printf "%-22s %s\n" "Total wall time:"  "${total_ms} ms"
printf "%-22s %s\n" "Effective rate:"   "${mbps} MB/s"

echo
echo "First 10 reads (ms bytes filename):"
head -10 "${LOG}"
echo
echo "Last 10 reads:"
tail -10 "${LOG}"

echo
echo "END TIME: $(date)"
