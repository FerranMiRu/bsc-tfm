#!/bin/bash
#SBATCH -J io_bench
#SBATCH --exclusive
#SBATCH --qos=acc_ehpc
#SBATCH --output=jobs-map/%j/%j.out
#SBATCH --error=jobs-map/%j/%j.err
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20

## EuroHPC allocation
#SBATCH --account=ehpc536
#
# Basic disk -> RAM benchmark on MN5 using dd, no O_DIRECT.
#
# Two tests:
#   1. Single-thread sequential read of one 4 GiB file (cached path; what one
#      zarr worker effectively sees once the page cache is warm).
#   2. ${N_STREAMS} parallel sequential reads, each from a SEPARATE 4 GiB file
#      (closer to what 12 dataloader workers do — independent GPFS streams,
#      no cross-stream cache reuse on the first pass).
#
# Why separate files for the parallel test: if all 12 dd processes read the
# same file, the kernel fetches each page once and the other 11 hit the page
# cache, inflating apparent throughput. Separate files force each stream to
# pull from GPFS independently.
#
# Caveats:
# - No O_DIRECT (per our research showing GPFS has a ~12x O_DIRECT penalty).
# - Page-cache state between job submissions on the same node is unpredictable.
#   To force a clean cold state, delete the test files (rm -f ${FILES_DIR}/*)
#   before resubmitting.
# - Test 1 leaves file_0 in cache, so file_0's stream in Test 2 will be fast.
#   With 12 streams this only shifts the aggregate by <10%.

set -eu

echo "START TIME: $(date)"
echo "HOST: $(hostname)"

source "${SLURM_SUBMIT_DIR}/scripts/utils.sh"

JOB_OUT_DIR="${SLURM_SUBMIT_DIR}/jobs-map/${SLURM_JOB_ID}"
mkdir -p "${JOB_OUT_DIR}"

N_STREAMS="${N_STREAMS:-12}"
SIZE_GB="${SIZE_GB:-4}"
SIZE_BYTES=$(( SIZE_GB * 1073741824 ))
SIZE_BLOCKS=$(( SIZE_GB * 1024 ))   # 1M blocks

FILES_DIR="/gpfs/scratch/ehpc736/${USER}/io_bench"
mkdir -p "${FILES_DIR}"

echo
echo "Config: N_STREAMS=${N_STREAMS} | SIZE_GB=${SIZE_GB} | FILES_DIR=${FILES_DIR}"

# --- Ensure N_STREAMS files of SIZE_GB exist ---
echo
echo "=== Ensuring ${N_STREAMS} test files of ${SIZE_GB} GiB each exist ==="
for stream_index in $(seq 0 $((N_STREAMS - 1))); do
    file="${FILES_DIR}/testfile_${SIZE_GB}g_${stream_index}.bin"
    if [ ! -f "${file}" ] || [ "$(stat -c %s "${file}")" -ne ${SIZE_BYTES} ]; then
        echo "Creating ${file}..."
        dd if=/dev/zero of="${file}" bs=1M count=${SIZE_BLOCKS} conv=fsync status=none
    fi
done
ls -lh "${FILES_DIR}/" | head -20

# --- Test 1: single-thread sequential read, no O_DIRECT ---
FILE0="${FILES_DIR}/testfile_${SIZE_GB}g_0.bin"
echo
echo "=== Test 1: single-thread sequential read | bs=1M | NO O_DIRECT ==="
echo "File: ${FILE0}"
dd if="${FILE0}" of=/dev/null bs=1M 2>&1

# --- Test 2: N parallel streams, separate files, no O_DIRECT ---
echo
echo "=== Test 2: ${N_STREAMS} parallel sequential reads | bs=1M | NO O_DIRECT ==="
echo "(each stream reads a different ${SIZE_GB} GiB file)"
echo

parallel_log="${JOB_OUT_DIR}/parallel_streams.log"
: > "${parallel_log}"

start_ns=$(date +%s%N)
for stream_index in $(seq 0 $((N_STREAMS - 1))); do
    file="${FILES_DIR}/testfile_${SIZE_GB}g_${stream_index}.bin"
    ( dd if="${file}" of=/dev/null bs=1M 2>&1 \
        | sed "s/^/[stream ${stream_index}] /" ) >> "${parallel_log}" &
done
wait
end_ns=$(date +%s%N)

elapsed_ms=$(( (end_ns - start_ns) / 1000000 ))
total_bytes=$(( N_STREAMS * SIZE_BYTES ))
aggregate_mbps=$(awk -v b="${total_bytes}" -v t="${elapsed_ms}" 'BEGIN {
    if (t > 0) printf "%.1f", (b / 1048576.0) / (t / 1000.0); else print "N/A"
}')
per_stream_mbps=$(awk -v agg="${aggregate_mbps}" -v n="${N_STREAMS}" 'BEGIN {
    printf "%.1f", agg / n
}')

echo "--- per-stream output (interleaved order, dd summary per stream) ---"
cat "${parallel_log}"

echo
echo "=== Test 2 summary ==="
printf "%-26s %s\n" "Streams:"               "${N_STREAMS}"
printf "%-26s %s\n" "Bytes per stream:"      "${SIZE_BYTES}"
printf "%-26s %s\n" "Aggregate bytes:"       "${total_bytes}"
printf "%-26s %s\n" "Wall time (last done):" "${elapsed_ms} ms"
printf "%-26s %s\n" "Aggregate throughput:"  "${aggregate_mbps} MB/s"
printf "%-26s %s\n" "Per-stream avg:"        "${per_stream_mbps} MB/s"

echo
echo "END TIME: $(date)"
