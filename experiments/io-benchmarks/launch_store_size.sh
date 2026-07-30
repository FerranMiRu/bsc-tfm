#!/bin/bash
#SBATCH -J store-size
#SBATCH --qos=acc_ehpc
#SBATCH --output=jobs-map/%j/%j.out
#SBATCH --error=jobs-map/%j/%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --account=ehpc536

set -eu

echo "START TIME: $(date)"

DATA="/gpfs/scratch/ehpc736/data"
CONCERTO_S="/gpfs/scratch/ehpc606/ai4land/CONCERTO"
CONCERTO_P="/gpfs/projects/ehpc606/ai4land/CONCERTO"

STORES=(
  "${DATA}/AI4LAND_NON_preprocessed-data-2000-2015.zarr"
  "${DATA}/AI4LAND_NON_preprocessed-data-1901-2015.zarr"
  "${DATA}/AI4LAND_merged-NON_preprocessed-data-2000-2015.zarr"
  "${DATA}/AI4LAND_merged_NON_preprocessed-data-1901-2015.zarr"
  "${CONCERTO_S}/luh2_1850-2015_cube_512.zarr"
  "${CONCERTO_P}/hilda_1899-2020_cube_512_single_channel_8_class_clean.zarr"
  "${CONCERTO_P}/static_cube_processed.zarr"
  "${CONCERTO_P}/koppen_geiger_annual_cube_1901-2020_512.zarr"
  "${CONCERTO_P}/population_histsoc_cube_1850-2020_512.zarr"
)

echo "size_on_disk,store"
for store in "${STORES[@]}"; do
  du -sh "${store}"
done

echo "END TIME: $(date)"
