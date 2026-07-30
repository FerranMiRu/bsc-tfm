## Week 1 (16F)

- Onboarding
- First contact with AI4Land Repo

## Week 2 (23F)

- CONCERTO
- Gained access to MN5 and set up environment
- Figure out modules and installation options
- Tried some runs of AI4Land

## Week 3 (2M)

- uv migration
- firsts runs to test cuda and pytorch

## Week 4 (9M)

- Profiling HPC applications training

## Week 5 (16M)

- uv migration: new merge from HPAI group required new dependencies
- torch comparison: module vs package (see https://gitlab.earth.bsc.es/ces/ai4land/-/merge_requests/40)
- according to https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html, cuda 12.8 should work with the GPUs in MN5 (H100)

## Week 6 (23M)

- mojo is unviable because they are focusing on inference rather than training right now
- found https://medium.com/@Modexa/8-pytorch-dataloader-tactics-to-max-out-your-gpu-22270f6f3fa8, seems an excellent place to start
- created optimization guide for after refactor:
  - use bf16 (look for references when TFM)
  - use torch.compile in the inner UNet (look for references when TFM)
  - cudnn.benchmark option (look for references when TFM)

## Week 7 (30 M)

- refactor of the source code to use src and create package
- implement optimization guide

## Week 8 (6A)

- optimization guide already implemented (most of it) by HPAI
- run with different values for optimization but no changes in time given dataloader
- start application of ruff

## Week 9 (13A)

- finish refactor and application of ruff

## Week 10 (20A)

- workshop for dataloader optimization
  - moved to a single zarr datastore (need to do tests to see impact on performance)
  - Jordi applied nsys traces but couldn’t see it yet

## Week 11 (27A)

- meeting with Jordi to create unified dataset without preprocessing and another with preprocessing for comparison
- started applying nsys profiling to get a baseline on mutlizarr store
- finally got the first nsys traces. Took 15 minutes but the profiling is just 2 minutes, with 99% of the time being dataloader
- looks like the sem_wait might be the cause for the behaviour

## Week 12 (4M)

- deep dive in traces:
  - seems like the main process is only around 50% efficient when actually doing work (SM Warp Occupancy), should be solved afterwrads
  - the other gpu metrics look good
  - there might be too many threads contending for cpu cores, to be looked after with OMP_NUM_THREADS=1 on the workers
- removed synchronizer from dataloader, the profiling run took only 85s (vs 136s previously) step11 still took >50s but it’s the only step in which this happened.
- firsts tests with single zarr store. Getting errors due to KG being “static”

## Week 13 (11M)

- Fixed KG errors, now the loss is 0.0, not sure why. Found the reason, we had to fill the nans in KG with a value.
- Checked the size of the datasets:
  - SingleZarr
    - 213G - /gpfs/scratch/ehpc736/data/AI4LAND-data-1960-2015.zarr/
    - 66G - /gpfs/scratch/ehpc736/data/AI4LAND_NON_preprocessed-data-2000-2015.zarr/
    - 64G - /gpfs/scratch/ehpc736/data/AI4LAND_preprocessed-data-2000-2015.zarr/
  - MultiZarr
    - 606G - /gpfs/scratch/ehpc606/ai4land/CONCERTO/luh2_1850-2015_cube_512.zarr
    - 4.5G - /gpfs/projects/ehpc606/ai4land/CONCERTO/hilda_1899-2020_cube_512_single_channel_8_class_clean.zarr
    - 4.1G - /gpfs/projects/ehpc606/ai4land/CONCERTO/static_cube_processed.zarr
    - 4.4G - /gpfs/projects/ehpc606/ai4land/CONCERTO/koppen_geiger_annual_cube_1901-2020_512.zarr
    - 61G - /gpfs/projects/ehpc606/ai4land/CONCERTO/population_histsoc_cube_1850-2020_512.zarr
- The loss is too different to be simple randomness, especially the validation one. Looking at the differences between the multizarr vs singlezarr vs preprocessed we see that HILDA should be remapped in the preprocessed one and it looks like it’s not, KG has value 31 when NaN in multizarr but 0 in the other two (which is correct?), normalization for all the other variables have changed.
- To check next week: can we trust the preprocessed dataset as what the data should return? If so, we can implement code that makes the non-processed path lead to the same data as the preprocessed path. When done, we can start profiling the performance differences.

## Week 14 (18M)

- Possible reference on zarr: https://www.earthdata.nasa.gov/s3fs-public/2024-05/ESDS-RFC-048%20Zarr%20Storage%20Specification%20V2%20v1.pdf
- Proper preprocessing and guards have been implemented following what was defined in issue #44 and implemented in here. Now the non-preprocessed and the processed dataset return the exact same values after the loader. The (+1e-8) were removed from the normalizations. Also missing the hilda targets that were clipped (instead of remapped) in the data pipeline of the preprocessed dataset so that dataset is unusable until further notice. Ensured that kg, population, aspect, elevation and slope have no nans.
- Meeting with EPICURE
  - reduce workers to 4/8 per gpu
    - try also 0 workers
  - memory usage is only 4GB means we need bigger batches
  - check for persistance workers
  - add tags inside step11:
    - how the get_item reprocessing is done
- Hilda targets have been fixed by Joan Verdi
- Found the difference in continuous and static between new and preprocessed singlezarr. In the pipeline `mean` and `std` are always taken instead of `mean_full_time_train` and `std_full_time_train`. This leads to small but noticeable differences. To discuss with Marina what is the correct one to take and potentially also apply it to the code.
- Store + Stores → Stores only

## Week 15 (25M)

- SingleZarr vs MultiZarr experiments: (initially synchronizer was still in for both single and multi)
  - first had some problems and had to change the name of the hilda_labels to lulc_states (already done in the debug and unified configs but not profiling)
  - **BATCH 1**: No changes to the main code. Didn’t have any metrics (loss) or validation so it was unsure the model really learned anything. The preprocessed one is perfectly balanced (looking at profiling) with no step taking any more time or dataloader appearing, main time is spent on forward and backward. In multizarr a single step took 74s to dataload
    - 41088388 - singlezarr: 1.3s
    - 41088390 - preprocessed: 0.8s
    - 41088508 - multizarr: 80s
  - **BATCH 2**: Now we see the same pattern as in multizarr in singlezarr/preprocessed. In both cases we see 2 steps (11/12, 11/13) taking 10-19 seconds of dataloader. The multizarr is exactly the same as before. Now we see the loss per epoch and we can confirm the model is learning. We suspect cache
    - 41093449 - singlezarr: 28.4s
    - 41093451 - preprocessed: 23.9s
    - 41093456 - multizarr: 74.3s
  - **BATCH 3**: Singlezarr took 10.1s and preprocseed 11.2s, while multizarr still took 82.4. It looks like some kind of cache might be at play but I’m still unsure. The losses are consistent across the 3 batches per dataset.
    - 41094289 - singlezarr
    - 41094291 - preprocessed
    - 41094292 - multizarr
  - **BATCH 4**: Same times as before more or less.
    - 41162051 - singlezarr
    - 41162052 - preprocessed
    - 41162053 - multizarr
  - **BATCH 5**: Singlezarr took 8s while preprocessed took 20s, and multizarr kept on the trend by taking 85s. This was launched immediately after batch4 so I suspect no cache is at play and the 1s run we got was just a fluke.
    - 41164024 - singlezarr
    - 41164025 - preprocessed
    - 41164026 - multizarr

- A pattern was found in which the batch12 is always the one to take extra time. Other batches can take some more time but 12 is the one that takes the most time and the only one that is consistent. A `warmup_batches` parameter has been added so that we can profile batches 6-10 instead of 11-15. If the pattern holds we can discard profiling intervention.

- **BATCH 6:** Mistake happened and the training stopped after the profiling window (batch 10). Need for repeat
  - 41164619 - singlezarr
  - 41164620 - preprocessed
  - 41164621 - multizarr
- **BATCH 7**: Now with the proper profiling we see that we have a hold up in step 11 rather than 12. So profiler was affecting the time it took to do step 12 for some reason. Now we suspect that this hold up is due to the profiling finishing (it usually takes 30 to 40s). Multizarr still had a hold up in step 12 but it was smaller (66s against the usual 80s). The 3 jobs took more or less the same time (multizarr took 0.774s while the other 2 took 0.784). That concludes that when we have the dataloaded, all take the same time as expected.
  - 41165374 - singlezarr
  - 41165375 - preprocessed
  - 41165376 - multizarr

- We have removed profiling for this next batch

- **BATCH 8**: The 12 step pattern has reappeared. In singlezarr and multizarr we also see a 3/4 batch time increase which is curious. The test with 50 batches should clear things up.
  - 41166558 - singlezarr
  - 41166559 - preprocessed
  - 41166561 - multizarr

- We now have increased the epoch size to 50 batches to check if the pattern repeats.

- **BATCH 9**: The pattern repeats cleanly every 12 with minor spikes in between. Plus we see that singlezarr and preprocessed have lower spikes all around. The big job with preprocessed has been launched even though there’s clipping on the hilda labels to see if the size of the store affects the size of the spikes. Indeed it lead to x1.9 time in the spikes
  - 41171635 - singlezarr
  - 41171636 - preprocessed
  - 41171637 - multizarr
  - 41172665 - preprocessed-big
- **BATCH 10**: To ensure the previous result we rerun the same jobs twice. We expect to confirm that a bigger zarr mean bigger spikes. In this case we saw a x3 increase (31s to 95s)
  - 41199507 - preprocessed
  - 41199508 - preprocessed-big
- **BATCH 11**: 41s to 97s. The size difference is confirmed. This also means that having a single datastore will probably lead to worse performance when we want to just use some features instead of all of them.
  - 41199507 - preprocessed
  - 41199508 - preprocessed-big

## Week 16 (1J)

- We continue with experiments:
  - we first added a toggle for the synchronizer
    - **BATCH 12**: Didn’t change anything.
      - 41318578 - with
      - 41318658 - without
