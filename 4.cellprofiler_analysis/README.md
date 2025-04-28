# Running the CellProfiler analysis pipeline

This pipeline can be run on local machine or on a High Performance Compute (HPC) cluster.
Instructions for both are as follows:
## Running on local machine
```bash
source run_cellprofiler_local.sh
```

Running on a local machine took:
~46.11 hours to run

Machine specs:
AMD Ryzen 9 5900X (24) @ 3.700GHz
128GB RAM
Pop!_OS 22.04 LTS x86_64

## Running on HPC cluster (SLURM)
```bash
sbatch run_cellprofiler_parent_HPC.sh
```

The HPC cluster is a SLURM scheduler. The script `run_cellprofiler_parent_HPC.sh` will submit the job to the queue and run the analysis pipeline on the cluster.
The script submits a single job for each FOV to run in parallel.
