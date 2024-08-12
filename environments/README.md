# Environments
This directory contains the necessary files to create the conda environments.

We have the following environments:
- [scDINO.yaml](scDINO.yaml): Environment for running scDINO
- [timelapse_env.yaml](timelapse_env.yaml): Environment for running the time-lapse analysis
- [R_env.yaml](R_env.yaml): Environment for running the R scripts
- [CellProfiling_env.yaml](CellProfiling_env.yaml): Environment for running the Cell Profiling pipeline

## Creating the environments
To create the environments, execute the following command:
```bash
source set_up_envs.sh
```
The environments will be activated when each shell script is run.


## SAM-2 environment
This environment is a bit tricky to create. The following steps are necessary:

```bash
source make_cuda_modules.sh
cd SAM2_install
source make_sam_env.sh
cd ../
```
