# Environments
This directory contains the necessary files to create the conda environments.

We have the following environments:
- [scDINO.yaml](scDINO.yaml): Environment for running scDINO
- [timelapse_env.yaml](timelapse_env.yaml): Environment for running the timelapse analysis
- [R_env.yaml](R_env.yaml): Environment for running the R scripts

## Creating the environments
To create the environments, execute the following command:
```bash
conda env create -f scDINO.yaml
conda env create -f timelapse_env.yaml
conda env create -f R_env.yaml
```
The environments will be activated when each shell script is run.
