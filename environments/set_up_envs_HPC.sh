#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=amilan
#SBATCH --qos=normal
#SBATCH --account=amc-general
#SBATCH --time=2:00:00
#SBATCH --output=sample-%j.out

module purge
module load mambaforge

# create conda environments needed for the project

# loop through all environment .yaml files in this directory
for file in $(ls -1 *.yaml); do
    # create conda environment from .yaml file
    mamba env create -f $file
done


# set up plugins for cellprofiler
/scratch/alpine/${USER}/CellProfiler-plugins/active_plugins

