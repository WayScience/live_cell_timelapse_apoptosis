#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=6
#SBATCH --partition=amilan
#SBATCH --qos=normal
#SBATCH --account=amc-general
#SBATCH --time=6:00:00
#SBATCH --output=sample-%j.out

module purge
module load anaconda

# convert the notebook to a python script
jupyter nbconvert --to=script --FilesWriter.build_directory=scripts/ notebooks/*.ipynb

# change the directory to the scripts folder
cd scripts/

echo "Starting IC processing"

# Run CellProfiler for IC processing
conda run -n cellprofiler_timelapse_env python perform_ic.py

# return the directory that script was run from
cd ../

echo "IC processing complete"
