#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=amilan
#SBATCH --qos=normal
#SBATCH --account=amc-general
#SBATCH --time=30:00
#SBATCH --output=ic_child-%j.out

module purge
module load anaconda

# convert the notebook to a python script
jupyter nbconvert --to=script --FilesWriter.build_directory=scripts/ notebooks/*.ipynb


dir=$1

cd scripts/ || exit

echo "Starting IC processing"

# Run CellProfiler for IC processing
conda run -n cellprofiler_timelapse_env python 0.perform_ic.py --input_dir $dir

# return the directory that script was run from
cd ../ || exit

echo "IC processing complete"
