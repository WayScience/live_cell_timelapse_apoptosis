#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --time=6:00:00
#SBATCH --partition=amilan
#SBATCH --qos=normal
#SBATCH --output=sample-%j.out

module purge
module load anaconda

# This script is used to run CellProfiler analysis on the timelapse images.
conda activate cellprofiler_timelapse_env
# convert the jupyter notebook to a python script
jupyter nbconvert --to=script --FilesWriter.build_directory=scripts/ notebooks/*.ipynb

conda deactivate


# change the directory to the scripts folder
cd scripts/

# run the python script
conda run -n cellprofiler_timelapse_env python run_cellprofiler_pipe_with_sam_outputs.py

# change the directory back to the orginal directory
cd ../

# End of the script
echo "CellProfiler analysis is completed."
