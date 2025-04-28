#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --time=2:00:00
#SBATCH --partition=amilan
#SBATCH --qos=normal
#SBATCH --output=CP_child-%j.out

module purge
module load anaconda

well_fov=$1

# This script is used to run CellProfiler analysis on the timelapse images.
conda activate cellprofiler_timelapse_env

cd scripts || exit

# run the python script
python run_cellprofiler_analysis.py --well_fov "$well_fov"

# change the directory back to the original directory
cd ../ || exit

conda deactivate

# End of the script
echo "CellProfiler analysis is completed."
