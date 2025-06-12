#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=1:00:00
#SBATCH --partition=amilan
#SBATCH --qos=normal
#SBATCH --output=CP_parent-%j.out

module purge
module load anaconda

# This script is used to run CellProfiler analysis on the timelapse images.
conda activate cellprofiler_timelapse_env
# convert the jupyter notebook to a python script
jupyter nbconvert --to=script --FilesWriter.build_directory=scripts/ notebooks/*.ipynb
cd scripts || exit
python get_well_fov_array.py
cd ../ || exit

# load the well_fov array
readarray -t well_fovs < ./well_fov_loading/well_fov_dirs.csv

for well_fov in "${well_fovs[@]}"; do
    number_of_jobs=$(squeue -u $USER | wc -l)
    while [ $number_of_jobs -gt 990 ]; do
        sleep 1s
        number_of_jobs=$(squeue -u $USER | wc -l)
    done
    sbatch run_cellprofiler_child_HPC.sh "$well_fov"
done

conda deactivate

# End of the script
echo "All CP jobs submitted."
