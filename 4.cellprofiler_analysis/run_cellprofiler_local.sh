#!/bin/bash

# This script is used to run CellProfiler analysis on the timelapse images.
conda activate cellprofiler_timelapse_env
# convert the jupyter notebook to a python script
jupyter nbconvert --to=script --FilesWriter.build_directory=scripts/ notebooks/*.ipynb
cd scripts || exit
python get_well_fov_array.py

# load the well_fov array
readarray -t well_fovs < ../well_fov_loading/well_fov_dirs.csv

for well_fov in "${well_fovs[@]}"; do
    # run the python script
    python run_cellprofiler_analysis.py --well_fov "$well_fov"
done

# change the directory back to the orginal directory
cd ../ || exit

conda deactivate

# End of the script
echo "CellProfiler analysis is completed."
