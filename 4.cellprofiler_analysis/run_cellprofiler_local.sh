#!/bin/bash

# This script is used to run CellProfiler analysis on the timelapse images.
conda activate cellprofiler_timelapse_env
# convert the jupyter notebook to a python script
jupyter nbconvert --to=script --FilesWriter.build_directory=scripts/ notebooks/*.ipynb
cd scripts || exit
python get_well_fov_array.py


# time the bash script
# start the timer
start=$(date +%s)
# load the well_fov array
readarray -t well_fovs < ../well_fov_loading/well_fov_dirs.csv
# get the length of the well_fov array
well_fovs_length=${#well_fovs[@]}
i=0
for well_fov in "${well_fovs[@]}"; do
    # run the python script
    echo "Running $i out of $well_fovs_length"
    python run_cellprofiler_analysis.py --well_fov "$well_fov"
    i=$((i + 1))
done

# end the timer
end=$(date +%s)
runtime=$((end - start))
echo "The runtime of the script is $runtime"
# change the directory back to the orginal directory
cd ../ || exit

conda deactivate

# End of the script
echo "CellProfiler analysis is completed."
