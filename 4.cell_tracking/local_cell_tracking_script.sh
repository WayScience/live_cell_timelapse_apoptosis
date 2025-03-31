#!/bin/bash

conda activate cell_tracking_env


jupyter nbconvert --to=script --FilesWriter.build_directory=scripts/ notebooks/*.ipynb

cd scripts/ || exit

well_fov_path="../../2.cellprofiler_ic_processing/illum_directory/timelapse/"
well_fovs=$(ls $well_fov_path)
echo "${well_fovs[@]}"

for well_fov in $well_fovs; do
    echo "Processing well_fov: $well_fov"
    python 0.nuclei_tracking.py --input_dir $well_fov_path$well_fov
done

cd ../ || exit

conda deactivate

echo "Cell tracking script completed"
