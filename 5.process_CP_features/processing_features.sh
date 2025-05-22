#!/bin/bash
# This script processes the timelapse data

# convert all notebooks to python files into the scripts folder
echo "Converting notebooks to Python scripts..."
jupyter nbconvert --to python --output-dir=scripts/ notebooks/*.ipynb
echo "Conversion complete."

conda activate cellprofiler_timelapse_env

cd scripts/ || exit

# get all fovs in the data folder
fovs=$(ls ../data/1.annotated_data/timelapse/*)
for well_fov in $fovs; do
    # get the fov name
    well_fov=$(basename "$well_fov")
    # split by underscore
    well_fov=$(echo "$well_fov" | cut -d'_' -f1-2)
    # check if the fov name is equal to the well_fov
        echo "Processing $well_fov..."
        # run the script for the fov
        python 0.merge_sc.py --well_fov "$well_fov"
        python 1.annotate_sc.py --well_fov "$well_fov"
        python 2.fuzzy_matching.py --well_fov "$well_fov"
done
python 3.combine_profiles.py
python 4.normalize.py
python 5.feature_select_sc.py
python 6.aggregation.py
python 7.whole_image_pipeline_processing.py



cd ../ || exit

conda deactivate

echo "Processing complete."
