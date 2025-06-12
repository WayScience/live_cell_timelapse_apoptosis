#!/bin/bash
# This script processes the timelapse data

# convert all notebooks to python files into the scripts folder
echo "Converting notebooks to Python scripts..."
jupyter nbconvert --to script --output-dir=scripts/ notebooks/*.ipynb
echo "Conversion complete."

conda activate cellprofiler_timelapse_env

cd scripts/ || exit


well_fovs_dir="../4.cellprofiler_analysis/well_fov_loading/well_fov_dirs.csv"

mapfile -t well_fovs < "$well_fovs_dir"

for well_fov in "${well_fovs[@]}"; do
    # check if the fov name is equal to the well_fov
    echo "Processing $well_fov..."
    # run the script for the fov
    # python 0.merge_sc.py --well_fov "$well_fov"
    python 1.annotate_sc.py --well_fov "$well_fov"
    python 2a.fuzzy_matching_timelapse.py --well_fov "$well_fov"
    python 2b.fuzzy_matching_endpoint.py --well_fov "$well_fov"
    echo "Fuzzy matching complete."
done

python 3.combine_profiles.py
echo "Combining profiles complete."
python 4.normalize.py
echo "Normalization complete."
python 5.feature_select_sc.py
echo "Feature selection complete."
python 6.aggregation.py
echo "Aggregation complete."
python 7.whole_image_pipeline_processing.py
echo "Whole image pipeline processing complete."


cd ../ || exit

conda deactivate

echo "Processing complete."
