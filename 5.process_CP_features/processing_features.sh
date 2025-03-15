#!/bin/bash
# This script processes the timelapse data

# convert all notebooks to python files into the scripts folder
echo "Converting notebooks to Python scripts..."
jupyter nbconvert --to python --output-dir=scripts/ notebooks/*.ipynb
echo "Conversion complete."

conda activate cellprofiler_timelapse_env

# check if the data is present in the data folder
# if present then remove the directory
if [ -d "data" ]; then
    echo "Data folder exists. Removing the data folder..."
    rm -r data
    echo "Data folder removed."
fi

well_fov="C-02_F0001"

cd scripts/ || exit

python 0.merge_sc.py --well_fov $well_fov
python 1.annotate_sc.py --well_fov $well_fov
python 2.fuzzy_matching.py --well_fov $well_fov
python 3.combine_profiles.py
python 4.normalize.py
python 5.feature_select_sc.py

cd ../ || exit

conda deactivate

echo "Processing complete."
