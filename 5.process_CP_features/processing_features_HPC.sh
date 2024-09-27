#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=15G
#SBATCH --time=6:00:00
#SBATCH --partition=amilan
#SBATCH --qos=normal
#SBATCH --output=alpine_std_out_std_err-%j.out


# This script processes the timelapse data
module purge
module load anaconda
module load mambaforge

mamba activate cellprofiler_timelapse_env

# check if the data is present in the data folder
# if present then remove the directory
if [ -d "data" ]; then
    echo "Data folder exists. Removing the data folder..."
    rm -r data
    echo "Data folder removed."
fi


# convert all notebooks to python files into the scripts folder
echo "Converting notebooks to Python scripts..."
jupyter nbconvert --to python --output-dir=scripts/ notebooks/*.ipynb
echo "Conversion complete."

cd scripts/

python 0.merge_sc.py
python 1.annotate_sc.py
python 2.normalize_sc_across_time.py
python 2.normalize_sc_within_time.py
python 3.feature_select_sc.py

cd ../

mamba deactivate

echo "Processing complete."
