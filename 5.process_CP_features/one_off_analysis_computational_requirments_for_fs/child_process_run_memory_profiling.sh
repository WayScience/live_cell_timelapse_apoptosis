#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --partition=amilan
#SBATCH --qos=normal
#SBATCH --account=amc-general
#SBATCH --time=1:00:00
#SBATCH --output=one_off_child-%j.out

module load miniforge
conda init bash
conda activate cellprofiler_timelapse_env

jupyter nbconvert --to python --output-dir=scripts/ notebooks/*.ipynb

cd scripts/ || exit

python 1.one_off_analysis_computational_requirments_for_fs.py --num_of_features $1 --num_of_cells_per_well $2 --num_of_groups $3 --num_of_replicates $4

cd ../ || exit

conda deactivate

echo "Done"
