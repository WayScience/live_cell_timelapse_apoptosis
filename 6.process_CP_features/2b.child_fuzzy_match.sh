#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --partition=amilan
#SBATCH --qos=normal
#SBATCH --account=amc-general
#SBATCH --time=1:00:00
#SBATCH --output=fuzzy_match_child-%j.out

module load miniforge
conda init bash
conda activate cellprofiler_timelapse_env


well_fov=$1

cd scripts/ || exit

python 2.fuzzy_matching.py --well_fov $well_fov

cd ../ || exit

conda deactivate

echo "Merging sc complete."
