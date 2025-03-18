#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=30:00
#SBATCH --partition=aa100
#SBATCH --gres=gpu:1
#SBATCH --output=cell_tracking-%j.out

module load miniforge
conda init bash
conda activate cell_tracking_env


jupyter nbconvert --to=script --FilesWriter.build_directory=scripts/ notebooks/*.ipynb

cd scripts/ || exit

input_dir=$1
python 0.nuclei_tracking.py --input_dir "$input_dir"

cd ../ || exit

conda deactivate

echo "Cell tracking script completed"
