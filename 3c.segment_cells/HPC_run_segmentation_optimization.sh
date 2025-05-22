#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --time=00:60:00
#SBATCH --partition=aa100
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --output=segment_optimization-%j.out

module load anaconda
conda init bash

conda activate timelapse_segmentation_env

cd scripts/ || exit

main_dir="../../2.cellprofiler_ic_processing/illum_directory/test_data/timelapse/"
terminal_dir="../../2.cellprofiler_ic_processing/illum_directory/test_data/endpoint/"


python 0.nuclei_segmentation_optimization.py --input_dir "$main_dir" --clip_limit 0.6
python 0.nuclei_segmentation_optimization.py --input_dir "$terminal_dir" --clip_limit 0.6

python 1.cell_segmentation_optimization.py --input_dir "$main_dir" --clip_limit 0.6

cd ../ || exit

conda deactivate

echo "Segmentation and tracking done."
