#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=1:00:00
#SBATCH --partition=amilan
#SBATCH --output=cell_tracking-%j.out

module load miniforge
conda init bash
conda activate cell_tracking_env


jupyter nbconvert --to=script --FilesWriter.build_directory=scripts/ notebooks/*.ipynb

cd scripts/ || exit

well_fov_path="../../2.cellprofiler_ic_processing/illum_directory/timelapse"
well_fovs=$(ls $well_fov_path)
echo "${well_fovs[@]}"


for file in "$well_fov_path"/*; do
    filename=$(basename "$file")
    well_fov="${filename#*MaxIP_}"
    echo "Well FOV: $well_fov"

    sbatch HPC_child_cell_tracking_script.sh $well_fov

done

cd ../ || exit

conda deactivate

echo "Cell tracking script completed"
