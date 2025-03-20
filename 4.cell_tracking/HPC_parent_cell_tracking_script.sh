#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=2:00:00
#SBATCH --partition=amilan
#SBATCH --output=cell_tracking-%j.out

module load miniforge
conda init bash
conda activate cell_tracking_env


jupyter nbconvert --to=script --FilesWriter.build_directory=scripts/ notebooks/*.ipynb

cd scripts/ || exit

well_fov_path="../../2.cellprofiler_ic_processing/illum_directory/timelapse/"
well_fovs=$(ls $well_fov_path)
echo "${well_fovs[@]}"

cd ../ || exit

for well_fov in $well_fovs; do
    echo "Processing well_fov: $well_fov"
    # check python script exit code
    full_path="${well_fov_path}${well_fov}"
    sbatch HPC_child_cell_tracking_script.sh $full_path
done

conda deactivate

echo "Cell tracking script completed"
