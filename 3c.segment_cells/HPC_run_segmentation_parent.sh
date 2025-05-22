#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=01:00:00
#SBATCH --partition=amilan
#SBATCH --qos=normal
#SBATCH --output=segment_parent-%j.out

# This script will work on a local machine that has enough VRAM to actually run the segmentation and tracking.
# Mine does not so we shall run this on the cluster on a NVIDIA a100 40GB VRAM GPU.

module load anaconda
conda init bash

conda activate timelapse_segmentation_env

# run the segmentation and tracking
echo "Submitting GPU jobs to segment and track objects in the images..."

jupyter nbconvert --to script --output-dir=scripts/ notebooks/*.ipynb

cd scripts/ || exit

# get the list of dirs in path
mapfile -t main_dirs < <(ls -d ../../2.cellprofiler_ic_processing/illum_directory/timelapse/*)
mapfile -t terminal_dirs < <(ls -d ../../2.cellprofiler_ic_processing/illum_directory/endpoint/*)

cd ../ || exit


if [ ${#main_dirs[@]} -ne ${#terminal_dirs[@]} ]; then
    echo "Error: The number of main directories and terminal directories do not match."
    exit 1
fi

touch job_ids.txt
jobs_submitted_counter=0
# run the pipelines
for i in "${!main_dirs[@]}"; do
    main_dir="${main_dirs[$i]}"
    terminal_dir="${terminal_dirs[$i]}"
    echo "Processing main directory: $main_dir with terminal directory: $terminal_dir"
    number_of_jobs=$(squeue -u $USER | wc -l)
    while [ $number_of_jobs -gt 990 ]; do # max jobs are 1000, 990 is a safe number
        sleep 1s
        number_of_jobs=$(squeue -u $USER | wc -l)
    done
    sbatch HPC_run_segmentation_child.sh "$main_dir" "$terminal_dir"

done


conda deactivate

echo "Submitted all jobs. $jobs_submitted_counter jobs submitted."
