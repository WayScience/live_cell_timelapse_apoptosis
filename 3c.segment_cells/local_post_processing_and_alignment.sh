#!/bin/bash

conda activate cellprofiler_timelapse_env

jupyter nbconvert --to script --output-dir=scripts/ notebooks/*.ipynb


cd scripts/ || exit

# get the list of dirs in path
mapfile -t main_dirs < <(ls -d ../../2.cellprofiler_ic_processing/illum_directory/timelapse/*)
mapfile -t terminal_dirs < <(ls -d ../../2.cellprofiler_ic_processing/illum_directory/endpoint/*)

# echo the number of directories
echo "Number of main directories: ${#main_dirs[@]}"
echo "Number of terminal directories: ${#terminal_dirs[@]}"

if [ ${#main_dirs[@]} -ne ${#terminal_dirs[@]} ]; then
    echo "Error: The number of main directories and terminal directories do not match."
    exit 1
fi


for i in "${!main_dirs[@]}"; do
    main_dir="${main_dirs[$i]}"
    terminal_dir="${terminal_dirs[$i]}"
    echo "Processing main directory: $main_dir with terminal directory: $terminal_dir"
    python 4.copy_cell_mask_over.py --final_timepoint_dir "$main_dir" --terminal_timepoint_dir "$terminal_dir"
    python 5.endpoint_manual_alignment.py --final_timepoint_dir "$main_dir" --terminal_timepoint_dir "$terminal_dir"
done

cd ../ || exit

conda deactivate

echo "Post-processing and alignment is completed."

