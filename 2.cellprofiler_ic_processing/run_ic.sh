#!/bin/bash

conda activate cellprofiler_timelapse_env
# convert the notebook to a python script
jupyter nbconvert --to=script --FilesWriter.build_directory=scripts/ notebooks/*.ipynb

# change the directory to the scripts folder
cd scripts/ || exit

echo "Starting IC processing"

for FOV_dir in ../../data/preprocessed_data/20231017ChromaLive_6hr_4ch_MaxIP/*; do
    # Run CellProfiler for IC processing
    python 0.perform_ic.py --input_dir $FOV_dir
done

for terminal_dirs in ../../data/preprocessed_data/20231017ChromaLive_endpoint_w_AnnexinV_2ch_MaxIP/*; do
    # Run CellProfiler for IC processing
    python 0.perform_ic.py --input_dir $terminal_dirs
done

python 1.process_ic_teminal_data.py

# return the directory that script was run from
cd ../ || exit

echo "IC processing complete"
