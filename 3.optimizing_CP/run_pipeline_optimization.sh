#!/bin/bash

# this script runs the pipeline optimization for the CP pipeline

mamba activate cellprofiler_timelapse_env

# run the pipeline optimization
cd scripts

# if logs directory does not exist, create it
if [ ! -d "../logs" ]; then
  mkdir ../logs
fi

python 0.tune_object_tracking_for_cell_type.py -t "LAP" -n 100 > ../logs/0.tune_object_tracking_for_cell_type.log
python 0.tune_object_tracking_for_cell_type.py -t "overlap" -n 100  > ../logs/0.tune_object_tracking_for_cell_type.log

python 1.assess_best_pipeline.py -t "LAP" > ../logs/1.assess_best_pipeline.log
python 1.assess_best_pipeline.py -t "overlap" > ../logs/1.assess_best_pipeline.log

cd ../

mamba deactivate

# end of script
echo "end of script"
