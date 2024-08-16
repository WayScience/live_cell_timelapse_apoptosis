#!/bin/bash
# This script calls other bash scripts to run scDINO analysis
# scDINO is a deep learning-based method for extracting representative features from single-cell microscopy images

conda activate scDINO_env

# Preprocess the data
# change the dir
cd 0.pre-process_images

# # # call the script
# # echo "Preprocessing the data..."
# source run_image_preprocess.sh

# change the dir
cd ../1.scDINO_run

# # run scDINO
# echo "Running scDINO..."
source run_scDINO.sh

# change the dir
cd ../2.scDINO_processing

# process the scDINO results
echo "Processing scDINO results..."
source process_scDINO.sh
# deploy shiny app
echo "Deploying shiny app..."
source deploy_shiny_app.sh

# change the dir
cd ../../

echo "scDINO analysis is complete!"
