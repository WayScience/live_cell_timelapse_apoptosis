#!/bin/bash
# this is a gold standard reproducible pipeline for the analysis of Live-Cell timelapse imaging data


############################################
# Download and pre-process the data
############################################
# echo "Starting the data download and pre-processing..."

# Download the data
# TODO: Add the link to the data once it is available

# pre-process the data
cd 1.pre_process_data/
source run_preprocessing.sh
cd ../

# run illumination correction
cd 2.cellprofiler_ic_processing
source run_ic.sh
cd ../

echo "Data download and pre-processing complete"

############################################
# CellProfiler analysis
############################################

# echo "Starting the CellProfiler analysis..."

# run SAM2
cd 3b.run_sam
mamba activate sam2_env
source run_sam2.sh
cd ../

mamba deactivate

# extract the features
mamba activate cellprofiler_timelapse_env
cd 4.cellprofiler_analysis
jupyter nbconvert --to python --output-dir=scripts/ notebooks/*.ipynb
cd scripts/
python run_cellprofiler_pipe_with_sam_outputs.py
cd ../

# process the features
cd 5.process_CP_features/
source processing_features.sh
cd ../

mamba deactivate

echo "CellProfiler analysis complete"
