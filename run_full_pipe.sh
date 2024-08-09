#!/bin/bash
# this is a gold standard reproducible pipeline for the analysis of Live-Cell timelapse imaging data


############################################
# Download and pre-process the data
############################################
echo "Starting the data download and pre-processing..."

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

echo "Starting the CellProfiler analysis..."

# run cellprofiler analysis
cd 3.cellprofiler_analysis
source run_cellprofiler_analysis.sh
cd ../

# process the features
cd 4.process_CP_features
source processing_features.sh
cd ../

echo "CellProfiler analysis complete"

############################################
# scDINO analysis
############################################

echo "Starting the scDINO analysis pipeline..."

# preprocess the data for scDINO
# run the scDINO analysis
# post-process the scDINO results
# deploy the scDINO results on a shiny app
cd 5.scDINO_analysis/
source run_scDINO_analysis.sh

echo "scDINO analysis complete"

############################################
# Harmonizing the CP and scDINO results
############################################

echo "Starting the harmonization of CP and scDINO results..."

# harmonize the CP and scDINO results
cd 6.data_harmonization/
source run_data_harmonization.sh
cd ../

echo "Harmonization of CP and scDINO results complete"

############################################

echo "Full pipeline complete"
