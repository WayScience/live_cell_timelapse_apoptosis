#!/bin/bash

# This script is used to deploy the shiny app to shinyapps.io

# activate the conda environment
conda activate R_timelapse_env

# convert the the notebooks to scripts
jupyter nbconvert --to=script --FilesWriter.build_directory=scripts/ notebooks/*.ipynb

# move to the scripts directory
cd scripts/ || exit

# run the deployment script
echo "Deploying the shiny app"
Rscript 4.deploy_shiny_app.r

# move back to the main directory
cd .. || exit

# deactivate the conda environment
conda deactivate

# check exit status
if [ $? -eq 0 ]; then
    echo "Deployment successful"
else
    echo "Deployment failed"
fi
