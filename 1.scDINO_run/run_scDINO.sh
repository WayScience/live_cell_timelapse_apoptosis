#!/bin/bash
# this script runs the pre-processing and scDINO pipeline

cd ../0.pre-process_images/

jupyter nbconvert --to=script --FilesWriter.build_directory=scripts/ notebooks/*.ipynb

cd scripts/

# python 0.pre-process_images.py

cd ../../1.scDINO_run/

# make logs directory if it doesn't exist
mkdir -p logs

snakemake -s only_downstream_snakefile --configfile="only_downstream_analyses.yaml" --until compute_CLS_features --cores all --rerun-incomplete > logs/CLS_token_run.log 2>&1

snakemake -s only_downstream_snakefile --configfile="only_downstream_analyses.yaml" --cores all --rerun-incomplete > logs/downstream_run.log 2>&1
