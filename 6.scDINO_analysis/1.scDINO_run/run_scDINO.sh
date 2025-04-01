#!/bin/bash
# this script runs the pre-processing and scDINO pipeline

conda activate scDINO_env

echo "Starting the scDINO analysis..."

# make logs directory if it doesn't exist
mkdir -p logs

# get start time
start=$(date +%s)

snakemake -s only_downstream_snakefile --configfile="only_downstream_analyses.yaml" --until compute_CLS_features --cores all --rerun-incomplete > logs/CLS_token_run.log 2>&1

snakemake -s only_downstream_snakefile --configfile="only_downstream_analyses.yaml" --cores all --rerun-incomplete > logs/downstream_run.log 2>&1

echo "scDINO analysis complete"

# conda deactivate

# get end time
end=$(date +%s)

# calculate runtime
runtime=$((end-start))

echo "Runtime: $runtime seconds"
