#!/bash/bin

conda activate cellprofiler_timelapse_env

jupyter nbconvert --to python --output-dir=scripts/ notebooks/*.ipynb

cd scripts/ || exit

python 1.one_off_analysis_computational_requirments_for_fs.py --num_of_featuresq $1 --num_of_cells_per_well $2 --num_of_groups $3 --num_of_replicates $4

cd ../ || exit

conda deactivate

echo "Done"
