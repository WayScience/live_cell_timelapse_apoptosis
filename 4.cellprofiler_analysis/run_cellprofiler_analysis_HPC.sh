#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --time=6:00:00
#SBATCH --partition=amilan
#SBATCH --qos=normal
#SBATCH --output=sample-%j.out

module purge
module load anaconda

# This script is used to run CellProfiler analysis on the timelapse images.
conda activate cellprofiler_timelapse_env
# convert the jupyter notebook to a python script
jupyter nbconvert --to=script --FilesWriter.build_directory=scripts/ notebooks/*.ipynb

# check if the CellProfiler-plugins directory exists
if [ -d "../../CellProfiler-plugins" ]; then
    echo "CellProfiler-plugins directory exists."
else
    echo "CellProfiler-plugins directory does not exist."
    git clone https://github.com/CellProfiler/CellProfiler-plugins.git

PLUGINS_DIR=../../../CellProfiler-plugins/active_plugins/
# change the directory to the scripts folder

cd live_cell_timelapse_apoptosis/4.cellprofiler_analysis/scripts/


# run the python script
python run_cellprofiler_pipe_with_sam_outputs.py -p $PLUGINS_DIR

# change the directory back to the orginal directory
cd ../

conda deactivate

# End of the script
echo "CellProfiler analysis is completed."
