#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=amilan
#SBATCH --qos=normal
#SBATCH --account=amc-general
#SBATCH --time=6:00:00
#SBATCH --output=ic_parent-%j.out

module purge
module load anaconda

# convert the notebook to a python script
jupyter nbconvert --to=script --FilesWriter.build_directory=scripts/ notebooks/*.ipynb

cd scripts/ || exit

# get a list of all dirs in the raw data folder
data_dir="../../data/test_data/20231017ChromaLive_6hr_4ch_MaxIP"
terminal_data="../../data/test_data/20231017ChromaLive_endpoint_w_AnnexinV_2ch_MaxIP"
# data_dir="../../../data/test_dir"
mapfile -t FOV_dirs < <(ls -d $data_dir/*)
mapfile -t terminal_dirs < <(ls -d $terminal_data/*)
echo length of plate_dirs: ${#FOV_dirs[@]}

cd ../ || exit


echo "Starting IC processing"

for FOV_dir in "${FOV_dirs[@]}"; do
	# get the number of jobs for the user
    number_of_jobs=$(squeue -u $USER | wc -l)
    while [ $number_of_jobs -gt 990 ]; do
        sleep 1s
        number_of_jobs=$(squeue -u $USER | wc -l)
    done
    sbatch run_ic_HPC_child.sh "$FOV_dir"
done

for terminal_dirs in "${terminal_dirs[@]}"; do
    # get the number of jobs for the user
    number_of_jobs=$(squeue -u $USER | wc -l)
    while [ $number_of_jobs -gt 990 ]; do
        sleep 1s
        number_of_jobs=$(squeue -u $USER | wc -l)
    done
    sbatch run_ic_HPC_child.sh "$terminal_dirs"
done


while [ $number_of_jobs -gt 2 ]; do
        sleep 1s
        number_of_jobs=$(squeue -u $USER | wc -l)
done

conda activate cellprofiler_timelapse_env
cd scripts/ || exit
conda run -n cellprofiler_timelapse_env python 1.process_ic_teminal_data.py
cd ../ || exit
conda deactivate

echo "IC processing jobs submitted"
