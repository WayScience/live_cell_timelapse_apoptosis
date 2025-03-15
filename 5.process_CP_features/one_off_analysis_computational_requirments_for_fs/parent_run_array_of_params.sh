#SBATCH --nodes=1
#SBATCH --partition=amilan
#SBATCH --qos=normal
#SBATCH --account=amc-general
#SBATCH --time=10:00:00
#SBATCH --output=ic_parent-%j.out


conda activate cellprofiler_timelapse_env

jupyter nbconvert --to python --output-dir=scripts/ notebooks/*.ipynb



# define array of parameters
num_features_path="./arrays/num_of_features.txt"
num_cells_path="./arrays/num_of_cells_per_well.txt"
num_groups_path="./arrays/num_of_groups.txt"
num_of_replicates_path="./arrays/num_of_replicates.txt"

mapfile -t num_features < "$num_features_path"
mapfile -t num_cells < "$num_cells_path"
mapfile -t num_groups < "$num_groups_path"
mapfile -t num_of_replicates < "$num_of_replicates_path"

cd scripts/ || exit

for feature_num in "${num_features[@]}"; do
    for num_cells in "${num_cells[@]}"; do
        for num_groups in "${num_groups[@]}"; do
            for num_of_replicates in "${num_of_replicates[@]}"; do
                echo "Running for feature_num: $feature_num, num_cells: $num_cells, num_groups: $num_groups, num_of_replicates: $num_of_replicates"
                number_of_jobs=$(squeue -u $USER | wc -l)
                while [ $number_of_jobs -gt 990 ]; do
                    sleep 1s
                    number_of_jobs=$(squeue -u $USER | wc -l)
                done
                sbatch --nodes=1 --partition=amilan --qos=normal --account=amc-general --time=1:00:00 --output=ic_child-%j.out child_process_run_memory_profiling.sh "$feature_num" "$num_cells" "$num_groups" "$num_of_replicates"
            done
        done
    done
done

cd ../ || exit

conda deactivate

echo "Done"
