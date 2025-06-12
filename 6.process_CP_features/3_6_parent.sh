#!/bin/bash
#SBATCH --nodes=1
#SBATCH --partition=amilan
#SBATCH --qos=normal
#SBATCH --account=amc-general
#SBATCH --time=24:00:00
#SBATCH --output=cytomining_parent-%j.out


job1=$(sbatch 3.combine.sh |  awk '{print $4}')
job2=$(sbatch --dependency=afterok:$job3 4.normalize.sh |  awk '{print $4}')
job3=$(sbatch --dependency=afterok:$job2 5.feature_select_sc.sh |  awk '{print $4}')
job4=$(sbatch --dependency=afterok:$job3 6.aggregate.sh |  awk '{print $4}')

echo "All jobs submitted."
