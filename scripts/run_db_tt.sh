#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --array=0-6
#SBATCH --job-name=tt_db
#SBATCH --output=/network/scratch/a/adam.burhan/logs/tt_db_%A_%a.out

set -euo pipefail

sequences=(
    "Barn"
    "Caterpillar"
    "Church"
    "Courthouse"
    "Ignatius"
    "Meetingroom"
    "Truck"
)

seq=${sequences[$SLURM_ARRAY_TASK_ID]}

repo_root=/home/mila/a/adam.burhan/repos/depth-aware-BA

data_root=$SCRATCH/datasets/tanks_and_temples
output_dir=$SCRATCH/experiments/depth-aware-ba/tt/$seq

mkdir -p $output_dir


echo "Running feature extraction and matching on $seq"
cd $repo_root
echo "commit: $(git rev-parse --short HEAD)"
uv run depthba-db \
    --config ${repo_root}/configs/db/tt_exhaustive.yaml \
    --data_root $data_root \
    --sequence $seq \
    --output_dir $output_dir 
echo "Done."
