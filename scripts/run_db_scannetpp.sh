#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=6:00:00
#SBATCH --array=0-8
#SBATCH --job-name=scannetpp_db_amb3r
#SBATCH --output=/network/scratch/a/adam.burhan/logs/scannetpp_db_amb3r/%A_%a.out

set -euo pipefail

sequences=(
    09c1414f1b
    0d2ee665be
    13c3e046d7
    1ada7a0617
    21d970d8de
    25f3b7a318
    27dd4da69e
    286b55a2bf
    31a2c91c43
)

seq=${sequences[$SLURM_ARRAY_TASK_ID]}

repo_root=/home/mila/a/adam.burhan/repos/depth-aware-BA

data_root=$SCRATCH/datasets/scannetpp/data/amb3r
output_dir=$SCRATCH/experiments/depth-aware-ba/scannetpp_amb3r/$seq/dslr

mkdir -p $output_dir


echo "Running feature extraction and matching on $seq"
cd $repo_root
echo "commit: $(git rev-parse --short HEAD)"
uv run depthba-db \
    --config ${repo_root}/configs/db/scannetpp_amb3r.yaml \
    --data_root $data_root \
    --sequence $seq \
    --output_dir $output_dir 
echo "Done."
