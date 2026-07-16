#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=6:00:00
#SBATCH --array=0-6
#SBATCH --job-name=eth3d_db
#SBATCH --output=/network/scratch/a/adam.burhan/logs/eth3d_db_%A_%a.out

set -euo pipefail

sequences=(
    # "courtyard"
    "kicker"
    "meadow"
    "office"
    "pipes"
    # "playground"
    "relief"
    "relief_2"
    # "terrace"
    # "terrains"
    # "electro"
    "delivery_area"
    # "facade"
)

seq=${sequences[$SLURM_ARRAY_TASK_ID]}

repo_root=/home/mila/a/adam.burhan/repos/depth-aware-BA

data_root=$SCRATCH/datasets/eth3d
output_dir=$SCRATCH/experiments/depth-aware-ba/eth3d/$seq

mkdir -p $output_dir


echo "Running feature extraction and matching on $seq"
cd $repo_root
echo "commit: $(git rev-parse --short HEAD)"
uv run depthba-db \
    --config ${repo_root}/configs/db/eth3d_${seq}.yaml \
    --data_root $data_root \
    --output_dir $output_dir 
echo "Done."
