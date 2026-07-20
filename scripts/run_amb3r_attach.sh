#!/bin/bash
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=4:00:00
#SBATCH --array=0-6
#SBATCH --job-name=tt_amb3r_attach
#SBATCH --output=/network/scratch/a/adam.burhan/logs/tt_attach_amb3r_%A_%a.out

set -euo pipefail

modes=(
    bimodal
    unimodal
)
sequences=(
    Barn
    Caterpillar
    Church
    Courthouse
    Ignatius
    Meetingroom
    Truck
)

for mode in "${modes[@]}"; do
    seq=${sequences[$SLURM_ARRAY_TASK_ID]}
    repo_root=$HOME/repos/depth-aware-BA/
    cd $repo_root

    db=$SCRATCH/experiments/depth-aware-ba/tt_amb3r/$seq/database.db
    config=${repo_root}/configs/depth/${mode}.yaml
    dump_dir=$SCRATCH/datasets/tanks_and_temples/amb3r/$seq/depth_bundles
    
    uv run depthba-attach \
        --config $config \
        --db $db \
        --dump_dir $dump_dir \
        --force
    
done