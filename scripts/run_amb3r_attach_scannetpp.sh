#!/bin/bash
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=4:00:00
#SBATCH --array=0-18
#SBATCH --job-name=scannetpp_amb3r_attach
#SBATCH --output=/network/scratch/a/adam.burhan/logs/scannetpp_attach_amb3r/%A_%a.out

set -euo pipefail

modes=(
    bimodal
    unimodal
)
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

for mode in "${modes[@]}"; do
    seq=${sequences[$SLURM_ARRAY_TASK_ID]}
    repo_root=$HOME/repos/depth-aware-BA/
    cd $repo_root

    db=$SCRATCH/experiments/depth-aware-ba/scannetpp_amb3r/$seq/database.db
    config=${repo_root}/configs/depth/${mode}.yaml
    dump_dir=$SCRATCH/datasets/scannetpp/data/amb3r/$seq/depth_bundles
    
    uv run depthba-attach \
        --config $config \
        --db $db \
        --dump_dir $dump_dir \
        --force
    
done