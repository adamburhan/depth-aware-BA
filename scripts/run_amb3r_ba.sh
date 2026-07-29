#!/bin/bash
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --array=0-62
#SBATCH --job-name=scannetpp_amb3r_ba
#SBATCH --output=/network/scratch/a/adam.burhan/logs/scannetpp_BA_amb3r/%A_%a.out

set -euo pipefail

configs=(
    none
    amb3r_gmm_global.yaml
    amb3r_unimodal_global.yaml
    amb3r_gmm_all.yaml
    amb3r_unimodal_all.yaml
    amb3r_gmm_local.yaml
    amb3r_unimodal_local.yaml
)
# sequences=(
#     Barn
#     Caterpillar
#     # Church
#     # Courthouse
#     Ignatius
#     Meetingroom
#     Truck
# )

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

n_seq=${#sequences[@]}
mode=${configs[$((SLURM_ARRAY_TASK_ID / n_seq))]}
seq=${sequences[$((SLURM_ARRAY_TASK_ID % n_seq))]}

repo_root=$HOME/repos/depth-aware-BA/
cd $repo_root

data=$SCRATCH/datasets/scannetpp/data/amb3r/$seq/dslr
db=$SCRATCH/experiments/depth-aware-ba/scannetpp_amb3r/$seq/dslr/database.db

case "$mode" in
    none)
        label=baseline; depth_arg="" ;;
    amb3r_gmm_global.yaml)
        label=gmm; depth_arg="--depthba_config ${repo_root}/configs/depthba/${mode}" ;;
    amb3r_unimodal_global.yaml)
        label=unimodal; depth_arg="--depthba_config ${repo_root}/configs/depthba/${mode}" ;;
    amb3r_gmm_all.yaml)
        label=gmm_all; depth_arg="--depthba_config ${repo_root}/configs/depthba/${mode}" ;;
    amb3r_unimodal_all.yaml)
        label=unimodal_all; depth_arg="--depthba_config ${repo_root}/configs/depthba/${mode}" ;;
    amb3r_gmm_local.yaml)
        label=gmm_local; depth_arg="--depthba_config ${repo_root}/configs/depthba/${mode}" ;;
    amb3r_unimodal_local.yaml)
        label=unimodal_local; depth_arg="--depthba_config ${repo_root}/configs/depthba/${mode}" ;;
esac

out=$SCRATCH/experiments/depth-aware-ba/scannetpp_amb3r/$seq/dslr/sfm_$label

echo "=== $seq / $label ==="
echo "commit: $(git rev-parse --short HEAD)"

uv run python -m depthba.backends.custom_incremental_pipeline \
    --database_path $db \
    --image_path $data/images \
    --output_path $out \
    $depth_arg