#!/bin/bash
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --job-name=snpp_amb3r_3dgs
#SBATCH --output=/network/scratch/a/adam.burhan/logs/snpp_3DGS_amb3r/%A_%a.out
#SBATCH --gres=gpu:l40s:1
#SBATCH --array=0-62%8

set -euo pipefail

modes=(
    baseline
    unimodal
    gmm
    unimodal_all
    gmm_all
    unimodal_local
    gmm_local
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
mode=${modes[$((SLURM_ARRAY_TASK_ID / n_seq))]}
seq=${sequences[$((SLURM_ARRAY_TASK_ID % n_seq))]}

module load cuda/12.1.1
source $SCRATCH/envs/gs3d/bin/activate
repo_root=$HOME/repos/gaussian-splatting

data_root=$SCRATCH/experiments/depth-aware-ba/3dgs_data_snpp/dslr
out_root=$SCRATCH/experiments/depth-aware-ba/3dgs_out_snpp/dslr
out=$out_root/${seq}_${mode}
mkdir -p $out_root

cd $repo_root
python train.py -s $data_root/${seq}_${mode} -m $out \
    --eval -r 1 --save_iterations 7000 30000 --test_iterations 7000 30000 --quiet --disable_viewer
# render.py --iteration takes ONE value (unlike train.py's --save_iterations),
# so loop; metrics.py then scores every test/ours_<iter> dir it finds.
for it in 7000 30000; do
    python render.py -m $out --iteration $it --skip_train
done
python metrics.py -m $out