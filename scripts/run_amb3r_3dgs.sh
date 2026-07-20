#!/bin/bash
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --job-name=tt_amb3r_3dgs
#SBATCH --output=/network/scratch/a/adam.burhan/logs/tt_3DGS_amb3r_%A_%a.out
#SBATCH --gres=gpu:l40s:1
#SBATCH --array=0-14

set -euo pipefail

modes=(
    baseline
    unimodal
    gmm
)
sequences=(
    Barn
    Caterpillar
    # Church
    # Courthouse
    Ignatius
    Meetingroom
    Truck
)

n_seq=${#sequences[@]}
mode=${modes[$((SLURM_ARRAY_TASK_ID / n_seq))]}
seq=${sequences[$((SLURM_ARRAY_TASK_ID % n_seq))]}

module load cuda/12.1.1
source $SCRATCH/envs/gs3d/bin/activate
repo_root=$HOME/repos/gaussian-splatting

data_root=$SCRATCH/experiments/depth-aware-ba/3dgs_data
out_root=$SCRATCH/experiments/depth-aware-ba/3dgs_out
out=$out_root/${seq}_${mode}
mkdir -p $out_root

cd $repo_root
python train.py -s $data_root/${seq}_${mode} -m $out \
    --eval -r 1 --save_iterations 7000 30000 --test_iterations 7000 30000 --quiet --disable_viewer
python render.py -m $out --iteration 7000 --skip_train
python render.py -m $out --iteration 30000 --skip_train
python metrics.py -m $out