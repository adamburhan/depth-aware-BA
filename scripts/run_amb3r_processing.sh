#!/bin/bash
#SBATCH --time=2:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --array=0-8
#SBATCH --job-name=process_amb3r
#SBATCH --output=/network/scratch/a/adam.burhan/logs/process_amb3_snpp/%A_%a.out

# sequences=(
# 	Caterpillar
# 	Church
# 	Ignatius
# 	Truck
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

seq=${sequences[$SLURM_ARRAY_TASK_ID]}

repo_root=$HOME/repos/depth-aware-BA/
cd $repo_root
source .venv/bin/activate
python scripts/process_amb3r_outputs.py \
	--npz ~/scratch/datasets/scannetpp/data/amb3r/$seq/dslr/scene_${seq}_results.npz \
	--out ~/scratch/datasets/scannetpp/data/amb3r/$seq/dslr/ \
	--rgb_dir ~/scratch/datasets/scannetpp/data/$seq/dslr/resized_undistorted_images
