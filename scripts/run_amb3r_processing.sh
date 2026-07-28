#!/bin/bash
#SBATCH --time=2:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --array=0-0

# sequences=(
# 	Caterpillar
# 	Church
# 	Ignatius
# 	Truck
# )

sequences=(
    09c1414f1b
)

seq=${sequences[$SLURM_ARRAY_TASK_ID]}

repo_root=$HOME/repos/depth-aware-BA/
cd $repo_root
source .venv/bin/activate
python scripts/process_amb3r_outputs.py \
	--npz ~/scratch/datasets/scannetpp/data/amb3r/$seq/scene_${seq}_results.npz \
	--out ~/scratch/datasets/scannetpp/data/amb3r/$seq \
	--rgb_dir ~/scratch/datasets/scannetpp/data/$seq/iphone/undistorted/images/
