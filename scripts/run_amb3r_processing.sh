#!/bin/bash
#SBATCH --time=2:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --array=0-3

sequences=(
	Caterpillar
	Church
	Ignatius
	Truck
)

seq=${sequences[$SLURM_ARRAY_TASK_ID]}

repo_root=$HOME/repos/depth-aware-BA/
cd $repo_root
python scripts/process_amb3r_outputs.py --npz ~/scratch/datasets/tanks_and_temples/amb3r/$seq/scene_${seq}_results.npz --out ~/scratch/datasets/tanks_and_temples/amb3r/$seq
