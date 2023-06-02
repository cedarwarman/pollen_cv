#!/bin/bash
#SBATCH --job-name=inference_array
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=32gb
#SBATCH --time=2:30:00
#SBATCH --account=rpalaniv
#SBATCH --partition=standard
#SBATCH --gres=gpu:1
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err
#SBATCH --array=1-34

dir_path=$(ls /xdisk/rpalaniv/cjperkins1/chronic_hs/052623_chronic_HS | sed -n $SLURM_ARRAY_TASK_IDp)

echo This is Slurm task $SLURM_ARRAY_TASK_ID, dir_path $dir_path
apptainer exec \
	--nv \
	--bind /xdisk/rpalaniv/cedar/cv \
	/xdisk/rpalaniv/cedar/cv/containers/pollen_cv.sif \
	python3 /home/u16/cedar/git/pollen_cv/python/pollen_inference_hpc.py \
		--checkpoint /xdisk/rpalaniv/cedar/cv/models/pub_models/two_pollen/checkpoint/ckpt-0 \
		--config /xdisk/rpalaniv/cedar/cv/models/pub_models/two_pollen/pipeline.config \
		--map /xdisk/rpalaniv/cedar/cv/models/pub_models/two_pollen/labelmap.pbtxt \
		--images $dir_path \
		--camera two \
		--output /xdisk/rpalaniv/cedar/cv/inference/2023-06-01_Calvin

