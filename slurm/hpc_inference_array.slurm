#!/bin/bash
#SBATCH --job-name=inference_array
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=32gb
#SBATCH --time=4:30:00
#SBATCH --account=rpalaniv
#SBATCH --partition=standard
#SBATCH --gres=gpu:1
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err
#SBATCH --array=1-1

#Set the number of runs that each SLURM task should do
PER_TASK=20

# Calculate the starting and ending values for this task based
# on the SLURM task and the number of runs per task.
START_NUM=$(( ($SLURM_ARRAY_TASK_ID - 1) * $PER_TASK + 1 ))
END_NUM=$(( $SLURM_ARRAY_TASK_ID * $PER_TASK ))

# Print the task and run range
echo This is task $SLURM_ARRAY_TASK_ID, which will do runs $START_NUM to $END_NUM

# Run the loop of runs for this task.
for (( run=$START_NUM; run<=$END_NUM; run++ )); do
	# dir_path=$(cat /xdisk/rpalaniv/cedar/cv/images/directory_paths/inference_dirs_2023-06-23.txt | sed -n ${run}p)
	dir_path=$(cat /xdisk/rpalaniv/cedar/cv/inference/2023-06-23_full_set/2023-10-23_inference_paths.txt | sed -n ${run}p)
	echo This is SLURM task $SLURM_ARRAY_TASK_ID, run number $run, dir_path $dir_path
	apptainer exec \
		--nv \
		--bind /xdisk/rpalaniv/cedar/cv \
		/xdisk/rpalaniv/cedar/cv/containers/pollen_cv.sif \
		python3 /home/u16/cedar/git/pollen_cv/python/pollen_inference_hpc.py \
			--checkpoint /xdisk/rpalaniv/cedar/cv/models/pub_models/combined_all/checkpoint/ckpt-0 \
			--config /xdisk/rpalaniv/cedar/cv/models/pub_models/combined_all/pipeline.config \
			--map /xdisk/rpalaniv/cedar/cv/models/pub_models/combined_all/labelmap.pbtxt \
			--images $dir_path \
			--output /xdisk/rpalaniv/cedar/cv/inference/2023-06-23_full_set_incomplete_wells \
			--save_images
done

