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

#Set the number of runs that each SLURM task should do
PER_TASK=100

# Calculate the starting and ending values for this task based
# on the SLURM task and the number of runs per task.
START_NUM=$(( ($SLURM_ARRAY_TASK_ID - 1) * $PER_TASK + 1 ))
END_NUM=$(( $SLURM_ARRAY_TASK_ID * $PER_TASK ))

# Print the task and run range
echo This is task $SLURM_ARRAY_TASK_ID, which will do runs $START_NUM to $END_NUM

# Run the loop of runs for this task.
for (( run=$START_NUM; run<=$END_NUM; run++ )); do
	dir_path=$(cat /xdisk/rpalaniv/cedar/inference/dir_paths/inference_dirs_2022-12-26.txt | sed -n ${run}p)
	echo This is SLURM task $SLURM_ARRAY_TASK_ID, run number $run, dir_path $dir_path
	apptainer exec \
		--nv \
		--bind /xdisk/rpalaniv/cedar/image_processing \
		/xdisk/rpalaniv/cedar/containers/pollen_cv/pollen_cv.sif \
		python3 /home/u16/cedar/git/pollen_cv/python/pollen_inference_hpc.py \
			--checkpoint /xdisk/rpalaniv/cedar/inference/models/2022-12-13_centernet/checkpoint/ckpt-0 \
			--config /xdisk/rpalaniv/cedar/inference/models/2022-12-13_centernet/pipeline.config \
			--map /xdisk/rpalaniv/cedar/inference/models/2022-12-13_centernet/2022-12-13_t013031.pbtxt \
			--images $dir_path \
			--output /xdisk/rpalaniv/cedar/inference/inference_out/inference_2022-12-26
done

