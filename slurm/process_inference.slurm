#!/bin/bash
#SBATCH --job-name=process_inference
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=2gb
#SBATCH --time=0:05:00
#SBATCH --account=rpalaniv
#SBATCH --partition=standard
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err
#SBATCH --array=1-249%50

# Sourcing my .bashrc so that conda works and the path is how it should be
source /home/u16/cedar/.bashrc

# Setting nullglob so that globs that match nothing don't make errors
shopt -s nullglob

conda activate process-inference

# Everything in this directory will be processed
cd /xdisk/rpalaniv/cedar/cv/inference/2023-06-23_full_set/

# Setting up the array
image_sequence=$(find . -maxdepth 1 -name '*run*' | sed -n ${SLURM_ARRAY_TASK_ID}p)

# Making sure it's not blank
if [ ${image_sequence} = "" ]
then
    echo "Image sequence is blank, exiting"
    exit 1
fi

printf "Processing run: ${image_sequence}\n"
save_path="${image_sequence}/processed_inference"
mkdir ${save_path}

for file in "${image_sequence}/predictions"/*.tsv
do
    python /home/u16/cedar/git/pollen_cv/python/process_inference.py \
		--inference ${file} \
		--output ${save_path}
done

# Cleaning up
conda deactivate
shopt -u nullglob

