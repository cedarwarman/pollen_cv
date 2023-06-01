#!/usr/bin/env bash

cd /home/git/models/research/

# From the tensorflow/models/research/ directory
PIPELINE_CONFIG_PATH="/home/git/pollen_cv/config/2023-04-18_centernet_both_tube_tip.config"
MODEL_DIR="/media/volume/sdb/models/2023-04-18_pub_models/combined_tube_tip"
CHECKPOINT_DIR="/media/volume/sdb/models/2023-04-18_pub_models/combined_tube_tip"
python object_detection/model_main_tf2.py \
    --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
    --model_dir=${MODEL_DIR} \
    --checkpoint_dir=${CHECKPOINT_DIR} \
    --alsologtostderr
#   --checkpoint_every_n=<int>
