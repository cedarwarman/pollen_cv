#!/usr/bin/env bash

cd /home/git/models/research/

# From the tensorflow/models/research/ directory
PIPELINE_CONFIG_PATH="/home/git/pollen_cv/config/2023-04-18_centernet_two_tube_tip.config"
MODEL_DIR="/media/volume/sdb/models/2023-04-18_pub_models/two_tube_tip"
python object_detection/model_main_tf2.py \
    --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
    --model_dir=${MODEL_DIR} \
    --alsologtostderr
#   --checkpoint_every_n=<int>
