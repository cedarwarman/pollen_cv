#!/usr/bin/env bash

cd /home/git/models/research/

# From the tensorflow/models/research/ directory
PIPELINE_CONFIG_PATH="/home/git/pollen_cv/config/centernet_hourglass104_pollen_classes_2023-03-23.config"
MODEL_DIR="/media/volume/sdb/models/2023-03-23_centernet_pollen"
CHECKPOINT_DIR="/media/volume/sdb/models/2023-03-23_centernet_pollen"
python object_detection/model_main_tf2.py \
    --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
    --model_dir=${MODEL_DIR} \
    --checkpoint_dir=${CHECKPOINT_DIR} \
    --alsologtostderr
#   --checkpoint_every_n=<int>
