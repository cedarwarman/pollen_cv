FROM tensorflow/tensorflow:latest-gpu

ARG DEBIAN_FRONTEND=noninteractive

# Install apt dependencies
RUN apt-get update && apt-get install -y \
    git \
    gpg-agent \
    python3-cairocffi \
    protobuf-compiler \
    python3-pil \
    python3-lxml \
    python3-tk \
    wget

# Add new user to avoid running as root and for the permissions to work
ARG USER_ID
ARG GROUP_ID

RUN addgroup --gid $GROUP_ID user
RUN useradd -ms /bin/bash -u $USER_ID -g $GROUP_ID tensorflow
USER tensorflow

# RUN useradd -ms /bin/bash tensorflow
# USER tensorflow
WORKDIR /home/tensorflow

# Copy this version of of the model garden into the image
# COPY --chown=tensorflow . /home/tensorflow/models
# New
COPY --chown=${USER_ID}:${GROUP_ID} . /home/tensorflow/models
RUN ls /home/tensorflow/models

# Compile protobuf configs
RUN (cd /home/tensorflow/models/research/ && protoc object_detection/protos/*.proto --python_out=.)
WORKDIR /home/tensorflow/models/research/

RUN cp object_detection/packages/tf2/setup.py ./
ENV PATH="/home/user/.local/bin:${PATH}"

RUN python -m pip install -U pip
RUN python -m pip install . imageio labelbox[data]

# Fixes a conflict between opencv version required by labelbox and opencv-headless version install by tensorflow
RUN python -m pip uninstall -y opencv-python-headless

ENV TF_CPP_MIN_LOG_LEVEL 3
