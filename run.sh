#!/usr/bin/env bash
xhost +local:docker

docker run -d -it \
    --name distortion-dev \
    --gpus all \
    --shm-size=16g \
    --net=host \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
    -v "$(pwd)":/workspace \
    -v /run/media:/run/media \
    -w /workspace \
    pytorch/pytorch:2.12.0-cuda13.2-cudnn9-devel \
    bash

