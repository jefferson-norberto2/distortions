xhost +local:docker

SCRIPT_DIR="$(pwd)"

# Run the Docker container with GUI and GPU support
docker run -it -d \
    --privileged \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    --name distortions_jmn \
    --gpus all \
    --network host \
    --mount type=bind,source=${SCRIPT_DIR},target=${SCRIPT_DIR} \
    --workdir=${SCRIPT_DIR} \
    distortions:latest bash

xhost -local:docker
