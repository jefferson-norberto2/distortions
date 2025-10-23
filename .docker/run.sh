xhost +local:docker

docker run -it -d \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    --name ubuntu \
    --gpus all \
    --network host \
    --mount type=bind,source=/home/jmn,target=/home/jmn/host \
    ubuntu:desktop

xhost -local:docker
