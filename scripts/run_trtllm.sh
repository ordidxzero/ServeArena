#!/bin/bash

CURRENT_DIR=$(pwd)

docker run --rm -it --name trtllm_container_yjheo --ipc=host --gpus all --ulimit memlock=-1 --ulimit stack=67108864 --volume "$CURRENT_DIR":/app trtllm_image_yjheo /bin/bash