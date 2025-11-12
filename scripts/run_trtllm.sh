#!/bin/bash

docker run --rm -it --name trtllm_container_yjheo --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --gpus=all nvcr.io/nvidia/tensorrt-llm/release:1.0.0