#!/bin/bash

CURRENT_DIR=$(pwd)

docker build -t vllm_image_yjheo -f "$CURRENT_DIR"/Dockerfile.vllm .