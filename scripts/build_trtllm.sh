#!/bin/bash

CURRENT_DIR=$(pwd)
TRTLLM_SOURCE_DIR_PATH="$CURRENT_DIR/_TensorRT-LLM"

# 폴더 존재 여부 확인
if [ ! -d "$TRTLLM_SOURCE_DIR_PATH" ]; then
    mkdir -p "$TRTLLM_SOURCE_DIR_PATH"
    cd "$TRTLLM_SOURCE_DIR_PATH"
    git clone -b v1.0.0 https://github.com/NVIDIA/TensorRT-LLM.git .
    git submodule update --init --recursive
else
    echo "폴더가 이미 존재합니다: $TRTLLM_SOURCE_DIR_PATH"
fi

docker build -t trtllm_image_yjheo -f "$CURRENT_DIR"/Dockerfile.trtllm .