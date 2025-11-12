#!/bin/bash

CURRENT_DIR=$(pwd)

docker run --rm -it --name vllm_container_yjheo --ipc=host --gpus all --volume "$CURRENT_DIR":/app --volume /raid/yjheo/.cache/huggingface:/huggingface --volume "$CURRENT_DIR"/auxiliary/vllm:/auxiliary --volume "$CURRENT_DIR"/auxiliary/vllm/llama.py:/.venv/lib/python3.10/site-packages/vllm/model_executor/models/llama.py --volume "$CURRENT_DIR"/auxiliary/vllm/loggers.py:/.venv/lib/python3.10/site-packages/vllm/v1/metrics/loggers.py vllm_image_yjheo /bin/bash