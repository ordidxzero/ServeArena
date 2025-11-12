from pathlib import Path
from serve_arena.vllm import VLLMRunner
from serve_arena.utils.command import CommandTemplate, CommandArgs
from serve_arena.utils.logger import ServeArenaLogger


server_cmd = CommandTemplate("vllm serve meta-llama/Llama-3.1-8B-Instruct --dtype bfloat16 --max-num-batched-tokens 131072 --max-model-len 131072")
benchmark_cmd = CommandTemplate("vllm bench serve --backend vllm --model meta-llama/Llama-3.1-8B-Instruct --endpoint /v1/completions --ignore-eos --dataset-name random --random-input-len {input_len} --random-output-len {output_len} --num-prompts 4096 --disable-tqdm --percentile-metrics ttft,tpot,itl,e2el --save-result --save-detailed --result-dir './outputs/benchmark' --result-filename {benchmark_detailed_log}")

def get_log_filename(device_prefix: str, prefix: str, args: CommandArgs):
    input_len = args.get("input_len", None)
    output_len = args.get("output_len", None)
    return f"{device_prefix}_{prefix}_in_{input_len}_out_{output_len}.log"

logger = ServeArenaLogger(Path('./outputs'), "v3_exp1", mapping=get_log_filename)

runner = VLLMRunner(benchmark_cmd=benchmark_cmd, server_cmd=server_cmd, envs={"VLLM_USE_V1": "1", "CUDA_VISIBLE_DEVICES": "0", 'VLLM_LOG_STATS_INTERVAL': '0.001'}, logger=logger)

CONFIGURATIONS = [(128, 128), (128, 2048), (2048, 128), (2048, 2048)]

for config in CONFIGURATIONS:
    input_len, output_len = config

    runner.run(input_len=input_len, output_len=output_len)