from pathlib import Path
from serve_arena.vllm import VLLMRunner
from serve_arena.utils.command import CommandTemplate, CommandArgs
from serve_arena.utils.nsys import NSYSWrapper
from serve_arena.utils.logger import ServeArenaLogger

# 49152
server_cmd = CommandTemplate("vllm serve meta-llama/Llama-3.2-1B-Instruct --dtype float32 --max-model-len 8192 --max-num-seqs 64 --max-num-batched-tokens 8192 --gpu-memory-utilization 0.1946 --no-enable-chunked-prefill")
benchmark_cmd = CommandTemplate("vllm bench serve --backend vllm --model meta-llama/Llama-3.2-1B-Instruct --endpoint /v1/completions --ignore-eos --dataset-name random --random-input-len {input_len} --random-output-len {output_len} --num-prompts {num_prompts} --disable-tqdm --percentile-metrics ttft,tpot,itl,e2el --save-result --save-detailed --result-dir './outputs/benchmark' --result-filename {benchmark_detailed_log}")

def get_log_filename(device_prefix: str, prefix: str, args: CommandArgs):
    input_len = args.get("input_len", None)
    output_len = args.get("output_len", None)
    return f"{device_prefix}_{prefix}_in_{input_len}_out_{output_len}.log"

logger = ServeArenaLogger(Path('./outputs'), "v3_nsys", mapping=get_log_filename)


CONFIGURATIONS = [(128, 128, 1024), (2048, 128, 1024), (128, 2048, 1024), (2048, 2048, 512)]
NSYS_CONFIGURATIONS = [(30, 90), (30, 420), (30, 120), (30, 360)]
i = 1
input_len, output_len, num_prompts = CONFIGURATIONS[i]
delay, duration = NSYS_CONFIGURATIONS[i]
nsys_wrapper = NSYSWrapper(delay=delay, duration=duration, output=f"in_{input_len}_out_{output_len}.nsys-rep")

server_cmd = nsys_wrapper.plug(server_cmd)

runner = VLLMRunner(benchmark_cmd=benchmark_cmd, server_cmd=server_cmd, envs={"VLLM_USE_V1": "1", "CUDA_VISIBLE_DEVICES": "0", 'VLLM_LOG_STATS_INTERVAL': '10'}, logger=logger)

runner.run(input_len=input_len, output_len=output_len, num_prompts=num_prompts)
# for config in CONFIGURATIONS:
#     input_len, output_len = config

#     runner.run(input_len=input_len, output_len=output_len)
#     VLLMRunner.clear_terminal()