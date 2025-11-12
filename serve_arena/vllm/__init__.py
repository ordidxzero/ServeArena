import time
from serve_arena.utils.command import CommandTemplate
from serve_arena.utils.logger import ServeArenaLogger
from serve_arena.utils.runner import ENV, BenchmarkRunner
from serve_arena.utils.pkg import find_package, find_package_version, is_version_at_least
from typing import Optional

class VLLMRunner(BenchmarkRunner):
    
    def __init__(self, benchmark_cmd: CommandTemplate, server_cmd: CommandTemplate, envs: Optional[ENV] = None, logger: Optional[ServeArenaLogger] = None):
        if not find_package("vllm"):
            raise Exception("vLLM is not installed")
        
        if not is_version_at_least("vllm", "0.10.2"):
            raise Exception("vLLM >= 0.10.2 is required")
        
        super().__init__('vllm', benchmark_cmd, server_cmd, envs, logger)

        print("=" * 30)
        print("Current vLLM Version: ", find_package_version("vllm"))
        print("=" * 30)

    
    def run(self, **kwargs: str | int | float):
        self.init(**kwargs)
        self.run_server('localhost', 8000)

        assert self._terminate_server is not None, "Server is not initialized"

        while True:
            if BenchmarkRunner.is_port_open('localhost', 8000):
                self.run_benchmark()
                break

            time.sleep(1)

        ENABLE_NSYS_PROFILE = "nsys" in self._cmd['server'].as_string()
        
        if ENABLE_NSYS_PROFILE:
            time.sleep(180)

        self._terminate_server()