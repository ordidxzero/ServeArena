import os, socket, subprocess, torch, gc, shlex
from threading import Thread
from typing import Dict, Literal, Optional, Callable
from pathlib import Path
from serve_arena.utils.command import CommandTemplate
from serve_arena.utils.logger import ServeArenaLogger
from uuid import UUID

RUNNER_TYPE = Literal['vllm', 'trtllm', 'sglang', 'tgi', 'llamacpp']
ENV = Dict[str, str]

class BenchmarkRunner:
    runner_type: RUNNER_TYPE
    envs: ENV
    benchmark_envs: ENV
    logger: Optional[ServeArenaLogger]
    
    _cmd: Dict[str, CommandTemplate]
    _is_server_ready: bool
    _is_benchmark_only: bool
    _server_process: Optional[subprocess.Popen[str]]
    _server_logging_thread: Optional[Thread]
    _server_logger_uid: Optional[UUID]
    _terminate_server: Optional[Callable[..., None]]
    _benchmark_process: Optional[subprocess.Popen[str]]

    def __init__(
            self,
            runner_type: RUNNER_TYPE,
            benchmark_cmd: CommandTemplate,
            server_cmd: Optional[CommandTemplate] = None,
            envs: Optional[ENV] = None,
            logger: Optional[ServeArenaLogger] = None
        ):
        self.runner_type = runner_type
        self.logger = logger
        self._cmd = {}

        self._cmd['benchmark'] = benchmark_cmd
        self._is_server_ready = False
        self._is_benchmark_only = server_cmd is None
        self._server_process = None
        self._server_logging_thread = None
        self._server_logger_uid = None
        self._terminate_server = None
        self._benchmark_process = None
        
        if server_cmd is not None:
            self._cmd['server'] = server_cmd

        self.envs = os.environ.copy()
        self.envs["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        if envs is not None:
            assert isinstance(envs, dict), "envs must be dictionary"
            self.envs.update(envs)

    def init(self, **kwargs: str | int | float):
        if self.logger is not None:
            self.logger.format(kwargs)
            kwargs['benchmark_detailed_log'] = self.logger.filename.replace(".log", '.json') # type: ignore
        
        for _, cmd in self._cmd.items():
            cmd.format(**kwargs)

        self._is_ready = True

    def run_server(self, host:str  = "localhost", port: int = 8000):
        assert self._is_benchmark_only == False, "Server does not required."
        assert BenchmarkRunner.is_port_open(host, port) == False, f"Post {port} is already used."

        if self._server_process is not None:
            assert self._server_process.poll() == 0, "Server does not closed successfully."
            self._server_process = None
        
        print("=" * 30)
        print(f"Server Command: {self._cmd['server'].as_string()}")
        print("=" * 30)

        self._server_process = subprocess.Popen(
            args=shlex.split(self._cmd['server'].as_string()),
            env=self.envs,
            encoding="utf-8",
            text=True,
            stderr=subprocess.STDOUT,
            stdout=subprocess.PIPE
        )
        self._is_server_ready = True

        def log_in_thread():
            assert self.logger is not None, "LoggerError in Thread"
            assert self._server_process is not None, "ServerProcess in Thread"

            uid = self.logger.open(server=True)
            self._server_logger_uid = uid
            while self._server_process.poll() is None:
                out = self._server_process.stdout.readline() # type: ignore
                self.logger.info(uid, out.rstrip())

        if self.logger is not None:
            from threading import Thread
            self._server_logging_thread = Thread(target=log_in_thread, daemon=True)
            self._server_logging_thread.start()

        def terminate():
            self._server_process.terminate() # type: ignore
            self._server_process.wait(timeout=60) # type: ignore
            self._is_server_ready = False
            if self._server_logging_thread is not None:
                self._server_logging_thread.join()
            
            if self.logger is not None and self._server_logger_uid is not None:
                self.logger.close(self._server_logger_uid)
            print("Terminated Server...")
            torch.cuda.empty_cache()
            gc.collect()

        self._terminate_server = terminate

    def run_benchmark(self):
        if not self._is_benchmark_only and not self._is_server_ready:
            raise Exception("Server is not ready.")
        
        if self._benchmark_process is not None:
            assert self._benchmark_process.poll() == 0, "Benchmark code does not closed successfully."
            self._benchmark_process = None

        print("=" * 30)
        print(f"Benchmark Command: {self._cmd['benchmark'].as_string()}")
        print("=" * 30)

        self._benchmark_process = subprocess.Popen(
            args=shlex.split(self._cmd['benchmark'].as_string()),
            env=self.envs,
            encoding="utf-8",
            text=True,
            stderr=subprocess.STDOUT,
            stdout=subprocess.PIPE
        )

        if self.logger is not None:
            uid = self.logger.open()
            while self._benchmark_process.poll() is None:
                out = self._benchmark_process.stdout.readline() # type: ignore
                self.logger.info(uid, out.rstrip())

            self.logger.close(uid)

        def terminate():
            self._benchmark_process.terminate() # type: ignore
            self._benchmark_process.wait(timeout=60) # type: ignore
            print("Terminate Benchmarking...")
            torch.cuda.empty_cache()
            gc.collect()

        terminate()

    @staticmethod
    def clear_terminal():
        os.system("clear")
    
    @staticmethod
    def is_port_open(host:str  = "localhost", port: int = 8000) -> bool:
        """포트가 열려 있는지 확인하는 메서드"""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(1)
            result = sock.connect_ex((host, port))
            return result == 0
    
    @staticmethod
    def is_docker() -> bool:
        """Docker 환경인지 체크하는 메서드"""
        cgroup = Path('/proc/self/cgroup')
        return Path('/.dockerenv').is_file() or (cgroup.is_file() and 'docker' in cgroup.read_text())