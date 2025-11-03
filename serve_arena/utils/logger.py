import torch
from pathlib import Path
from typing import Dict, Optional, Callable
from serve_arena.utils.command import CommandArgs
from io import TextIOWrapper

class ServeArenaLogger:
    dir: Path
    prefix: str
    device_prefix: str
    filename: Optional[str]
    mapping: Callable[[str, str, CommandArgs], str]
    _file: TextIOWrapper
    
    def __init__(self, dir: Path, prefix: str, mapping: Callable[[str, str, CommandArgs], str]):
        self.dir = dir
        self.prefix = prefix
        self.filename = None
        self.device_prefix = self.get_device_prefix()
        self.mapping = mapping

    def format(self, args: CommandArgs):
        self.filename = self.mapping(self.device_prefix, self.prefix, args)

        if not self.filename.endswith(".log"):
            self.filename += '.log'

    def open(self, server: bool = False):
        if server == True:
            self._file = open(self.get_server_log_path(absolute=True).as_posix(), "w")
        else:
            self._file = open(self.get_benchmark_log_path(absolute=True).as_posix(), "w")
    
    def info(self, line: str):
        self._file.write(line + '\n')

    def close(self):
        assert self._file is not None, "File did not open."
        self._file.close()
    

    def get_server_log_path(self, absolute: bool = False) -> Path:
        assert self.filename is not None, "Filename is None."
        p = self.dir / 'server'

        if not p.exists():
            p.mkdir(parents=True)

        p = p / self.filename
        
        if absolute:
            return p.absolute()
        
        return p

    def get_benchmark_log_path(self, absolute: bool = False) -> Path:
        assert self.filename is not None, "Filename is None."
        p = self.dir / 'benchmark'

        if not p.exists():
            p.mkdir(parents=True)

        p = p / self.filename
        
        if absolute:
            return p.absolute()
        
        return p

    def get_device_prefix(self):
        device_name = torch.cuda.get_device_name(0)
        device_name = device_name.replace("NVIDIA ", "")
        device_name = device_name.replace(" ", "_")
        return device_name