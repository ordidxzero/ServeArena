import torch
from uuid import uuid4, UUID
from pathlib import Path
from typing import Dict, Optional, Callable
from serve_arena.utils.command import CommandArgs
from io import TextIOWrapper

class ServeArenaLogger:
    dir: Path
    prefix: str
    device_prefix: str
    filename: Optional[str]
    mapping: Optional[Callable[[str, str, CommandArgs], str]]
    _file: Dict[UUID, TextIOWrapper]
    
    def __init__(self, dir: Path, prefix: Optional[str] = None, mapping: Optional[Callable[[str, str, CommandArgs], str]] = None):
        self._file = {}
        self.dir = dir
        self.prefix = '' if prefix is None else prefix
        self.filename = None
        self.device_prefix = self.get_device_prefix()
        self.mapping = mapping

    def format(self, args: CommandArgs):
        if self.mapping is not None:
            self.filename = self.mapping(self.device_prefix, self.prefix, args)
        else:
            self.filename = f'{self.device_prefix}_{self.prefix}_output.log'

        if not self.filename.endswith(".log"):
            self.filename += '.log'

    def open(self, server: bool = False) -> UUID:
        uid = uuid4()
        if server == True:
            self._file[uid] = open(self.get_server_log_path(absolute=True).as_posix(), "w")
        else:
            self._file[uid] = open(self.get_benchmark_log_path(absolute=True).as_posix(), "w")
        
        return uid
    
    def info(self, uid: UUID, line: str):
        if line != "":
            self._file[uid].write(line + '\n')

    def close(self, uid: UUID):
        assert self._file[uid] is not None, "File did not open."
        self._file[uid].close()

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