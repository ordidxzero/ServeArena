from dataclasses import dataclass
from serve_arena.utils.command import CommandTemplate
from typing import Optional

@dataclass
class NSYSWrapper:
    trace: str = 'cuda,nvtx'
    output: str = ''
    cuda_graph_trace: str = 'node'
    delay: int = 50
    duration: Optional[int] = None
    gpu_metrics_devices: str = 'all'

    def plug(self, cmd: CommandTemplate) -> CommandTemplate:
        cmd_skeleton = cmd.get_skeleton()
        nsys_cmd = self.get_nsys_cmd()

        return CommandTemplate(nsys_cmd + cmd_skeleton)
    
    def get_nsys_cmd(self) -> str:
        nsys_cmd = f'nsys profile -t {self.trace} -o {self.output} --cuda-graph-trace={self.cuda_graph_trace} --cuda-memory-usage=true --gpu-metrics-devices={self.gpu_metrics_devices} --force-overwrite=true --sample=none --cpuctxsw=none --delay {self.delay} '
        
        if self.duration is not None:
            nsys_cmd += f'--duration {self.duration} '
        
        return nsys_cmd