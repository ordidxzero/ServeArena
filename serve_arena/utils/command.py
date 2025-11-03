import os
from typing import Dict, List, Optional
from string import Formatter, Template


CommandArgs = Dict[str, str | int | float]

class CommandTemplate:
    _skeleton: Template
    _arg_names: List[str]
    _cmd: Optional[str]

    def __init__(self, template: str, partial_variables: CommandArgs = {}):
        self._skeleton = Template(Template(template).safe_substitute(**partial_variables))
        self._arg_names = self._extract_template_vars()
        self._cmd = None

    def as_string(self) -> str:
        assert self._cmd is not None, "cmd is empty."

        return self._cmd
    
    def get_skeleton(self) -> str:
        return self._skeleton.safe_substitute()
    
    def format(self, **kwargs: str | int | float):
        """
        필요한 변수들이 완전히 주어졌을 때, 템플릿을 기반으로 명령어 문자열을 생성합니다.

        Args:
            **kwrags (str | int | float): 템플릿에 삽입될 변수들
        """
        args: CommandArgs = {}

        for key, value in kwargs.items():
            if key in self._arg_names:
                args[key] = value
        self._cmd = self._skeleton.template.format(**args)

    def _extract_template_vars(self) -> List[str]:
       """
       템플릿 문자열로부터 `{변수}` 형식의 변수명을 추출합니다.

       Returns:
           List[str]: 추출된 변수명들의 리스트
       """
       return [fn for _, fn, _, _ in Formatter().parse(self._skeleton.template) if fn is not None]