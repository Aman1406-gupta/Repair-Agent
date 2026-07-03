from dataclasses import dataclass

@dataclass
class Patch:
    methodName: str
    start_line: int
    end_line: int
    replacement: str
    original: str