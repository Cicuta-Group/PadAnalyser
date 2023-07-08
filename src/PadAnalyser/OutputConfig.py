
import dataclasses

@dataclasses.dataclass
class OutputConfig:
    output_dir: str
    work_dir: str
    debug_dir: str
    logging_file: str

    ignore_cache: bool
    process_count: int


# replace the module with the the class
import sys
sys.modules[__name__] = OutputConfig