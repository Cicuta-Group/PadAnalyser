
import dataclasses
from typing import Optional

@dataclasses.dataclass
class OutputConfig:
    output_dir: str
    work_dir: str
    debug_dir: str
    mask_dir: str

    logging_file: str

    cache_segmentation: bool
    cache_dataframe: bool

    process_count: Optional[int] # number of processes to run in paralell. 1 for single threaded. None for as many as possible. 