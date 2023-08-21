
import dataclasses
from typing import Optional
import os

@dataclasses.dataclass
class OutputConfig:
    output_dir: str
    work_dir: str
    debug_dir: str
    mask_dir: str

    logging_file: str

    cache_segmentation: bool
    cache_dataframe: bool

    clear_dirs: bool # clear output and debug directories on run. If false, old files will be overwritten.
    process_count: Optional[int] # number of processes to run in paralell. 1 for single threaded. None for as many as possible. 

    def _make_dir(self, dir_name: str):
        if dir_name and not os.path.exists(dir_name):
            os.makedirs(dir_name)

    def __post_init__(self):
        if self.process_count is None:
            self.process_count = os.cpu_count() - 1
        if self.process_count < 1:
            self.process_count = 1
    
        self._make_dir(self.output_dir)
        self._make_dir(self.work_dir)
        self._make_dir(self.debug_dir)
        self._make_dir(self.mask_dir)