
import dataclasses
from copy import copy

'''
Class to concicely pass information about how debug should be handled.
Make lots of local copies for different subcontexts.
'''
@dataclasses.dataclass
class DInfo:
    label: str
    live_plot: bool = False # produce matplotlib graphs
    file_plot: bool = False
    video: bool = True # if it should output video debug file
    image_dir: str = None # output debug images to spesified directory
    video_dir: str = None # output debug videos to spesified directory
    crop: tuple[tuple[int]] = None # region to crop output images to
    printing: bool = False
    

    def append_to_label(self, text):
        cp = copy(self)
        cp.label = f'{cp.label}_{text}'
        return cp

    def with_live_plot(self, active=True):
        cp = copy(self)
        cp.live_plot = active
        return cp

    def with_file_plot(self, active=True):
        cp = copy(self)
        cp.file_plot = active
        return cp

    def disabled(self):
        cp = copy(self)
        cp.live_plot = False
        cp.file_plot = False
        cp.video = False
        return cp

