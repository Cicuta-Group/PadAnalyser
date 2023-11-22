
import dataclasses
from copy import copy

import matplotlib.pyplot as plt
import os

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
    crop: tuple[tuple[int]] = None # region to crop output images to, (x0,x1), (y0,y1)
    printing: bool = False
    
    def replace_label(self, text):
        cp = copy(self)
        cp.label = text
        return cp

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

    def tidy_and_save_plot(self, name):
        if self.file_plot:
            plt.savefig(os.path.join(self.image_dir, f'{name}.png'), bbox_inches='tight', dpi=300)
        if not self.live_plot:
            plt.close()

