
from typing import List, Union
from PIL import Image
import numpy as np
import cv2 as cv

import dataclasses
from abc import ABC, abstractmethod

Frame = np.ndarray


@dataclasses.dataclass
class ZStack:
    frames: List[Frame]
    times: List[int] # in seconds
    label: str

    @property
    def z_count(self) -> int:
        return len(self.times)
    
    @property
    def time(self) -> int:
        return self.times[0]


# Interface for sets of frames to be analyzed as a timelapse. 
# Objects that contain the time-lapse information for a given field of view. Can include z-stacks or not. 

class FrameSet(ABC):
    
    ### Methods subclasses should implement

    @abstractmethod
    def __init__(self, label: str, metadata: dict = None): # initializer that enables user to spesify the data for this frame set
        self.label = label
        self.metadata = metadata

    def get_frame(self, index: int) -> Frame:
        pass
    
    def get_time(self, index: int) -> int: # returns list of timestamps in seconds
        pass
    
    def get_frame_label(self, index: int) -> int:
        pass

    def get_frame_count(self) -> int:
        pass
    

    ### Methods that allow users to interact with subclasses thourgh iterators etc.
    
    def get_frame_and_time(self, index: int): # -> Tuple[Frame, int]
        return (self.get_frame(index), self.get_time(index))

    # enables checking length with len(frameSetInstance)
    def __len__(self) -> int:
        return self.get_frame_count()
    
    # enables access to frames and time touple using square brackets
    def __getitem__(self, key: int):
        if isinstance(key, int):
            return self.get_frame_and_time(key)

        elif isinstance(key, slice):
            # Get the start, stop, and step from the slice
            return [self.get_frame_and_time(ii) for ii in range(*key.indices(len(self)))]
        
        else:
            raise TypeError(f'Could not rexognize iterator key {key} for {str(self)}. Must be int or slice, is {type(key)}.')

    # init iterator so you can loop over objects in memory efficient manner
    def __iter__(self):
        self._iterator_count = 0
        return self

    # enables itterator behaviour
    def __next__(self):
        if self._iterator_count < len(self):
            self._iterator_count += 1
            return self[self._iterator_count-1] # start at zero
        else:
            raise StopIteration

    def get_frame_labels(self):
        return [self.get_frame_label(i) for i in range(len(self))]




class TiffFrameSet(FrameSet):

    def __init__(self, file_paths: List[List[str]], times_in_seconds: List[int], frame_labels: List[str], **kwargs): # list of frames at different times. If using z-stacks, times is a list of lists. Outer list is timepoints, inner list is z-stacks.
        if len(file_paths) != len(times_in_seconds):
            raise Exception(f"Number of files {len(file_paths)} different from number of times {len(times_in_seconds)}")
        
        self.filenames = file_paths
        self.times = times_in_seconds
        self.frame_labels = frame_labels if frame_labels else [str(i) for i in range(len(file_paths))]
        super().__init__(**kwargs)


    def get_frame(self, index: int) -> Frame: # -> Tuple[Frame, int]: # returns frame and time in seconds
        return [np.array(Image.open(f)) for f in self.filenames[index]]
    
    def get_time(self, index: int) -> int: # returns list of timestamps in seconds
        return self.times[index]

    def get_frame_label(self, index: int) -> str:
        return self.frame_labels[index]

    def get_frame_count(self) -> int:
        return len(self.times)

    def __str__(self) -> str:
        return f'TiffFrameSet with {len(self.times)} frames'

    def __repr__(self) -> str:
        return f'TiffFrameSet(frame=[{self.filenames[0]}, ...], times=[{self.times[0]}, ...])'


class PngFrameSet(FrameSet):

    def __init__(self, file_paths: List[str], times_in_seconds: List[int], frame_labels: List[str], **kwargs): # list of frames at different times. If using z-stacks, times is a list of lists. Outer list is timepoints, inner list is z-stacks.
        if len(file_paths) != len(times_in_seconds):
            raise Exception(f"Number of files {len(file_paths)} different from number of times {len(times_in_seconds)}")
        
        self.filenames = file_paths
        self.times = times_in_seconds
        self.frame_labels = frame_labels if frame_labels else [str(i) for i in range(len(file_paths))]
        super().__init__(**kwargs)


    def get_frame(self, index: int) -> Frame: # returns frame 
        filename = self.filenames[index]
        frame = cv.imread(filename, cv.IMREAD_UNCHANGED) # benchmarked to be faster than PIL
        return frame

    def get_time(self, index: int) -> int: # returns list of timestamps in seconds
        return self.times[index]

    def get_frame_label(self, index: int) -> str:
        return self.frame_labels[index]

    def get_frame_count(self) -> int:
        return len(self.times)

    def __str__(self) -> str:
        return f'PngFrameSet with {len(self.times)} frames'

    def __repr__(self) -> str:
        return f'PngFrameSet(frame=[{self.filenames[0]}, ...], times=[{self.times[0]}, ...])'



# class MovieFrameSet(FrameSet):
#     def __init__(self, filename: str, index_set: List[int]): # index_set is a list of frame indices to use
#         pass
    
#     def __init__(self, filename: str, index_set: List[List[int]]): # if using z-stacks, index_set is a list if lists. Outer list is timepoints, inner list is z-stacks.
#         pass
