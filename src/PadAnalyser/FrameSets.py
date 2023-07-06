
import numpy as np
Frame = np.ndarray

from typing import Touple, List, Union

# Interface for sets of frames to be analyzed as a timelapse. 
# Objects that contain the time-lapse information for a given field of view. Can include z-stacks or not. 

class FrameSet:
    def getNextFrame() -> Touple[Frame, int]: # returns frame and time in seconds
        pass
    
    def getTimestamps() -> List[int]: # returns list of timestamps in seconds
        pass
    
    def getFrameCount() -> int:
        pass



class TiffFrameSet(FrameSet):

    def __init__(self, file_paths: Union[List[str], List[List[str]]], times_in_seconds: List[int]): # list of frames at different times. If using z-stacks, times is a list of lists. Outer list is timepoints, inner list is z-stacks.
        self.filenames = file_paths
        self.times = times_in_seconds

    def getNextFrame() -> Touple[Frame, int]: # returns frame and time in seconds
        pass
    
    def getTimestamps() -> List[int]: # returns list of timestamps in seconds
        pass

    def getFrameCount() -> int:
        pass


    def get_frame(self, i):
        return np.array(Image.open(self.image_paths[i]))
    
    def __len__(self):
        return len(self.image_paths)
    
    # enables access to frames using square brackets
    def __getitem__(self, key):
        if isinstance(key, slice):
            # Get the start, stop, and step from the slice
            return [self.get_frame(ii) for ii in range(*key.indices(len(self)))]
        elif isinstance(key, int):
            return self.get_frame(key)
        else:
            raise TypeError

    def __iter__(self):
        self._iterator_count = 0
        return self

    # enables itterator behaviour
    def __next__(self):
        if self._iterator_count < len(self):
            self._iterator_count += 1
            return self[self._iterator_count]
        else:
            raise StopIteration

    def frame_time(self, i, true_time=False) -> float:
        return i/30



# class MovieFrameSet(FrameSet):
#     def __init__(self, filename: str, index_set: List[int]): # index_set is a list of frame indices to use
#         pass
    
#     def __init__(self, filename: str, index_set: List[List[int]]): # if using z-stacks, index_set is a list if lists. Outer list is timepoints, inner list is z-stacks.
#         pass
