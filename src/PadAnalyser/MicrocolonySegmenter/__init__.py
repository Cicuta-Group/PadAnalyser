SEGMENTATION_VERSION = 'S1.23'
DATAFRAME_VERSION = 'D1.13'
PLOT_VERSION = 'P1.21'
MODEL_VERSION = 'M1.03'


from PadAnalyser import FrameSets, OutputConfig
import SegmentPads, SegmentPadsRunner

def segment_frame_set(frame_set: FrameSets.FrameSet, output_config: OutputConfig):

    
    print(frame_set, output_config)