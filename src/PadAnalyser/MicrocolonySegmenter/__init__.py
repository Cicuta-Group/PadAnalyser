SEGMENTATION_VERSION = 'S1.23'
DATAFRAME_VERSION = 'D1.13'
PLOT_VERSION = 'P1.21'
MODEL_VERSION = 'M1.03'

# typehint types
from PadAnalyser.FrameSets import FrameSet
from PadAnalyser.OutputConfig import OutputConfig
import pandas as pd
from typing import List

import MKUtils
import tqdm
import logging
import os
import json
import shutil
from functools import partial
from multiprocessing import Pool
from . import MKTimeseriesAnalyzer, DataProcessor
from MKSegmentUtils import DInfo
import MKAnalysisUtils


def segment_frame_set(frame_set: FrameSet, output_config: OutputConfig) -> pd.DataFrame:
    """Segment colonies and single cells of frame set.

    Produces two intermediate files with segmentation output and dataframe output, that can both be used at cache to speed up expensive calculations.

    Args:
        frame_set (FrameSet): _description_
        output_config (OutputConfig): _description_

    Returns:
        pd.DataFrame: _description_
    """
    
    label = frame_set.label
    print(label, frame_set, output_config)

    MKUtils.configure_logger(filepath=output_config.logging_file, label=f'subprocess', file_mode='a', stream_level='WARN') # reconfigure in each multiprocessing pool

    frameset_output_dir = frame_set.output_dir
    frameset_segmentation_dir = os.path.join(frameset_output_dir, 'segmentation')
    frameset_dataframe_dir = os.path.join(frameset_output_dir, 'dataframe')
    
    # make directories if they do not exist
    MKUtils.generate_directory(frameset_segmentation_dir, clear=False)
    MKUtils.generate_directory(frameset_dataframe_dir, clear=False)

    frameset_segmentation_file = os.path.join(frameset_segmentation_dir, label + '.json')
    frameset_dataframe_file = os.path.join(frameset_dataframe_dir, label + '.json') # file where data from this experiment is stored

    # try to recover dataframe from cache file
    if output_config.cache_dataframe:
        try:
            return pd.read_json(path_or_buf=frameset_dataframe_file)
        except (ValueError, FileNotFoundError):
            pass
        except Exception:
            logging.exception(f'Could not load dataframe.')


    # try to recover segmentation information from file
    if output_config.cache_segmentation:
        try:
            with open(frameset_segmentation_file, 'r') as f: file_contents = f.read()
            logging.info('Loaded log file.')
            return json.loads(file_contents)
        except FileNotFoundError:
            pass
        except json.decoder.JSONDecodeError as e:
            logging.debug(f'Could not decode json from file {frameset_segmentation_file}, {e}')
        except Exception as e:
            logging.exception('Could not load segmentation data from cache.')


    dinfo = DInfo(
        label=label,
        live_plot=False,
        file_plot=None,
        video=None,
        image_dir=None,
        video_dir=None,
        crop=False,
        printing=False,
    )

    logging.info(f'Analysing {label}')
    data = MKTimeseriesAnalyzer.analyze_time_seriess(
        frame_set=frame_set, 
        mask_folder=output_config.mask_dir, 
        label=label, 
        dinfo=dinfo,
    )
    for key, value in frame_set.metadata.items():
        data[key] = value
    
    MKAnalysisUtils.write_dict_to_json(
        data=data,
        filename=frameset_segmentation_file,
    )
    logging.info(f'Done analyzing {label}')

    # convert to dataframe and add growth rate etc.
    logging.info(f'Making dataframe for {label}')
    df = DataProcessor.dataframe_from_data_series(
        data=data,
        metadata=frame_set.metadata
    )
    df.to_json(frameset_dataframe_file)
    logging.info(f'Finished making dataframe for {label}')

    return df
    

def segment_frame_sets(frame_sets: List[FrameSet], output_config: OutputConfig):
        
    # try to load from individual data-files or segment from scratch
    logging.info('Starting analysis')
    
    dataframe_file = os.path.join(output_config.output_dir, 'dataframe.json')        

    with Pool(processes=output_config.process_count) as pool: 

        # Analyze videos to produce timeseries data vectors
        dataframes = list(
            tqdm.tqdm( # display progress bar, https://stackoverflow.com/a/45276885/1502517
                pool.imap_unordered( # map with generator, trigger evaluation using outer list
                    partial(
                        segment_frame_set,
                        output_config=output_config,
                    ),
                    zip(frame_sets)
                ), total=len(frame_sets)
            )
        )

    df = pd.concat(dataframes, axis=0)
    df.reset_index(inplace=True)
    df.to_json()

    # set round_time based on smallest time with given round_index
    # find time series that starts with smallest value (i.e. is from first imaged location) for round_times
    # time_series = [s['time'] for s in data_series]
    # index_of_first_time_series = np.argmin([s[0] for s in time_series])
    # round_times = np.array(time_series[index_of_first_time_series])
    
    return df

