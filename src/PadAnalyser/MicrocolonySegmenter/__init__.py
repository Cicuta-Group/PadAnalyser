SEGMENTATION_VERSION = 'S1.23'
DATAFRAME_VERSION = 'D1.14'
PLOT_VERSION = 'P1.21'
MODEL_VERSION = 'M1.03'

# typehint types
from PadAnalyser.FrameSet import FrameSet
from PadAnalyser.OutputConfig import OutputConfig
import pandas as pd
from typing import List, Type, Optional

import MKUtils
import tqdm
import logging
import os
import json
from functools import partial
from multiprocessing import Pool
from . import MKTimeseriesAnalyzer, DataProcessor
from .DInfo import DInfo
from . import MKAnalysisUtils


def segment_frame_set_to_dict(frame_set: FrameSet, output_config: OutputConfig) -> dict:
    
    label = frame_set.label
    clear_dirs = output_config.clear_dirs
    output_dir = output_config.output_dir
    debug_dir = output_config.debug_dir
    mask_dir = output_config.mask_dir

    image_dir = MKUtils.join_and_make_path(debug_dir, 'img', clear=clear_dirs)
    video_dir = MKUtils.join_and_make_path(debug_dir, 'mov', clear=clear_dirs)
    segmentation_file = MKUtils.join_and_make_path(output_dir, 'segmentation', f'{label}_{SEGMENTATION_VERSION}.json')

    # try to recover segmentation information from file
    if output_config.cache_segmentation:
        try:
            with open(segmentation_file, 'r') as f: file_contents = f.read()
            logging.info('Loaded log file.')
            return json.loads(file_contents)
        except FileNotFoundError:
            pass
        except json.decoder.JSONDecodeError as e:
            logging.debug(f'Could not decode json from file {segmentation_file}, {e}')
        except Exception as e:
            logging.exception('Could not load segmentation data from cache.')

    dinfo = DInfo(
        label=label,
        live_plot=False,
        file_plot=True,
        video=True,
        image_dir=image_dir,
        video_dir=video_dir,
        crop=None,
        printing=False,
    )

    logging.info(f'Analysing {label}')
    data = MKTimeseriesAnalyzer.analyze_time_seriess(
        frame_set=frame_set,
        mask_folder=mask_dir, 
        label=label, 
        dinfo=dinfo,
    )
    
    MKAnalysisUtils.write_dict_to_json(
        data=data,
        filename=segmentation_file,
    )
    logging.info(f'Done analyzing {label}')

    return data


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
    
    # Logging dir setup
    loggin_dir = os.path.dirname(output_config.logging_file)
    MKUtils.generate_directory(loggin_dir)
    MKUtils.configure_logger(filepath=output_config.logging_file, label=f'subprocess', file_mode='a', stream_level='WARN') # reconfigure in each multiprocessing pool
        
    data = segment_frame_set_to_dict(
        frame_set=frame_set, 
        output_config=output_config, 
    )
    
    # convert to dataframe and add growth rate etc.
    logging.info(f'Making dataframe for {label}')
    df = DataProcessor.dataframe_from_data_series(
        data=data,
        label=label,
        metadata=frame_set.metadata
    )
    logging.info(f'Finished making dataframe for {label}')

    return df
    

def dataframe_filepath(directory: str) -> str:
    return os.path.join(directory, f'dataframe_{DATAFRAME_VERSION}.json')

def load_dataframe(dataframe_file: str) -> Optional[pd.DataFrame]:
    try:
        return pd.read_json(path_or_buf=dataframe_file)
    except (ValueError, FileNotFoundError):
        pass
    except Exception:
        logging.exception(f'Could not load dataframe {dataframe_file}.')


def segment_frame_sets(frame_sets: List[FrameSet], output_config: OutputConfig):
        
    # try to load from individual data-files or segment from scratch
    logging.info('Starting analysis')
    
    dataframe_file = dataframe_filepath(directory=output_config.output_dir)
    if output_config.cache_dataframe:
        df = load_dataframe(dataframe_file=dataframe_file)
        if df is not None:
            logging.info(f'Loaded dataframe from cache {dataframe_file}.')
            return df
        
    try:
        with Pool(processes=output_config.process_count) as pool:

            # Analyze videos to produce timeseries data vectors
            dataframes = list(
                tqdm.tqdm( # display progress bar, https://stackoverflow.com/a/45276885/1502517
                    pool.imap_unordered( # map with generator, trigger evaluation using outer list
                        partial(
                            segment_frame_set,
                            output_config=output_config,
                        ),
                        frame_sets
                    ), total=len(frame_sets)
                )
            )
    except KeyboardInterrupt:
        print("Detected KeyboardInterrupt. Terminating workers.")
        pool.terminate()
        pool.join()
        return None

    df = pd.concat(dataframes, axis=0)
    df.reset_index(inplace=True)
    df.to_json(dataframe_file) # save to file 
    
    return df