import os, logging, argparse, json, shutil
from decouple import config
# from dotenv import load_dotenv
# load_dotenv()

from functools import partial
from multiprocessing import Pool
import tqdm
import numpy as np
import pandas as pd

from MKImageAnalysis import MKTimeseriesAnalyzer
from MKImageAnalysis.MKAnalysisUtils import i0_and_z_count, find_movie_filepath, write_dict_to_json, load_points_from_experiment
from MKImageAnalysis.MKSegmentUtils import  DInfo
from TemikaXML.Movie import Movie
import MKUtils

from MKImageAnalysis import Plotter, PlotPads, DataProcessor, SEGMENTATION_VERSION, DATAFRAME_VERSION, experiment_folder_name

PROCESS_COUNT = int(config('PROCESS_COUNT', 0))
if PROCESS_COUNT < 1: PROCESS_COUNT = None

def process_index_series(data, movie_file_path, output_dir, use_cache, mask_folder, dinfo: DInfo):
    
    MKUtils.configure_logger(dinfo.video_dir, LOGGING_FILENAME, label=f'subprocess', file_mode='a', stream_level='WARN') # reconfigure in each multiprocessing pool
    # if np.random.uniform() < 0.02: dinfo = dinfo.with_file_plot() # output all files for given percentage of FOWs

    index_series, label, pad_name = data
    logging.debug(f'{index_series=}, {label=}, {pad_name=}')
    
    output_filename = f'{label}.json'
    output_filepath = os.path.join(output_dir, output_filename)
    dinfo = dinfo.append_to_label(label)
    m = Movie(movie_file_path)
    
    if use_cache:
        try:
            with open(output_filepath, 'r') as f: file_contents = f.read()
            logging.info('Loaded log file.')
            return json.loads(file_contents)
        except (FileNotFoundError, json.decoder.JSONDecodeError) as e:
            logging.debug(f'Could not find or load file {output_filepath}, {e}')
        except Exception:
            logging.exception('Could not load from cache.')

    logging.info(f'Analyzing {label}')
    data = MKTimeseriesAnalyzer.analyze_time_seriess(m, index_series, mask_folder=mask_folder, label=label, dinfo=dinfo)
    logging.info(f'Done analyzing {label}')

    data['filename'] = movie_file_path
    data['label'] = label
    data['pad_name'] = pad_name
    data['index_series'] = index_series
    
    write_dict_to_json(
        data=data,
        filename=output_filepath,
    )

    return data


# import signal
# def init_worker():
#     signal.signal(signal.SIGINT, signal.SIG_IGN)
# initializer=init_worker passed to Pool

def generate_dataframe(movie_file_path, output_dir, experiment, dinfo, all_index_series, labels, pad_names, experiment_map, dataframe_file, mask_folder, use_cache):
    
    # try to recover full dataframe from cache file
    if use_cache:
        try:
            return pd.read_json(path_or_buf=dataframe_file)
        except (ValueError, FileNotFoundError):
            pass
        except Exception:
            logging.exception(f'Could not load dataframe.')
        
    # try to load from individual data-files or segment from scratch
    logging.info('Starting analysis')
    with Pool(processes=PROCESS_COUNT) as pool:

        # Analyze videos to produce timeseries data vectors
        data_series = list(
            tqdm.tqdm( # display progress bar, https://stackoverflow.com/a/45276885/1502517
                pool.imap_unordered( # map with generator, trigger evaluation using outer list
                    partial(
                        process_index_series,
                        movie_file_path=movie_file_path, 
                        output_dir=output_dir,
                        use_cache=use_cache,
                        mask_folder=mask_folder,
                        dinfo=dinfo,
                    ),
                    zip(all_index_series, labels, pad_names)
                ), total=len(all_index_series)
            )
        )

        # data_series = [
        #     process_index_series(
        #         (index_series, label, pad_name),
        #         movie_file_path=movie_file_path, 
        #         output_dir=output_dir,
        #         use_cache=use_cache,
        #         mask_folder=mask_folder,
        #         dinfo=dinfo,
        #     ) for index_series, label, pad_name in zip(all_index_series, labels, pad_names)
        # ]
        
        # find time series that starts with smallest value (i.e. is from first imaged location) for round_times
        time_series = [s['time'] for s in data_series]
        index_of_first_time_series = np.argmin([s[0] for s in time_series])
        round_times = np.array(time_series[index_of_first_time_series])
        
        ### Compute dataframe
        dfs = list(
            tqdm.tqdm( # display progress bar, https://stackoverflow.com/a/45276885/1502517
                pool.imap_unordered( # map with generator, trigger evaluation using outer list
                    partial(
                        DataProcessor.dataframe_from_data_series,
                        round_times=round_times,
                        experiment=experiment,
                        experiment_map=experiment_map,
                        segmentation_version=SEGMENTATION_VERSION,
                        process=True,
                    ),
                    data_series
                ), total=len(data_series)
            )
        )

    df = pd.concat(dfs, axis=0)
    df.reset_index(inplace=True)
    df.to_json(dataframe_file)

    return df


def find_stack_indices_from_times(m, end_index):
    ts = np.array([m.frame_time(i=i, true_time=True) for i in range(end_index)])
    dts = ts[1:] - ts[:-1]
    transitions = dts > 1

    last_true = 0
    dis = []
    for i, value in enumerate(transitions):
        di = i-last_true
        if value and di > 1:
            dis.append(di)
            last_true = i

    dis = dis
    i0s = [0] + list(np.cumsum(dis)+1)

    return i0s, dis

def compute_index_sets(experiment, movie, n_lim, experiment_map, points):
    z_count = experiment_map.get('experiments', {}).get(experiment, {}).get('z_stack_frames', None)
    i0_0 = experiment_map.get('experiments', {}).get(experiment, {}).get('first_frame', 0)
    variable_zstack_size = experiment_map.get('experiments', {}).get(experiment, {}).get('variable_zstack_size', False)

    if not z_count: i0_0, z_count = i0_and_z_count(movie)
    last_use_index = len(movie)
    last_use_index = last_use_index - last_use_index % (len(points) * z_count) # tuncate to last complete round that was imaged 
    end_index = min(last_use_index, n_lim * len(points) * z_count) if n_lim else last_use_index

    if variable_zstack_size:
        logging.info('Computing variable length z-stacks.')
        blocks = find_stack_indices_from_times(movie, end_index)
        index_blocks = [[int(i0+i) for i in range(l)] for i0, l in zip(*blocks)]
        fows = len(points['pad_name'])
        return [index_blocks[i:end_index:fows] for i in range(fows)]


    i0s = range(i0_0, end_index, len(points)*z_count)
    
    logging.info(f'Index series parameters: {i0_0=}, {i0s=}, {z_count=}, {len(movie)=}')
    logging.info(f'Total timesteps per FOW={end_index/z_count/len(points):.1f}, for analysis={len(i0s)}')
    logging.info(f'Analyzing until {movie.frame_time(end_index-1, true_time=False)}')

    return [[[i0 + z + z_count*i for z in range(z_count)] for i0 in i0s] for i, _ in enumerate(points['pad_name'])]



LOGGING_FILENAME = 'SegmentPads.log'

def main(experiment, n_lim, m_lim, all_ignore_cache, debug_output, tiff_start_index, tiff_count):

    # 1. Get directories
    output_directory = config('OUTPUT_DIRECTORY') # '/Users/mkals/Library/CloudStorage/OneDrive-UniversityofCambridge/data/'
    work_directory = config('WORK_DIRECTORY')
    input_movie_directory = config('INPUT_MOVIE_DIRECTORY')
    input_points_directory = config('INPUT_POINTS_DIRECTORY')

    # 2. Append experiment name to directories
    folder_name =  experiment_folder_name(experiment)
    output_directory = os.path.join(output_directory, folder_name)
    work_directory = os.path.join(work_directory, folder_name)

    if all_ignore_cache:
        if os.path.exists(work_directory): shutil.rmtree(work_directory)
        if os.path.exists(output_directory): shutil.rmtree(output_directory)

    # 2. Prepare output directories
    debug_dir = os.path.join(output_directory, 'mov')
    output_dir = os.path.join(work_directory, 'json')
    img_dir = os.path.join(work_directory, 'img')
    mask_dir = os.path.join(work_directory, 'all_masks')
    
    MKUtils.generate_directory(debug_dir)
    MKUtils.generate_directory(output_dir)
    MKUtils.generate_directory(img_dir)
    MKUtils.generate_directory(mask_dir)

    dataframe_file = os.path.join(output_directory, f'dataframe_{DATAFRAME_VERSION}.json')
    
    # 3. Configure logging
    MKUtils.configure_logger(debug_dir, LOGGING_FILENAME, label='main', file_mode='a')
    logging.info(f'Executing SegmentPads.py with {experiment}: {n_lim=}, {m_lim=}, {all_ignore_cache=}, {debug_output=}')

    dinfo = DInfo (
        label='seg',
        live_plot=False, 
        file_plot=debug_output, 
        video=True,
        image_dir=img_dir,
        video_dir=debug_dir, 
        crop=None,
        printing=False,
    )

    # 4. Load resources
    try:
        movie_file_path = find_movie_filepath(experiment, input_movie_directory)
        m = Movie(movie_file_path)
    except Exception:
        logging.exception('Could not import movie file.')
        return

    
    if tiff_count: # 5. Generate still images for reference
        from MKImageAnalysis import movie_to_tiff
        movie_to_tiff.main(movie=movie_file_path, interval=1, n_lim=tiff_count, start_index=tiff_start_index, output_directory=os.path.join(output_directory, 'tiffs'))
        return

    try:
        points = load_points_from_experiment(experiment, input_points_directory)
        len(points) # throws if points is NoneType -> if that is the case, break
    except Exception:
        logging.exception('Could not import points file.')
        return

    try:
        experiment_map = Plotter.mapping_for_experiment(experiment)
    except Exception:
        logging.exception('Could not load experiment map.')
        return

    # 6. Compute index sets for FOW, z-stack
    all_index_series = compute_index_sets(
        experiment=experiment,
        movie=m,
        n_lim=n_lim,
        points=points,
        experiment_map=experiment_map,
    )

    if m_lim: all_index_series = all_index_series[:m_lim]
    
    # 7. Generate pad names
    fl_br_label = 'br' # TODO: add more channels
    pad_names = points['pad_name']
    labels = [f'{experiment}_{i}_{p.strip()}_{fl_br_label}' for i, p in enumerate(pad_names)]
    
    # Segment or load dataframe from cache
    df = generate_dataframe(
        movie_file_path = movie_file_path, 
        output_dir = output_dir, 
        experiment = experiment, 
        all_index_series = all_index_series, 
        labels = labels, 
        pad_names = pad_names, 
        experiment_map = experiment_map, 
        dataframe_file = dataframe_file, 
        use_cache = not all_ignore_cache,
        mask_folder = mask_dir,
        dinfo = dinfo, 
    )

    PlotPads.main(experiment=experiment, df=df)


# python -m MKImageAnalysis BE83 -n 5 -m 1