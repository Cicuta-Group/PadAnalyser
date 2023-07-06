import os, logging, datetime
from decouple import config
# from dotenv import load_dotenv
# load_dotenv()

import argparse, re
import natsort

from MKImageAnalysis import SegmentPads, DataUtils
import MKUtils


def main(debug_output, all_ignore_cache, n_lim, m_lim, tiff_start_index, tiff_count):
    
    data_folder = config('OUTPUT_DIRECTORY') # '/Users/mkals/Library/CloudStorage/OneDrive-UniversityofCambridge/data/'
    movie_folder = config('INPUT_MOVIE_DIRECTORY') # '/Users/mkals/Library/CloudStorage/OneDrive-UniversityofCambridge/data/'
    
    logging_folder = DataUtils.append_path(data_folder, 'runlogs')

    # 3. Configure logging
    MKUtils.configure_logger(logging_folder, datetime.datetime.now().strftime('mylogfile_%H_%M_%d_%m_%Y.log'), label='main', file_mode='w')
    logging.info(f'Executing SegmentPadsRunner.py with {n_lim=}, {m_lim=}, {all_ignore_cache=}, {debug_output=}')

    # experiments = [key for experiment_map in experiment_maps for key in experiment_map['experiments'].keys()][::-1]
    regex = re.compile(r'^BE\d+$')
    experiments = [f for f in os.listdir(movie_folder) if regex.match(f)]
    experiments = natsort.natsorted(experiments, alg=natsort.ns.IGNORECASE)[::-1]
    logging.info(f'Experiments in data directory: {", ".join(experiments)}.')

    # manual override of which exepriments to run
    #experiments = ['BE124', 'BE123', 'BE120']
    #experiments = [e for e in experiments if int(e[2:]) < 136]

    logging.info(f'Analysing experiments: {", ".join(experiments)}.')

    for experiment in experiments:
        try:
            logging.info(f'Segmenting {experiment}.')
            SegmentPads.main(
                experiment=experiment, 
                all_ignore_cache=all_ignore_cache,
                debug_output=debug_output,
                n_lim=n_lim,
                m_lim=m_lim,
                tiff_start_index=tiff_start_index,
                tiff_count=tiff_count,
            )
            logging.info(f'Completed segmentation of {experiment}.')

        except Exception:
            logging.exception(f'Exception occured for {experiment} during segmentation.')


if __name__ == '__main__':
    
    # 0. Load input arguments from user.
    parser = argparse.ArgumentParser(description='Run analysis and plotting for all data-sets.')
    parser.add_argument('--debug', '-d', type=bool, default=False, const=True, nargs='?', help='Produce debug output.')
    parser.add_argument('--all', '-a', type=bool, default=False, const=True, nargs='?', help='Do not used cached data.')
    parser.add_argument('--tiff_count', '-t', type=int, default=0, nargs='?', help='Generate tiffs instead.')
    args = parser.parse_args()
    
    main(
        debug=args.debug,
        all_ignore_cache=args.all,
        tiff_count=args.tiff_count,
    )
