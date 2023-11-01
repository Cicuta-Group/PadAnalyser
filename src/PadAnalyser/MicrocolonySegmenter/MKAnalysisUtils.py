
import os

import pandas as pd
import numpy as np
import json
import codecs
import itertools

import numpy as np


def i0_and_z_count(m, check_steps=30):
    check_steps = min(len(m), check_steps) # ensure do not exceed length of movie

    ts = np.array([m.frame_time(i, true_time=True) for i in range(check_steps)]) # s, get frame times from movie
    dts = ts[1:] - ts[:-1] # s, time delta between frames

    dt_lim = min(dts)*2.5
    under = dts < dt_lim # under dt_lim
    under_counts = np.diff(np.where(np.concatenate(([under[0]], under[:-1] != under[1:], [True])))[0])[::2]
    under_counts = under_counts[:-1] # disguard last entry, as could be partial
    
    z_count = round(np.mean(under_counts)) + 1 # count the peak in between as well
    i0 = 0 if under[0] else 1 # starting index - if offset from zero

    return i0, z_count


class NumpyArrayEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        return json.JSONEncoder.default(self, obj)


def write_dict_to_json(data, filename):
    json.dump(
        data,
        codecs.open(filename, 'w', encoding='utf-8'), separators=(',', ':'), 
        sort_keys=True, 
        indent=4,
        cls=NumpyArrayEncoder,
    ) ### this saves the array in .json format


def querystring(key, val):
    if val != val: return f'`{key}` != `{key}`'

    val_string = f'"{val}"' if isinstance(val, str) else f'{val}'
    return f'`{key}` == {val_string}' 


def dfs_for_unique(df, key):
    options = df[key].unique()
    return list(zip(options, [df.query(querystring(key, val)) for val in options]))


'''
Not used at the moment, but could be usefull for later
'''
# def compute_all_index_series(movie, z_stack_count, dual_illumination, analyze_fluorescence):
#     '''
#     Normally just brightfield data. Flag if flourescent images acialable as well. 
#     Assumptions: if BF + FL, then BF first
#     Returns 3d array with dimensions:
#     - starting-index of analysis
#     - jump between z-stack stacks
#     - indices in z-stack
#     '''

#     images_per_position = 2 if dual_illumination else 1
#     analyze_fl_offset = 1 if analyze_fluorescence else 0 
#     total_frame_count = len(movie)
    
#     offset_index, di = interval_from_movie(movie, check_steps=10_000)
#     position_count = di // z_stack_count // images_per_position
#     print(f'Analyzed movie to find first frame at {offset_index} and frame interval {di}. Total of {position_count} fields of view imaged.')

#     starting_indices = [analyze_fl_offset+i*z_stack_count*images_per_position for i in range(position_count)] # initial offset, constant added
#     time_series_indices = [i for i in range(0, total_frame_count, position_count*z_stack_count*images_per_position)] # indices for identical set of frames
#     z_stack_indices = [i*images_per_position for i in range(z_stack_count)] # indices for z-stack

#     return [[[offset_index + i0 + i + z for z in z_stack_indices] for i in time_series_indices] for i0 in starting_indices] # position, timeseries, z-stack


# def interval_from_movie(m, check_steps=10000):
#     # Pick out capture interval (number of frames in movie per cycle)
#     # based on time-infomration of frames.
#     # Needs to capture at least two full cycles.
#     try:
#         check_steps = min(len(m), check_steps) # ensure do not exceed length of movie
#         ts = np.array([m.frame_time(i, true_time=True) for i in range(check_steps)]) # s, get frame times from movie
#         dts = ts[1:] - ts[:-1] # s, time delta between frames
#         peaks = dts > 0.8 * dts.max() # pick out peaks
#         peak_indices = np.array(np.where(peaks)[0]) # get index of peaks
#         peak_index_delta = peak_indices[1:] - peak_indices[:-1] # compute index differences between peaks
#         di = peak_index_delta[0]
#         i0 = peak_indices[0]-di+1
#         return i0, di
#     except Exception as e:
#         print('Could not infer interval from movie.', e)

def sanetize_filename(f):
    f = f.replace('/', '')
    f = f.replace('%', '')
    f = f.replace('*', '')
    f = f.replace(':', '')
    f = f.replace('<', '')
    f = f.replace('>', '')
    f = f.replace('|', '')
    f = f.replace('"', '')
    f = f.replace("'", '')
    f = f.replace(" ", '_')
    return f