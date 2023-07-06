
import os, logging 
import pandas as pd

from MKImageAnalysis import MKAnalysisUtils, DataUtils

import json
from decouple import config

SERIES_KEY = 'labelid'

def mapping_for_experiment(experiment):
    input_points_directory = config('INPUT_POINTS_DIRECTORY')
    mapping_file = os.path.join(input_points_directory, 'BE_condition_map.json') # '/Users/mkals/Developer/Cambridge/BacterialImaging/point_sets/BE_condition_map.json'
    with open(mapping_file, 'r') as f:
        d = f.read()
        d = d.replace('Âµ', 'µ')
        experiment_maps = json.loads(d)

    for experiment_map in experiment_maps:
        if experiment in experiment_map['experiments'].keys():
            return experiment_map

    raise Exception(f'Error - could not find experiment map for {experiment}.')


def find_numeric_and_text_keys(experiment_map):
    condition_sets = experiment_map['condition_sets']
    all_keys = list(condition_sets.keys())
    if len(all_keys) == 0: raise Exception("No series labels found in experiment map.")

    numeric_keys = []
    text_keys = []
    for k in all_keys:
        values = list(condition_sets[k].values())
        if isinstance(values[0], str): text_keys.append(k)
        else: numeric_keys.append(k)

    return numeric_keys, text_keys



first = lambda x: x.iloc[0]

# Produce standardized dataframe that can be compared across experiments.
# Time in hours
def df_from_reduced_property(df, properties, copy_keys):
       
    copy_keys = [k for k in copy_keys if k] # remove none
    
    return df.groupby(SERIES_KEY).agg({
        'time_days': first, 
        'round_time_days': first, 
        'experiment': first, 
        SERIES_KEY: first, 
        'row': first, 
        'col': first, 
        'replicate': first, 
        **{k: first for k in copy_keys},
        **{p: 'mean' for p in properties},
    })


def generate_growth_rate_stats_df(df, numeric_keys, text_keys, ykeys, experiment, experiment_category, instrument, filename, output_folder):
    keys = numeric_keys+text_keys+['row', 'col', 'pad_name']

    df_gr = df_from_reduced_property( # one row per colony
        df=df,
        properties=ykeys,
        copy_keys=keys,
    )
    
    stats_df = pd.DataFrame()
    for values, df_query in DataUtils.groupby(df_gr, keys):
        if len(df_query) == 0: continue # do not add rows for combinations with zero series
        row = {
            'experiment': experiment,
            'experiment_category': experiment_category,
            'instrument': instrument,
            'series_count': len(df_query),
            'numeric_keys': numeric_keys,
            'text_keys': text_keys,
            **{f'{k}_mean': df_query[k].mean() for k in ykeys},
            **{f'{k}_std': df_query[k].std() for k in ykeys},
            **{k:v for k,v in zip(keys, DataUtils.listify(values))},
        }
        
        if stats_df.empty:
            stats_df = pd.DataFrame(columns=row.keys()) # add column names first round
            
        stats_df.loc[stats_df.shape[0]] = row.values()

    # logging.info(stats_df)
    filepath = os.path.join(output_folder, MKAnalysisUtils.sanetize_filename(f'{filename}.json'))
    stats_df.to_json(path_or_buf=filepath)
    
    return stats_df



