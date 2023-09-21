import os, logging
import scipy.signal
import pandas as pd
import numpy as np
from collections import defaultdict

from . import MKSegmentUtils, DataUtils #, PLOT_VERSION, DATAFRAME_VERSION, SEGMENTATION_VERSION

# try:
#     import CellClassifier
# except ImportError:
#     print('Could not import CellClassifier')

SERIES_KEY = 'labelid'

# inplace manipulation of dataframe to make ready for plotting
def process_dataframe(df):

    # numeric_keys, text_keys = Plotter.find_numeric_and_text_keys(experiment_map=experiment_map)
    # logging.debug(f'{numeric_keys=}, {text_keys=}')
    
    # Filter rows
    df.query('id >= 0', inplace=True) # remove all without id
    
    # Add and manipulate columns
    df['ss_area_total'] = df['ss_area_mean'].mul(df['ss_area_count'])
    
    # add_label_columns(df=df, experiment_map=experiment_map) # add labels desribing experiment (antibiotic etc.)
    # add_replicate_column(df=df, keys=numeric_keys+text_keys)
    add_time_bin_column(df=df, interval=30)
    
    # low_pass_filter_column(df=df, column='colony_area')
    # low_pass_filter_column(df=df, column='ss_area_total')
    # low_pass_filter_column(df=df, column='ss_area_count')
    add_growth_rate_time_series_columns(df=df, fit_to_key='colony_area')
    add_growth_rate_time_series_columns(df=df, fit_to_key='ss_area_total')
    add_growth_rate_time_series_columns(df=df, fit_to_key='ss_area_count')

    if not 'ss_area_total_growth_rate' in df.columns: return

    add_termination_marker_columns(df=df)

    add_colony_lysis_columns(df)
    add_present_for_duration_column(df)

    # Filtering that can only be done at the end
    # df.query('colony_on_border == False', inplace=True)
    # df.dropna(subset=[COLONY_GROWTH_RATE_KEY], inplace=True) # has all fields filled
    
    # CellClassifier.add_is_debris_score_column(df=df)



def dataframe_from_data_series(data: dict, label: str, metadata: dict) -> pd.DataFrame:
    
    keys = ['colony_area', 'colony_arc_length', 'colony_on_border', 'colony_id', 'colony_ID', 'colony_name'] # for all keys, see MKTImeseriesAnalyzer EOF
    ss_keys = MKSegmentUtils.SS_STATS_KEYS
    
    id_key = 'colony_id'
    ss_ids_key = 'ss_id'

    df = pd.DataFrame()
    times = data['time'] # loop over 
    
    for i, (time, ids, properties_for_ids, ss_ids, ss_properties) in enumerate(zip(
        times, 
        data[id_key], 
        zip(*[data[key] for key in keys]), 
        data[ss_ids_key], 
        zip(*[data.get(key, []) for key in ss_keys]
    ))): # loop over time
        for j, (id, peroperties) in enumerate(zip(ids, zip(*properties_for_ids))): # loop over ids
            
            # filter to only include ss rows that pertain to this
            ss_filtered = [ps for ss_id, ps in zip(ss_ids, zip(*ss_properties)) if ss_id==id]
            ss_df = pd.DataFrame(ss_filtered, columns=ss_keys, dtype=float)
            
            # convert this to dict with each stat value as a key
            ss_stats = ss_df.describe()

            ss_dict = {f'{col_name}_{stat}': value for col_name, col in ss_stats.items() for stat, value in col.items()}
            
            interval = 60*15
            round_time = int(time//interval)*interval # bin to every 15 min - todo: update? 

            row = {
                'time': time, # seconds
                'time_hours': time/60/60, # hours
                'time_days': time/60/60/24, # days
                'round_time': round_time, # sec
                'round_time_hours': round_time/60/60, # hour
                'round_time_days': round_time/60/60/24, # days
                'id': id,
                'label': label,
                SERIES_KEY: f'{label}_{id}',
                'time_index': i,
                'id_index': j,
                **metadata,
                **{key: val for key, val in zip(keys, peroperties)},
                **ss_dict,
            }

            if df.empty: df = pd.DataFrame(columns=row.keys()) # add column names first round
            df.loc[df.shape[0]] = row.values()

    # TODO: how to deal with experiment_map? 
    if not df.empty: 
        process_dataframe(df=df) # process dataframe from threading jobs - makes this multiprocessed as well

    return df



def single_cell_dataframe_from_data_series(data: dict, label: str, metadata: dict) -> pd.DataFrame:
    
    multi_keys = ['ss_area', 'ss_aspect_ratio', 'ss_aspect_ratio_max_width', 'ss_centroid', 'ss_dist_sums', 'ss_length', 'ss_max_width', 'ss_width']
    single_keys = ['time', 'ss_count']

    print('Ignored keys: ', [key for key in data.keys() if key not in multi_keys + single_keys])

    time_list_of_multi_dicts = [{k:v for k, v in zip(multi_keys, values)} for values in zip(*[data[key] for key in multi_keys])]
    time_list_of_single_dicts = [{k:v for k, v in zip(single_keys, values)} for values in zip(*[data[key] for key in single_keys])]
    
    dfs = []

    for time_index, (multi_dicts, single_dicts) in enumerate(zip(time_list_of_multi_dicts, time_list_of_single_dicts)):
        df_multi = pd.DataFrame(multi_dicts)
        
        for k, v in {**single_dicts, **metadata, 'label': label}.items():
            df_multi[k] = v
        df_multi['time_index'] = time_index
        dfs.append(df_multi)

    df = pd.concat(dfs, ignore_index=True)

    return df


#### Data loading

def load_df_from_file(dataframe_file):
    try:
        return pd.read_json(path_or_buf=dataframe_file)
    except ValueError:
        logging.error(f'Dataframe not generated. Trying to load {dataframe_file}')
    except Exception:
        logging.exception(f'Could not load dataframe {dataframe_file}.')

# def load_stats_dataframe(experiment: str):
#     data_folder = config('OUTPUT_DIRECTORY')
#     dataframe_file = os.path.join(data_folder, experiment_folder_name(experiment), f'graphs_{PLOT_VERSION}', f'{experiment} growth rate stats.json')
#     return load_df_from_file(dataframe_file)

# def load_dataframe(experiment: str, segmentation_version=SEGMENTATION_VERSION, dataframe_version=DATAFRAME_VERSION):
#     data_folder = config('OUTPUT_DIRECTORY')
#     dataframe_file = os.path.join(data_folder, experiment_folder_name(experiment, segmentation_version), f'dataframe_{dataframe_version}.json')
#     return load_df_from_file(dataframe_file)

def replace_col_name(df, old_name, new_name):
    if old_name in df.columns:
        if new_name in df.columns:
            df[new_name] = df[new_name].combine_first(df[old_name])
        else:
            df[new_name] = df[old_name]


# def load_experiments(experiments: list[str], segmentation_version=SEGMENTATION_VERSION, dataframe_version=DATAFRAME_VERSION, repeat_keys=None):
#     experiment_maps = [Plotter.mapping_for_experiment(experiment=experiment) for experiment in experiments]
#     numeric_keys_lists, text_keys_lists = zip(*[Plotter.find_numeric_and_text_keys(experiment_map=experiment_map) for experiment_map in experiment_maps])

#     numeric_keys = list(set([key for keys in numeric_keys_lists for key in keys]))
#     text_keys = list(set([key for keys in text_keys_lists for key in keys]))
    
#     dfs = [load_dataframe(experiment, segmentation_version, dataframe_version) for experiment in experiments]
#     dfs = [d for d in dfs if d is not None]
#     if len(dfs) == 0: raise ValueError(f'Could not load any dataframes for experiments {experiments}.')    
    
#     df = pd.concat(dfs, axis=0)
#     df.reset_index(inplace=True)

#     replace_col_name(df, 'Ampicilin concentration (µg/ml)', 'Ampicillin concentration (µg/ml)')
#     replace_col_name(df, 'Tetracyclin concentration (µg/ml)', 'Tetracycline concentration (µg/ml)')

#     add_time_bin_column(df=df, interval=30) # cheap, so can do again here
#     add_repeat_column(df=df, numeric_keys=numeric_keys, text_keys=text_keys, repeat_keys=repeat_keys)

#     return df, numeric_keys, text_keys


# def load_experiments_stats(experiments):
#     experiment_map = Plotter.mapping_for_experiment(experiment=experiments[0])
#     numeric_keys, text_keys = Plotter.find_numeric_and_text_keys(experiment_map=experiment_map)

#     dfs = [load_stats_dataframe(e) for e in experiments]
#     dfs = [d for d in dfs if not isinstance(d, type(None))]
#     if len(dfs) == 0: raise ValueError(f'Could not load any dataframes for {experiments}')
    
#     df = pd.concat(dfs, axis=0)
#     df.reset_index(inplace=True)
#     return df, numeric_keys, text_keys


#### Condition mapping and information extraction

def condition_label(text_keys, text_values, numeric_key):

    labels = [f'{k}={DataUtils.to_str(v)}' if float(v) != 1.0 else k for k,v in zip(text_keys, text_values) if DataUtils.is_float(v)]
    labels += [f'{k}={v}' for k,v in zip(text_keys, text_values) if not DataUtils.is_float(v) and v != 'nan']

    if not labels:
        labels = [f'{k.lower()}={v if v.isupper() else v.lower()}' for k,v in zip(text_keys, text_values)]
    
    numeric_label = numeric_key.split('concentration')[0].strip() if isinstance(numeric_key, str) else numeric_key # remove concentration and unit from title
    label = ', '.join(DataUtils.listify(numeric_label) + labels)
    
    lowercase_first_char = lambda s: s[:1].lower() + s[1:] if s else ''
    return lowercase_first_char(label.strip())


def add_antibiotic_and_concentration_columns(df, antibiotic_keys):
    """Adds columns for antibiotic and concentration to dataframe
    Assumes only one antibiotic per pad
    """
    df["Antibiotic"] = df[antibiotic_keys].idxmax(1).str.split(' ').str[0]
    df["Concentration"] = df[antibiotic_keys].max(1)


### Mathods to add columns to dataframe

def add_termination_marker_columns(df):
    # Add marks where colonies hit border
    df.loc[:,'colony_on_border_start'] = False
    df['colony_on_border'] = df['colony_on_border'] # convert from <class bool> to bool
    for _, df_query in df.groupby(SERIES_KEY): # get dataframe containing data for only one colony
        # df_sorted = df_query.sort_values('time')
        d = pd.to_numeric(df_query['colony_on_border'].shift(-1)) # assign to row before so that termination mark is not excluded by filter
        if not d.empty and d.any(): # will return first index of false if all are false, so check for any
            on_border_start_index = d.idxmax()
            df.loc[on_border_start_index, 'colony_on_border_start'] = True
        
    # Add marks where colonies dissapear
    df.loc[:,'colony_has_dissapeared'] = False
    IDs = df['colony_name'].unique()
    for _, df_query in df.groupby(SERIES_KEY): # get dataframe containing data for only one colony
        _ID = df_query.iloc[-1]['colony_name']
        if not any([_ID != ID and ID.startswith(_ID) for ID in IDs]):
            last_index = df_query.last_valid_index() #list(df_query['colony_has_dissapeared'].index)[-1]
            df.loc[last_index, 'colony_has_dissapeared'] = True

# 
def add_is_debris_column(df):
    # Overall problem: 
    # - There are debris points that are recognized - how to deal with these? 
    #   1) Patch: if growth, remove non-growth. If non-growth, keep all.
    #   2) Solid solution: 
    #     - for each ID, output starting image of that cell
    #     - based on that image, a human can label as present or not
    #       - how to label? Place image in folder? Add to filename?
    #      Tools required: 
    #       - during segmentation, save images
    #       - gui to show images, keystrokes for labels (classes overlaid as text coding to numbers 1,2,3...) Labels: bacteria, bacteria-seg-issue, debis, unknown
    #       - during plotting, read dir set and correspond labels with colony identities
    #
    # - Colonies are only tracked well for a given period of time. 
    #   1) If robust removal of non-cells, it does not matter that much as averages and std. will be correct until end of tracking.
    #   2) Could also add a "not enough information" filter that removes points when there less than n colonies left in tracking.
    
    logging.info('Starting add_is_debris_column.')

    # for value, df_query in df.groupby(SERIES_KEY):
    #     # df.loc[df_query.index, 'is_debris'] = df_query['colony_area'].iloc[0] < 4
    #     # df.loc[df_query.index, 'is_debris'] = df_query[COLONY_GROWTH_RATE_KEY].std() > 0.2
    #     # df.loc[df_query.index, 'is_debris'] = df_query['ss_area_count'].std() > 0
    #     # df.loc[df_query.index, 'is_debris'] = df_query['ss_area_count'].mean() > 0
    #     df.loc[df_query.index, 'is_debris'] = df_query[COLONY_GROWTH_RATE_KEY].mean() < 0.2 and df_query['Streptomycin concentration (ug/ml)'].iloc[0] < 2

    logging.info("Completed add_is_debris_column.")




# b, a = scipy.signal.butter(2, 0.1, 'low') # needs 9 data-points
b, a = scipy.signal.butter(1, 0.1, 'low') # needs 6 data-points
BUTTER_MIN_DATAPOINTS = 6

def low_pass_filter_column(df, column):
    raw_column = f'{column}_raw'
    
    if raw_column not in df:
        df.rename(columns={column: raw_column}, inplace=True)

    df_q = df.query('colony_on_border == False') if ('colony_on_border' in df) else df
    df_q = df_q.dropna(subset=[raw_column])

    for _, df_group in df_q.groupby(SERIES_KEY): # get dataframe containing data for only one colony
        if len(df_group.index) <= BUTTER_MIN_DATAPOINTS: continue # too short to filter
        filtered_areas = scipy.signal.filtfilt(b, a, np.log2(df_group[raw_column])) # perform filtering in log-space
        df.loc[df_group.index, column] = np.exp2(filtered_areas) 

    # if no groups, then add area column back in
    if not column in df: df[column] = np.NaN


def single_fit_simple(xs, ys):
    fit = scipy.stats.linregress(xs, ys)
    return fit[0]

from .GrowthRateTools.fitderiv import fitderiv
import matplotlib.pyplot as plt

def add_growth_rate_time_series_columns(df, fit_to_key):
        
    for _, dfg in df.groupby(SERIES_KEY): # get dataframe containing data for only one colony
        
        if len(dfg.index) < 6: continue # too short to fit
        
        # prepare data vectors
        ts = np.array(dfg['time'])/60/60 # time in hours
        prevelance = np.array(dfg[fit_to_key]) # prevelance = areas for single cell imaging, and OD for liquid culture
        
        try:
            q = fitderiv(ts, prevelance, logs=True, stats=False, showstaterrors=False, warn=False)
            df.loc[dfg.index, f'{fit_to_key}_growth_rate'] = q.df
            df.loc[dfg.index, f'{fit_to_key}_growth_rate_var'] = q.dfvar
        except Exception as e:
            logging.debug(f'Failed to fit growth rate {len(ts)=} {len(prevelance)=} {e}')

        # # plot results
        # plt.figure()
        # plt.subplot(2,1,1)
        # q.plotfit('f', ylabel= 'log(OD)')
        # plt.subplot(2,1,2)
        # q.plotfit('df', ylabel= 'growth rate')
        # plt.show()


def add_replicate_column(df, keys):
    # for each combination of key values, look for number of pads
    # assign the pads a number from 1 to n, and add that to a new dataframe column

    id_key = 'pad_name'

    # combine keys 
    df['conditions'] = df.apply(lambda r: "_".join([str(x) for x in r[keys]]), axis=1)
    df_pads = df.drop_duplicates(subset=[id_key])

    pad_to_condition = {r[id_key]: r['conditions'] for i, r in df_pads.iterrows()}
    
    condition_to_pads = defaultdict(list)
    for pad, condition in pad_to_condition.items():
        condition_to_pads[condition].append(pad)
    
    pad_to_replicate = {}
    for condition, pads in condition_to_pads.items():
        for i, pad in enumerate(pads):
            pad_to_replicate[pad] = i+1 # start replicate at 1

    df.loc[:, 'replicate'] = df[id_key].map(pad_to_replicate)


def add_time_bin_column(df, interval: int): # interval in minutes

    # human readable range
    round_times = df['round_time'].unique()/60
    bins = list(range(int(np.min(round_times)), int(np.max(round_times)), interval))
    labels = [f'{a}-{b} min' for a,b in zip(bins, bins[1:])]
    df['time_bin'] = pd.cut(df['round_time']/60, bins, labels=labels, include_lowest=True, right=False)

    # float number, rounding down to nearest interval
    TIME_BIN_30 = 30 # minutes
    TIME_BIN_15 = 15 # minutes
    df['time_bin_30min'] = df['round_time_hours'].apply(lambda x: f'{(float(x)*60//TIME_BIN_30)*TIME_BIN_30/60}').astype(float)
    df['time_bin_15min'] = df['round_time_hours'].apply(lambda x: f'{(float(x)*60//TIME_BIN_15)*TIME_BIN_15/60}').astype(float)
    

def groupby(df, keys, **kwargs):
    if len(keys) == 0: return [('', df)]
    return df.groupby(keys, **kwargs)

def add_repeat_column(df, numeric_keys, text_keys, repeat_keys=None): # add a new number for each new experiment
    if repeat_keys == None: repeat_keys = ['experiment']

    for numeric_key in numeric_keys:
        # print(numeric_key)
        dfq = df.dropna(subset=[numeric_key])

        for _, dfg in groupby(dfq, text_keys, dropna=False):
            for i, (_, dfgg) in enumerate(dfg.groupby(repeat_keys, dropna=False)):
                df.loc[dfgg.index, 'repeat'] = i+1
                # print(i+1, len(dfgg.index))

    if not 'repeat' in df.columns: # if no numeric keys
        for _, dfg in groupby(df, text_keys, dropna=False):
            for i, (_, dfgg) in enumerate(dfg.groupby(repeat_keys, dropna=False)):
                df.loc[dfgg.index, 'repeat'] = i+1

    df['repeat'] = df['repeat'].astype('int')


# Add a new repeat column where rows are split into separte repeats.
# Half the control pads are distributed evenly across the repeats
def add_repeat_column_ignore_control(df):

    las_idx = 1
    for _, dfq in df.groupby('experiment'):
        
        # set main repeat count not taking control into account
        dfq_exp_rows = dfq.query('col != 12').groupby('row')
        for j, (_, dfg) in enumerate(dfq_exp_rows):
            df.loc[dfg.index, 'repeat'] = las_idx + j 

        for j, (_, dfg) in enumerate(dfq.query('col == 12').groupby('row')):
            df.loc[dfg.index, 'repeat'] = las_idx + j % len(dfq_exp_rows)
        
        las_idx = las_idx + len(dfq_exp_rows)
    
    df['repeat'] = df['repeat'].astype('int')

def add_normalized_column(df, key, numeric_keys, text_keys):
    for experiment, dfe in df.groupby('experiment'):
        for numeric_key in numeric_keys:
            for values, df_query in DataUtils.groupby(dfe, text_keys):
                free_mean_value = df_query.query(f'`{numeric_key}` == 0')[key].mean()
                df.loc[df_query.index, f'{key}_normalized'] = df_query[key]/free_mean_value


# normalize to mean value 
def add_normalized_to_self_column(df, key, numeric_keys, text_keys):
    for experiment, dfe in df.groupby('experiment'):
        for values, df_query in DataUtils.groupby(dfe, numeric_keys + text_keys):
            start_group = df_query.groupby('round_time_hours').agg({key: 'mean'})
            free_mean_value = start_group.iloc[1]
            df.loc[df_query.index, f'{key}_normalized'] = df_query[key]/float(free_mean_value)






def colony_agg_dataframe(df, numeric_keys, text_keys):

    dfs = []

    for label, df_fow in df.groupby('label'):

        # reduce ID list to only last generation colonies present at time cuttoff
        IDs = list(df_fow['colony_ID'].unique())
        parent_IDs = [
            ID_a for ID_a in IDs
            if not any([ID_a.startswith(ID_b) and ID_a != ID_b for ID_b in IDs])
        ]

        for ID in parent_IDs:

            df_fow
            a = df_fow['colony_ID'].str.startswith(f'{ID}.')
            b = (df_fow['colony_ID'] == ID)

            df_ID_group = df_fow.loc[np.logical_or(df_fow['colony_ID'].str.startswith(f'{ID}.'), (df_fow['colony_ID'] == ID))]
            
            df_t = df_ID_group.groupby('round_time_hours').agg({
                'colony_area': 'sum',
                'ss_area_total': 'sum',
                'ss_area_count': 'sum',
                'colony_ID': lambda _: ID,
                'colony_name': lambda _: ID,
                'colony_id': lambda _: ID,
                SERIES_KEY: lambda _: label + ID,
                'label': lambda _: label,
                'time': lambda x: x.iloc[0], 
                'experiment': lambda x: x.iloc[0],
                **{k: lambda x: x.iloc[0] for k in numeric_keys + text_keys},
            })
            
            dfs.append(df_t)
            

    df_o = pd.concat(dfs, axis=0)
    df_o.sort_values('time', inplace=True)
    df_o.reset_index(inplace=True)

    low_pass_filter_column(df=df_o, column='colony_area')
    low_pass_filter_column(df=df_o, column='ss_area_total')
    low_pass_filter_column(df=df_o, column='ss_area_count')

    add_growth_rate_time_series_columns(df=df_o, fit_to_key='colony_area')
    add_growth_rate_time_series_columns(df=df_o, fit_to_key='ss_area_total')
    add_growth_rate_time_series_columns(df=df_o, fit_to_key='ss_area_count')

    return df_o


def add_colony_lysis_columns(df):
    # Look for time when area for each colony is largest 
    # If that is not the last timestep, call that the termination time
    # Track the area at this time, in addition to the number of cells and area of individual cells
    # Do this for area based on colony envalope and single cell area

    end_time = df['round_time_hours'].max()

    for id, dfq in df.groupby(SERIES_KEY):
        
        max_index = dfq.index.max()
        min_index = dfq.index.min()

        # max_colony_area_idx = dfq['ss_area_total_raw'].idxmax()
        max_colony_area_idx = dfq['colony_area'].idxmax()
        df.loc[dfq.index, 'colony_area_idxmax'] = max_colony_area_idx
                
        cmax_at_eop = max_colony_area_idx == max_index
        present_at_end = dfq.loc[max_index, 'time_hours'] >= 4
        present_at_start = dfq.loc[min_index, 'time_hours'] < 0.2
        
        cmax_colony_area = dfq.loc[max_colony_area_idx, 'colony_area']
        cmax_cell_count = dfq.loc[max_colony_area_idx, 'ss_area_count']
        cmax_time = dfq.loc[max_colony_area_idx, 'round_time_hours']

        colony_undergoes_lysis = not cmax_at_eop
        
        if not (colony_undergoes_lysis and present_at_end): 
            # project growth and area
            mean_area_growth_rate = dfq['ss_area_total_growth_rate'].mean()
            mean_count_growth_rate = dfq['ss_area_count_growth_rate'].mean()

            cmax_colony_area = mean_area_growth_rate * (end_time - cmax_time) + cmax_colony_area
            cmax_cell_count = mean_count_growth_rate * (end_time - cmax_time) + cmax_colony_area
            cmax_time = end_time

        df.loc[dfq.index, 'cmax_colony_area'] = cmax_colony_area
        df.loc[dfq.index, 'cmax_cell_count'] = cmax_cell_count
        df.loc[dfq.index, 'cmax_time'] = cmax_time
        df.loc[dfq.index, 'cmax_at_eop'] = cmax_at_eop
        df.loc[dfq.index, 'present_at_end'] = present_at_end
        df.loc[dfq.index, 'present_at_start'] = present_at_start
        df.loc[dfq.index, 'Colony lysis'] = colony_undergoes_lysis

    # If child colonies lyse, mark parents for lysis as well
    df['labelID'] = df['label'].astype(str) + df['colony_ID'].astype(str)

    for label, dfg in df.groupby('label'): # we know no colonies will have ancestors on different pads
        idx_dfg_lysis = dfg.query('`Colony lysis` == True').groupby(SERIES_KEY, as_index=False)
        idx_dfg_no_lysis = dfg.query('`Colony lysis` == False').groupby(SERIES_KEY, as_index=False)

        # one row per column
        for indices_small, df_small in idx_dfg_no_lysis:
            #df_small_row =  # find all IDs that do not lyse - we are interested to see if they actually do lyse after merge
            for indices_merged, df_merged in idx_dfg_lysis:
                if df_merged.iloc[0]['labelID'] in df_small.iloc[0]['labelID'].split('.'):
                    df.loc[df_small.index, 'Colony lysis'] = True # must be true, as we are only looking for merged coloies that lyse


def add_present_for_duration_column(df):
    for _, d in df.groupby(SERIES_KEY):
        df.loc[d.index, 'present_for_duration'] = d['round_time_hours'].max() - d['round_time_hours'].min()

### Plot data from mutliple experiments at the same time

# def shared_graph_dir(experiments):
#     data_folder = config('OUTPUT_DIRECTORY')
#     experiment_name = '_'.join(experiments)
#     return os.path.join(data_folder, f'graphs_{PLOT_VERSION}', experiment_name)


import re

def load_platereader_df(filename, experiment, tlim):

    with open(filename, 'rb') as f:
        data = f.read()

    data = data.splitlines()[1].decode('utf-8')[:-1] # remove trailing character
    # print(data)

    ds = [s.strip() for s in data.split(' ') if s]

    last_was_text = False
    title = ''

    output = {}

    for d in ds:
        is_text = re.match(r'^-?\d+(?:\.\d+)$', d) == None

        if last_was_text and is_text: continue # skip double text

        if is_text:
            last_was_text = True
            title = d
            output[title] = []
            continue

        last_was_text = False
        output[title].append(float(d))

    times = output['t']
    temperature = output['C']

    output.pop('t')
    output.pop('C')

    print(f'max_time = {times[tlim]/60/60:.2f} hours')

    plt.figure(figsize=(12,10), dpi=300)

    dicts = []

    for k, v in output.items():
        row = k[0]
        col = int(k[1:])
        ri = ord(row) - ord('A')
        ci = col - 1

        for t0,v0 in zip(times[:tlim], v):
            dicts.append({
                'experiment': experiment,
                'time_days': t0/60/60/24,
                'round_time_days': t0/60/60/24,
                'round_time': t0,
                'round_time_hours': t0/60/60,
                'time': t0,
                'row': row,
                'col': col,
                'OD600': v0,
                'pad_name': f'{row}{col}',
                'labelid': f'{experiment}_{row}{col}'
            })

        ROWS = 16
        COLS = 24
        i = COLS*ri+ci+1
        # print(ri, ci, i)
        plt.subplot(ROWS, COLS, i)
        plt.plot(times[:tlim], v[:tlim])
        plt.gca().axes.get_xaxis().set_visible(False)
        plt.gca().axes.get_yaxis().set_visible(False)
        plt.ylim((0,2))
        

    # make dataframe from list of dicts
    return pd.DataFrame.from_records(dicts)

    # for experiment, dfe in df.groupby('experiment'):
    #     for values, df_query in dfe.groupby('labelid'):
    #         start_group = df_query.groupby('round_time_hours').agg({'OD600': 'mean'})
    #         free_mean_value = start_group.iloc[1]
    #         df.loc[df_query.index, f'OD600'] = df_query['OD600']