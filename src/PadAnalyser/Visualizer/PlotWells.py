# Plotting functions for platereader data

from decouple import config
import argparse

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os, sys, json
import logging
import glob

import MKUtils

sys.path.insert(1, os.getcwd())
from MKImageAnalysis import PlotPads, MKAnalysisUtils, Plotter


def read_platereader_data(filename):

    df = pd.read_csv(filename, header=6)
    df = df.rename(columns={'Unnamed: 0': 'row', 'Unnamed: 1': 'col'})
    df['Time'] = df['row'] + df['col'].astype(str)
    # df.set_index('Well', inplace=True)
    df.drop('row', inplace=True, axis=1)
    df.drop('col', inplace=True, axis=1)
    df.set_index('Time', inplace=True)
    df = df.T

    def string_to_min(s):
        parts = s.split(' ')
        t = int(parts[0]) * 60
        if len(parts) == 4: t += int(parts[2])
        return t

    df['Time (min)'] = df.index.map(string_to_min)
    df.set_index('Time (min)', inplace=True)
    return df


def ploting_df_from_platereader_df(df, experiment):
    df2 = pd.DataFrame()
    for time, row in df.iterrows():
        for well, value in zip(row.keys(), row.values):
            # m = mapping[well]

            row = {
                'row': well[0],
                'col': int(well[1:]),
                'Well': well,
                'pad_name': well, # to be compatible with pad plotting code
                'Time (min)': time, # in minutes
                'time': time*60.0, # in seconds, as float
                'time_hours': time/60, # in hours
                'time_days': time/60/24, # in days
                'round_time_days': time/60/24, 
                'experiment': experiment,
                'OD600': value,
            }

            if df2.empty:
                df2 = pd.DataFrame(columns=row.keys()) # add column names first round
            
            df2.loc[df2.shape[0]] = row.values()
    return df2


def plot_vs_time(df, ykey, ylabel, title, numeric_key, output_folder):

    filename = title
    plt.figure()
    PlotPads.setup_figure()
    PlotPads.setup_x_as_time_axis()
    sns.lineplot(data=df, x='time_days', y=ykey, hue=numeric_key, legend='full')
    plt.ylabel(ylabel)
    plt.title(title)

    legend = plt.legend(title=numeric_key, bbox_to_anchor=(1, 0.5), loc='center left', frameon=False)

    plt.savefig(
        os.path.join(output_folder, MKAnalysisUtils.sanetize_filename(f'{filename}.png')), 
        bbox_extra_artists=[legend], 
        bbox_inches='tight',
    )
    plt.close()


def load_platereader_dataframe(experiment_map, input_file, experiment):
    df_raw = read_platereader_data(input_file)
    df = ploting_df_from_platereader_df(df_raw, experiment)
    
    numeric_keys, text_keys = Plotter.find_numeric_and_text_keys(experiment_map=experiment_map)
    
    Plotter.add_label_columns(df, experiment_map=experiment_map)
    Plotter.add_replicate_column(df=df, keys=numeric_keys+text_keys)
    Plotter.low_pass_filter_column(df=df, column='OD600')
    Plotter.add_growth_rate_time_series_columns(df=df, fit_to_key='OD600')

    return df

# Produce standardized dataframe that can be compared across experiments.
# time in hours
# def df_from_reduced_property(df, t_start, t_end, series_key, property, copy_keys):
    
#     df_time = df
#     if t_start: df_time = df_time.query(f'{t_start} <= time_hours')
#     if t_end: df_time = df_time.query(f'time_hours <= {t_end}')

#     copy_keys = [k for k in copy_keys if k] # remove none
    
#     df = pd.DataFrame() # overload df from argument - not needed anymore
#     for _, df_query in dfs_for_unique(df_time, series_key):
#         r = df_query.iloc[0][['time_days', 'round_time_days', 'experiment', series_key, 'row', 'col', *copy_keys]]
#         property_mean = df_query[property].mean()
        
#         # if math.isnan(property_mean): continue # only add rows where mean is not none
        
#         row = {
#             property: property_mean,
#             **r.to_dict(),
#         }
#         if df.empty: df = pd.DataFrame(columns=row.keys()) # add column names first round
#         df.loc[df.shape[0]] = row.values()
    
#     return df


def make_plots(experiment, experiment_map, input_file, output_folder):
 
    # numeric_keys -> use options for kiven key as series/x-axis
    # text_keys -> produce different garphs for different string values for these keys
    numeric_keys, text_keys = Plotter.find_numeric_and_text_keys(experiment_map=experiment_map)
    logging.info(f'{numeric_keys=}, {text_keys=}')

    plot_options = experiment_map.get('plot_options', {})
    experiment_info = experiment_map['experiments'][experiment]
    experiment_category = experiment_info.get('experiment_category', 'Unknown')
    instrument = experiment_info.get('instrument', 'Unknown')
    t_start, t_end = experiment_info.get('exponential_time_hours', [1, 2]) # in hours

    df = load_platereader_dataframe(
        experiment_map=experiment_map,
        input_file=input_file,
        experiment=experiment,
    )


    for val, df_query in df.groupby(text_keys):
        for numeric_key in numeric_keys:
            
            condition_text = f'{numeric_key}, {", ".join(val)}'

            plot_vs_time(
                df=df_query,
                ykey='OD600',
                ylabel='Optical density (OD600)',
                title=f'OD600 ({condition_text})',
                numeric_key=numeric_key,
                output_folder=output_folder,
            )
            
            plot_vs_time(
                df=df_query, 
                ykey='colony_growth_rates',
                ylabel='Growth rates',
                title=f'Growth rates ({condition_text})',
                numeric_key=numeric_key, 
                output_folder=output_folder,
            )

    
    Plotter.generate_growth_rate_stats_df(
        df=df,
        numeric_keys=numeric_keys, 
        text_keys=text_keys, 
        t_start=t_start,
        t_end=t_end,
        series_key='Well',
        ykey='colony_growth_rates',
        experiment=experiment, 
        experiment_category=experiment_category, 
        instrument=instrument,
        filename=f'{experiment} growth rate stats', 
        output_folder=output_folder,
    )


def main(experiment, **kwargs):

    # Collect environment variables
    experiment = experiment
    data_folder = config('OUTPUT_DIRECTORY') # '/Users/mkals/Library/CloudStorage/OneDrive-UniversityofCambridge/data/'
    input_points_directory = config('INPUT_POINTS_DIRECTORY')
    mapping_file = os.path.join(input_points_directory, 'BE_condition_map.json') # '/Users/mkals/Developer/Cambridge/BacterialImaging/point_sets/BE_condition_map.json'
    
    input_folder = os.path.join(data_folder, experiment)
    input_file = os.path.join(input_folder, f'{experiment}.csv')

    output_folder = os.path.join(input_folder, 'graphs')

    # Configure logging
    logging_file = os.path.join(data_folder, experiment, 'PlotWells.log')
    logging.basicConfig(
        filename=logging_file, 
        encoding='utf-8', 
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        filemode='w',
    )
    
    # Load experiment map
    with open(mapping_file, 'r') as f:
        experiment_maps = json.load(f)
    experiment_map = Plotter.mapping_for_experiment(experiment, experiment_maps)

    MKUtils.generate_directory(output_folder)
    for f in glob.glob(os.path.join(output_folder, '*')):
        try:
            os.remove(f)
        except:
            logging.exception(f'Could not remove file {f}.')

    # Perform plotting
    logging.info(f'Generating plots for {experiment}.')
    make_plots(
        experiment=experiment,
        experiment_map=experiment_map,
        input_file=input_file,
        output_folder=output_folder,
    )
    logging.info(f'Completed generating plots for {experiment}.')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Extract information from .movie files.')
    parser.add_argument('experiment', type=str, help='Experiment name.')
    # parser.add_argument('--no_cache', '-n', type=bool, default=False, const=True, nargs='?', help='Do not used cached dataframes.')
    args = parser.parse_args()

    main(experiment=args.experiment)