
import json, os
import argparse
import numpy as np
import pandas as pd
import scipy.stats
import matplotlib.pyplot as plt
import seaborn as sns
import MKUtils

import colorcet as cc
COLOR_PALETTE = cc.glasbey + cc.glasbey_dark + cc.glasbey_light + cc.glasbey_bw + cc.glasbey_hv # 256*5=1280 distinct colors in scrambeled order

def color_for_number(number):
    return COLOR_PALETTE[number%len(COLOR_PALETTE)]

def load_text(file):
    with open(file, 'r') as f:
        return f.read()


# either pass expeirment directory with traililng / or path to .movie file
def load_files_in_folder(input_folder):
    
    files = [n for n in os.listdir(input_folder) if '.json' in n]

    data = [load_text(os.path.join(input_folder, f)) for f in files]
    
    ds = []
    for d, f in zip(data, files):
        try:
            ds.append(json.loads(d))
        except Exception as e:
            print(f'Could not load {f}.', e)

    # data = [json.loads(d) for d in data]
    return ds
    

def extract_from_data(data, key, id):
    return np.array(data[key][id])

def data_to_df(input_folder, debug_folder, points=None, debug=True):

    pads_data = load_files_in_folder(input_folder) # array of data from all pads
    pad_names = np.array(points[:,-1]) if points else [d['pad_name'] for d in pads_data]
    pads = [Pad(p) for p in pad_names]
    
    data_colony_vector_keys = [k for k in pads_data[0].keys() if 'colony_' in k]

    # vector with single value properties for each colony

    df = pd.DataFrame()#columns=['pad', 'row', 'col', 'colony_id' 'doubling_time', 'growth_rate', 'tracking_count', 'tracking_time', 'initial_area', 'final_area'])

    # pad = A1
    # row = A
    # col = 1
    # colony id = unique id to each colony on each pad
    # stats...

    # get data into table

    for data, pad in zip(pads_data, pads):

        times = data['times'] # timesteps for data
        counts = data['counts'] # number of colonies found for each timestep

        colony_ids = data[data_colony_vector_keys[0]].keys()

        doubling_times = []

        # colony
        for colony_id in colony_ids:

            areas = extract_from_data(data=data, key='colony_areas', id=colony_id)

            normalized_areas = areas/areas[0]

            if len(normalized_areas) > 10:
                ts = times[:len(normalized_areas)]
                fit = scipy.stats.linregress(ts, np.log2(normalized_areas))
                doubling_time = 1/fit[0]/60 if fit else -1 # in minutes
            else:
                doubling_time = -1

            growth_rate = np.log(2)/doubling_time # mass added per minute

            table_data = {
                'pad': pad.name,
                'row': pad.row,
                'col': pad.col,
                'colony_id': colony_id,
                'doubling_time': doubling_time,
                'growth_rate': growth_rate,
                'tracking_count': len(areas),
                'tracking_time': times[len(areas)-1],
                'initial_area': areas[0],
                'final_area': areas[-1],
                # cannot determine division time from colony statistics!
            }

            df = df.append(pd.Series(table_data), ignore_index=True)
    
            if debug:
                doubling_times.append(doubling_time)
                plt.plot(times[:len(normalized_areas)], normalized_areas, '-', label=colony_id)

        if debug:
            # plot growth-rate trendlines
            #min_dt = np.min(doubling_times)
            #max_dt = np.max(doubling_times)
            #mean_dt = np.mean(doubling_times)
            
            #x_vals = np.array(plt.xlim())
            # plt.plot(x_vals, 2**(x_vals/min_dt), '--', label=f'$T_{{d, \\textnormal{{ min}}}}$={min_dt:.0f} min', color='0')
            # plt.plot(x_vals, 2**(x_vals/mean_dt), '-.', label=f'$T_{{d, \\textnormal{{ mean}}}}$={mean_dt:.0f} min', color='0')
            # plt.plot(x_vals, 2**(x_vals/max_dt), ':', label=f'$T_{{d, \\textnormal{{ max}}}}$={max_dt:.0f} min', color='0')

            plt.title(f'Colony growth for {pad.name}')
            plt.xlabel('Time (min)')
            plt.ylabel('Normalized colony area ($A$/$A_0$)')
            # plt.legend()
            plt.yscale('log', base=2)

            plt.savefig(os.path.join(debug_folder, f'area_time_plot_{pad.name}.jpeg'))

    return df



def plot_boxplot(filename, df, y_label, group_cols, x_tick_labels=None):
    plt.figure(figsize=(8, 5), dpi=300)

    sns.set(style="whitegrid")

    x_label = 'col' if group_cols else 'row'

    sns.boxplot(x=x_label, y=y_label, data=df, showfliers = False, linewidth=1, color='0.9')
    sns.swarmplot(x=x_label, y=y_label, hue='row', data=df)

    if x_tick_labels:
        plt.gca().set_xticklabels(labels=x_tick_labels, rotation=0)

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.savefig(filename)



    # itterating over set sorts it, col in sorted order as well
    # for index, (col, name) in enumerate(zip(set(point_cols), col_names)):
    #     cols = point_cols == col
        
    #     plt.figure()
    #     growth_rates = []
    #     growth_rates_labels = []
    #     labels = set()

    #     # print(f'{index=}, {col=}, {name=}')
    #     for i in np.array(range(len(cols)))[cols]:
    #         # print(f'{i=}')
    #         if not len(data) > i:
    #             print(f'Data for {i} missing.')
    #             continue

    #         d = data[i]
    #         n = point_names[i]
    #         row = n[:1]
    #         color = np.array(color_for_number(ord(row)))/255.

    #         first = True

    #         times = np.array(d['times'])/60
            
    #         for key, area in d['colony_normalized_areas'].items():
    #             if len(area) > 100:
    #                 labels.add(row)
    #                 area = area[:len(times)]
    #                 times = times[:len(area)]

    #                 fit = scipy.stats.linregress(times, np.log2(area))
    #                 if fit != 0:
    #                     colony_doudoubling_time = 1/fit[0] # dubling times in minutes
    #                 else:
    #                     colony_doudoubling_time = -1

    #                 # print(colony_doudoubling_time)

    #                 if 10 < colony_doudoubling_time < 60*5:
    #                     growth_rates.append(colony_doudoubling_time)
    #                     growth_rates_labels.append(row)
    #                     i = 0 if pd.isnull(grdf.index.max()) else grdf.index.max() + 1
    #                     grdf.loc[i] = [row, col, colony_doudoubling_time]

    #                     if first: # only one legend for each color
    #                         plt.plot(times, area, '-', color=color, label=row)
    #                         first = False
    #                     else:
    #                         plt.plot(times, area, '-', color=color)
        
    #     all_growth_rates.append(growth_rates)
    #     all_growth_rates_labels.append(growth_rates_labels)

    #     # plot growth-rate trendlines
    #     min_gr = np.min(growth_rates)
    #     max_gr = np.max(growth_rates)
    #     mean_gr = np.mean(growth_rates)
        
    #     x_vals = np.array(plt.xlim())
    #     plt.plot(x_vals, 2**(x_vals/min_gr), '--', label=f'min, {min_gr:.0f} min')
    #     plt.plot(x_vals, 2**(x_vals/mean_gr), '--', label=f'mean, {mean_gr:.0f} min')
    #     plt.plot(x_vals, 2**(x_vals/max_gr), '--', label=f'max, {max_gr:.0f} min')

    #     plt.title(f'Colony growth for {name}')
    #     plt.xlabel('Time (min)')
    #     plt.ylabel('Noemalized colony area ($A$/$A_0$)')
    #     #plt.labels(set(point_rows[cols]))
    #     plt.legend()
    #     plt.yscale('log', base=2)

    #     plt.savefig(os.path.join(plot_folder, f'Growth {name}.png'), dpi=300)



def plot(filename):

    experiment_dir = os.path.dirname(filename)
    input_folder = os.path.join(experiment_dir, 'output')
    plot_folder = os.path.join(experiment_dir, 'plot')
    plot_debug_folder = os.path.join(plot_folder, 'debug')

    MKUtils.generate_directory(plot_folder)
    MKUtils.generate_directory(plot_debug_folder)

    df = data_to_df(input_folder, plot_debug_folder)
    
    if len(df) == 0:
        print('no data, skipping final plot')

    # filter colonies with too few points
    filter = df['tracking_count'] > 20
    df = df[filter]

    try:
        plot_boxplot(filename=os.path.join(plot_folder, 'doubling_time_rows'), df=df, y_label='doubling_time', group_cols=False, x_tick_labels=None)
        plot_boxplot(filename=os.path.join(plot_folder, 'doubling_time_cols'), df=df, y_label='doubling_time', group_cols=True, x_tick_labels=None)
    except Exception as e:
        print('Error making boxplots. ', e)


def main():

    # 0. Load input arguments from user.

    parser = argparse.ArgumentParser(description='Extract information from .movie files.')
    parser.add_argument('movie', type=str,
                        help='Filename for source .movie file.')
    args = parser.parse_args()

    plot(filename=args.movie)


if __name__ == '__main__':
    main()