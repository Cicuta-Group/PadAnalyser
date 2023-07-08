import os, glob
import MKUtils
import matplotlib.dates as mdates

from decouple import config
import argparse

from matplotlib.ticker import FuncFormatter
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import math
from typing import List # for type hints

from scipy import optimize

import matplotlib
matplotlib.use("Agg") # to avoid "failed to allocate bitmap" error
matplotlib.rcParams['svg.fonttype'] = 'none'
plt.rcParams['legend.loc'] = 'center left'

import sigfig

from MKImageAnalysis import MKAnalysisUtils, DataProcessor, DataUtils, Plotter, PLOT_VERSION, experiment_folder_name

import logging
from multiprocessing import pool

# import warnings
# warnings.filterwarnings("ignore", category=DeprecationWarning) 
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
warnings.filterwarnings('ignore')


# goal: as simple as possible, plotting of a set of standard figures for sample platform
# For more complex setups, go to manual plotting for now, can be standardised later. 

SERIES_KEY = 'labelid'

# Three options to measure growth rate by:
# COLONY_GROWTH_RATE_KEY = 'colony_area_growth_rate'
COLONY_GROWTH_RATE_KEY = 'ss_area_total_growth_rate'
# COLONY_GROWTH_RATE_KEY = 'ss_area_count_growth_rate'


PRESENTATION = True

def experiment_description(experiments):

    experiment = experiments[0] 

    # load experiment map
    experiment_map = {} # TODO: dynamically get experiment map

    strings = [
        "Bacteria: {0}".format(experiment_map.get('bacteria', '$\it{E. coli}$ K-12 MG1655')),
        "Growth media: {0}".format(experiment_map.get('growth_media', 'LB Broth (Miller)')),
        "Pad: {0}".format(experiment_map.get('pad_setup', '0.8% w/v Agarose')),
        "Incubation temperature: {0}˙C".format(experiment_map.get('incubation_temperature', '37')),
    ]
    return '\n'.join(strings)

def save_figure(filename, output_folder, figure = None, live=True, bbox_extra_artists=None):
    
    if filename == None or output_folder == None: return
    if bbox_extra_artists == None: bbox_extra_artists = []
    if figure == None: figure = plt.gcf()

    output_folder = output_folder.replace(' ', '_')

    if not os.path.exists(output_folder): os.makedirs(output_folder)
    filepath = os.path.join(output_folder, MKAnalysisUtils.sanetize_filename(filename))

    plt.rcParams['svg.fonttype'] = 'none'
    plt.figure(figure.number)
    plt.savefig(
        filepath + '.png', 
        bbox_extra_artists=bbox_extra_artists,
        bbox_inches='tight',
        dpi=300,
    )
    plt.savefig(
        filepath + '.svg',
        bbox_extra_artists=bbox_extra_artists,
        bbox_inches='tight',
        format='svg',
    )
    
    print(f'Saved figure "{filepath}.svg"')
    if not live: plt.close(fig=figure)



#### Plot generation funcitons ####

def setup_figure():
    if PRESENTATION:
        plt.gcf().set_size_inches(6, 3.6)
        plt.gcf().set_dpi(600)
    else:
        plt.gcf().set_size_inches(10, 6)
        plt.gcf().set_dpi(300)
        sns.reset_defaults()
    # sns.set_theme()

def setup_x_as_time_axis():
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H'))
    plt.gca().xaxis.set_major_locator(mdates.HourLocator())
    plt.gca().xaxis.set_minor_locator(mdates.MinuteLocator(byminute=[15,30,45]))
    plt.gca().set_xlim(left=0)
    plt.xlabel('Time since experiment start (hour)')
    plt.gca().grid(visible=True, which='minor', color='w', linewidth=0.5)


def plot_boxplot(df, output_folder, hue_label, ykey, xkey, ylabel, swarm_hue_label, title, xorder=None, ylim=None, set_xticklabels=True, swarmplot_dot_size=2, legend_location='upper right', presentation=True, live=False):

    sns.reset_defaults()
    sns.reset_orig()

    experiments = df['experiment']
    dfq = df.dropna(subset=[xkey]) # remove NAN values on x-axis
    
    # ensure bars appear in right order
    if xorder is None:
        xorder = list(dfq[xkey].unique())
        xorder.sort()
        if 'None' in xorder: 
            xorder.remove('None')
            xorder.insert(0, 'None') # put None at the start

    if swarm_hue_label:
        color = 'lightgray'
        sns.set_palette("rocket", n_colors=len(dfq[swarm_hue_label].unique()))
    else:
        color = None

    plt.figure()
    if hue_label:
        sns.set_palette("rocket", n_colors=len(dfq[hue_label].unique()))
    else:
        sns.set_palette("rocket_r", n_colors=len(dfq[xkey].unique()))

    sns.boxplot(x=xkey, order=xorder, y=ykey, hue=hue_label, data=dfq, showfliers=False, color=color, linewidth=1) # color=0.9

    if swarm_hue_label: # too much to have this and mutliple hues for box-plot
        df_g = dfq.groupby(SERIES_KEY).agg({
            ykey: 'mean',
            **{k: lambda x: x.iloc[0] for k in [xkey, hue_label, swarm_hue_label, SERIES_KEY, 'colony_name', 'label'] if k in df.columns},
        })
        sns.swarmplot(x=xkey, order=xorder, y=ykey, hue=swarm_hue_label, legend="auto", data=df_g.sort_values(by=swarm_hue_label), size=swarmplot_dot_size, dodge=True) #"row"
    
    setup_figure()
    legend_title = swarm_hue_label or hue_label or ''
    if legend_title == 'time_bin': legend_title = 'Time'
    if presentation:
        legend = plt.legend(title=legend_title.capitalize(), loc=legend_location)
    else:
        legend = plt.legend(title=legend_title.capitalize(), bbox_to_anchor=(1, 0.5), loc='center left', frameon=False)

    if set_xticklabels: plt.gca().set_xticklabels([DataUtils.to_str(text.get_text()) for text in plt.gca().get_xticklabels()])
    
    plt.xticks(rotation=90)
    plt.title(title)
    plt.ylabel(ylabel)

    if ylim: plt.ylim(ylim)

    # plt.yscale('log')
    # plt.ylim([0, 3000])

    # if not PRESENTATION:
        # b = plt.gca().text(1.03, 0.0, experiment_description(experiments), transform=plt.gca().transAxes, verticalalignment='bottom', fontsize='small')
        # _ = plt.gca().text(1.03, -0.1, ', '.join(experiments), transform=plt.gca().transAxes, verticalalignment='top', fontsize='x-small', color='0.6')
    
    save_figure(
        figure=plt.gcf(),
        filename=title,
        bbox_extra_artists=[legend],# if PRESENTATION else [legend, b],
        output_folder=output_folder,
        live=live,
    )


def plot_param_over_time(df, ykey, ylabel, title, log2, ylim_zero, plot_mean, output_folder, live=False):
    
    filename = title

    # df_c = df_c.sort_values(['label', 'id'], ascending=False, inplace=False)
    label_order = df['label'].unique().sort()

    fig, axs = plt.subplots(2, 1, figsize=(15, 5), sharex=True, gridspec_kw={'height_ratios': [3, 1]})

    plt.axes(axs[0])
    
    # plot data
    sns.lineplot(data=df, x='time_days', y=ykey, hue='label', hue_order=label_order, style='id', legend='full', zorder=10, err_kws={'edgecolor': 'none'})
    sns.scatterplot(data=df.query(f'colony_on_border_start == True'), x='time_days', y=ykey, marker='o', color='k', legend=None, zorder=20)
    sns.scatterplot(data=df.query(f'colony_has_dissapeared == True'), x='time_days', y=ykey, marker='X', color='k', legend=None, zorder=21)
    if plot_mean: sns.lineplot(data=df, x='round_time_days', y=ykey, ci='sd', palette="muted", legend=None, zorder=22, err_kws={'edgecolor': 'none'})
    
    # To check colony areas after they are on_edge
    # g = sns.relplot(data=df_c, kind='line', x='time_hours', y='colony_area', hue='label', style='id', size='colony_on_border', size_order=[True, False], legend='full')

    # update legend
    handles, labels = plt.gca().get_legend_handles_labels()
    handles = [h for l, h in zip(labels, handles) if 'BE' in l]
    labels = [DataUtils.pad_label_to_string(l) for l in labels if 'BE' in l]
    
    legend_0 = plt.gca().get_legend() # only remove legend after, else empty handles list
    if legend_0: legend_0.remove()
    legend = plt.legend(handles=handles, labels=labels, bbox_to_anchor=(1, 0.5), loc='center left', frameon=False)
    
    if log2: plt.yscale('log', base=2)
    if ylim_zero: plt.gca().set_ylim(bottom=0)

    setup_figure()
    setup_x_as_time_axis()
    
    plt.title(title)
    plt.ylabel(ylabel)
    
    plt.axes(axs[1])
    ts = df['round_time_days'].unique()
    ts.sort()
    colony_counts = [len(df.query(f'round_time_days == {t}').dropna(subset=[ykey]).index) for t in ts]
    plt.plot(ts, colony_counts)
    setup_figure()
    setup_x_as_time_axis()
    plt.ylabel('Colony count')
    
    save_figure(
        figure=plt.gcf(),
        filename=filename,
        bbox_extra_artists=[legend],
        output_folder=output_folder,
        live=live,
    )
    


def plot_overview_over_time(df, series_label, title, ykey, ylabel, style_label, log2, ylim_zero, output_folder, live=False):
    # df.sort_values([series_label], ascending=False, inplace=True)

    series_order = df[series_label].unique().sort()

    xkey = 'time_bin_30min' # 'round_time_days'

    plt.figure()
    setup_figure()
    # setup_x_as_time_axis()

    sns.lineplot(data=df, x=xkey, y=ykey, hue=series_label, hue_order=series_order, style=style_label, ci='sd', palette="muted", err_kws={'edgecolor': 'none'}) # ci='sd'


    # update legend
    handles, labels = plt.gca().get_legend_handles_labels()
    labels = [DataUtils.to_str(l) for l in labels]
    
    legend_0 = plt.gca().get_legend() # only remove legend after, else empty handles list
    if legend_0: legend_0.remove()
    if PRESENTATION:
        legend = plt.legend(handles=handles, labels=labels, loc='upper right')
    else:
        legend = plt.legend(handles=handles, labels=labels, bbox_to_anchor=(1, 0.5), loc='center left', frameon=False)
    
    # plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel('Time (hours)')

    if log2: plt.yscale('log', base=2)
    if ylim_zero: plt.gca().set_ylim(bottom=0)

    save_figure(
        figure=plt.gcf(),
        filename=title,
        bbox_extra_artists=[legend],
        output_folder=output_folder,
        live=live,
    )


def plot_areas_for_numeric_key(df, numeric_key, title, xkey, ykey, ylabel, output_folder, xlim = None, for_presentation=False, is_od=False, figsize=(6,6), live=False):

    colors = sns.color_palette('rocket', n_colors=len(df[numeric_key].dropna().unique()))
    colors.reverse()

    sns.reset_defaults()
    fig, (ax0, ax1) = plt.subplots(nrows=2, sharex=True, gridspec_kw={'height_ratios': [1, 1], 'hspace': 0.04})

    if for_presentation:
        plt.gcf().set_size_inches(*figsize)
        plt.gcf().set_dpi(300)
    else:
        plt.gcf().set_size_inches(10, 10)
        plt.gcf().set_dpi(300)

    # round numeric key to 2 sigfigs in legend
    df = df.copy()
    df[numeric_key] = df[numeric_key].apply(lambda x: sigfig.round(x, sigfigs=2, type=str))
    
    hue_order = df[numeric_key].unique()
    hue_order = sorted(hue_order, key=lambda x: float(x))

    plt.axes(ax0)
    sns.lineplot(data=df, x=xkey, y=ykey, hue=numeric_key, marker='.', markeredgewidth=0, hue_order=hue_order, palette=colors, ci='sd', legend='full', err_kws={'edgecolor': 'none'})
    

    if not is_od:
        # only keep last entry by xkey
        dfg = df.groupby([numeric_key, xkey], as_index=False).mean()
        for v, dfgg in dfg.groupby(numeric_key):
            dfg.loc[dfgg.index, 'is_last'] = dfgg[xkey] == dfgg[xkey].max()
        dfgq = dfg.query(f'`is_last` == True')
        sns.scatterplot(data=dfgq, x=xkey, y=ykey, hue=numeric_key, hue_order=hue_order, palette=colors, legend=None, zorder=22)

    if for_presentation:
        # l = plt.legend(loc='upper left', title=numeric_key, ncol=3)
        l = plt.legend(
            loc="lower center",
            bbox_to_anchor=(.5, 1), 
            title=numeric_key, 
            ncol=4,
            frameon=False,
        )
    else:
        l = plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., title=numeric_key)
    
    plt.ylabel(ylabel)
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: sigfig.round(x, sigfigs=1, type=str)))
    
    if not is_od:
        plt.yscale('log')
        plt.ylim((2, 10000))

    if xlim != None: plt.xlim((0,xlim))

    plt.axes(ax1)
    sns.lineplot(data=df, x=xkey, y=f'{ykey}_growth_rate', hue=numeric_key, hue_order=hue_order, palette=colors, ci='sd', marker='.', markeredgewidth=0, legend=None, err_kws={'edgecolor': 'none'})
    plt.ylabel('Colony growth rate (per hour)')
    plt.xlabel('Time (hours)')
    if not is_od:
        plt.ylim((-1, 2.2))
    if xlim != None: plt.xlim((0,xlim))

    if not is_od:
        sns.scatterplot(data=dfgq, x=xkey, y=f'{ykey}_growth_rate', hue=numeric_key, hue_order=hue_order, palette=colors, legend=None, zorder=22)

    save_figure(
        figure=plt.gcf(),
        filename=title,
        output_folder=output_folder,
        bbox_extra_artists=(l,),
        live=live,
    )

    return ax0, ax1



def plot_lineplot_for_numeric_key(df, numeric_key, title, xkey, ykey, ylabel, output_folder, style_key=None, xlim=None, ylim=None, logy_base=None, figsize=(6,6), live=False):

    colors = sns.color_palette('rocket', n_colors=len(df[numeric_key].dropna().unique()))
    colors.reverse()

    sns.reset_defaults()
    plt.figure(figsize=figsize, dpi=300)

    # round numeric key to 2 sigfigs in legend
    df = df.copy()
    df[numeric_key] = df[numeric_key].apply(lambda x: sigfig.round(x, sigfigs=2, type=str))
    
    hue_order = df[numeric_key].unique()
    hue_order = sorted(hue_order, key=lambda x: float(x))

    sns.lineplot(data=df, x=xkey, y=ykey, hue=numeric_key, hue_order=hue_order, style=style_key, marker='.', markeredgewidth=0, palette=colors, ci='sd', legend='full', err_kws={'edgecolor': 'none'})

    # only keep last entry by xkey
    dfg = df.groupby([numeric_key, xkey], as_index=False).mean()
    for v, dfgg in dfg.groupby(numeric_key):
        dfg.loc[dfgg.index, 'is_last'] = dfgg[xkey] == dfgg[xkey].max()
    dfgq = dfg.query(f'`is_last` == True')
    sns.scatterplot(data=dfgq, x=xkey, y=ykey, hue=numeric_key, hue_order=hue_order, palette=colors, legend=None, zorder=22)

    l = plt.legend(
        loc="lower center",
        bbox_to_anchor=(.5, 1), 
        title=numeric_key, 
        ncol=4,
        frameon=False,
    )
    
    plt.ylabel(ylabel)
    plt.xlabel('Time (hours)')
    
    if ylim != None: plt.ylim(ylim)
    if xlim != None: plt.xlim(xlim)

    if logy_base != None: 
        plt.yscale('log', base=logy_base)
    else:
        plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: sigfig.round(x, sigfigs=1, type=str)))

    save_figure(
        figure=plt.gcf(),
        filename=title,
        output_folder=output_folder,
        bbox_extra_artists=(l,),
        live=live,
    )



# Small function to produce count plots before and after filtering
def plot_counts(df, df_raw, cell_count, output_folder, live=False):

    if cell_count:
        ylabel = 'Cell count'
        ykey = 'cell_count'
    else:
        ylabel = 'Colony count'
        ykey = 'colony_count'

    fig = plt.figure()
    setup_figure()

    gs = fig.add_gridspec(3, hspace=0)
    axs = gs.subplots(sharex=True, sharey=False)

    experiments = df['experiment'].unique()
    
    colors = sns.color_palette("rocket", 2)

    for experiment, ax in zip(experiments, axs):
        
        # set ax as current
        plt.axes(ax)

        # plot number of colonies/cells per timestep
        dfs = df.query(f'experiment == "{experiment}"').groupby(['round_time']).agg(
            cell_count = ('ss_area_count', 'sum'),
            colony_count = ('colony_area', 'count'),
        )
        dfs.index = dfs.index/60 # seconds to minutes
        
        df_raws = df_raw.query(f'experiment == "{experiment}"').groupby(['round_time']).agg(
            cell_count = ('ss_area_count', 'sum'),
            colony_count = ('colony_area', 'count'),
        )
        df_raws.index = df_raws.index/60 # seconds to minutes

        plt.plot(dfs.index, dfs[ykey], 'o-', color=colors[0], label='Filtered')
        plt.plot(df_raws.index, df_raws[ykey], 'o-', color=colors[1], label='Raw')
        plt.xlabel('Time (min)'); plt.ylabel(ylabel)
        plt.title('Counts in experiment ' + experiment)
        plt.legend()
        # if not last_ax: last_ax = ax
        # else: 
        #     ax.sharex(last_ax)
        #     ax.set_xticklabels([])

    # share x axis
    plt.gca().set_ylim(bottom=0)
    plt.gca().set_ylim(top=dfs[ykey].max()*1.1)


    save_figure(
        figure=plt.gcf(),
        filename=ylabel,
        output_folder=output_folder,
        bbox_extra_artists=[],
        live=live,
    )



def plot_is_cell_score_filter(df, output_folder, numeric_key, title, live=False):
    
    df_g = df.groupby(SERIES_KEY).agg({
        COLONY_GROWTH_RATE_KEY: 'max',
        SERIES_KEY: lambda x: x.iloc[0],
        'colony_name': lambda x: x.iloc[0],
        'label': lambda x: x.iloc[0],
        numeric_key: lambda x: x.iloc[0],
        'is_cell_score': lambda x: x.iloc[0]>0.5,
    })

    xorder = df[numeric_key].unique()
    xorder.sort()

    plt.figure()
    g = sns.swarmplot(x=numeric_key, order=xorder, y=COLONY_GROWTH_RATE_KEY, hue='is_cell_score', data=df_g, size=2)
    setup_figure()
    plt.gca().set_xticklabels([DataUtils.to_str(text.get_text()) for text in plt.gca().get_xticklabels()])
    plt.xticks(rotation=90)
    plt.title(title)
    plt.ylabel('Colony growth rate (per hour)')

    if PRESENTATION:
        legend = plt.legend(title='', loc='upper right')
    else:
        legend = plt.legend(title='', bbox_to_anchor=(1, 0.5), loc='center left', frameon=False)
    
    for t, l in zip(legend.texts, ['Debris', 'Colony']):
        t.set_text(l)

    save_figure(
        figure=plt.gcf(),
        filename=title,
        bbox_extra_artists=[legend],
        output_folder=output_folder,
        live=live,
    )



def compute_ic90(h,n):
    return h * 9**(1/n)

def plot_hill_fits_from_df(df, numeric_key, output_folder, title, growth_rate_key=COLONY_GROWTH_RATE_KEY, ic90=True, floating=False, live=False, label_loc='upper right'):
    
    growth_rate_mean_key = f'{growth_rate_key}_mean'
    growth_rate_std_key = f'{growth_rate_key}_std'

    df[numeric_key] = pd.to_numeric(df[numeric_key], errors='coerce')
    df_b = df.dropna(subset=[numeric_key, growth_rate_mean_key, growth_rate_std_key], how='any')

    plt.figure()
    # plt.title(title)

    if len(df_b[numeric_key].index) == 0: 
        print(f'Error, no data {title}')
        return
    
    xmin, xmax = DataUtils.min_max(df_b[numeric_key])
    ymin, ymax = -0.4, 2.4
    # ymin = min(df_b[growth_rate_mean_key] - df_b[growth_rate_std_key])
    # ymax = max(df_b[growth_rate_mean_key] + df_b[growth_rate_std_key])

    fit_params = []
    
    repeat_groups = list(DataUtils.groupby(df_b, ['repeat']))
    
    experiments = df['experiment'].unique()
    colors = sns.color_palette('rocket', n_colors=len(repeat_groups))

    for i, (repeat, df_c) in enumerate(repeat_groups): 
        x_data = df_c[numeric_key]
        y_data = df_c[growth_rate_mean_key]
        y_err = df_c[growth_rate_std_key]
        
        if len(x_data) < 5:
            continue

        # hill_function = lambda x, a,h,n,b: a/(h+(x)**n)+b # hill function
        hill_function_full = lambda x, a,h,n,b: a*(1-1/(1+(h/x)**n))+b # hill function
        if floating:
            hill_function = hill_function_full
            bounds = ((0.2, 0.0, 1, -0.2), (3, 10, 50, 0.2)) # lower limits, higher limits
            initial_guess = (1.6, 0, 1, 1)
        else:
            hill_function = lambda x, a,h,n: hill_function_full(x,a,h,n,0) # hill function
            # bounds = ((0.2, 1e-3, 1), (2, 1e2, 40)) # lower limits, higher limits
            bounds = ((0.2, 2), (1e-3, 1e2), (0.1, 40)) # lower limits, higher limits
            initial_guess = (1.6, 1, 1)


        # Objective function with regularization term
        def objective(params, x_data, y_data, alpha):
            y_pred = hill_function(x_data, *params)
            residuals = y_data - y_pred
            regularization = alpha * params[2]**2  # Regularization term (penalizes larger values of n)
            return np.sum(residuals**2) + regularization

        # Set the regularization weight (alpha)
        alpha = 0.001

        # Fit the curve using the minimize function with the objective function and regularization term
        

        try:
            result = optimize.minimize(objective, initial_guess, bounds=bounds, args=(x_data, y_data, alpha))
            coefficients = result.x
            
            # print(coefficients)
            # # fit can fail if no optimal solution is found
            # coefficients, confusion_matrix = optimize.curve_fit(
            #     f=hill_function,
            #     xdata=x_data,
            #     ydata=y_data,
            #     bounds=bounds,
            #     # sigma=y_err,
            #     # absolute_sigma=True,
            #     p0=p0,
            #     method='trf',
            # )

            # compute chi squared of fit 
            expected_value = hill_function(x_data, *coefficients)

            # SST = Σ(Yi - Ymean)² for all i
            # SSR = Σ(Yi - Ypred_i)² for all i
            # R² = 1 - (SSR/SST)
            SST = np.sum((y_data - np.mean(y_data))**2)
            SSR = np.sum((y_data - expected_value)**2)
            R2 = 1 - (SSR/SST)
            print(f'R2 {numeric_key.split(" ")[0]}, {R2}')

            # chi_squared = np.sum((y_data - expected_value)**2 / expected_value)
            # print(f'Chi squared {numeric_key.split(" ")[0]}, {chi_squared}')

            if ic90:
                halfway = compute_ic90(h=coefficients[1], n=coefficients[2]) # 9 is the IC90
            else:
                halfway = coefficients[1]
            
            # print(f'{title:20s}, h={coefficients[1]:.4f}, n={coefficients[2]:.2f}')

            if not DataUtils.is_float(halfway): halfway = 0
            fit_params.append({'a': coefficients[0], 'h': coefficients[1], 'n':coefficients[2]})

        except RuntimeError as e:
            coefficients = [1,1,1]
            halfway = 0

        # xs = np.linspace(np.partition(x_data, 1)[1], max(x_data), 10000)
        
        xmin = min([n for n in df_b[numeric_key].unique() if n != 0])
        
        xs = np.logspace(
            math.floor(math.log10(xmin)) - 1,
            math.ceil(math.log10(xmax)),
            1000,
        )

        xs = np.append([0], xs)

        ic_label = '$\mathregular{IC_{90}}$' if ic90 else '$\mathregular{IC_{50}}$'
        
        if len(df_b['repeat'].unique()) == 1: # only one repeat
            plt.errorbar(x=x_data, y=y_data, yerr=y_err, fmt='o', label=f'{ic_label}={sigfig.round(halfway, sigfigs=2):g} ug/ml', capsize=2, color=colors[i])
        else:
            plt.errorbar(x=x_data, y=y_data, yerr=y_err, fmt='o', label=f'Repeat {repeat[0] if repeat else "1"}, {ic_label}={sigfig.round(halfway, sigfigs=2):g}ug/ml', capsize=2, color=colors[i])
        
        if halfway > 0 and halfway < xmax*2:
            plt.plot(xs, [hill_function(x, *coefficients) for x in xs], color=colors[i])
            plt.vlines(halfway, ymin=ymin, ymax=ymax, color=colors[i], linestyles='dotted')
        
    # plt.gca().set_xscale('symlog', )
    min_nonzero_x = min([n for n in df_b[numeric_key].unique() if n != 0])
    order_of_magnitude = 10**math.floor(math.log10(min_nonzero_x)) # rounds 0.6 to 0.1 and 23 to 10
    plt.xscale('symlog', linthresh=min_nonzero_x)
    formatter = FuncFormatter(lambda y, _: '{:.16g}'.format(y))
    plt.gca().xaxis.set_major_formatter(formatter)

    plt.xlabel(numeric_key)
    plt.ylabel('Growth rate (per hour)')
    plt.ylim(ymin, ymax)

    items = []
    if label_loc != None:
        legend = plt.legend(title='', loc=label_loc)    
        items.append(legend)
    
    setup_figure()
    
    # halfway_points = np.array(halfway_points)
    # IC_text = f'Mean IC90={sigfig.round(halfway_points.mean(), uncertainty=halfway_points.std())} ug/ml'

    # if not PRESENTATION:
    #     _ = plt.gca().text(1.03, 0.3, IC_text, transform=plt.gca().transAxes, verticalalignment='top')
    #     b = plt.gca().text(1.03, 0.0, experiment_description(experiments), transform=plt.gca().transAxes, verticalalignment='bottom', fontsize='small')
    #     _ = plt.gca().text(1.03, -0.1, ', '.join(experiments), transform=plt.gca().transAxes, verticalalignment='top', fontsize='x-small', color='0.6')

    save_figure(
        figure=plt.gcf(),
        filename=title,
        bbox_extra_artists=items,
        output_folder=output_folder,
        live=live,
    )

    return fit_params


def make_growth_rate_hill_plots(df, growth_rate_key, numeric_keys, text_keys, output_folder, mean_time=2.5, offset=0.5, ignore_repeat=False, from_growth_rate=True, label_loc='lower left', live=False):

    dfq = filter_time(df, mean_time-offset, mean_time+offset)

    fit_params = {}

    for numeric_key in numeric_keys: 
        for text_values, df_query in DataUtils.groupby(dfq, text_keys):

            dfqq = df_query.dropna(subset=[numeric_key])
            
            # one row for each colony
            df_g = dfqq.groupby([numeric_key, 'repeat', 'labelid'], as_index=False).agg({
                growth_rate_key: 'mean',
                'experiment': 'first',
            })

            df_g['shrinking'] = df_g[growth_rate_key] <= 0
            df_g['shrinking'] = df_g['shrinking'].astype(float)*100

            df_g[growth_rate_key] = np.maximum(df_g[growth_rate_key], 0)
            
            # one row for each condition with colony-wide variation preserved
            df_gg = df_g.groupby([numeric_key, 'repeat'], as_index=False).agg(
                growth_rate_mean = (growth_rate_key, 'mean'),
                growth_rate_std = (growth_rate_key, 'std'),
                shrinking_mean = ('shrinking', 'mean'),
                shrinking_std = ('shrinking', 'std'),
                experiment = ('experiment', 'first'),
            )

            if ignore_repeat: df_gg['repeat'] = 1

            condition_text = DataProcessor.condition_label(text_keys=text_keys, text_values=text_values, numeric_key=numeric_key)

            fit_params[condition_text] = plot_hill_fits_from_df(
                title=f'Growth rate ({condition_text})',
                growth_rate_key='growth_rate' if from_growth_rate else 'shrinking',
                output_folder=output_folder,
                df=df_gg, 
                numeric_key=numeric_key,
                label_loc=label_loc,
                live=live,
            )

    return fit_params

import math
from scipy import optimize
LYSIS_RATE_KEY = 'Lysis rate (%)'

def plot_lysis_rate_hill_funtion(df, numeric_key, output_folder, title, live=False):

    df[numeric_key] = pd.to_numeric(df[numeric_key], errors='coerce')
    df_b = df.dropna(subset=[numeric_key, LYSIS_RATE_KEY], how='any')

    plt.figure()
    plt.title(title)
             
    x_data = df_b[numeric_key]
    y_data = df_b[LYSIS_RATE_KEY]
    
    hill_function_full = lambda x, a,h,n,b: a/(1+(h/x)**n)+b # hill function
    hill_function = lambda x, h,n: hill_function_full(x,100,h,n,0) # hill function

    try:
        # fit can fail if no optimal solution is found
        coefficients, confusion_matrix = optimize.curve_fit(
            f=hill_function,
            xdata=x_data,
            ydata=y_data,
            # bounds=bounds,
            # sigma=y_err,
            # absolute_sigma=True,
            # p0=p0,
            method='trf',
        )

    except RuntimeError as e:
        coefficients = [0,1]
    
    h, n = coefficients
    if n != 0 and h != 0: 
        upper_lim = h / 9**(1/n)
        loewr_lim = h / (1/9)**(1/n)

    else:
        upper_lim = 0
        loewr_lim = 0
    
    width_h = abs(h - upper_lim)
    width_l = abs(h - loewr_lim)
    std = (width_h + width_l) / 2
    
    try:
        print(title, f'({sigfig.round(h, std)})')
    except Exception as e:
        print(e)

    xs = np.linspace(np.partition(x_data, 1)[1], max(x_data), 10000)
    # xs = np.logspace(-4, 4, 10000)

    marker = 'o' #if j == 'WT' else 'x'
    linestyle = '--' #if marker == 'x' else '-'

    colors = sns.color_palette('rocket', n_colors=len([1]))
    i = 0
    plt.plot(xs, [hill_function(x, *coefficients) for x in xs], color=colors[i], linestyle=linestyle)
    plt.scatter(x=x_data, y=y_data, color=colors[i], marker=marker, s=12)
    p = plt.plot([], [], linestyle + marker, color=colors[i])

    # make dashed line
    plt.vlines(h, ymin=0, ymax=100, color=colors[i], linestyles='dotted')
    plt.gca().fill_between([upper_lim, loewr_lim], 0, 100, alpha=0.2)
    # plt.gca().fill_between([halfway-width/2, halfway+width/2], 0, 100, alpha=0.5)

    plt.gca().set_xscale('log')
    # min_nonzero_x = min([n for n in df_b[numeric_key].unique() if n != 0])
    # order_of_magnitude = 10**math.floor(math.log10(min_nonzero_x)) # rounds 0.6 to 0.1 and 23 to 10
    # plt.xscale('symlog', linthresh=order_of_magnitude)
    # formatter = FuncFormatter(lambda y, _: '{:.16g}'.format(y))
    # plt.gca().xaxis.set_major_formatter(formatter)

    plt.xlabel(numeric_key)
    plt.ylabel('Lysis rate (%)')

    # if False:
    #     legend = plt.legend(title='', loc='lower left')
    # else:
    legend = plt.legend(title='', bbox_to_anchor=(1, 0.5), loc='center left', frameon=False)
    setup_figure()
    
    # plt.gca().xaxis.set_minor_formatter(mticker.ScalarFormatter())
    plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: sigfig.round(x, sigfigs=1, type=str)))

    # halfway_points = np.array(halfway_points)
    # IC_text = f'Mean IC90={sigfig.round(halfway_points.mean(), uncertainty=halfway_points.std())} µg/ml'

    # if not PRESENTATION:
    #     _ = plt.gca().text(1.03, 0.3, IC_text, transform=plt.gca().transAxes, verticalalignment='top')
    #     b = plt.gca().text(1.03, 0.0, experiment_description(experiments), transform=plt.gca().transAxes, verticalalignment='bottom', fontsize='small')
    #     _ = plt.gca().text(1.03, -0.1, ', '.join(experiments), transform=plt.gca().transAxes, verticalalignment='top', fontsize='x-small', color='0.6')

    save_figure(
        figure=plt.gcf(),
        filename=title,
        bbox_extra_artists=[legend],
        output_folder=output_folder,
        live=live,
    )


def plot_lysis_rate(df, numeric_key, group_keys, title, output_folder, live=True):

    # check fate of parent - if they lyse, then all their children should be marked as lysed
    dfg = df.groupby(SERIES_KEY).last()

    dfg.query('present_at_start == True', inplace=True) # only consider colonies present from the start
    dfg.query('present_for_duration > 1', inplace=True) # only consider colonies present from the start

    dfqq = dfg.groupby([numeric_key, 'repeat']+group_keys, as_index=False).apply(lambda x: pd.Series({LYSIS_RATE_KEY: (x['Colony lysis'] == True).mean()*100}))
    
    if len(group_keys) == 0:
        plot_lysis_rate_hill_funtion(dfqq, numeric_key, output_folder, title=title, live=True)
        return

    for keys, d in dfqq.groupby(group_keys):
        plot_lysis_rate_hill_funtion(d, numeric_key, output_folder, title=f'{title} ({", ".join(keys)})', live=True)



def plot_comparisons(df, x_key, y_keys, y_labels, title, output_folder, figsize=(5,5), hue_key=None, hue_map=None, presentation=True, live=False):

    sns.reset_defaults()
    fig = plt.figure(figsize=figsize, dpi=300)
    
    if presentation: 
        setup_figure()

    gs = fig.add_gridspec(len(y_keys), hspace=0 if presentation else 0.1)
    axs = gs.subplots(sharex=True, sharey=False)

    if df[x_key].dtype == 'O':
        xorder = list(df[x_key].unique())
        xorder.sort()
        if 'None' in xorder: 
            xorder.remove('None')
            xorder.insert(0, 'None') # put None at the start

    else: 
        xorder = None

    for i, (y_key, ax) in enumerate(zip(y_keys, axs)):

        hmap = hue_map[y_key] if hue_map != None else None   
        label = y_labels[y_key]
        
        plt.axes(ax)

        sns.boxplot(
            data=df,
            x=x_key,
            y=y_key,
            hue=hue_key, # set hue based on mean if not specified
            # style=STRAIN_LABEL,
            order=xorder,
            showfliers=False,
            palette=hmap,
            
        )
        plt.ylabel(label)
        # rotate x labels

        plt.xticks(rotation=90)
        # plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: sigfig.round(x, sigfigs=2, type=str)))
        plt.gca().set_xticklabels([DataUtils.to_str(text.get_text()) for text in plt.gca().get_xticklabels()])

        legend = None
        if i == 0: 
            if hue_key != None:
                legend = plt.gca().legend(loc='upper left', bbox_to_anchor=(1, 1), frameon=False)
        
        if i < len(axs)-1:
            plt.gca().set_xlabel('')

    for ax in axs:
        yticks = ax.get_yticks().tolist()
        print(yticks)
        if len(yticks) > 2:  # Ensure there are more than two ticks
            ax.set_yticks(yticks[1:-1])

    # for ax in axs:
    #     ymin, ymax = ax.get_ylim()
    #     yrange = ymax - ymin
    #     ax.set_ylim(ymin - 0.12*yrange, ymax + 0.12*yrange)
    #     print('set limits', ymin - 0.12*yrange, ymax + 0.12*yrange)
    # plt.tight_layout()

    save_figure(
        figure=plt.gcf(),
        filename=title,
        bbox_extra_artists=[legend] if legend != None else None,
        output_folder=output_folder,
        live=live,
    )



# time in hours, filter out all data points outside of the time range
def filter_time(df, start_time, end_time, inplace=False) -> pd.DataFrame:
    if not inplace:
        if start_time != None: df = df.query(f'{start_time} <= round_time_hours')
        if end_time != None: df = df.query(f'round_time_hours < {end_time}')
    else:
        if start_time != None: df.query(f'{start_time} <= round_time_hours', inplace=True)
        if end_time != None: df.query(f'round_time_hours < {end_time}', inplace=True)
    
    return df

# time in hours
def filter_for_plotting(df: pd.DataFrame, start_time: float, end_time: float, numeric_keys: List[str], text_keys: List[str], min_present_count: int = 5, remove_on_border: bool = True):
    
    exclude_labels = []
    for experiment, df_q in df.groupby(['experiment']):
        experiment_map = Plotter.mapping_for_experiment(experiment)
        experiment_info = experiment_map['experiments'][experiment]
        exclude_pads = experiment_info.get('exclude_pads', [])
        for p in exclude_pads:
            exclude_labels += df_q.query('pad_name == @p')['label'].unique().tolist()

    df.query('label != @exclude_labels', inplace=True)
    
    df.query('colony_area_growth_rate_var < 0.5', inplace=True) # remove where fit was not good

    df.query('is_cell_score > 0.5', inplace=True)
    if remove_on_border: df.query('colony_on_border == False', inplace=True)
    filter_time(df, start_time, end_time, inplace=True)

    # only include (and resport statistics) when there are at least N colonies present
    for numeric_key in numeric_keys:
        for _, dfg in DataProcessor.groupby(df, text_keys, dropna=False):
            for _, dfgg in dfg.groupby([numeric_key, 'time_bin_30min']): # time_bin_small to count between experiments, round_time to count in experiment only
                df.loc[dfgg.index, 'present_count'] = len(dfgg.index)
    
    if not 'present_count' in df.columns: 
        for _, dfg in DataProcessor.groupby(df, text_keys, dropna=False):
            for _, dfgg in dfg.groupby(['time_bin_30min']): # time_bin_small to count between experiments, round_time to count in experiment only
                df.loc[dfgg.index, 'present_count'] = len(dfgg.index)
    
    df.query(f'present_count >= {min_present_count}', inplace=True)



class WorkProcesser:    
    def __init__(self):
        self.p = pool.Pool() # processes automatically set to core-count

    def add(self, target, args=(), kwargs=()):
        self.p.apply_async(func=target, args=args, kwds=kwargs)

    def join(self):
        self.p.close()
        self.p.join()


def plots_from_dataframe(df, experiment_map, experiment, output_folder):
    
    if not isinstance(df, pd.DataFrame) or df.empty:
        logging.error('Trying to plot from empty dataframe.')
        return

    numeric_keys, text_keys = Plotter.find_numeric_and_text_keys(experiment_map=experiment_map)
    all_keys = numeric_keys + text_keys

    logging.info(f'{numeric_keys=}, {text_keys=}')

    plot_options = experiment_map.get('plot_options', {})
    experiment_info = experiment_map['experiments'][experiment]
    experiment_category = experiment_info.get('experiment_category', 'Unknown')
    instrument = experiment_info.get('instrument', 'Unknown')
    t_start, t_end = experiment_info.get('exponential_time_hours', [0, 2]) # in hours
    exclude_pads = experiment_info.get('exclude_pads', [])

    # TEMP - remove
    if 'Ampicilin concentration (µg/ml)' in df.columns:
        df['Ampicillin concentration (µg/ml)'] = df['Ampicilin concentration (µg/ml)']
    DataProcessor.add_time_bin_column(df=df, interval=30) # cheap, so can do again here

    if not 'repeat' in df.columns:
        print('Adding repeat column')
        DataProcessor.add_repeat_column(df, numeric_keys=numeric_keys, text_keys=text_keys, repeat_keys=['experiment', 'row'])
    print(df['repeat'].unique())

    w = WorkProcesser()

    # Filter
    # principle: only filter for time once at this top level to ensure time-fitering is consistent
    df_raw = df.copy(deep=False)
    filter_for_plotting(df, start_time=0, end_time=5, numeric_keys=numeric_keys, text_keys=text_keys, min_present_count=4, remove_on_border=True)

    w.add(plot_counts, kwargs={
        'output_folder': DataUtils.append_path(output_folder, 'counts'),
        'df': df, 
    })

    # Plots per column
    for col, df_query in df.groupby('col'): # if multiple mutually exclusive numeric keys, the other's data will be null and not included in plots            
        w.add(plot_boxplot, kwargs={
            'title': f'Growth rates for column {col}',
            'output_folder': DataUtils.append_path(output_folder, 'col'),
            'df': df_query,
            'hue_label': None,
            'swarm_hue_label': 'replicate',
            'xkey': 'row',
            'ykey': COLONY_GROWTH_RATE_KEY,
            'ylabel': 'Colony growth rate (per hour)',
        })
        
    # Plot per row
    for row, df_query in df.groupby('row'): # if multiple mutually exclusive numeric keys, the other's data will be null and not included in plots            
        w.add(plot_boxplot, kwargs={
            'title': f'Growth rates for row {row}',
            'output_folder': DataUtils.append_path(output_folder, 'row'),
            'df': df_query,
            'hue_label': None,
            'swarm_hue_label': 'replicate',
            'xkey': 'col',
            'ykey': COLONY_GROWTH_RATE_KEY,
            'ylabel': 'Colony growth rate (per hour)',
        })
    

    # Plots per numeric_key
    for numeric_key in numeric_keys: # if multiple mutually exclusive numeric keys, the other's data will be null and not included in plots
        
        for text_key in text_keys:

            condition_text = numeric_key
            if len(text_keys) > 1: condition_text += f' and {text_key}'
            
            w.add(plot_boxplot, kwargs={
                'title': f'Growth rates ({condition_text}, 1-2h)',
                'output_folder': DataUtils.append_path(output_folder, 'growth rate text keys'),
                'df': filter_time(df, 1, 2),
                'hue_label': text_key,
                'swarm_hue_label': None,
                'xkey': numeric_key,
                'ykey': COLONY_GROWTH_RATE_KEY,
                'ylabel': 'Colony growth rate (per hour)',
            })
        
        
        df_q = df.dropna(subset=[numeric_key], how='all')
        for text_values, df_query in DataUtils.groupby(df_q, text_keys):

            condition_text = DataProcessor.condition_label(text_keys=text_keys, text_values=text_values, numeric_key=numeric_key)
            # DataUtils.print_full(df_query.iloc[0])

            w.add(plot_areas_for_numeric_key, kwargs={
                'title': f'Timeplot ({condition_text})', 
                'output_folder': DataUtils.append_path(output_folder, 'timeplots'),
                'df': df_query, 
                'numeric_key': numeric_key, 
                'xkey': 'time_bin',
                'ykey': 'colony_area',
                'ylabel': 'Colony area (µm²)',
            })

            w.add(plot_boxplot, kwargs={
                'title': f'Growth rate after 2 hours ({condition_text})',
                'output_folder': DataUtils.append_path(output_folder, 'growth rate 2h'),
                'df': filter_time(df_query, 1.75, 2.25),
                'hue_label': None,
                'swarm_hue_label': 'time_bin',
                'xkey': numeric_key,
                'ykey': COLONY_GROWTH_RATE_KEY,
                'ylabel': 'Colony growth rate (per hour)',
            })

            w.add(plot_boxplot, kwargs={
                'title': f'Growth rates over time ({condition_text})',
                'output_folder': DataUtils.append_path(output_folder, 'growth rate time'),
                'df': df_query,
                'hue_label': 'time_bin',
                'swarm_hue_label': None,
                'xkey': numeric_key,
                'ykey': COLONY_GROWTH_RATE_KEY,
                'ylabel': 'Colony growth rate (per hour)',
            })

            w.add(plot_comparisons, kwargs={
                'title': f'Cell statistics ({condition_text})',
                'output_folder': DataUtils.append_path(output_folder, 'cell stats'),
                'df': df_query,
                'hue_key': 'time_bin',
                'x_key': numeric_key,
                'y_keys': [
                    'ss_area_mean',
                    'ss_length_mean',
                    'ss_width_mean',
                ],
                'y_labels': {
                    'ss_area_mean': 'Mean area (µm²)',
                    'ss_length_mean': 'Mean length (µm)',
                    'ss_width_mean': 'Mean width (µm)',
                },
            })


        # For each numeric key, loop over raw dataframe
        for text_values, df_query in DataUtils.groupby(df_raw, text_keys):
            condition_text = DataProcessor.condition_label(text_keys=text_keys, text_values=text_values, numeric_key=numeric_key)

            w.add(plot_is_cell_score_filter, kwargs={
                'title': f'Growth rate filter ({condition_text})',
                'output_folder': DataUtils.append_path(output_folder, 'is cell score'), 
                'df': df_query,
                'numeric_key': numeric_key,
            })

    
    # Plots per text_key
    if not numeric_keys:
        for text_key in text_keys:

            w.add(plot_boxplot, kwargs={
                'title': f'Growth rates ({text_key}, 1-2h)',
                'output_folder': DataUtils.append_path(output_folder, 'growth rate text key'),
                'df': filter_time(df, 1, 2),
                'hue_label': None,
                'swarm_hue_label': 'time_bin',
                'xkey': text_key,
                'ykey': COLONY_GROWTH_RATE_KEY,
                'ylabel': 'Colony growth rate (per hour)',
            })
            
            w.add(plot_boxplot, kwargs={
                'title': f'Growth rates ({text_key}, 1-2h)',
                'output_folder': DataUtils.append_path(output_folder, 'growth rate per pad'),
                'df': filter_time(df, 1, 2),
                'hue_label': text_key,
                'swarm_hue_label': 'time_bin',
                'xkey': 'pad_label',
                'ykey': COLONY_GROWTH_RATE_KEY,
                'ylabel': 'Colony growth rate (per hour)',
            })
    
    

    # Plots per key
    logging.info('Generating colony individual timeplots')
    for values, df_query in DataUtils.groupby(df, all_keys):
        
        condition_text = DataProcessor.condition_label(text_keys=all_keys, text_values=values, numeric_key=None)

        w.add(plot_param_over_time, kwargs={
            'title': f'Areas ({condition_text})',
            'output_folder': DataUtils.append_path(output_folder, 'time vs areas'),
            'df': df_query, 
            'ykey': 'colony_area',
            'ylabel': 'Colony area (µm²)', 
            'log2': True,
            'ylim_zero': False,
            'plot_mean': False,
        })
        
        # w.add(plot_param_over_time, kwargs={
        #     'title': f'Growth rates ({condition_text})',
        #     'output_folder': DataUtils.append_path(output_folder, 'time vs growth rate'),
        #     'df': df_query, 
        #     'ykey': COLONY_GROWTH_RATE_KEY, 
        #     'ylabel': 'Colony growth rate (per hour)', 
        #     'log2': False,
        #     'ylim_zero': True,
        #     'plot_mean': True,
        # })

    w.add(make_growth_rate_hill_plots, kwargs={
        'df': df,
        'numeric_keys': numeric_keys,
        'text_keys': text_keys,
        'growth_rate_key': 'colony_area_growth_rate',
        'output_folder': DataUtils.append_path(output_folder, 'growth rate fits 2p1'),
        'mean_time': 2,
        'offset': 1,
    })
    w.add(make_growth_rate_hill_plots, kwargs={
        'df': df,
        'numeric_keys': numeric_keys,
        'text_keys': text_keys,
        'growth_rate_key': 'colony_area_growth_rate',
        'output_folder': DataUtils.append_path(output_folder, 'growth rate fits 3p1'),
        'mean_time': 3,
        'offset': 1,
    })

    # logging.info('Generating colony overview timeplots')
    # for numeric_key in numeric_keys:
    #     for values, df_query in Plotter.groupby(df, text_keys):

    #         condition_text = condition_label(text_keys=text_keys, text_values=values, numeric_key=numeric_key)


    #         ## Colony area

    #         plot_overview_over_time(
    #             df=df_query,
    #             series_label=numeric_key,
    #             style_label=None,
    #             title=f'Colony area overview ({condition_text}) cs',
    #             ykey='colony_area',
    #             ylabel='Colony area (square pixels)',
    #             log2=True,
    #             ylim_zero=False,
    #             output_folder=output_folder,
    #         )

    #         plot_overview_over_time(
    #             df=df_query,
    #             series_label=numeric_key,
    #             style_label=None,
    #             title=f'Colony growth rate overview ({condition_text}) cs',
    #             ykey=COLONY_GROWTH_RATE_KEY,
    #             ylabel='Colony growth rate (per hour)',
    #             log2=False,
    #             ylim_zero=False,
    #             output_folder=output_folder,
    #         )

    #         # colony_area
    #         # ss_area_total
    #         # ss_area_count


    #         ## Single cell area

    #         plot_overview_over_time(
    #             df=df_query,
    #             series_label=numeric_key,
    #             style_label=None,
    #             title=f'Colony area overview ({condition_text}) ss',
    #             ykey='ss_area_total',
    #             ylabel='Colony area (square pixels)',
    #             log2=True,
    #             ylim_zero=False,
    #             output_folder=output_folder,
    #         )

    #         plot_overview_over_time(
    #             df=df_query,
    #             series_label=numeric_key,
    #             style_label=None,
    #             title=f'Colony growth rate overview ({condition_text}) ss',
    #             ykey='ss_area_total_growth_rate',
    #             ylabel='Colony growth rate (per hour)',
    #             log2=False,
    #             ylim_zero=False,
    #             output_folder=output_folder,
    #         )
            

    #         ## Sigle cell count

    #         plot_overview_over_time(
    #             df=df_query,
    #             series_label=numeric_key,
    #             style_label=None,
    #             title=f'Colony count ({condition_text})',
    #             ykey='ss_area_total',
    #             ylabel='Colony count (#)',
    #             log2=True,
    #             ylim_zero=False,
    #             output_folder=output_folder,
    #         )
            
    #         plot_overview_over_time(
    #             df=df_query,
    #             series_label=numeric_key,
    #             style_label=None,
    #             title=f'Colony doubling rate overview ({condition_text})',
    #             ykey='ss_area_total_growth_rate',
    #             ylabel='Colony growth rate (per hour)',
    #             log2=False,
    #             ylim_zero=False,
    #             output_folder=output_folder,
    #         )


    ## Plots based on statistics dataframe

    df_stats = Plotter.generate_growth_rate_stats_df(
        df=df, 
        numeric_keys=numeric_keys, 
        text_keys=text_keys,
        ykeys=['colony_area_growth_rate', 'ss_area_total_growth_rate', 'ss_area_total_growth_rate'],
        experiment=experiment, 
        experiment_category=experiment_category, 
        instrument=instrument,
        filename=f'{experiment} growth rate stats', 
        output_folder=output_folder,
    )

    # for numeric_key in numeric_keys: 
    #     for text_values, df_query in DataUtils.groupby(df_stats, text_keys):

    #         condition_text = DataProcessor.condition_label(text_keys=text_keys, text_values=text_values, numeric_key=numeric_key)
            

    #         # if 'concentration' in numeric_key.lower():
    #         #     plot_hill_fits_from_df(
    #         #         title=f'Colony growth rate ({condition_text})',
    #         #         output_folder=DataUtils.append_path(output_folder, 'growth rate fits'),
    #         #         df=df_query, 
    #         #         numeric_key=numeric_key,
    #         #     )
            
    w.join()



def main(experiment, no_cache=False, quick=False, df=None, segmentation_version=None, plot_version=None, **kwargs):

    # Collect environment variables
    graph_dir = DataProcessor.get_graph_folder(experiment=experiment, segmentation_version=segmentation_version, plot_version=plot_version)

    # create and clear contents of graph directory
    MKUtils.configure_logger(graph_dir, 'PlotPads.log', label='main', file_mode='w')
    logging.info('Started')

    graph_files = glob.glob(os.path.join(graph_dir, '*.png'))

    # Do not analyze 
    if not no_cache and len(graph_files) > 4: 
        logging.info(f'{len(graph_files)} graph files allready present.')
        return 

    for f in graph_files:
        try:
            os.remove(f)
        except:
            logging.exception(f'Could not remove file {f}.')

    logging.info(f'Executing PlotPads with {experiment=}')
    
    # Load experiment map
    experiment_map = Plotter.mapping_for_experiment(experiment)
    
    if df is not None:
        logging.info(f'Dataframe passed as argument')
    else: 
        logging.info(f'Attempting to load dataframe for experiment {experiment}')
        df = DataProcessor.load_dataframe(experiment)
        assert not isinstance(df, type(None))
        
    if quick: df.query('id < 20', inplace=True)

    logging.info(f'Generating plots for {experiment}.')
    plots_from_dataframe(
        df=df,
        experiment_map=experiment_map,
        experiment=experiment,
        output_folder=graph_dir,
    )

    logging.info(f'Completed generating plots for {experiment}.')


if __name__ == '__main__':

    # Load input arguments from user.
    parser = argparse.ArgumentParser(description='Extract information from .movie files.')
    parser.add_argument('experiment', type=str, help='Experiment name.')
    parser.add_argument('--all', '-a', type=bool, nargs='?', const=True, default=False, 
                        help='Do not skip files that have already been generated.')
    parser.add_argument('--quick', '-q', type=bool, nargs='?', const=True, default=False, 
                        help='Query subset of dataframe to generate output fast.')
    
    args = parser.parse_args()

    main(
        experiment=args.experiment, 
        no_cache=args.all,
        quick=args.quick,
    )


'''
Plotting data from lots of experiments at once
DFs need columns: 
  - experiment name
  - category (what type of eperiment this is [Sample platform, Platereader, Agarose coverslip sandwitch, Etest])
  - growth rate (mean + std) for each condition
  - condition description (antibiotic concentration etc)

These dataframes will be small, and purely exported for the purpoce of plotting together.

How is this plotting controlled? Separate file? Dataframe contains only info to be plotted. 

Join all dataframes together and plot using seabron. -> this is to be done this afternoon. 
'''
