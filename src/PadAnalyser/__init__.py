SEGMENTATION_VERSION = 'S1.23'
DATAFRAME_VERSION = 'D1.13'
PLOT_VERSION = 'P1.21'
MODEL_VERSION = 'M1.03'

def experiment_folder_name(experiment, segmentation_version=SEGMENTATION_VERSION):
    return f'{experiment}_{segmentation_version}'