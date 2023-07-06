
import math
import sigfig
import pandas as pd


def is_float(element: str) -> bool:
    try:
        f = float(element)
        return not (math.isnan(f) or math.isinf(f))
    except ValueError:
        return False

def to_str(s) -> str:
    if isinstance(s, str) and not is_float(s): # check if string is string is of a float
        # print(f'Not float to round {s}, {type(s)}')
        return s
    try:
        return sigfig.round(s, sigfigs=2, type=str, warn=False)
    except Exception as e:
        print(f'Could not round {s}, {type(s)}, {e}')
        pass
    return str(s)


# make into list of not already iterable
def listify(l):
    if isinstance(l, str): return [l]
    try:
        _ = iter(l)
    except TypeError:
        if l is None: return []
        return [l]
    return [str(e) for e in l]


# should return touple of values, group dataframe. Values must be list of values that applies to group dataframe. 
def groupby(df, keys):
    keys = [k for k in keys if k in df] # filter out keys that are not present in this dataframe
    if not keys: return [([], df)]
    return ((listify(values), groups) for values, groups in df.groupby(keys, dropna=False)) 


def min_max(a):
    return min(a), max(a)



def pad_label_to_string(s):
    e = s.split('_')
    return f'Pad {e[2]}, {e[0]}-{e[1]}'


import MKUtils
import os 

def append_path(path: str, *folder_name: str) -> str:
    new_path = os.path.join(path, *folder_name)
    MKUtils.generate_directory(new_path)
    return new_path



# Print dataframe object in full without truncation
def print_full(x):
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 2000)
    pd.set_option('display.float_format', '{:20,.2f}'.format)
    pd.set_option('display.max_colwidth', None)
    print(x)
    pd.reset_option('display.max_rows')
    pd.reset_option('display.max_columns')
    pd.reset_option('display.width')
    pd.reset_option('display.float_format')
    pd.reset_option('display.max_colwidth')