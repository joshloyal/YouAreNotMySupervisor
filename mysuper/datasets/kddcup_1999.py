import os
import pandas as pd

from mysuper.datasets import utils as data_utils


root = os.path.abspath(os.path.dirname(__file__))
DATA_NAME = 'data'
FEATURE_NAME = os.path.join(root, DATA_NAME, 'kddcup.names')
KDD_NAME = os.path.join(root, DATA_NAME, 'kddcup.data_10_percent')


def feature_names():
    feature_names = []
    with open(FEATURE_NAME, 'r') as f:
        for line in f:
            line = line.strip().strip('.')
            features = line.split(',')
            if len(features) == 1:
                feature_names.append(features[0].split(':')[0])
            else:
                pass

    return feature_names


def fetch_kdd99(original_dataframe=False, only_numeric=False):
    df = pd.read_csv(KDD_NAME, header=None)
    df.pop(41)
    df.columns = feature_names()

    dtype_groups = data_utils.dtype_dict(df)

    if original_dataframe:
        if only_numeric:
            return df[dtype_groups['float64']]
        return df

    if only_numeric:
        return data_utils.mean_impute_numerics(df[dtype_groups['float64']])
    else:
        return df
