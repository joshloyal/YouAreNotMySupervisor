from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import division

import os

import pandas as pd

from mysuper.datasets import utils as data_utils


root = os.path.abspath(os.path.dirname(__file__))
DATA_NAME = 'data'
FILE_NAME = os.path.join(root, DATA_NAME, 'cars.csv')


def fetch_cars(no_processing=False):
    df = pd.read_csv(FILE_NAME)
    numeric_cols = data_utils.numeric_columns(df)

    if no_processing:
        return df[numeric_cols]
    else:
        data = data_utils.mean_impute_numerics(df[numeric_cols])
        return pd.DataFrame(data, columns=df[numeric_cols].columns)


data = fetch_cars()
