from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import division

import os

import numpy as np
import pandas as pd

from mysuper.datasets import utils as data_utils


root = os.path.abspath(os.path.dirname(__file__))
DATA_NAME = 'data'
FILE_NAME = os.path.join(root, DATA_NAME, '10k_diabetes.csv')


def fetch_10kdiabetes(only_numerics=False, only_categoricals=False, one_hot=False):
    # load data
    df = pd.read_csv(FILE_NAME)

    # don't use the target for clustering for now
    y = df.pop('readmitted').values.astype(np.int)

    # remove text columns
    df = df.drop(['diag_1_desc', 'diag_2_desc', 'diag_3_desc'], axis=1)

    # call before replace None...
    numeric_cols = data_utils.numeric_columns(df)
    if numeric_cols:
        num_x = data_utils.mean_impute_numerics(df[numeric_cols])
        numerics = pd.DataFrame(num_x, columns=df[numeric_cols].columns)


    # replace NaN
    categorical_cols = data_utils.categorical_columns(df)
    categorical_df = data_utils.replace_null(df[categorical_cols], value='NaN')

    if one_hot:
        categoricals = pd.get_dummies(categorical_df, drop_first=True)
    else:
        categoricals = data_utils.label_encode(categorical_df)


    if only_numerics:
        return numerics
    elif only_categoricals:
        return categorical
    else:
        return pd.concat([numerics, categoricals], axis=1)
