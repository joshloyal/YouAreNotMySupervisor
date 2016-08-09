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


def fetch_10kdiabetes(original_dataframe=False, only_numerics=False):
    # load data
    df = pd.read_csv(FILE_NAME)
    if original_dataframe:
        return df

    # don't use the target for clustering for now
    y = df.pop('readmitted').values.astype(np.int)

    # remove text columns
    df = df.drop(['diag_1_desc', 'diag_2_desc', 'diag_3_desc'], axis=1)

    # call before replace None...
    numeric_cols = data_utils.numeric_columns(df)
    if numeric_cols:
        num_x = data_utils.mean_impute_numerics(df[numeric_cols])


    dtype_groups = data_utils.dtype_dict(df)

    # replace NaN
    data_utils.replace_null(df, inplace=True)

    cat_X, num_X = None, None
    if 'object' in  dtype_groups:
        cat_X = data_utils.dict_encode(df[dtype_groups['object']], sparse=False)



    if only_numerics:
        return pd.DataFrame(num_x, columns=df[numeric_cols].columns)

    if cat_X is not None and num_X is not None:
        X = np.c_[cat_X, num_X]
    elif cat_X is not None:
        X = cat_X
    else:
        X = num_X

    return X
