import numpy as np
import pandas as pd
import six

from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Imputer


def null_filter(x):
    return True


def unflatten_list(llist):
    return [l for sublist in llist for l in sublist]


def dtype_dict(dataframe, dtype_filter=None):
    column_series = dataframe.columns.to_series()
    dtype_groups = column_series.groupby(dataframe.dtypes).groups

    if dtype_filter is None:
        dtype_filter = null_filter

    return {k.name: v for k, v in dtype_groups.items() if dtype_filter(k.name)}


def is_numeric(dtype):
    if isinstance(dtype, six.string_types):
        try:
            dtype = np.dtype(dtype)
        except TypeError:
            return False

    return np.issubdtype(dtype, np.number)


def is_categorical(dtype):
    return not is_numeric(dtype)


def numeric_columns(dataframe):
    return unflatten_list(dtype_dict(dataframe, is_numeric).values())


def categorical_columns(dataframe):
    return unflatten_list(dtype_dict(dataframe, is_categorical).values())


def mean_impute_numerics(dataframe):
    numerics = dataframe.values

    # impute with mean
    imputer = Imputer(strategy='mean', missing_values='NaN')
    numerics = imputer.fit_transform(numerics)

    # apply standard scalar
    scaler = StandardScaler()
    numerics = scaler.fit_transform(numerics)

    return numerics


def replace_null(dataframe, value='NaN', inplace=False):
    dataframe[pd.isnull(dataframe)] = value
    return dataframe


def dict_encode(dataframe, sparse=True):
    cat_dict = dataframe.T.to_dict().values()
    vectorizer = DictVectorizer(sparse=sparse)
    return vectorizer.fit_transform(cat_dict)


def label_encode(dataframe, inplace=False):
    if not inplace:
        dataframe = dataframe.copy()

    for column in dataframe.columns:
        if dataframe[column].dtype == np.object:
            encoder = LabelEncoder()
            dataframe[column] = encoder.fit_transform(dataframe[column])

    return dataframe

