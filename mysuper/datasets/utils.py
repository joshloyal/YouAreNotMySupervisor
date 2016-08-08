import numpy as np
import pandas as pd

from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Imputer

def dtype_dict(dataframe):
    column_series = dataframe.columns.to_series()
    dtype_groups = column_series.groupby(dataframe.dtypes).groups
    return {k.name: v for k, v in dtype_groups.items()}



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
    if not inplace:
        dataframe = dataframe.copy()

    for column in dtype_dict(dataframe)['object']:
        dataframe[pd.isnull(dataframe[column])] = value

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

