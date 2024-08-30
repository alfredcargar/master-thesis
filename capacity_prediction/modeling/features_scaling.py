"""
Utilities used to scale the features.

Additional information:
    https://scikit-learn.org/stable/auto_examples/preprocessing/plot_all_scaling.html#plot-all-scaling-standard-scaler-section

Author: smas
Last update: 06/10/2023
"""
from typing import Tuple

import pandas as pd
from pandas import DataFrame
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler, RobustScaler, PowerTransformer


def standard_scaler(df: pd.DataFrame) -> tuple[DataFrame, StandardScaler]:
    """
    This function performs a standard scaling of the features

    Notes:
        1. StandardScaler removes the mean and scales the data to unit variance
        2. Outliers have an influence when computing the empirical mean and standard deviation
    """

    scaler = preprocessing.StandardScaler()
    df_scaled = scaler.fit_transform(df)
    df_scaled = pd.DataFrame(df_scaled, columns=df.columns)
    return df_scaled, scaler


def robuts_scaler(df: pd.DataFrame) -> tuple[DataFrame, RobustScaler]:
    """
    This function performs a robust scaling of the features

    Notes:
        1. RobustScaler removes the median and scales the data according to the quantile range
            and are therefore not influenced by a small number of very large marginal outliers
    """

    scaler = preprocessing.RobustScaler()
    df_scaled = scaler.fit_transform(df)
    df_scaled = pd.DataFrame(df_scaled, columns=df.columns)
    return df_scaled, scaler


def power_transformer(df: pd.DataFrame, method: str = 'yeo-johnson') -> tuple[DataFrame, PowerTransformer]:
    """
    This function performs a power transformation of the features

    Args:
        df: dataframe, contains all the features to be analyzed
        method: 'str', to chose from 'yeo-johnson' or 'box-cox'

    Notes:
        1. PowerTransformer applies a power transformation to each feature to make the data more Gaussian-like
            in order to stabilize variance and minimize skewness
    """

    scaler = preprocessing.PowerTransformer(method=method)
    df_scaled = scaler.fit_transform(df)
    df_scaled = pd.DataFrame(df_scaled, columns=df.columns)
    return df_scaled, scaler
