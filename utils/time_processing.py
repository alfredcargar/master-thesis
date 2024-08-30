"""
Utilities related to date and time processing

Author: smas
Last update: 21/09/2023
"""

import pandas as pd
from datetime import date


def get_observations_for_specific_year(dataframe: pd.DataFrame, year: int, from_year_to_active: bool = False) -> pd.DataFrame:
    """
    Get the observations from the input dataset for a specific year

    Args:
        dataframe: Input dataframe
        year: Year of interest
        from_year_to_active: Boolean indicating if you want to obtain the samples from the year that are active

    Returns:
        A dataframe with the observations for the specific year

    Notes:
        The boolean variable fro m_year_to_active returns information if the Date_From of the sample belongs to the year
    """

    assert "Date_From" in dataframe.columns, "Date_From column not found"
    assert "Date_To" in dataframe.columns, "Date_To column not found"

    dataframe['Date_from_year'] = pd.to_datetime(dataframe.iloc[:-1]['Date_From']).dt.year
    dataframe['Date_to_year'] = pd.to_datetime(dataframe.iloc[:-1]['Date_To']).dt.year
    dataframe.at[dataframe.index[-1], 'Date_from_year'] = pd.to_datetime(dataframe.iloc[-1]['Date_From']).year

    dataframe_year_from = dataframe[dataframe['Date_from_year'] == year]
    dataframe_year_to = dataframe[dataframe['Date_to_year'] == year]
    dataframe_year_from_to = dataframe[(year <= dataframe['Date_to_year']) & (year >= dataframe['Date_from_year'])]

    dataframe_year = pd.concat([dataframe_year_from, dataframe_year_to, dataframe_year_from_to], ignore_index=True)

    if not from_year_to_active:
        dataframe_year = dataframe_year[dataframe_year['Date_To'] != date(9999, 12, 31)]

    dataframe_year = dataframe_year.drop(['Date_from_year', 'Date_to_year'], axis=1)
    dataframe_year = dataframe_year.drop_duplicates()
    dataframe_year.sort_values(by='Date_From', inplace=True)
    dataframe_year = dataframe_year.reset_index().drop(['index'], axis=1)

    return dataframe_year
