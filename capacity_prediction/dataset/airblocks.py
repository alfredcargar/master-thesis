"""
Utilities to extract information from Airblocks composing a sector

Author: smas
Last update: 17/10/2023
"""

import pandas as pd
import datetime
import numpy as np


def volume_airblocks_composing_sector_average(date_from: datetime.date, date_to: datetime.date,
                                              df_spatial_airblocks_specific_sector: pd.DataFrame,
                                              debug_mode: bool = False) -> [float, float]:
    """
    Compute the volume of the airblocks composing the sector based on the AVERAGE volume of the airblocks for the
        different possible periods in [date_from, date_to].

    Args:
        date_from: Date from of the interval
        date_to: Date to of the interval
        df_spatial_airblocks_specific_sector: DataFrame with the airblocks composing the sector
        debug_mode: If True, it prints some information

    Returns:
        The volume of the airblocks composing the sector

    Notes:
        1. The intervals of the operational/spatial information do not match the intervals of the airblocks. Thus,
            the volume is approximated using the FIRST set of airblocks in DWH used to compose the sector.
    """

    assert "Date_From" in df_spatial_airblocks_specific_sector.columns and "Date_From" in \
           df_spatial_airblocks_specific_sector, "Date_From column not found"
    assert "Date_To" in df_spatial_airblocks_specific_sector.columns and "Date_To" in \
           df_spatial_airblocks_specific_sector, "Date_To column not found"
    assert "LowerBound" in df_spatial_airblocks_specific_sector.columns, "LowerBound column not found"
    assert "UpperBound" in df_spatial_airblocks_specific_sector.columns, "UpperBound column not found"
    assert "Polygon_area" in df_spatial_airblocks_specific_sector.columns, "Polygon_area column not found"

    df_airblocks_interval = get_validity_periods_for_date_from_date_to(date_from, date_to,
                                                                       df_spatial_airblocks_specific_sector,
                                                                       debug_mode=debug_mode)

    unique_date_from = np.unique(df_airblocks_interval['Date_From'].values)

    # Compute the average volume of the airblocks composing the sector.
    #  It is possible to have multiple valid periods of airblocks for the requested [date_from date_to].
    possible_volumes = []
    number_of_airblocks = []
    for date_from in unique_date_from:
        df_airblocks = df_airblocks_interval[df_airblocks_interval['Date_From'] == date_from]
        df_airblocks.loc[:, 'volume_airblock'] = df_airblocks.loc[:, 'Polygon_area'] * \
                                                 (df_airblocks.loc[:, 'UpperBound'] - df_airblocks.loc[:, 'LowerBound'])

        volume = float(np.sum(df_airblocks['volume_airblock'].values))
        number_airblocks = len(df_airblocks)

        if debug_mode:
            print("++++++++++++++++++++++++++++")
            print(date_from)
            print(df_airblocks)
            print(f"volume: {volume} | number_airblocks: {number_airblocks}")

        possible_volumes.append(volume)
        number_of_airblocks.append(number_airblocks)

    return np.mean(possible_volumes), round(np.mean(number_of_airblocks), 2)


def volume_airblocks_composing_sector_average_first_valid_period(date_from: datetime.date, date_to: datetime.date,
                                                                 df_spatial_airblocks_specific_sector: pd.DataFrame,
                                                                 debug_mode: bool = False) -> [float, int]:
    """
    Compute the volume of the airblocks composing the sector. It uses the FIRST valid interval based on the
    date_from

    Args:
        date_from: Date from of the interval
        date_to: Date to of the interval
        df_spatial_airblocks_specific_sector: DataFrame with the airblocks composing the sector
        debug_mode: If True, it prints some information

    Returns:
        The volume of the airblocks composing the sector

    Notes:
        1. The intervals of the operational/spatial information do not match the intervals of the airblocks. Thus,
            the volume is approximated using the FIRST set of airblocks in DWH used to compose the sector.
    """

    assert "Date_From" in df_spatial_airblocks_specific_sector.columns and "Date_From" in \
           df_spatial_airblocks_specific_sector, "Date_From column not found"
    assert "Date_To" in df_spatial_airblocks_specific_sector.columns and "Date_To" in \
           df_spatial_airblocks_specific_sector, "Date_To column not found"
    assert "LowerBound" in df_spatial_airblocks_specific_sector.columns, "LowerBound column not found"
    assert "UpperBound" in df_spatial_airblocks_specific_sector.columns, "UpperBound column not found"
    assert "Polygon_area" in df_spatial_airblocks_specific_sector.columns, "Polygon_area column not found"

    df_airblocks_interval = df_spatial_airblocks_specific_sector[
        (date_from >= df_spatial_airblocks_specific_sector['Date_From']) &
        (date_from < df_spatial_airblocks_specific_sector['Date_To'])]

    assert len(df_airblocks_interval) != 0, f"No airblocks found for the period {date_from} - {date_to}"

    df_airblocks_interval.loc[:, 'Area_airblock'] = df_airblocks_interval.loc[:, 'Polygon_area'] * \
                                                    (df_airblocks_interval.loc[:,
                                                     'UpperBound'] - df_airblocks_interval.loc[:, 'LowerBound'])

    area_sector = float(np.sum(df_airblocks_interval['Area_airblock'].values))
    number_airblocks = len(df_airblocks_interval)

    if debug_mode:
        print("----------------------------")
        print(f"date_from: {date_from} || date_to: {date_to}")
        print(f"area_sector: {area_sector}")
        print(f"df_airblocks_interval: {df_airblocks_interval}")

    return area_sector, number_airblocks


def check_if_volume_with_steps(date_from: datetime.date, date_to: datetime.date,
                               df_spatial_airblocks_specific_sector: pd.DataFrame,
                               debug_mode: bool = False) -> [bool]:
    """
    It extracts the lowest and highest bound for a specific sector from the FIRST interval where of validity that the
        airblocks where valid for the provided DATE_FROM.

    Args:
        date_from: Date from the intervals
        date_to: Date to the intervals
        df_spatial_airblocks_specific_sector: DataFrame with the airblocks composing the sector
        debug_mode: If True, it prints some information

    Returns:
        1 if the airblocks composing the volume have steps, 0 otherwise
    """

    assert "Date_From" in df_spatial_airblocks_specific_sector.columns and "Date_From" in \
           df_spatial_airblocks_specific_sector, "Date_From column not found"
    assert "Date_To" in df_spatial_airblocks_specific_sector.columns and "Date_To" in \
           df_spatial_airblocks_specific_sector, "Date_To column not found"
    assert "LowerBound" in df_spatial_airblocks_specific_sector.columns, "LowerBound column not found"
    assert "UpperBound" in df_spatial_airblocks_specific_sector.columns, "UpperBound column not found"

    df_airblocks_interval = get_validity_periods_for_date_from_date_to(date_from, date_to,
                                                                       df_spatial_airblocks_specific_sector,
                                                                       debug_mode=debug_mode)

    lowest_bound = np.min(df_airblocks_interval['LowerBound'].values)
    highest_bound = np.max(df_airblocks_interval['UpperBound'].values)

    lower_bounds_values = df_airblocks_interval['LowerBound'].to_numpy()
    upper_bounds_values = df_airblocks_interval['UpperBound'].to_numpy()

    if (lower_bounds_values == lowest_bound).all() and (upper_bounds_values == highest_bound).all():
        return 0
    else:
        return 1


def lowest_highest_bound_specific_sector(date_from: datetime.date, date_to: datetime.date,
                                             df_spatial_airblocks_specific_sector: pd.DataFrame,
                                             debug_mode: bool = False) -> [int, int]:
    """
    It extracts the lowest and highest bound between all possible airblocks in the period [Date_From, Date_To] .

    Args:
        date_from: Date from the intervals
        date_to: Date to the intervals
        df_spatial_airblocks_specific_sector: DataFrame with the airblocks composing the sector
        debug_mode: If True, it prints debug information

    Returns:
        The lowest and highest bound for a specific sector
    """

    assert "Date_From" in df_spatial_airblocks_specific_sector.columns and "Date_From" in \
           df_spatial_airblocks_specific_sector, "Date_From column not found"
    assert "Date_To" in df_spatial_airblocks_specific_sector.columns and "Date_To" in \
           df_spatial_airblocks_specific_sector, "Date_To column not found"
    assert "LowerBound" in df_spatial_airblocks_specific_sector.columns, "LowerBound column not found"
    assert "UpperBound" in df_spatial_airblocks_specific_sector.columns, "UpperBound column not found"

    df_airblocks_interval = get_validity_periods_for_date_from_date_to(date_from, date_to,
                                                                       df_spatial_airblocks_specific_sector,
                                                                       debug_mode=debug_mode)

    lowest_bound = np.min(df_airblocks_interval['LowerBound'].values)
    highest_bound = np.max(df_airblocks_interval['UpperBound'].values)

    return lowest_bound, highest_bound


def get_validity_periods_for_date_from_date_to(date_from: datetime.date, date_to: datetime.date,
                                               df_spatial_airblocks_specific_sector: pd.DataFrame,
                                               debug_mode: bool = False) -> [int, int]:
    """
    Given the Date_From and Date_to it returns the valid information for the period

    Args:
        date_from: Date from the intervals
        date_to: Date to the intervals
        df_spatial_airblocks_specific_sector: DataFrame with the airblocks composing the sector
        debug_mode: If True, it prints debug information

    Returns:
        Information valid in the periods [date_from, date_to]

    Example:
        1. ATCUnitCode = 'LECL', SectorCode = 'LECLVAP'
        2. Operational information valid from 2019-03-28 to 2022-03-24
        3. Airblocks active used from 2018-04-26 to 2022-03-24
    """

    assert "Date_From" in df_spatial_airblocks_specific_sector.columns and "Date_From" in \
           df_spatial_airblocks_specific_sector, "Date_From column not found"
    assert "Date_To" in df_spatial_airblocks_specific_sector.columns and "Date_To" in \
           df_spatial_airblocks_specific_sector, "Date_To column not found"

    if debug_mode:
        print("----------------------------")
        print(f"date_from: {date_from} || date_to: {date_to}")

    # Exact same periods #
    df_airblocks_interval_period_equal = df_spatial_airblocks_specific_sector[
        (df_spatial_airblocks_specific_sector['Date_From'] == date_from) &
        (df_spatial_airblocks_specific_sector['Date_To'] == date_to)]
    if debug_mode:
        print("===== EQUAL")
        print(df_airblocks_interval_period_equal[
                  ['ATCUnitCode', 'SectorCode', 'LowerBound', 'UpperBound', 'Date_From', 'Date_To']])
    if len(df_airblocks_interval_period_equal) != 0:
        df_airblocks_interval_period_equal['overlap'] = 'equal'
        return df_airblocks_interval_period_equal

    # Active #
    elif date_to == datetime.date(9999, 12, 31):
        df_airblocks_interval_period_active = df_spatial_airblocks_specific_sector[
            (df_spatial_airblocks_specific_sector['Date_To'] > date_from)]
        if debug_mode:
            print("===== ACTIVE")
            print(df_airblocks_interval_period_active[
                      ['ATCUnitCode', 'SectorCode', 'LowerBound', 'UpperBound', 'Date_From', 'Date_To']])
        assert len(df_airblocks_interval_period_active) != 0, f"No airblocks found in ACTIVE {date_from} - {date_to}"
        df_airblocks_interval_period_equal['overlap'] = 'active'
        return df_airblocks_interval_period_active

    # Multiple segments
    else:
        # Left overlap #
        df_airblocks_interval_period_left = df_spatial_airblocks_specific_sector[
            (df_spatial_airblocks_specific_sector['Date_From'] < date_from) &
            (df_spatial_airblocks_specific_sector['Date_To'] > date_from) &
            (df_spatial_airblocks_specific_sector['Date_To'] <= date_to)]
        df_airblocks_interval_period_left['overlap'] = 'left'

        # Complete overlap #
        df_airblocks_interval_period_complete = df_spatial_airblocks_specific_sector[
            (df_spatial_airblocks_specific_sector['Date_From'] < date_from) &
            (df_spatial_airblocks_specific_sector['Date_To'] > date_to)]
        df_airblocks_interval_period_complete['overlap'] = 'complete'

        # Inside #
        df_airblocks_interval_period_inside = df_spatial_airblocks_specific_sector[
            (df_spatial_airblocks_specific_sector['Date_From'] >= date_from) &
            (df_spatial_airblocks_specific_sector['Date_To'] <= date_to)]
        df_airblocks_interval_period_inside['overlap'] = 'inside'

        # Right overlap #
        df_airblocks_interval_period_right = df_spatial_airblocks_specific_sector[
            (df_spatial_airblocks_specific_sector['Date_From'] >= date_from) &
            (df_spatial_airblocks_specific_sector['Date_From'] < date_to) &
            (df_spatial_airblocks_specific_sector['Date_To'] > date_to)]
        df_airblocks_interval_period_right['overlap'] = 'right'

        df_airblocks_interval_period = pd.concat([df_airblocks_interval_period_left,
                                                  df_airblocks_interval_period_complete,
                                                  df_airblocks_interval_period_inside,
                                                  df_airblocks_interval_period_right])

        if debug_mode:
            print("===== INTERACTION")
            print(df_airblocks_interval_period[
                      ['ATCUnitCode', 'SectorCode', 'LowerBound', 'UpperBound', 'Date_From', 'Date_To', 'overlap']])

        assert len(df_airblocks_interval_period) != 0, f"No airblocks found in INTERACTION {date_from} - {date_to}"

        return df_airblocks_interval_period
