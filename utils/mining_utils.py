"""
Data mining utils
acg
"""

import pandas as pd
from pandas import DataFrame
import datetime
import numpy as np


def merge_operational_capacity_for_specific_sector_to_spatial_structures(
        df_airspace_spatial_specific_sector: pd.DataFrame, df_airspace_operational_specific_sector: pd.DataFrame,
        debug_mode: bool = False) -> pd.DataFrame:

    """
    Given a DataFrame with spatial information and another DataFrame with the capacity (operational information),
        it returns a DataFrame with the spatial information and the operational information merged into one row
        per valid period.

    Args:
        df_airspace_spatial_specific_sector: DataFrame with spatial information
        df_airspace_operational_specific_sector: DataFrame with operational information (capacity)
        debug_mode: Boolean to activate debug mode

    Returns:
        DataFrame with spatial and the capacity merged into one row per valid period

    Notes:
        1. The approach is based on the overlying between the intervals of validity of the spatial and the capacity
            (operational information).
        2. The intervals of the capacity must be consecutive and non-overlapping for optimal results
            2.1 Use capacity_prediction/airspace_structures_operational.py/
                    compact_airspace_structures_operational_by_capacity_date_from_date_to() to compact the capacity
                    intervals
        3. This merge will create the final intervals of validity
    """

    assert "LevelType" in df_airspace_spatial_specific_sector.columns, "LevelType column not found"
    assert "ATCUnitCode" in df_airspace_spatial_specific_sector.columns, "ATCUnitCode column not found"
    assert "ATCType" in df_airspace_spatial_specific_sector.columns, "ATCType column not found"
    assert "SectorCode" in df_airspace_spatial_specific_sector.columns, "SectorCode column not found"
    assert "Date_From" in df_airspace_spatial_specific_sector.columns and "Date_From" in \
           df_airspace_operational_specific_sector, "Date_From column not found"
    assert "Date_To" in df_airspace_spatial_specific_sector.columns and "Date_To" in \
           df_airspace_operational_specific_sector, "Date_To column not found"
    assert "Active" in df_airspace_spatial_specific_sector.columns, "Active column not found"
    assert "Capacity" in df_airspace_operational_specific_sector.columns, "Capacity column not found"

    df_dataset = pd.DataFrame()  # Initialize output DataFrame

    # Shared information #
    new_row = {
        'LevelType': df_airspace_spatial_specific_sector.iloc[0]['LevelType'],
        'ATCUnitCode': df_airspace_spatial_specific_sector.iloc[0]['ATCUnitCode'],
        'SectorCode': df_airspace_spatial_specific_sector.iloc[0]['SectorCode'],
        'ATCType': df_airspace_spatial_specific_sector.iloc[0]['ATCType'],
    }

    for index_spatial, row_spatial in df_airspace_spatial_specific_sector.iterrows():

        if debug_mode:
            print("-------------------------------------")
            print(f"level_type_spatial: {row_spatial['LevelType']} | unit_code_spatial: {row_spatial['ATCUnitCode']}")
            print(f"date_from_spatial: {row_spatial['Date_From']} | date_to_spatial: {row_spatial['Date_To']}")

        # Exact same periods #
        df_info_operational_period_equal = df_airspace_operational_specific_sector[
            (df_airspace_operational_specific_sector['Date_From'] == row_spatial['Date_From']) &
            (df_airspace_operational_specific_sector['Date_To'] == row_spatial['Date_To'])]

        if len(df_info_operational_period_equal) != 0:
            if debug_mode:
                print("===== EQUAL")
                print(df_info_operational_period_equal)

            new_row['Polygon_area'] = row_spatial['Polygon_area']
            new_row['Perimeter'] = row_spatial['Perimeter']
            new_row['NumberOfVertices'] = row_spatial['NumberOfVertices']
            new_row['Centroid_Latitude'] = row_spatial['Centroid_Latitude']
            new_row['Centroid_Longitude'] = row_spatial['Centroid_Longitude']
            new_row['Capacity'] = df_info_operational_period_equal['Capacity'].values[0]
            new_row['Date_From'] = row_spatial['Date_From']
            new_row['Date_To'] = row_spatial['Date_To']

            df_dataset = pd.concat([df_dataset, pd.DataFrame([new_row])], ignore_index=True)

        # Active interval #
        elif row_spatial['Active']:
            df_info_operational_period_active = df_airspace_operational_specific_sector[
                (df_airspace_operational_specific_sector['Date_To'] > row_spatial['Date_From'])]

            if debug_mode:
                print("===== ACTIVE")
                print(df_info_operational_period_active)

            date_from = row_spatial['Date_From']
            for index_operational, row_operational in df_info_operational_period_active.iterrows():
                new_row['Polygon_area'] = row_spatial['Polygon_area']
                new_row['Perimeter'] = row_spatial['Perimeter']
                new_row['NumberOfVertices'] = row_spatial['NumberOfVertices']
                new_row['Centroid_Latitude'] = row_spatial['Centroid_Latitude']
                new_row['Centroid_Longitude'] = row_spatial['Centroid_Longitude']
                new_row['Capacity'] = row_operational['Capacity']
                new_row['Date_From'] = date_from
                new_row['Date_To'] = row_operational['Date_To']

                df_dataset = pd.concat([df_dataset, pd.DataFrame([new_row])], ignore_index=True)

                # The date from of the next iteration is the date to of the previous one
                if not row_operational['Active']:
                    date_from = row_operational['Date_To']
        else:
            # Left overlap #
            df_info_operational_period_left = df_airspace_operational_specific_sector[
                (df_airspace_operational_specific_sector['Date_From'] < row_spatial['Date_From']) &
                (df_airspace_operational_specific_sector['Date_To'] > row_spatial['Date_From']) &
                (df_airspace_operational_specific_sector['Date_To'] <= row_spatial['Date_To'])]
            df_info_operational_period_left['overlap'] = 'left'

            # Complete overlap #
            df_info_operational_period_complete = df_airspace_operational_specific_sector[
                (df_airspace_operational_specific_sector['Date_From'] < row_spatial['Date_From']) &
                (df_airspace_operational_specific_sector['Date_To'] > row_spatial['Date_To'])]
            df_info_operational_period_complete['overlap'] = 'complete'

            # Inside #
            df_info_operational_period_inside = df_airspace_operational_specific_sector[
                (df_airspace_operational_specific_sector['Date_From'] >= row_spatial['Date_From']) &
                (df_airspace_operational_specific_sector['Date_To'] < row_spatial['Date_To'])]
            df_info_operational_period_inside['overlap'] = 'inside'

            # Right overlap #
            df_info_operational_period_right = df_airspace_operational_specific_sector[
                (df_airspace_operational_specific_sector['Date_From'] >= row_spatial['Date_From']) &
                (df_airspace_operational_specific_sector['Date_From'] < row_spatial['Date_To']) &
                (df_airspace_operational_specific_sector['Date_To'] >= row_spatial['Date_To'])]
            df_info_operational_period_right['overlap'] = 'right'

            df_info_operational_period = pd.concat([df_info_operational_period_left,
                                                    df_info_operational_period_complete,
                                                    df_info_operational_period_inside,
                                                    df_info_operational_period_right])

            if debug_mode:
                print("===== INTERACTION")
                print(df_info_operational_period[['Capacity', 'Date_From', 'Date_To', 'overlap']])

            for index_operation, row_operational in df_info_operational_period.iterrows():
                new_row['Polygon_area'] = row_spatial['Polygon_area']
                new_row['Perimeter'] = row_spatial['Perimeter']
                new_row['NumberOfVertices'] = row_spatial['NumberOfVertices']
                new_row['Centroid_Latitude'] = row_spatial['Centroid_Latitude']
                new_row['Centroid_Longitude'] = row_spatial['Centroid_Longitude']
                new_row['Capacity'] = row_operational['Capacity']

                if row_operational['overlap'] == 'complete':
                    new_row['Date_From'] = row_spatial['Date_From']
                    new_row['Date_To'] = row_spatial['Date_To']
                elif row_operational['overlap'] == 'left':
                    new_row['Date_From'] = row_spatial['Date_From']
                    new_row['Date_To'] = row_operational['Date_To']
                elif row_operational['overlap'] == 'inside':
                    new_row['Date_From'] = row_operational['Date_From']
                    new_row['Date_To'] = row_operational['Date_To']
                elif row_operational['overlap'] == 'right':
                    new_row['Date_From'] = row_operational['Date_From']
                    new_row['Date_To'] = row_spatial['Date_To']

                df_dataset = pd.concat([df_dataset, pd.DataFrame([new_row])], ignore_index=True)

    return df_dataset

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


def features_cfmu_clustered_routes_specific_sector(df_dataset_specific_sector: pd.DataFrame,
                                                   df_cfmu_clustered_routes_specific_sector: pd.DataFrame) -> \
        pd.DataFrame:
    """
    Add information about the available clustered routes in the sector

    Args:
        df_dataset_specific_sector:
        df_cfmu_clustered_routes_specific_sector:

    Returns:
        Original DataFrame with the information about the clustered routes in the sector
    Notes:
        1. The information is added to the entire dataset as it is identical for each sector and each interval
    """

    # Number of fluxes #
    df_dataset_specific_sector = add_num_fluxes(df_dataset_specific_sector, df_cfmu_clustered_routes_specific_sector)

    # Attitudes #
    df_dataset_specific_sector = add_attitudes(df_dataset_specific_sector, df_cfmu_clustered_routes_specific_sector)

    # Number of flights #
    df_dataset_specific_sector = add_num_flights(df_dataset_specific_sector, df_cfmu_clustered_routes_specific_sector)

    # Time in route #
    df_dataset_specific_sector = add_time_in_route(df_dataset_specific_sector, df_cfmu_clustered_routes_specific_sector)

    # Latitudes and longitudes #
    df_dataset_specific_sector = add_latitudes_and_longitudes(df_dataset_specific_sector, df_cfmu_clustered_routes_specific_sector)

    return df_dataset_specific_sector


def add_latitudes_and_longitudes(df_dataset_specific_sector: pd.DataFrame,
                                 df_cfmu_clustered_routes_specific_sector: pd.DataFrame) -> pd.DataFrame:
    """ Add the latitudes and longitudes of the fluxes in the sector to the dataset """

    if len(df_cfmu_clustered_routes_specific_sector) != 0:
        df_dataset_specific_sector.loc[:, 'all_latIN_fluxes'] = df_dataset_specific_sector.apply(
            lambda x: df_cfmu_clustered_routes_specific_sector['latIN'].values.tolist(), axis=1)
        df_dataset_specific_sector.loc[:, 'all_lngIN_fluxes'] = df_dataset_specific_sector.apply(
            lambda x: df_cfmu_clustered_routes_specific_sector['lngIN'].values.tolist(), axis=1)
        df_dataset_specific_sector.loc[:, 'all_latOUT_fluxes'] = df_dataset_specific_sector.apply(
            lambda x: df_cfmu_clustered_routes_specific_sector['latOUT'].values.tolist(), axis=1)
        df_dataset_specific_sector.loc[:, 'all_lngOUT_fluxes'] = df_dataset_specific_sector.apply(
            lambda x: df_cfmu_clustered_routes_specific_sector['lngOUT'].values.tolist(), axis=1)

        df_dataset_specific_sector.loc[:, 'avg_latIN_fluxes'] = np.average(
            df_cfmu_clustered_routes_specific_sector['latIN'].values)
        df_dataset_specific_sector.loc[:, 'avg_lngIN_fluxes'] = np.average(
            df_cfmu_clustered_routes_specific_sector['lngIN'].values)
        df_dataset_specific_sector.loc[:, 'avg_latOUT_fluxes'] = np.average(
            df_cfmu_clustered_routes_specific_sector['latOUT'].values)
        df_dataset_specific_sector.loc[:, 'avg_lngOUT_fluxes'] = np.average(
            df_cfmu_clustered_routes_specific_sector['lngOUT'].values)
    else:
        df_dataset_specific_sector.loc[:, 'all_latIN_fluxes'] = np.nan
        df_dataset_specific_sector.loc[:, 'all_lngIN_fluxes'] = np.nan
        df_dataset_specific_sector.loc[:, 'all_latOUT_fluxes'] = np.nan
        df_dataset_specific_sector.loc[:, 'all_lngOUT_fluxes'] = np.nan

        df_dataset_specific_sector.loc[:, 'avg_latIN_fluxes'] = np.nan
        df_dataset_specific_sector.loc[:, 'avg_lngIN_fluxes'] = np.nan
        df_dataset_specific_sector.loc[:, 'avg_latOUT_fluxes'] = np.nan
        df_dataset_specific_sector.loc[:, 'avg_lngOUT_fluxes'] = np.nan

    return df_dataset_specific_sector


def add_time_in_route(df_dataset_specific_sector: pd.DataFrame, df_cfmu_clustered_routes_specific_sector: pd.DataFrame) -> \
    pd.DataFrame:
    """ Add the time in route of the fluxes in the sector to the dataset """

    if len(df_cfmu_clustered_routes_specific_sector) != 0:
        df_dataset_specific_sector.loc[:, 'sum_avgTimeInRoute'] = np.sum(
            df_cfmu_clustered_routes_specific_sector['avgTimeInRoute'].values)
        # df_dataset_specific_sector.loc[:, 'min_avgTimeInRoute'] = np.min(
        #     df_cfmu_clustered_routes_specific_sector['avgTimeInRoute'].values)
        df_dataset_specific_sector.loc[:, 'max_avgTimeInRoute'] = np.max(
            df_cfmu_clustered_routes_specific_sector['avgTimeInRoute'].values)
        df_dataset_specific_sector.loc[:, 'median_avgTimeInRoute'] = np.median(
            df_cfmu_clustered_routes_specific_sector['avgTimeInRoute'].values)
        df_dataset_specific_sector.loc[:, 'std_avgTimeInRoute'] = round(float(np.std(
            df_cfmu_clustered_routes_specific_sector['avgTimeInRoute'].values)), 3)
        df_cfmu_clustered_routes_specific_sector['avgTimeInRoute_divided_num_flight'] = \
            df_cfmu_clustered_routes_specific_sector['avgTimeInRoute'] / df_cfmu_clustered_routes_specific_sector['nFlights']
        # Average of the average route time with respect the number of flights in the flux
        df_dataset_specific_sector.loc[:, 'sum_avgTimeInRoute_divided_num_flight'] = np.sum(
            df_cfmu_clustered_routes_specific_sector['avgTimeInRoute_divided_num_flight'].values)
        df_dataset_specific_sector.loc[:, 'percentile25_avgTimeInRoute'] = round(float(np.percentile(
            df_cfmu_clustered_routes_specific_sector['avgTimeInRoute'].values, 25)), 3)
        df_dataset_specific_sector.loc[:, 'percentile75_avgTimeInRoute'] = round(float(np.percentile(
            df_cfmu_clustered_routes_specific_sector['avgTimeInRoute'].values, 75)), 3)
    else:
        df_dataset_specific_sector.loc[:, 'sum_avgTimeInRoute'] = np.nan
        # df_dataset_specific_sector.loc[:, 'min_avgTimeInRoute'] = np.nan
        df_dataset_specific_sector.loc[:, 'max_avgTimeInRoute'] = np.nan
        df_dataset_specific_sector.loc[:, 'median_avgTimeInRoute'] = np.nan
        df_dataset_specific_sector.loc[:, 'std_avgTimeInRoute'] = np.nan
        df_dataset_specific_sector.loc[:, 'sum_avgTimeInRoute_divided_num_flight'] = np.nan
        df_dataset_specific_sector.loc[:, 'percentile25_avgTimeInRoute'] = np.nan
        df_dataset_specific_sector.loc[:, 'percentile75_avgTimeInRoute'] = np.nan

    return df_dataset_specific_sector


def add_num_flights(df_dataset_specific_sector: pd.DataFrame, df_cfmu_clustered_routes_specific_sector: pd.DataFrame) -> \
        pd.DataFrame:
    """ Add the number of fluxes in the sector to the dataset """

    if len(df_cfmu_clustered_routes_specific_sector) != 0:
        df_dataset_specific_sector.loc[:, 'sum_nFlights'] = np.sum(
            df_cfmu_clustered_routes_specific_sector['nFlights'].values)
        # df_dataset_specific_sector.loc[:, 'min_nFlights'] = np.min(
        #     df_cfmu_clustered_routes_specific_sector['nFlights'].values)
        df_dataset_specific_sector.loc[:, 'max_nFlights'] = np.max(
            df_cfmu_clustered_routes_specific_sector['nFlights'].values)
        df_dataset_specific_sector.loc[:, 'avg_nFlights'] = np.average(
            df_cfmu_clustered_routes_specific_sector['nFlights'].values)
        df_dataset_specific_sector.loc[:, 'median_nFlights'] = np.median(
            df_cfmu_clustered_routes_specific_sector['nFlights'].values)
        df_dataset_specific_sector.loc[:, 'std_nFlights'] = round(float(
            np.std(df_cfmu_clustered_routes_specific_sector['nFlights'].values)), 3)
        df_dataset_specific_sector.loc[:, 'percentile25_nFlights'] = round(float(
            np.percentile(df_cfmu_clustered_routes_specific_sector['nFlights'].values, 25)), 3)
        df_dataset_specific_sector.loc[:, 'percentile75_nFlights'] = round(float(
            np.percentile(df_cfmu_clustered_routes_specific_sector['nFlights'].values, 75)), 3)
    else:
        df_dataset_specific_sector.loc[:, 'sum_nFlights'] = np.nan
        # df_dataset_specific_sector.loc[:, 'min_nFlights'] = np.nan
        df_dataset_specific_sector.loc[:, 'max_nFlights'] = np.nan
        df_dataset_specific_sector.loc[:, 'avg_nFlights'] = np.nan
        df_dataset_specific_sector.loc[:, 'median_nFlights'] = np.nan
        df_dataset_specific_sector.loc[:, 'std_nFlights'] = np.nan
        df_dataset_specific_sector.loc[:, 'percentile25_nFlights'] = np.nan
        df_dataset_specific_sector.loc[:, 'percentile75_nFlights'] = np.nan

    return df_dataset_specific_sector


def add_attitudes(df_dataset_specific_sector: pd.DataFrame, df_cfmu_clustered_routes_specific_sector: pd.DataFrame) -> \
    pd.DataFrame:
    """ Add the attitudes of the fluxes in the sector to the dataset """

    if len(df_cfmu_clustered_routes_specific_sector) != 0:

        df_dataset_specific_sector.loc[:, 'all_attitudesIN'] = df_dataset_specific_sector.apply(
            lambda x: df_cfmu_clustered_routes_specific_sector['attitudIN'].values.tolist(), axis=1)
        df_dataset_specific_sector.loc[:, 'all_attitudesOUT'] = df_dataset_specific_sector.apply(
            lambda x: df_cfmu_clustered_routes_specific_sector['attitudOUT'].values.tolist(), axis=1)

        all_attitudes_in = df_cfmu_clustered_routes_specific_sector['attitudIN'].values.tolist()
        all_attitudes_out = df_cfmu_clustered_routes_specific_sector['attitudOUT'].values.tolist()
        attitude_type = {'cruise_cruise': 0,  # cruise-cruise
                         'cruise_descend_comb': 0,  # cruise-descend | descend-cruise
                         'cruise_climb_comb': 0,  # cruise-climb | climb-cruise
                         'decent_descend': 0,  # decent-descend
                         'climb_climb': 0,  # climb-climb
                         'descend_climb_comb': 0}  # climb_descend | descend_climb

        for attitude_in, attitudes_out in zip(all_attitudes_in, all_attitudes_out):
            if attitude_in == 'CRUISE' and attitudes_out == 'CRUISE':
                attitude_type['cruise_cruise'] += 1
            elif (attitude_in == 'CRUISE' and attitudes_out == 'DESCEND') or (
                    attitude_in == 'DESCEND' and attitudes_out == 'CRUISE'):
                attitude_type['cruise_descend_comb'] += 1
            elif (attitude_in == 'CRUISE' and attitudes_out == 'CLIMB') or (
                    attitude_in == 'CLIMB' and attitudes_out == 'CRUISE'):
                attitude_type['cruise_climb_comb'] += 1
            elif attitude_in == 'DESCEND' and attitudes_out == 'DESCEND':
                attitude_type['decent_descend'] += 1
            elif attitude_in == 'CLIMB' and attitudes_out == 'CLIMB':
                attitude_type['climb_climb'] += 1
            elif (attitude_in == 'DESCEND' and attitudes_out == 'CLIMB') or (
                    attitude_in == 'CLIMB' and attitudes_out == 'DESCEND'):
                attitude_type['descend_climb_comb'] += 1
            else:
                raise ValueError('Attitude not found')

        for key in attitude_type.keys():
            df_dataset_specific_sector.loc[:, f'attitude_{key}'] = attitude_type[key]

    else:
        df_dataset_specific_sector.loc[:, 'all_attitudesIN'] = np.nan
        df_dataset_specific_sector.loc[:, 'all_attitudesOUT'] = np.nan
        df_dataset_specific_sector.loc[:, 'attitude_cruise_cruise'] = np.nan
        df_dataset_specific_sector.loc[:, 'attitude_cruise_descend_comb'] = np.nan
        df_dataset_specific_sector.loc[:, 'attitude_cruise_climb_comb'] = np.nan
        df_dataset_specific_sector.loc[:, 'attitude_decent_descend'] = np.nan
        df_dataset_specific_sector.loc[:, 'attitude_climb_climb'] = np.nan
        df_dataset_specific_sector.loc[:, 'attitude_descend_climb_comb'] = np.nan

    return df_dataset_specific_sector


def add_num_fluxes(df_dataset_specific_sector: pd.DataFrame, df_cfmu_clustered_routes_specific_sector: pd.DataFrame) -> \
    pd.DataFrame:
    """ Add the number of fluxes in the sector to the dataset """

    if len(df_cfmu_clustered_routes_specific_sector) != 0:
        df_dataset_specific_sector.loc[:, 'Num_fluxes'] = len(df_cfmu_clustered_routes_specific_sector)
    else:
        df_dataset_specific_sector.loc[:, 'Num_fluxes'] = np.nan

    return df_dataset_specific_sector


def features_cfmu_clustered_routes_interacting_spacific_sector(
        df_dataset_specific_sector: pd.DataFrame, df_cfmu_clustered_routes_interacting_specific_sector: pd.DataFrame,
        list_all_clustered_routes_specific_sector: list) \
        -> pd.DataFrame:
    """
    Add information about the interacting routes inside a specific sector

    Args:
        df_dataset_specific_sector: Main DataFrame with the dataset
        df_cfmu_clustered_routes_interacting_specific_sector: DataFrame with the interacting routes inside the sector
        list_all_clustered_routes_specific_sector: List with all the clustered routes inside the sector

    Returns:
        Original DataFrame with the information about the interacting routes inside the sector

    Notes:
        1. NaN values are returned if there is no information from fluxes. -1 is returned if there is no information
            for the specific interaction type
    """

    df_dataset_specific_sector = add_interacting_fluxes_by_type(df_dataset_specific_sector,
                                                                df_cfmu_clustered_routes_interacting_specific_sector,
                                                                list_all_clustered_routes_specific_sector)

    df_dataset_specific_sector = add_number_of_flights_interacting_fluxes_by_type(df_dataset_specific_sector,
                                                                                  df_cfmu_clustered_routes_interacting_specific_sector)

    df_dataset_specific_sector = add_average_time_route_interacting_fluxes_by_type(df_dataset_specific_sector,
                                                                                   df_cfmu_clustered_routes_interacting_specific_sector)

    df_dataset_specific_sector = add_complexity_interacting_fluxes_by_type(df_dataset_specific_sector,
                                                                           df_cfmu_clustered_routes_interacting_specific_sector)

    df_dataset_specific_sector = add_complexity_interacting_fluxes(df_dataset_specific_sector,
                                                                   df_cfmu_clustered_routes_interacting_specific_sector)

    return df_dataset_specific_sector


def add_complexity_interacting_fluxes(df_dataset_specific_sector, df_cfmu_clustered_routes_interacting_specific_sector):

    if len(df_cfmu_clustered_routes_interacting_specific_sector) != 0:
        # Basic complexity
        df_dataset_specific_sector.loc[:, f'max_complexity'] = \
            3 * df_dataset_specific_sector.loc[:, f'max_complexity_type_3'] + \
            2 * df_dataset_specific_sector.loc[:, f'max_complexity_type_2'] + \
            1 * df_dataset_specific_sector.loc[:, f'max_complexity_type_1']
        df_dataset_specific_sector.loc[:, f'sum_complexity'] = \
            3 * df_dataset_specific_sector.loc[:, f'sum_complexity_type_3'] + \
            2 * df_dataset_specific_sector.loc[:, f'sum_complexity_type_2'] + \
            1 * df_dataset_specific_sector.loc[:, f'sum_complexity_type_1']
        df_dataset_specific_sector.loc[:, f'median_complexity'] = \
            3 * df_dataset_specific_sector.loc[:, f'median_complexity_type_3'] + \
            2 * df_dataset_specific_sector.loc[:, f'median_complexity_type_2'] + \
            1 * df_dataset_specific_sector.loc[:, f'median_complexity_type_1']
        df_dataset_specific_sector.loc[:, f'std_complexity'] = \
            3 * df_dataset_specific_sector.loc[:, f'std_complexity_type_3'] + \
            2 * df_dataset_specific_sector.loc[:, f'std_complexity_type_2'] + \
            1 * df_dataset_specific_sector.loc[:, f'std_complexity_type_1']
        df_dataset_specific_sector.loc[:, f'percentile25_complexity'] = \
            3 * df_dataset_specific_sector.loc[:, f'percentile25_complexity_type_3'] + \
            2 * df_dataset_specific_sector.loc[:, f'percentile25_complexity_type_2'] + \
            1 * df_dataset_specific_sector.loc[:, f'percentile25_complexity_type_1']
        df_dataset_specific_sector.loc[:, f'percentile75_complexity'] = \
            3 * df_dataset_specific_sector.loc[:, f'percentile75_complexity_type_3'] + \
            2 * df_dataset_specific_sector.loc[:, f'percentile75_complexity_type_2'] + \
            1 * df_dataset_specific_sector.loc[:, f'percentile75_complexity_type_1']

        # Harmonic complexity
        df_dataset_specific_sector.loc[:, f'max_harmonic_complexity'] = \
            3 * df_dataset_specific_sector.loc[:, f'max_harmonic_complexity_type_3'] + \
            2 * df_dataset_specific_sector.loc[:, f'max_harmonic_complexity_type_2'] + \
            1 * df_dataset_specific_sector.loc[:, f'max_harmonic_complexity_type_1']
        df_dataset_specific_sector.loc[:, f'sum_harmonic_complexity'] = \
            3 * df_dataset_specific_sector.loc[:, f'sum_harmonic_complexity_type_3'] + \
            2 * df_dataset_specific_sector.loc[:, f'sum_harmonic_complexity_type_2'] + \
            1 * df_dataset_specific_sector.loc[:, f'sum_harmonic_complexity_type_1']
        df_dataset_specific_sector.loc[:, f'median_harmonic_complexity'] = \
            3 * df_dataset_specific_sector.loc[:, f'median_harmonic_complexity_type_3'] + \
            2 * df_dataset_specific_sector.loc[:, f'median_harmonic_complexity_type_2'] + \
            1 * df_dataset_specific_sector.loc[:, f'median_harmonic_complexity_type_1']
        df_dataset_specific_sector.loc[:, f'std_harmonic_complexity'] = \
            3 * df_dataset_specific_sector.loc[:, f'std_harmonic_complexity_type_3'] + \
            2 * df_dataset_specific_sector.loc[:, f'std_harmonic_complexity_type_2'] + \
            1 * df_dataset_specific_sector.loc[:, f'std_harmonic_complexity_type_1']
        df_dataset_specific_sector.loc[:, f'percentile25_harmonic_complexity'] = \
            3 * df_dataset_specific_sector.loc[:, f'percentile25_harmonic_complexity_type_3'] + \
            2 * df_dataset_specific_sector.loc[:, f'percentile25_harmonic_complexity_type_2'] + \
            1 * df_dataset_specific_sector.loc[:, f'percentile25_harmonic_complexity_type_1']
        df_dataset_specific_sector.loc[:, f'percentile75_harmonic_complexity'] = \
            3 * df_dataset_specific_sector.loc[:, f'percentile75_harmonic_complexity_type_3'] + \
            2 * df_dataset_specific_sector.loc[:, f'percentile75_harmonic_complexity_type_2'] + \
            1 * df_dataset_specific_sector.loc[:, f'percentile75_harmonic_complexity_type_1']

    else:
        df_dataset_specific_sector.loc[:, f'max_complexity'] = np.nan
        df_dataset_specific_sector.loc[:, f'sum_complexity'] = np.nan
        df_dataset_specific_sector.loc[:, f'median_complexity'] = np.nan
        df_dataset_specific_sector.loc[:, f'std_complexity'] = np.nan
        df_dataset_specific_sector.loc[:, f'percentile25_harmonic_complexity'] = np.nan
        df_dataset_specific_sector.loc[:, f'percentile75_harmonic_complexity'] = np.nan

        df_dataset_specific_sector.loc[:, f'max_harmonic_complexity'] = np.nan
        df_dataset_specific_sector.loc[:, f'sum_harmonic_complexity'] = np.nan
        df_dataset_specific_sector.loc[:, f'median_harmonic_complexity'] = np.nan
        df_dataset_specific_sector.loc[:, f'std_harmonic_complexity'] = np.nan
        df_dataset_specific_sector.loc[:, f'percentile25_harmonic_complexity'] = np.nan
        df_dataset_specific_sector.loc[:, f'percentile75_harmonic_complexity'] = np.nan

    return df_dataset_specific_sector


def add_complexity_interacting_fluxes_by_type(
        df_dataset_specific_sector: pd.DataFrame, df_cfmu_clustered_routes_interacting_specific_sector: pd.DataFrame) \
        -> pd.DataFrame:
    """
    Compute metrics about the complexity of the interacting fluxes by interaction type

    Notes:
        1. The complexity is estimates as a weighted measure between the number of flights and the average time in route
            (num. flight A x num. flight B) / (avg. time in route A + avg. time in route B)
        2. flow_interacting_type=None it is not filtered by the interaction type
    """

    # Basic complexity
    df_dataset_specific_sector = _metrics_complexity_specific_interaction_type(
        df_dataset_specific_sector, df_cfmu_clustered_routes_interacting_specific_sector, flow_interacting_type=None)
    df_dataset_specific_sector = _metrics_complexity_specific_interaction_type(
        df_dataset_specific_sector, df_cfmu_clustered_routes_interacting_specific_sector, flow_interacting_type=1)
    df_dataset_specific_sector = _metrics_complexity_specific_interaction_type(
        df_dataset_specific_sector, df_cfmu_clustered_routes_interacting_specific_sector, flow_interacting_type=2)
    df_dataset_specific_sector = _metrics_complexity_specific_interaction_type(
        df_dataset_specific_sector, df_cfmu_clustered_routes_interacting_specific_sector, flow_interacting_type=3)

    # Harmonic complexity
    df_dataset_specific_sector = _metrics_harmonic_complexity_specific_interaction_type(
        df_dataset_specific_sector, df_cfmu_clustered_routes_interacting_specific_sector, flow_interacting_type=None)
    df_dataset_specific_sector = _metrics_harmonic_complexity_specific_interaction_type(
        df_dataset_specific_sector, df_cfmu_clustered_routes_interacting_specific_sector, flow_interacting_type=1)
    df_dataset_specific_sector = _metrics_harmonic_complexity_specific_interaction_type(
        df_dataset_specific_sector, df_cfmu_clustered_routes_interacting_specific_sector, flow_interacting_type=2)
    df_dataset_specific_sector = _metrics_harmonic_complexity_specific_interaction_type(
        df_dataset_specific_sector, df_cfmu_clustered_routes_interacting_specific_sector, flow_interacting_type=3)

    return df_dataset_specific_sector


def _metrics_harmonic_complexity_specific_interaction_type(df_dataset_specific_sector: pd.DataFrame,
                                                           df_cfmu_clustered_routes_interacting_specific_sector: pd.DataFrame,
                                                           flow_interacting_type: int) -> pd.DataFrame:
    """ Compute metrics about the harmonic complexity of the interacting fluxes per interaction type"""

    if len(df_cfmu_clustered_routes_interacting_specific_sector) != 0:

        if flow_interacting_type is not None:
            df_interacting_fluxes_type = df_cfmu_clustered_routes_interacting_specific_sector[
                df_cfmu_clustered_routes_interacting_specific_sector['flowInteractingType'] == flow_interacting_type]
        else:
            df_interacting_fluxes_type = df_cfmu_clustered_routes_interacting_specific_sector

        if len(df_interacting_fluxes_type) != 0:
            df_interacting_fluxes_type['harmonic_complexity'] = \
                2 / (1 / (1 / df_interacting_fluxes_type['nFlights'] + 1 / df_interacting_fluxes_type['nFlightsInteracting']) + 1 / (1 / df_interacting_fluxes_type['avgTimeInRoute'] + 1 / df_interacting_fluxes_type['avgTimeInRouteInteracting']))

            df_dataset_specific_sector.loc[:, f'max_harmonic_complexity_type_{flow_interacting_type}'] = \
                np.max(df_interacting_fluxes_type['harmonic_complexity'].values)
            df_dataset_specific_sector.loc[:, f'sum_harmonic_complexity_type_{flow_interacting_type}'] = \
                np.sum(df_interacting_fluxes_type['harmonic_complexity'].values)
            df_dataset_specific_sector.loc[:, f'median_harmonic_complexity_type_{flow_interacting_type}'] = \
                np.median(df_interacting_fluxes_type['harmonic_complexity'].values)
            df_dataset_specific_sector.loc[:, f'std_harmonic_complexity_type_{flow_interacting_type}'] = \
                round(float(np.std(df_interacting_fluxes_type['harmonic_complexity'].values)), 3)
            df_dataset_specific_sector.loc[:, f'percentile25_harmonic_complexity_type_{flow_interacting_type}'] = \
                round(float(np.percentile(df_interacting_fluxes_type['harmonic_complexity'].values, 25)), 3)
            df_dataset_specific_sector.loc[:, f'percentile75_harmonic_complexity_type_{flow_interacting_type}'] = \
                round(float(np.percentile(df_interacting_fluxes_type['harmonic_complexity'].values, 75)), 3)

        else:
            df_dataset_specific_sector.loc[:, f'max_harmonic_complexity_type_{flow_interacting_type}'] = 0
            df_dataset_specific_sector.loc[:, f'sum_harmonic_complexity_type_{flow_interacting_type}'] = 0
            df_dataset_specific_sector.loc[:, f'median_harmonic_complexity_type_{flow_interacting_type}'] = 0
            df_dataset_specific_sector.loc[:, f'std_harmonic_complexity_type_{flow_interacting_type}'] = 0
            df_dataset_specific_sector.loc[:, f'percentile25_harmonic_complexity_type_{flow_interacting_type}'] = 0
            df_dataset_specific_sector.loc[:, f'percentile75_harmonic_complexity_type_{flow_interacting_type}'] = 0

    else:
        df_dataset_specific_sector.loc[:, f'max_harmonic_complexity_type_{flow_interacting_type}'] = np.nan
        df_dataset_specific_sector.loc[:, f'sum_harmonic_complexity_type_{flow_interacting_type}'] = np.nan
        df_dataset_specific_sector.loc[:, f'median_harmonic_complexity_type_{flow_interacting_type}'] = np.nan
        df_dataset_specific_sector.loc[:, f'std_harmonic_complexity_type_{flow_interacting_type}'] = np.nan
        df_dataset_specific_sector.loc[:, f'percentile25_harmonic_complexity_type_{flow_interacting_type}'] = np.nan
        df_dataset_specific_sector.loc[:, f'percentile75_harmonic_complexity_type_{flow_interacting_type}'] = np.nan

    return df_dataset_specific_sector


def _metrics_complexity_specific_interaction_type(df_dataset_specific_sector: pd.DataFrame,
                                                  df_cfmu_clustered_routes_interacting_specific_sector: pd.DataFrame,
                                                  flow_interacting_type: int) -> pd.DataFrame:
    """ Compute metrics about the complexity of the interacting fluxes per interaction type"""

    if len(df_cfmu_clustered_routes_interacting_specific_sector) != 0:

        if flow_interacting_type is not None:
            df_interacting_fluxes_type = df_cfmu_clustered_routes_interacting_specific_sector[
                df_cfmu_clustered_routes_interacting_specific_sector['flowInteractingType'] == flow_interacting_type]
        else:
            df_interacting_fluxes_type = df_cfmu_clustered_routes_interacting_specific_sector

        if len(df_interacting_fluxes_type) != 0:
            df_interacting_fluxes_type['complexity'] = \
                (df_interacting_fluxes_type['nFlights'] + df_interacting_fluxes_type['nFlightsInteracting']) * \
                (df_interacting_fluxes_type['avgTimeInRoute'] + df_interacting_fluxes_type['avgTimeInRouteInteracting'])

            df_dataset_specific_sector.loc[:, f'max_complexity_type_{flow_interacting_type}'] = \
                np.max( df_interacting_fluxes_type['complexity'].values)
            df_dataset_specific_sector.loc[:, f'sum_complexity_type_{flow_interacting_type}'] = \
                np.sum(df_interacting_fluxes_type['complexity'].values)
            df_dataset_specific_sector.loc[:, f'median_complexity_type_{flow_interacting_type}'] = \
                np.median(df_interacting_fluxes_type['complexity'].values)
            df_dataset_specific_sector.loc[:, f'std_complexity_type_{flow_interacting_type}'] = \
                round(float(np.std(df_interacting_fluxes_type['complexity'].values)), 3)
            df_dataset_specific_sector.loc[:, f'percentile25_complexity_type_{flow_interacting_type}'] = \
                round(float(np.percentile(df_interacting_fluxes_type['complexity'].values, 25)), 3)
            df_dataset_specific_sector.loc[:, f'percentile75_complexity_type_{flow_interacting_type}'] = \
                round(float(np.percentile(df_interacting_fluxes_type['complexity'].values, 75)), 3)

        else:
            df_dataset_specific_sector.loc[:, f'max_complexity_type_{flow_interacting_type}'] = 0
            df_dataset_specific_sector.loc[:, f'sum_complexity_type_{flow_interacting_type}'] = 0
            df_dataset_specific_sector.loc[:, f'median_complexity_type_{flow_interacting_type}'] = 0
            df_dataset_specific_sector.loc[:, f'std_complexity_type_{flow_interacting_type}'] = 0
            df_dataset_specific_sector.loc[:, f'percentile25_complexity_type_{flow_interacting_type}'] = 0
            df_dataset_specific_sector.loc[:, f'percentile75_complexity_type_{flow_interacting_type}'] = 0

    else:
        df_dataset_specific_sector.loc[:, f'max_complexity_type_{flow_interacting_type}'] = np.nan
        df_dataset_specific_sector.loc[:, f'sum_complexity_type_{flow_interacting_type}'] = np.nan
        df_dataset_specific_sector.loc[:, f'median_complexity_type_{flow_interacting_type}'] = np.nan
        df_dataset_specific_sector.loc[:, f'std_complexity_type_{flow_interacting_type}'] = np.nan
        df_dataset_specific_sector.loc[:, f'percentile25_complexity_type_{flow_interacting_type}'] = np.nan
        df_dataset_specific_sector.loc[:, f'percentile75_complexity_type_{flow_interacting_type}'] = np.nan

    return df_dataset_specific_sector


def add_average_time_route_interacting_fluxes_by_type(
        df_dataset_specific_sector: pd.DataFrame, df_cfmu_clustered_routes_interacting_specific_sector: pd.DataFrame) \
        -> pd.DataFrame:
    """ Compute metrics about the average time of the interacting fluxes by interaction type """

    df_dataset_specific_sector = _metrics_average_time_route_specific_interaction_type(
        df_dataset_specific_sector, df_cfmu_clustered_routes_interacting_specific_sector, flow_interacting_type=1)
    df_dataset_specific_sector = _metrics_average_time_route_specific_interaction_type(
        df_dataset_specific_sector, df_cfmu_clustered_routes_interacting_specific_sector, flow_interacting_type=2)
    df_dataset_specific_sector = _metrics_average_time_route_specific_interaction_type(
        df_dataset_specific_sector, df_cfmu_clustered_routes_interacting_specific_sector, flow_interacting_type=3)

    return df_dataset_specific_sector


def _metrics_average_time_route_specific_interaction_type(
    df_dataset_specific_sector: pd.DataFrame, df_cfmu_clustered_routes_interacting_specific_sector: pd.DataFrame,
    flow_interacting_type: int) -> DataFrame:
    """ Compute metrics about the average time in route of the interacting fluxes per interaction type """

    if len(df_cfmu_clustered_routes_interacting_specific_sector) != 0:
        df_interacting_fluxes_type = df_cfmu_clustered_routes_interacting_specific_sector[
            df_cfmu_clustered_routes_interacting_specific_sector['flowInteractingType'] == flow_interacting_type]

        if len(df_interacting_fluxes_type) != 0:
            df_interacting_fluxes_type['avgTimeInRoute_combined'] = \
                df_interacting_fluxes_type['avgTimeInRoute'] + df_interacting_fluxes_type['avgTimeInRouteInteracting']
            df_interacting_fluxes_type['nFlights_combined'] = \
                df_interacting_fluxes_type['nFlights'] + df_interacting_fluxes_type['nFlightsInteracting']
            df_interacting_fluxes_type['avgTimeInRoute_divided_num_flights_combined'] = \
                df_interacting_fluxes_type['avgTimeInRoute_combined'] / df_interacting_fluxes_type['nFlights_combined']

            # df_dataset_specific_sector.loc[:, f'min_avg_time_route_type_{flow_interacting_type}_combined'] = \
            #     np.min(df_interacting_fluxes_type['avgTimeInRoute_combined'].values)
            df_dataset_specific_sector.loc[:, f'max_avg_time_route_type_{flow_interacting_type}_combined'] = \
                np.max(df_interacting_fluxes_type['avgTimeInRoute_combined'].values)
            df_dataset_specific_sector.loc[:, f'sum_avg_time_route_type_{flow_interacting_type}_divided_num_flight_combined'] = \
                np.sum(df_interacting_fluxes_type['avgTimeInRoute_divided_num_flights_combined'].values)
            df_dataset_specific_sector.loc[:, f'median_avg_time_route_type_{flow_interacting_type}_combined'] = \
                round(float(np.median(df_interacting_fluxes_type['avgTimeInRoute_combined'].values)), 3)
            df_dataset_specific_sector.loc[:, f'std_avg_time_route_type_{flow_interacting_type}_combined'] = \
                round(float(np.std(df_interacting_fluxes_type['avgTimeInRoute_combined'].values)), 3)
            df_dataset_specific_sector.loc[:, f'percentile25_avg_time_route_type_{flow_interacting_type}_combined'] = \
                round(float(np.percentile(df_interacting_fluxes_type['avgTimeInRoute_combined'].values, 25)), 3)
            df_dataset_specific_sector.loc[:, f'percentile75_avg_time_route_type_{flow_interacting_type}_combined'] = \
                round(float(np.percentile(df_interacting_fluxes_type['avgTimeInRoute_combined'].values, 75)), 3)
        else:
            # df_dataset_specific_sector.loc[:, f'min_avg_time_route_type_{flow_interacting_type}_combined'] = 0
            df_dataset_specific_sector.loc[:, f'max_avg_time_route_type_{flow_interacting_type}_combined'] = 0
            df_dataset_specific_sector.loc[:, f'sum_avg_time_route_type_{flow_interacting_type}_divided_num_flight_combined'] = 0
            df_dataset_specific_sector.loc[:, f'median_avg_time_route_type_{flow_interacting_type}_combined'] = 0
            df_dataset_specific_sector.loc[:, f'std_avg_time_route_type_{flow_interacting_type}_combined'] = 0
            df_dataset_specific_sector.loc[:, f'percentile25_avg_time_route_type_{flow_interacting_type}_combined'] = 0
            df_dataset_specific_sector.loc[:, f'percentile75_avg_time_route_type_{flow_interacting_type}_combined'] = 0
    else:
        # df_dataset_specific_sector.loc[:, f'min_avg_time_route_type_{flow_interacting_type}_combined'] = np.nan
        df_dataset_specific_sector.loc[:, f'max_avg_time_route_type_{flow_interacting_type}_combined'] = np.nan
        df_dataset_specific_sector.loc[:, f'sum_avg_time_route_type_{flow_interacting_type}_divided_num_flight_combined'] = np.nan
        df_dataset_specific_sector.loc[:, f'median_avg_time_route_type_{flow_interacting_type}_combined'] = np.nan
        df_dataset_specific_sector.loc[:, f'std_avg_time_route_type_{flow_interacting_type}_combined'] = np.nan
        df_dataset_specific_sector.loc[:, f'percentile25_avg_time_route_type_{flow_interacting_type}_combined'] = np.nan
        df_dataset_specific_sector.loc[:, f'percentile75_avg_time_route_type_{flow_interacting_type}_combined'] = np.nan

    return df_dataset_specific_sector


def add_number_of_flights_interacting_fluxes_by_type(
        df_dataset_specific_sector: pd.DataFrame, df_cfmu_clustered_routes_interacting_specific_sector: pd.DataFrame) \
        -> pd.DataFrame:
    """ Features about the number of flights for each interacting flux by interaction type """

    df_dataset_specific_sector = _metrics_number_interacting_flights_specific_interaction_type(
        df_dataset_specific_sector, df_cfmu_clustered_routes_interacting_specific_sector, flow_interacting_type=1)
    df_dataset_specific_sector = _metrics_number_interacting_flights_specific_interaction_type(
        df_dataset_specific_sector, df_cfmu_clustered_routes_interacting_specific_sector, flow_interacting_type=2)
    df_dataset_specific_sector = _metrics_number_interacting_flights_specific_interaction_type(
        df_dataset_specific_sector, df_cfmu_clustered_routes_interacting_specific_sector, flow_interacting_type=3)

    return df_dataset_specific_sector


def _metrics_number_interacting_flights_specific_interaction_type(
        df_dataset_specific_sector: pd.DataFrame, df_cfmu_clustered_routes_interacting_specific_sector: pd.DataFrame,
        flow_interacting_type: int) -> DataFrame:
    """ Compute metrics about the number of flights according to the requested interaction type"""

    if len(df_cfmu_clustered_routes_interacting_specific_sector) != 0:
        interacting_fluxes_type = df_cfmu_clustered_routes_interacting_specific_sector[
            df_cfmu_clustered_routes_interacting_specific_sector['flowInteractingType'] == flow_interacting_type]
        num_flights_fluxes = interacting_fluxes_type['nFlights'].values
        num_flights_fluxes_interacting = interacting_fluxes_type['nFlightsInteracting'].values
        num_flights = np.concatenate([num_flights_fluxes, num_flights_fluxes_interacting])

        if num_flights.size != 0:
            # df_dataset_specific_sector.loc[:, f'min_num_flights_interacting_fluxes_type_{flow_interacting_type}'] = \
            #     np.min(num_flights)
            df_dataset_specific_sector.loc[:, f'max_num_flights_interacting_fluxes_type_{flow_interacting_type}'] = \
                np.max(num_flights)
            df_dataset_specific_sector.loc[:, f'avg_num_flights_interacting_fluxes_type_{flow_interacting_type}'] = \
                np.average(num_flights)
            df_dataset_specific_sector.loc[:, f'median_num_flights_interacting_fluxes_type_{flow_interacting_type}'] = \
                round(float(np.median(num_flights)), 3)
            df_dataset_specific_sector.loc[:, f'std_num_flights_interacting_fluxes_type_{flow_interacting_type}'] = \
                round(float(np.std(num_flights)), 3)
            df_dataset_specific_sector.loc[:, f'percentile25_num_flights_interacting_fluxes_type_{flow_interacting_type}'] = \
                round(float(np.percentile(num_flights, 25)), 3)
            df_dataset_specific_sector.loc[:, f'percentile75_num_flights_interacting_fluxes_type_{flow_interacting_type}'] = \
                round(float(np.percentile(num_flights, 75)), 3)
        else:
            # df_dataset_specific_sector.loc[:, f'min_num_flights_interacting_fluxes_type_{flow_interacting_type}'] = 0
            df_dataset_specific_sector.loc[:, f'max_num_flights_interacting_fluxes_type_{flow_interacting_type}'] = 0
            df_dataset_specific_sector.loc[:, f'avg_num_flights_interacting_fluxes_type_{flow_interacting_type}'] = 0
            df_dataset_specific_sector.loc[:, f'median_num_flights_interacting_fluxes_type_{flow_interacting_type}'] = 0
            df_dataset_specific_sector.loc[:, f'std_num_flights_interacting_fluxes_type_{flow_interacting_type}'] = 0
            df_dataset_specific_sector.loc[:, f'percentile25_num_flights_interacting_fluxes_type_{flow_interacting_type}'] = 0
            df_dataset_specific_sector.loc[:, f'percentile75_num_flights_interacting_fluxes_type_{flow_interacting_type}'] = 0
    else:
        # df_dataset_specific_sector.loc[:, f'min_num_flights_interacting_fluxes_type_{flow_interacting_type}'] = np.nan
        df_dataset_specific_sector.loc[:, f'max_num_flights_interacting_fluxes_type_{flow_interacting_type}'] = np.nan
        df_dataset_specific_sector.loc[:, f'avg_num_flights_interacting_fluxes_type_{flow_interacting_type}'] = np.nan
        df_dataset_specific_sector.loc[:, f'median_num_flights_interacting_fluxes_type_{flow_interacting_type}'] = np.nan
        df_dataset_specific_sector.loc[:, f'std_num_flights_interacting_fluxes_type_{flow_interacting_type}'] = np.nan
        df_dataset_specific_sector.loc[:, f'percentile25_num_flights_interacting_fluxes_type_{flow_interacting_type}'] = np.nan
        df_dataset_specific_sector.loc[:, f'percentile75_num_flights_interacting_fluxes_type_{flow_interacting_type}'] = np.nan

    return df_dataset_specific_sector


def add_interacting_fluxes_by_type(df_dataset_specific_sector: pd.DataFrame,
                                   df_cfmu_clustered_routes_interacting_specific_sector: pd.DataFrame,
                                   list_all_clustered_routes_specific_sector: list) -> pd.DataFrame:
    """
    Features about the number of interacting fluxes by interaction type {0, 1, 2, 3}

    Notes:
        1. Note: The values are the same for all the rows because the DataFrame contains information
            from a specific sector
        2. All values are divided by 2 to remove duplicated information.
            Example: 1vs3 and 3vs1 -> it should only be counted one time
        3. Average and median values are not decided by 2 because there is no impact from duplicated information
    """

    if len(df_cfmu_clustered_routes_interacting_specific_sector) != 0:

        # Overall number of fluxes
        df_dataset_specific_sector.loc[:, 'num_interacting_fluxes'] = \
            len(df_cfmu_clustered_routes_interacting_specific_sector) / 2

        # Non-interacting
        list_clustered_routes_interacting = list(df_cfmu_clustered_routes_interacting_specific_sector['clusteredRouteKey'].values)
        noninteracting_clusteredRouteKey = [route_key for route_key in list_all_clustered_routes_specific_sector if
                                            route_key not in list_clustered_routes_interacting]
        df_dataset_specific_sector.loc[:, 'num_interacting_fluxes_type_0'] = len(noninteracting_clusteredRouteKey) / 2

        # Interacting fluxes
        df_dataset_specific_sector.loc[:, 'num_interacting_fluxes_type_1'] = \
            len(df_cfmu_clustered_routes_interacting_specific_sector[
                    df_cfmu_clustered_routes_interacting_specific_sector['flowInteractingType'] == 1]) / 2
        df_dataset_specific_sector.loc[:, 'num_interacting_fluxes_type_2'] = \
            len(df_cfmu_clustered_routes_interacting_specific_sector[
                df_cfmu_clustered_routes_interacting_specific_sector['flowInteractingType'] == 2]) / 2
        df_dataset_specific_sector.loc[:, 'num_interacting_fluxes_type_3'] = \
            len(df_cfmu_clustered_routes_interacting_specific_sector[
                df_cfmu_clustered_routes_interacting_specific_sector['flowInteractingType'] == 3]) / 2

        df_dataset_specific_sector.loc[:, 'avg_interacting_type'] = round(float(np.average(
            df_cfmu_clustered_routes_interacting_specific_sector['flowInteractingType'].values)), 3)

        df_dataset_specific_sector.loc[:, 'median_interacting_type'] = round(float(np.median(
            df_cfmu_clustered_routes_interacting_specific_sector['flowInteractingType'].values)), 3)

    else:
        df_dataset_specific_sector.loc[:, 'num_interacting_fluxes'] = np.nan
        df_dataset_specific_sector.loc[:, 'num_interacting_fluxes_type_0'] = np.nan
        df_dataset_specific_sector.loc[:, 'num_interacting_fluxes_type_1'] = np.nan
        df_dataset_specific_sector.loc[:, 'num_interacting_fluxes_type_2'] = np.nan
        df_dataset_specific_sector.loc[:, 'num_interacting_fluxes_type_3'] = np.nan
        df_dataset_specific_sector.loc[:, 'avg_interacting_type'] = np.nan
        df_dataset_specific_sector.loc[:, 'median_interacting_type'] = np.nan

    return df_dataset_specific_sector
