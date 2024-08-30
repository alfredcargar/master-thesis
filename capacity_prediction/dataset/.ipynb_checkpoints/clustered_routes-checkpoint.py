"""
Utilities to extract information from Clustered routes

Author: smas
Last update: 24/10/2023
"""

import pandas as pd
import numpy as np


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
