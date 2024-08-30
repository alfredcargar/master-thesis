"""
Utilities to extract information from Clustered routes interacting

Author: smas
Last update: 19/10/2023
"""

import pandas as pd
import numpy as np
from pandas import DataFrame


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
