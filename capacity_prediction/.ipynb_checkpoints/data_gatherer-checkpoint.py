"""
Set of functionalities to get information from the DWH

Author: smas
Last update: 19/09/2023
"""

import pyodbc
import pandas as pd


def get_oc_ec_groupby_date_traffic_volume(connection_dwh: pyodbc.connect, date_from: str, date_to: str) -> pd.DataFrame:
    """ Get the traffic volume for a given year """

    assert date_from[0:4] == "2022", f"Year {date_from[0:4]} not supported. Only 2022"

    query_oc_ec_groupby_date_traffic_volume = f'''
                                                declare @dateFrom date = '{date_from}'
                                                declare @dateTo date = '{date_to}'
                                                
                                                ;with dates as (
                                                    select dateKey
                                                    from CalendarDate
                                                    where date>=@dateFrom and date<@dateTo
                                                )
                                                
                                                select
                                                    trafficVolumeCode, windowDateFromKey
                                                    ,  MAX(occupancyInWindow_1) as max_occupancyInWindow_1
                                                    ,  MAX(entriesInWindow_1) as max_entriesInWindow_1
                                                    ,  MAX(occupancyInWindow_5) as max_occupancyInWindow_5
                                                    , MAX(entriesInWindow_5) as max_entriesInWindow_5
                                                    ,  MAX(occupancyInWindow_10) as max_occupancyInWindow_10
                                                    , MAX(entriesInWindow_10) as max_entriesInWindow_10
                                                    ,  MAX(entriesInWindow_60) as max_entriesInWindow_60
                                                from dates
                                                inner join dwh.dbo.cfmuEntriesAndOccupancies_Facts ec_oc
                                                    on dates.dateKey=ec_oc.windowDateFromKey
                                                inner join dwh.dbo.dimCFMUtrafficVolume volume
                                                    on ec_oc.trafficVolumeKey=volume.trafficVolumeKey
                                                group by
                                                    trafficVolumeCode, windowDateFromKey
                                               '''

    df_oc_ec_groupby_date_traffic_volume = pd.read_sql(query_oc_ec_groupby_date_traffic_volume, connection_dwh)

    return df_oc_ec_groupby_date_traffic_volume


def get_all_interacting_cfmu_clustered_routes(connection_dwh: pyodbc.connect, year: int) -> pd.DataFrame:
    """ Get the clustered routes for a given year """

    assert year == 2022, f"Year {year} not supported. Only 2022"

    query_all_cfmu_clustered_routes_interacting = f'''
                                                    select 
                                                        clusters.SectorCode, clusters.attitudIN, clusters.attitudOUT
                                                        , clustersInteracting.attitudIN attitudINinteracting, clustersInteracting.attitudOUT attitudOUTinteracting
                                                        , interaction.clusteredRouteKey, interaction.clusteredRouteKeyInteracting
                                                        , interaction.flowInteractingSeverity, interaction.flowInteractingType
                                                        , clusters.nFlights, clustersInteracting.nFlights nFlightsInteracting
                                                        , clusters.avgTimeInRoute, clustersInteracting.avgTimeInRoute avgTimeInRouteInteracting
                                                    from ANALISIS.desarrollo.FlujosCFMUClusteredRoutesInteraction_Facts_ComplejidadStructure interaction
                                                    inner join ANALISIS.desarrollo.dimFlujosCFMUClusteredRoutes clusters
                                                        on interaction.clusteredRouteKey=clusters.clusteredRouteKey
                                                    inner join ANALISIS.desarrollo.dimFlujosCFMUClusteredRoutes clustersInteracting
                                                        on interaction.clusteredRouteKeyInteracting=clustersInteracting.clusteredRouteKey
                                                   '''

    df_all_cfmu_clustered_routes_interacting = pd.read_sql(query_all_cfmu_clustered_routes_interacting, connection_dwh)

    return df_all_cfmu_clustered_routes_interacting


def get_all_cfmu_clustered_routes(connection_dwh: pyodbc.connect, year: int) -> pd.DataFrame:
    """ Get the clustered routes for a given year """

    assert year == 2022, f"Year {year} not supported. Only 2022"

    query_all_cfmu_clustered_routes = f'''
                                    select 
                                        clusters.clusteredRouteKey,
                                        clusters.SectorCode, 
                                        clusters.attitudIN, clusters.attitudOUT, 
                                        clusters.nFlights, 
                                        clusters.avgTimeInRoute, 
                                        clusters.latIN, clusters.lngIN, clusters.latOUT, clusters.lngOUT
                                    from ANALISIS.desarrollo.dimFlujosCFMUClusteredRoutes clusters
                                   '''

    df_all_cfmu_clustered_routes = pd.read_sql(query_all_cfmu_clustered_routes, connection_dwh)

    return df_all_cfmu_clustered_routes


def get_spatial_airblocks_specific_atc_unit_code(connection_dwh: pyodbc.connect, atc_unit_code: str) -> pd.DataFrame:
    """ Get the spatial information for all airblocks in a specific ATC unit code """

    query_spatial_airblocks_specific_sector = f'''
                                                select
                                                    LevelType, ATCUnitCode, ATCType, SectorCode
                                                    , sp.Polygon.STArea () Polygon_area
                                                    , LowerBound, UpperBound, Date_From, Date_To, Active
                                                from 
                                                    dwh.dbo.AirspaceStructuresSpatial sp
                                                where 
                                                    sp.ATCUnitCode='{atc_unit_code}'
                                                    and sp.LevelType='Airblock'
                                                '''

    df_spatial_airblocks_specific_sector = pd.read_sql(query_spatial_airblocks_specific_sector, connection_dwh)

    return df_spatial_airblocks_specific_sector


def get_airspace_structures_operational_all_sectors(connection_dwh: pyodbc.connect) -> pd.DataFrame:
    """ Get the operational information for all sectors available in the DWH """

    query_all_operational_sectors = f'''
                                     select
                                         op.LevelType, op.ATCUnitCode, op.ATCType, op.SectorCode,
                                         op.Capacity, op.Date_From, op.Date_To, op.Active
                                     from dwh.dbo.AirspaceStructuresOperational op
                                     where op.LevelType='Sector'
                                     '''

    df_all_operational_sectors = pd.read_sql(query_all_operational_sectors, connection_dwh)

    return df_all_operational_sectors


def get_airspace_structures_spatial_for_all_sectors(connection_dwh: pyodbc.connect) -> pd.DataFrame:
    """
    Get the spatial information for all sectors available in the DWH

    Notes:
        1. We are working with SqlGeography data type, and it does not have a direct STCentroid method. Instead, you can
        use a combination of other functions to calculate the centroid of a SqlGeography polygon. Here's an alternative
        approach using STPointN and STNumPoints:
            - STNumPoints calculates the number of points in the polygon.
            - STPointN is used to fetch the point at the middle position, which should approximate the centroid of the polygon
    """

    # query_all_spatial_sectors = f'''
    #                             select
    #                                 sp.LevelType, sp.ATCUnitCode, sp.ATCType, sp.SectorCode,
    #                                 sp.Polygon.STArea () Polygon_area,
    #                                 sp.Date_From, sp.Date_To, sp.Active
    #                             from dwh.dbo.AirspaceStructuresSpatial sp
    #                             where sp.LevelType='Sector'
    #                             '''

    query_all_spatial_sectors = f'''
                                SELECT
                                    sp.LevelType, sp.ATCUnitCode, sp.ATCType, sp.SectorCode, 
                                    sp.Date_From, sp.Date_To, sp.Active,
                                    sp.Polygon.STArea() Polygon_area,
                                    sp.Polygon.STLength() AS Perimeter,
                                    sp.Polygon.STNumPoints() AS NumberOfVertices,
                                    sp.Polygon.STPointN(sp.Polygon.STNumPoints() / 2).Lat AS Centroid_Latitude,
                                    sp.Polygon.STPointN(sp.Polygon.STNumPoints() / 2).Long AS Centroid_Longitude
                                FROM dwh.dbo.AirspaceStructuresSpatial sp 
                                WHERE sp.LevelType = 'Sector';
                                '''

    df_all_spatial_sectors = pd.read_sql(query_all_spatial_sectors, connection_dwh)

    return df_all_spatial_sectors
