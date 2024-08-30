"""
Queries for first data extraction
    1) sectors
    2) entries and occupancies
acg
"""
import pandas as pd
import pyodbc

def select_sectors_spatial(connection_dwh: pyodbc.connect) -> pd.DataFrame:
    query = f'''
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
    return pd.read_sql(query, connection_dwh)


def select_sectors_operational(connection_dwh: pyodbc.connect) -> pd.DataFrame:
    query = f'''
            select
            op.LevelType, op.ATCUnitCode, op.ATCType, op.SectorCode,
            op.Capacity, op.Date_From, op.Date_To, op.Active
            from dwh.dbo.AirspaceStructuresOperational op
            where op.LevelType='Sector'
            '''
    return pd.read_sql(query, connection_dwh)


def select_ent_occ_groupby_sector_day(connection_dwh: pyodbc.connect, date_from: str, date_to: str) -> pd.DataFrame:
    """
    gets all sectors with average entries and occupancies by day.
    check: Maybe should be MAX instead?
    """
    query = f'''
            ;with dates as (
            select a.date, a.dateKey, b.timeKey, b.time
            from dwh.dbo.CalendarDate a
            inner join dwh.dbo.CalendarTime b ON 1=1
            where date >= '{date_from}' and date < '{date_to}'
              and timeKey % 2000 = 0 -- every 20 minutes
            ),
            spatialSectors as (
                select
                    sp.SectorCode,
                    sp.Date_From,
                    sp.Date_To,
            		--sp.Polygon.STLength() AS Perimeter,
                    sp.Active
                from dwh.dbo.AirspaceStructuresSpatial sp 
                where sp.LevelType = 'Sector'
            )
            select 
                dates.dateKey,
                volume.trafficVolumeCode,
                min(ss.Date_From) AS Date_From,
                max(ss.Date_To) AS Date_To,
            	--max(ss.Perimeter),
                max(ec_oc.occupancyInWindow_1) as max_occupancyInWindow_1,
                avg(ec_oc.occupancyInWindow_1) as occupancyInWindow_1,
                avg(ec_oc.entriesInWindow_1) as entriesInWindow_1,
                max(ec_oc.entriesInWindow_1) as max_entriesInWindow_1,
                avg(ec_oc.occupancyInWindow_5) as occupancyInWindow_5,
                max(ec_oc.occupancyInWindow_5) as max_occupancyInWindow_5,
                avg(ec_oc.entriesInWindow_5) as entriesInWindow_5,
                max(ec_oc.entriesInWindow_5) as max_entriesInWindow_5,
                avg(ec_oc.occupancyInWindow_10) as occupancyInWindow_10,
                max(ec_oc.occupancyInWindow_10) as max_occupancyInWindow_10,
                avg(ec_oc.entriesInWindow_10) as entriesInWindow_10,
                max(ec_oc.entriesInWindow_10) as max_entriesInWindow_10,
                avg(ec_oc.entriesInWindow_60) as entriesInWindow_60,
                max(ec_oc.entriesInWindow_60) as max_entriesInWindow_60
            from dates
            inner join dwh.dbo.cfmuEntriesAndOccupancies_Facts ec_oc
                on dates.dateKey = ec_oc.windowDateFromKey
                and dates.timeKey = ec_oc.windowTimeFromKey
            inner join dwh.dbo.dimCFMUtrafficVolume volume
                on ec_oc.trafficVolumeKey = volume.trafficVolumeKey
            inner join spatialSectors ss
                on volume.trafficVolumeCode = ss.SectorCode
                and dates.date between ss.Date_From and ss.Date_To
            group by 
                dates.dateKey, 
                volume.trafficVolumeCode, ss.Date_From, ss.date_To
            order by 
                volume.trafficVolumeCode, 
                dates.dateKey;
            '''
    return pd.read_sql(query, connection_dwh)


def select_meteo_events_groupby_day(connection_dwh: pyodbc.connect, icao: str, date_from: str, date_to: str) -> pd.DataFrame:
    query = f'''
            WITH weatherAggregates AS (
            SELECT 
                CAST(mr.observationTime AS DATE) AS datekey, 
                mm.icao,
                AVG(mm.temperature) AS temperature,
                AVG(mm.qnh) AS atm_pressure,
                AVG(mw.knots) AS wind_speed,
                AVG(dmc.height) AS cloud_height
            FROM 
                meteoMetarMiscellaneousWeather_Facts mm
            INNER JOIN 
                dimmeteowind mw ON mm.meteoWindKey = mw.meteoWindKey
            INNER JOIN 
                [dimMeteoMETARreport] mr ON mm.metarReportKey = mr.metarKey
            INNER JOIN 
                [meteoMetarCloud_Facts] mc ON mm.metarReportKey = mc.metarReportKey
            INNER JOIN 
                dimmeteocloud dmc ON mc.meteoCloudKey = dmc.meteoCloudKey
            INNER JOIN 
                meteoMetarWeatherPhenomena_Facts mwp ON mm.metarreportkey = mwp.metarreportkey
            INNER JOIN 
                dimmeteoweatherphenomena dmp ON mwp.meteoWeatherPhenKey = dmp.meteoWeatherPhenKey
            WHERE 
                mr.observationtime > '{date_from}' AND mr.observationtime < '{date_to}'
                AND mm.meteoTrendKey IN (1,2)
                and mm.icao = '{icao}'
            GROUP BY 
                CAST(mr.observationTime AS DATE), mm.icao
        ),
        lastPhenomenon AS (
            SELECT 
                CAST(mr.observationTime AS DATE) AS datekey,
                dmp.phenomenon1, 
                dmp.intensity,
                ROW_NUMBER() OVER (PARTITION BY CAST(mr.observationTime AS DATE) ORDER BY mr.observationTime DESC) AS rn
            FROM 
                dimMeteoMETARreport mr
            INNER JOIN 
                meteoMetarWeatherPhenomena_Facts mwp ON mr.metarKey = mwp.metarreportkey
            INNER JOIN 
                dimmeteoweatherphenomena dmp ON mwp.meteoWeatherPhenKey = dmp.meteoWeatherPhenKey
            WHERE 
                mr.observationtime > '2022-01-01' AND mr.observationtime < '2022-07-01'
                AND dmp.phenomenon1 IS NOT NULL
            )
            SELECT 
                wa.datekey, --wa.icao,
                wa.temperature as avg_temperature,
                wa.atm_pressure as avg_atmpressure,
                wa.wind_speed as avg_wind_speed,
                wa.cloud_height as avg_cloud_height,
                lp.phenomenon1 as event,
                lp.intensity
            FROM 
                weatherAggregates wa
            LEFT JOIN 
                (SELECT datekey, phenomenon1, intensity
                 FROM lastPhenomenon
                 WHERE rn = 1) lp ON wa.datekey = lp.datekey
            ORDER BY 
                wa.datekey;
            '''
    return pd.read_sql(query, connection_dwh)


def select_clustered_routes(connection_dwh: pyodbc.connect, year: int) -> pd.DataFrame:
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

    