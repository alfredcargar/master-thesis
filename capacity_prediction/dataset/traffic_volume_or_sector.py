"""
Utilities to extract information from Traffic Volume and Sectors

Author: smas
Last update: 24/10/2023
"""

import pandas as pd


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
