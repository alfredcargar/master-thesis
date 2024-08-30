"""
Utilities related to the model in the DWH called AirspaceStructuresOperational

Author: smas
Last update: 30/08/2023
"""

import pandas as pd
import numpy as np
import datetime


def compact_operational_by_capacity_date_from_date_to_for_specific_sector(
        df_airspace_structures_operational: pd.DataFrame) -> pd.DataFrame:

    """
    Given a DataFrame with multiple capacity and valid periods, it returns a DataFrame with one capacity and one valid
        period per row
    
    Args:
        df_airspace_structures_operational: DataFrame with multiple capacity and 
            valid periods (per sectorization) 

    Returns:
        DataFrame with one capacity and one valid period per row
    """

    assert "Capacity" in df_airspace_structures_operational.columns, "Capacity column not found"
    assert "Date_From" in df_airspace_structures_operational.columns, "Date_From column not found"
    assert "Date_To" in df_airspace_structures_operational.columns, "Date_To column not found"

    df_airspace_structures_operational_compacted = pd.DataFrame()  # Initialize output DataFrame

    unique_capacity_values = np.unique(df_airspace_structures_operational['Capacity'].values)

    for capacity_value in unique_capacity_values:
        df_for_capacity_value = df_airspace_structures_operational[
            df_airspace_structures_operational['Capacity'] == capacity_value]
        min_date_from = np.min(df_for_capacity_value['Date_From'])
        max_date_to = np.max(df_for_capacity_value['Date_To'])

        new_row = {'LevelType': df_for_capacity_value.iloc[0]['LevelType'],
                   'ATCUnitCode': df_for_capacity_value.iloc[0]['ATCUnitCode'],
                   'SectorCode': df_for_capacity_value.iloc[0]['SectorCode'],
                   'Capacity': df_for_capacity_value.iloc[0]['Capacity'],
                   'Date_From': min_date_from,
                   'Date_To': max_date_to,
                   'Active': df_for_capacity_value.iloc[0]['Active']}

        df_airspace_structures_operational_compacted = pd.concat([df_airspace_structures_operational_compacted,
                                                                  pd.DataFrame([new_row])], ignore_index=True)

    df_airspace_structures_operational_compacted = df_airspace_structures_operational_compacted.sort_values(by=['Date_From'])

    return df_airspace_structures_operational_compacted
