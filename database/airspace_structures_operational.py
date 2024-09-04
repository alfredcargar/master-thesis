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

    assert "capacity" in df_airspace_structures_operational.columns, "capacity column not found"
    assert "date_from" in df_airspace_structures_operational.columns, "date_from column not found"
    assert "date_to" in df_airspace_structures_operational.columns, "date_to column not found"

    df_airspace_structures_operational_compacted = pd.DataFrame()  # Initialize output DataFrame

    unique_capacity_values = np.unique(df_airspace_structures_operational['capacity'].values)

    for capacity_value in unique_capacity_values:
        df_for_capacity_value = df_airspace_structures_operational[
            df_airspace_structures_operational['capacity'] == capacity_value]
        min_date_from = np.min(df_for_capacity_value['date_from'])
        max_date_to = np.max(df_for_capacity_value['date_to'])

        new_row = {'level_type': df_for_capacity_value.iloc[0]['level_type'],
                   'atcunit_code': df_for_capacity_value.iloc[0]['atcunit_code'],
                   'sector_code': df_for_capacity_value.iloc[0]['sector_code'],
                   'capacity': df_for_capacity_value.iloc[0]['capacity'],
                   'date_from': min_date_from,
                   'date_to': max_date_to,
                   'active': df_for_capacity_value.iloc[0]['active']}

        df_airspace_structures_operational_compacted = pd.concat([df_airspace_structures_operational_compacted,
                                                                  pd.DataFrame([new_row])], ignore_index=True)

    df_airspace_structures_operational_compacted = df_airspace_structures_operational_compacted.sort_values(by=['date_from'])

    return df_airspace_structures_operational_compacted
