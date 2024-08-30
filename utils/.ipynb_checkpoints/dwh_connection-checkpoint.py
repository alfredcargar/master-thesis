"""
Utilities related to connection to the DWH in CRIDA

Author: smas
Last update: 24/08/2023
"""

import pyodbc


def get_dwh_connection(host: str, dwh_database: str, user_id: str, password: str) -> pyodbc.connect:
    """
    Create a connection to the DWH

    Args:
        host: Host of the DWH
        dwh_database: Name of the database
        user_id: User ID
        password: Password of the user ID

    Returns:
        A connection to the DWH

    Note:
        1. Remember to close the connection
    """

    try:
        connection_dwh = pyodbc.connect(
            'Driver=ODBC Driver 17 for SQL Server;Server=' + host + ';Database=' + dwh_database + ';UID=' + user_id +
            ';PWD=' + password + '')
    except:
        raise ConnectionError(f"Impossible to connect to {host}")

    return connection_dwh
