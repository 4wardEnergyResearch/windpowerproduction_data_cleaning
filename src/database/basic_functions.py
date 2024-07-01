# -*- coding: utf-8 -*-

"""12.01.2023, Version 1.2.

Creates SQlite Data Base:
Data Base can be edited and observed e. g. with DB Browser
(https://bodo-schoenfeld.de/python-und-sqlite3/)

Reads in Metadata and Turbine Power Curves for all wind power operators.

@author: Nina Bisko, Johanna Ganglbauer
"""

# python modules
import os
import sqlite3
from collections import Counter
from typing import Tuple, List, Union

import numpy as np
import pandas as pd

import sys
project_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
# Insert project directory to path
if project_dir not in sys.path:
    sys.path.insert(0, project_dir)
from options import *


# functions
def create_empty_database(path: str) -> None:
    """Create empty database.

    Contains one table called metadata with relevant information on each wind turbine.
    Another table called power evaluation is constructed, which saves information on
    performance of wind2power models.
    """
    # Verbindung zur Datenbank aufbauen - wenn noch keine Datenbank vorhanden ist, wird diese angelegt:
    verbindung = sqlite3.connect(FolderPaths.database_path)
    # Cursor-Objekt erzeugen:
    zeiger = verbindung.cursor()

    # SQL-Anweisung als String vorbereiten:
    sql_anweisung = """
    CREATE TABLE IF NOT EXISTS metadaten (
    [turbine ID] TEXT UNIQUE,
    [windpark ID] TEXT,
    [operator ID] TEXT,
    [turbine type] TEXT,
    [provider ID] TEXT,
    [latitude] REAL,
    [longitude] REAL,
    [height] REAL,
    [hub height] REAL,
    [diameter] REAL,
    [nominal power] REAL,
    [wind sensor 1] TEXT,
    [wind sensor 2] TEXT,
    [installation date] TEXT,
    [first timestamp] TEXT,
    [last timestamp] TEXT,
    [total values] TEXT,
    [invalid values] INTEGER
    );"""

    # Datenbankblatt mit Spalten aus Sql-Anweisung an Zeiger übergeben und ausführen:
    zeiger.execute(sql_anweisung)
    verbindung.commit()

    # SQL-Anweisung als String vorbereiten:
    sql_anweisung = """
    CREATE TABLE IF NOT EXISTS powerevaluation (
    [turbine ID] TEXT UNIQUE,
    [BIAS] REAL,
    [MSE] REAL,
    [RMSE] REAL,
    [flagged values] INTEGER
    );"""

    # Datenbankblatt mit Spalten aus Sql-Anweisung an Zeiger übergeben und ausführen:
    zeiger.execute(sql_anweisung)
    verbindung.commit()

    # close connections
    verbindung.close()


def create_empty_turbine_power_curve_table(path: str, name: str) -> None:
    """Create empty table labeled with turbine ID with power curve information."""
    # Verbindung zur Datenbank aufbauen - wenn noch keine Datenbank vorhanden ist, wird diese angelegt:
    verbindung = sqlite3.connect(FolderPaths.database_path)
    # Cursor-Objekt erzeugen:
    zeiger = verbindung.cursor()
    # SQL-Anweisung als String vorbereiten:
    sql_anweisung = "CREATE TABLE IF NOT EXISTS '" + name + \
        """' (
            [windspeed] REAL UNIQUE,
            [power] REAL
            );"""

    # Datenbankblatt mit Spalten aus Sql-Anweisung an Zeiger übergeben und ausführen:
    zeiger.execute(sql_anweisung)
    verbindung.commit()
    verbindung.close()


def put_metadata_frame_into_database(path: str, metadata: pd.DataFrame):
    """Read in dataframe to metadata data base."""
    verbindung = sqlite3.connect(FolderPaths.database_path)
    zeiger = verbindung.cursor()
    # Iterate over all rows from data frame.
    for i, row in metadata.iterrows():
        zeiger.execute("INSERT OR IGNORE INTO metadaten VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", tuple(row))
        verbindung.commit()
    verbindung.close()


def put_power_evaluation_line_into_database(path: str, power_evaluation_line: Tuple):
    """Insert results from power evaluation to powerevaluation table of data base."""
    verbindung = sqlite3.connect(FolderPaths.database_path)
    zeiger = verbindung.cursor()
    zeiger.execute("INSERT OR IGNORE INTO powerevaluation VALUES (?,?,?,?,?)",
                   power_evaluation_line)
    verbindung.commit()
    verbindung.close()


def complete_metadata_from_timesereis(path: str, turbine_id: str, dates: List[str], total_values: Union[float, int], invalid_values: Union[float, int]) -> None:
    """Provide first time stamp, last time stamp and number of invalid values from time series data."""
    verbindung = sqlite3.connect(FolderPaths.database_path)
    zeiger = verbindung.cursor()
    sql_anweisung = "UPDATE metadaten SET [first timestamp] = '" + dates[0] + \
        "', [last timestamp] = '" + dates[1] + \
        "', [total values] = '" + str(total_values) + \
        "', [invalid values] = '" + str(invalid_values) + \
        "' WHERE [turbine ID] = '" + turbine_id + "';"
    zeiger.execute(sql_anweisung)
    verbindung.commit()
    verbindung.close()


def put_turbine_power_curve_into_database(path: str, tablename: str, powercurve: pd.DataFrame):
    """Read in dataframe to selected turbine power curve table in data base."""
    verbindung = sqlite3.connect(FolderPaths.database_path)
    zeiger = verbindung.cursor()
    sql_anweisung = "INSERT OR IGNORE INTO '" + tablename + "' VALUES (?,?)"
    # Iterate over all rows from data frame.
    for i, row in powercurve.iterrows():
        zeiger.execute(sql_anweisung, tuple(row))
        verbindung.commit()
    verbindung.close()
