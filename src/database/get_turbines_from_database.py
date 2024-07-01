"""11.04.2023, Version 1.0.

Selects turbines from meta data based on certain criteria.

@author: Nina Bisko, Johanna Ganglbauer
"""

# python modules
import sqlite3
import os
import sys
project_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
# Insert project directory to path
if project_dir not in sys.path:
    sys.path.insert(0, project_dir)
from options import *


def get_all_turbines(path: str) -> None:
    """Print all turbines listed in metadata table of database."""
    verbindung = sqlite3.connect(FolderPaths.database_path)
    zeiger = verbindung.cursor()
    zeiger.execute("SELECT [turbine ID] FROM metadaten")
    inhalt = zeiger.fetchall()
    print(inhalt)
    verbindung.close()


def get_certain_turbines(path: str, name: str) -> None:
    """Select turbines of certain type."""
    verbindung = sqlite3.connect(FolderPaths.database_path)
    spalten_name = "[turbine type]"
    liste_werte = []
    command = "SELECT [turbine ID] FROM metadaten WHERE " + spalten_name + " = '" + name + "';"
    zeiger = verbindung.cursor()
    zeiger.execute(command)
    verbindung.commit()
    zeilen = zeiger.fetchall()
    for zeile in zeilen:
        liste_werte.append(zeile[0])
    verbindung.commit()
    verbindung.close()


if __name__ == "__main__":
    path = FolderPaths.data_dir
    get_all_turbines(path=path)
    get_certain_turbines(path=path, name='E82_3MW')
