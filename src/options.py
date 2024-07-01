"""01.07.2024, Version 1.0.
Contains options for the data cleaning routines.
@author: Stefan Janisch
"""

import os
import multiprocessing

# FOLDER PATHS ################################################################
class FolderPaths:
    src_dir = os.path.dirname(__file__)
    project_dir = os.path.dirname(src_dir)

    # Input data directory. The database is saved here.
    data_dir = os.path.join(project_dir, 'example_data')

    # Fodler containing input time series
    input_dir = os.path.join(data_dir, 'input_time_series')

    # Folder containing turbine and cleaning metadata
    metadata_dir = os.path.join(data_dir, 'input_metadata')
    database_path = os.path.join(metadata_dir, 'turbine_metadata.db')
    cleaning_config_path = os.path.join(metadata_dir, 'data_cleaning_config.xlsx')

    # Folder containing flagged data
    flagged_dir = os.path.join(data_dir, 'output_flagged_time_series')

    # Folder containing diagnostic plots and tables
    diagnostics_dir = os.path.join(data_dir, 'output_diagnostics')

# PARALLEL PROCESSING #########################################################
class ParallelProcessing:
    num_conc_proc = 1 # int(multiprocessing.cpu_count()/2)

# DIAGNOSTICS #################################################################
class Diagnostics:
    # Threshold percentage of flagged data to mark the turbine as suspicious
    threshold_flagging = 10 # percent