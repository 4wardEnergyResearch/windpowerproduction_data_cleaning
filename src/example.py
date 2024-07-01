"""01.07.2024, Version 1.0.

This is an exemplary use case of the data cleaning routines.

@author: Stefan Janisch
"""

# 1. IMPORTS ##################################################################
import os
from typing import List
import pandas as pd
import concurrent.futures

src_dir = os.path.dirname(__file__)
from options import *

if __name__ == "__main__":

# 2. DATA CLEANING ############################################################
    import data_cleaning.compute_power_and_flag as cpf
    # Set plotting to True to generate explanatory plots for the different stages
    cpf.main(plotting=False)

# 3. DIAGNOSTICS ##############################################################
# 3.1. Read in flagged files ##################################################
    import data_cleaning.create_report as cr
    cr.main()