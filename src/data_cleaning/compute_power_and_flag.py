# -*- coding: utf-8 -*-

"""01.07.2024, Version 1.0.

Computes wind power based on prepared data and power curve.
Flags values, which do not match observation at all.
Saves new time series including wind power calculation and flag
columns to new .csv file in flagged folder.

@author: Nina Bisko, Johanna Ganglbauer, Stefan Janisch
"""


import os
import sys
import sqlite3
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
import windpowerlib as wpl
import plotly.graph_objects as go
import scipy.signal as signal
import scipy.ndimage as ndimage
import concurrent

project_dir = os.path.dirname(os.path.dirname(__file__))
# Insert project directory to path
if project_dir not in sys.path:
    sys.path.insert(0, project_dir)

from database.basic_functions import put_power_evaluation_line_into_database
import data_cleaning.timeseries_cleaning as tsc
from options import *

def print_red(text: str) -> None:
    """Print text in red."""
    print(f"\033[91m{text}\033[00m")

def print_green(text: str) -> None:
    """Print text in green."""
    print(f"\033[92m{text}\033[00m")

# disable warnings
pd.options.mode.chained_assignment = None


def get_turbine_data_from_id(path: str, turbine_id: str) -> Tuple:
    """Return relevant information of turbine from ID (and data base)."""
    # connects to data base and returns relevant values from table meta data
    verbindung = sqlite3.connect(FolderPaths.database_path)
    spalten_name = "[turbine ID]"
    command = "SELECT [turbine type], [nominal power], [hub height], [diameter] FROM metadaten WHERE" \
        + spalten_name + " = '" + turbine_id + "';"
    zeiger = verbindung.cursor()
    zeiger.execute(command)
    verbindung.commit()
    zeilen = zeiger.fetchall()
    verbindung.close()
    return zeilen[0]


def get_turbine_power_curve_from_type(path: str, turbine_type: str) -> pd.DataFrame:
    """Extract power curve data from data base and convert to given format of windpowerlib."""
    # get data from data base
    verbindung = sqlite3.connect(FolderPaths.database_path)
    command = "SELECT * FROM '" + turbine_type + "';"
    zeiger = verbindung.cursor()
    zeiger.execute(command)
    verbindung.commit()
    zeilen = zeiger.fetchall()
    verbindung.close()

    # convert data to format of windpowerlib
    power_curve_df = pd.DataFrame(zeilen)
    power_curve_df.columns = ["wind_speed", "value"]
    power_curve_df["value"] = power_curve_df["value"] * 1e3
    return power_curve_df


def initialize_turbine(filepath: str) -> Tuple[str, Optional[wpl.WindTurbine], Optional[List[float]]]:
    """Create Wind Turbine of WindPowerLib from database."""
    # gets turbine ID and path of database from file name
    head, tail = os.path.split(filepath)
    turbine_id = tail[:-4]
    path, _ = os.path.split(head)

    # gets relevant information out of data base
    turbine_data = get_turbine_data_from_id(path=path, turbine_id=turbine_id)
    # for now power curve of "M1800-600/150" is missing, skip calculation until data is provided
    powercurve_df = get_turbine_power_curve_from_type(path=path, turbine_type=turbine_data[0])

    windspeed_cut_in = powercurve_df["wind_speed"][(powercurve_df["value"] > 0)].iloc[0]
    windspeed_cut_out = powercurve_df["wind_speed"][((powercurve_df["wind_speed"] > 10) & (powercurve_df["value"] == 0))].iloc[0]
    windspeed_rated = powercurve_df["wind_speed"][(powercurve_df["value"] >= turbine_data[1] * 1e3 * 0.99)].iloc[0]

    # initializes turbine of windpowerlib
    my_turbine = {
        "turbine_type": turbine_data[0],
        "nominal_power": turbine_data[1] * 1e3,  # in W
        "hub_height": turbine_data[2],  # in m
        "rotor_diameter": turbine_data[3],  # in m
        "power_curve": powercurve_df,
    }
    spec = [windspeed_cut_in, windspeed_cut_out, windspeed_rated, turbine_data[0]]
    # initialize WindTurbine object
    return turbine_id, wpl.WindTurbine(**my_turbine), spec

def seperate_by_temperature(data: pd.DataFrame, ws_cut_in: float, ws_rated: float, ws_cut_out: float, nominal_power: float):
    
    data["Flag Computed"] = 0
    # Wind speed
    data.loc[((data["windspeed [m/s]"] < 0) | (data["windspeed [m/s]"] > 40)), "Flag Computed"] = 1

    # Power
    data.loc[((data["power [kW]"] / nominal_power <= -0.05) | (data["power [kW]"] / nominal_power > 1.05)), "Flag Computed"] = 1

    # Flag power values around 0 for wind speeds above 2*cut-in speed
    data.loc[((data["windspeed [m/s]"] > ws_cut_in * 2) & (data["power [kW]"] < 0.005 * nominal_power)), "Flag Computed"] = 1

    # extract only unflagged data
    group_data = data[data["Flag Computed"] == 0]
    temp_data = group_data.groupby(pd.cut(group_data["temperature [C]"], np.arange(-10,11,1)))
   
    fig, ax = plt.subplots(dpi=150)
    cmap = mpl.colormaps['viridis']
    colors = cmap(np.linspace(0.5, 1, len(temp_data)))
    for i, (name, group) in enumerate(temp_data):
        group.plot(ax=ax, x=["windspeed [m/s]"], y=["power [kW]"], kind="scatter", 
                       marker="*", s=1e-2, label=name, color=colors[i])
    ax.set_xlabel('Wind Speed (m/s)')
    ax.set_ylabel('Power Output (kW)')
    ax.legend()
    plt.show()

def calc_central_values(grouped: dict) -> dict:
    """Calculate the central values of the power bins.

    Parameters
    ----------
    grouped: dict
        A dictionary containing the grouped data.

    Returns
    -------
    central_values : dict
        The central values of the power bins.
    """
    central_values = {}

    for key, df in grouped:
        central_values[key] = (key.left + key.right) / 2

    return central_values


def create_static_plot(data: pd.DataFrame, sigma_mult: float, medians: dict, bins_power_cv: dict, int_lower: dict, int_upper: dict):
    """Generate a scatter plot showing wind speed vs power output.

    The function differentiates between data points inside and outside a given
    standard deviation (sigma) interval, and visualizes them using different markers.

    Parameters
    ----------
    data : pandas.DataFrame
        A dataframe containing the data. Expected to have columns "windspeed [m/s]",
        "power [kW]", and "Flag Computed".
    sigma_mult : float
        The multiplier for the standard deviation interval. Used for labeling.
    medians : dict
        A dictionary containing the median values for each power bin.
    bins_power_cv : dict
        A dictionary containing the power bins.
    int_lower : dict
        A dictionary containing the lower interval values for each power bin.
    int_upper : dict
        A dictionary containing the upper interval values for each power bin.

    Returns
    -------
    None
        The function directly shows the static plot.
    """
    # Filtering data based on the flag
    inside_data = data[data["Flag Computed"] == 0]
    outside_data = data[data["Flag Computed"] == 5]

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot inside and outside data
    ax.scatter(inside_data["windspeed [m/s]"], inside_data["power [kW]"], s=5, alpha=0.1, label=f"inside {sigma_mult}-sig interval")
    ax.scatter(outside_data["windspeed [m/s]"], outside_data["power [kW]"], s=5, alpha=0.1, label=f"outside {sigma_mult}-sig interval")

    # Plot medians and intervals
    ax.plot(list(medians.values()), list(bins_power_cv.values()), color="C3", label="median", linewidth=1)
    ax.plot(list(int_lower.values()), list(bins_power_cv.values()), color="C4", label="lower interval", linewidth=2)
    ax.plot(list(int_upper.values()), list(bins_power_cv.values()), color="C5", label="upper interval", linewidth=2)

    # Set title, labels, and other properties
    ax.set_title("Wind Speed vs Power Output")
    ax.set_xlabel("Wind Speed (m/s)")
    ax.set_ylabel("Power Output (kW)")
    ax.legend(loc='upper left')

    # Show the plot
    plt.tight_layout()
    plt.show()

def plot_sg_diagnostics(bins_power_cv, medians_sg, int_lower, int_upper, int_lower_medfilt, int_upper_medfilt):
    """Diagnostic plots for setting the savitzky-golay filter parameters (sigma clipping)
    Plots the median (filtered) and the confidence intervals (unfiltered and filtered)
    """
    plt.figure()
    plt.plot(list(bins_power_cv.values()), medians_sg.values(), label="median (filtered)", color="C0", linestyle="--", alpha=0.2)
    plt.plot(list(bins_power_cv.values()), int_lower.values(), label="lower interval (unfiltered)", color="C1", linestyle="--", alpha=0.2)
    plt.plot(list(bins_power_cv.values()), int_upper.values(), label="upper interval (unfiltered)", color="C2", linestyle="--", alpha=0.2)
    plt.plot(list(bins_power_cv.values()), int_lower_medfilt.values(), label="lower interval (sg filtered)", color="C1", alpha=1)
    plt.plot(list(bins_power_cv.values()), int_upper_medfilt.values(), label="upper interval (sg filtered)", color="C2", alpha=1)
    plt.legend()

def flag_invalid_data(data: pd.DataFrame, nominal_power: float, ws_cut_in: float):
    """Flag invalid data points in the data frame.

    Parameters
    ----------
    data : pandas.DataFrame
        A dataframe containing the data. Expected to have columns "windspeed [m/s]",
        "power [kW]", and "Flag Computed".
    nominal_power : float
        The nominal power of the turbine.
    ws_cut_in : float
        The cut-in wind speed of the turbine.
    
    Returns
    -------
    data : pandas.DataFrame
        The dataframe with the flagged data points.
    """
    # introduce new column Flag Computed
    data["Flag Computed"] = 0

    # = = = = = VALIDITY of data = = = = =
    # - - - Univariate extreme values - - -
    # Wind speed
    data.loc[((data["windspeed [m/s]"] < 0) | (data["windspeed [m/s]"] > 40)), "Flag Computed"] = 1

    # Power
    data.loc[((data["power [kW]"] / nominal_power <= -0.05) | (data["power [kW]"] / nominal_power > 1.05)), "Flag Computed"] = 1

    # Flag power values around 0 for wind speeds above 2*cut-in speed
    data.loc[((data["windspeed [m/s]"] > ws_cut_in * 2) & (data["power [kW]"] < 0.005 * nominal_power)), "Flag Computed"] = 1

    return data

# BOOKMARK: Goretti
def compute_flag_goretti(data: pd.DataFrame, ws_cut_in: float, ws_rated: float, ws_cut_out: float, nominal_power: float, k_up=3, k_low=3):
    """Flag wind turbine data based on method provided by goretti.

    Here power values of certain wind speed intervals are cleaned based on threshold values. The interval between cut-in and rated speed is cleaned
    based on quantiles with the quantile value provided as input. The bin size (wind speed interval in which quantile is evaluated) varies with the
    amount of data provided - to have outcomes not changind with the amount of data provided.

    Base Code: https://github.com/ggoretti/data_cleaning

    Variables
    ----------
    data_raw : DataFrame
        Raw data to be cleaned.
    ws_cut_in : float
        Cut-in wind speed in m/s.
    ws_rated : float
        Rated wind speed in m/s.
    ws_cut_out : float
        Cut-out wind speed in m/s.
    nominal_power : float
        Nominal power of the wind turbine
    k_up : float, default=1.5
        Multiplier of IQR to define upper threshold for outlier detection.
    k_low : float, default=1.5
        Multiplier of IQR to define lower threshold for outlier detection.

    Returns
    ----------
    data : pandas.DataFrame
        Cleaned data.

    Notes
    -----
    Wind speed values are in m/s.
    Power values are normalised by rated power.
    """
    # = = = = = DATA CLEANING = = = = =
    # introduce new column Flag Computed
    data["Flag Computed"] = 0

    # = = = = = VALIDITY of data = = = = =
    # - - - Univariate extreme values - - -
    # Wind speed
    data.loc[((data["windspeed [m/s]"] < 0) | (data["windspeed [m/s]"] > 40)), "Flag Computed"] = 1

    # Power
    data.loc[((data["power [kW]"] / nominal_power <= -0.05) | (data["power [kW]"] / nominal_power > 1.05)), "Flag Computed"] = 1

    # - - - Bivariate extreme values - - -
    # 1. Flag instances of high power output for wind speed in [0, ws_cut_in - 0.5] with 2
    data.loc[((data["windspeed [m/s]"] >= 0) & (data["windspeed [m/s]"] < ws_cut_in - 0.5) & (data["power [kW]"] / nominal_power > 0.04)),
             "Flag Computed"] = 2

    # 2. Flag instances of non-zero power for wind speed > ws_cut_out + 0.5 with 2
    data.loc[((data["windspeed [m/s]"] >= ws_cut_out + 0.5) & (data["power [kW]"] > 0)), "Flag Computed"] = 2

    # 3. Flag instances of zero power output for wind speed in [ws_cut_in+2,
    #    ws_cut_out-0.5] with 3
    data.loc[((data["windspeed [m/s]"] > ws_cut_in + 2) & (data["windspeed [m/s]"] < ws_rated + 0.5) &
              (data["power [kW]"] / nominal_power < 0.04)), "Flag Computed"] = 3

    # 4. Flag instances of low power output (<95% of P_nom) for wind speed in
    #    [ws_rated + 0.5, ws_cut_out - 0.5] with 3
    data.loc[((data["windspeed [m/s]"] > ws_rated + 0.5) & (data["windspeed [m/s]"] < ws_cut_out - 0.5) &
              (data["power [kW]"] / nominal_power < 0.95)), "Flag Computed"] = 3

    # 5. Remove instances of high power output for wind speed in [ws_cut_in+0.5, ws_rated]
    group_data = data[data["Flag Computed"] == 0]
    bin_width = (ws_cut_out - ws_cut_in) * 1.5e3 / \
        (len(group_data["power [kW]"]) - group_data["power [kW]"].isna().sum())  # [m/s]

    # 6.1 Group by wind speed values, bin width=bin_width.
    grouped = group_data.groupby(
        pd.cut(group_data["windspeed [m/s]"], np.arange(ws_cut_in + 0.5, ws_rated + 0.5, bin_width))
        )
    for key, df in grouped:
        # 5.2 Calculate outlier threshold (Q3 + 2.5*IQR) for each group
        if not df["power [kW]"].dropna().empty:
            q25, q75 = np.percentile(df["power [kW]"].dropna(), [25, 75])
            iqr = q75 - q25
            thresh_up = q75 + k_up*iqr
            thresh_low = q25 - k_low * iqr

            # 5.3 Flag instances where power is above the threshold
            data.loc[((data["windspeed [m/s]"] > key.left) & (data["windspeed [m/s]"] <= key.right) &
                      (data["power [kW]"] > thresh_up)), "Flag Computed"] = 4

            # 6.3 Flag instances where power output is below the threshold
            data.loc[((data["windspeed [m/s]"] > key.left) & (data["windspeed [m/s]"] <= key.right) &
                      (data["power [kW]"] < thresh_low)), "Flag Computed"] = 4

    return data

# BOOKMARK: Quantiles
def compute_flag_quantiles(data: pd.DataFrame, ws_cut_in: float, ws_rated: float, ws_cut_out: float,
                           nominal_power: float, plot: bool, number_of_quantiles: int,
                           threshold_fall: float, cut_threshold: float = np.nan):
    """Flag wind turbine data based on quantiles.

    Here the quantiles are followed by the lowest (0.002) as long as their derivation is positive.
    When it oscillates down, the next higher quantile is chosen (from lower wind speeds to higher wind speeds).

    The code is inspired by, although the methodoloy changed (no power curve fitting, onle derivation as indication for quantile change):
    https://ayrtonb.github.io/Merit-Order-Effect/ug-03-power-curve/

    Variables
    ----------
    data_raw : DataFrame
        Raw data to be cleaned.
    ws_cut_in : float
        Cut-in wind speed in m/s.
    ws_rated : float
        Rated wind speed in m/s.
    ws_cut_out : float
        Cut-out wind speed in m/s.
    nominal_power : float
        Nominal power of the wind turbine
    plot: bool
        If True the choice of quantiles is plot.
    number_of_quantiles: int
        Number of quantiles used for the evaluation of the lower limit for
        data cutting. Considering more quantiles means a finer resolution in the cut,
        and a higher computation time.
    threshold_fall: float
        Threshold value for the decision if a quantile is considered as "falling"
        in comparison to the value of the previous bin. Considering low values 
        (large magnitudes with negative signs) makes the method less sensitive.
    cut_threshold: float, default = nan
        Windspeed threshold under which quantile based flagging is applied. If the wind
        speed exceeds this threshold, the data is not flagged by the quantile based
        routine. If it is set to anything other than a number, quantile based flagging
        is applied to all data points.

    Returns
    ----------
    data : pandas.DataFrame
        Cleaned data.

    Notes
    -----
    Wind speed values are in m/s.
    Power values are normalised by rated power.
    """
    # extract only unflagged data
    group_data = data[data["Flag Computed"] == 0]

    # FLAGGING BASED ON QUANTILES ONLY
    # create wind speed vector with step size 0.1 until rated wind speed and step size 1 
    # between rated wind speed and cut-out wind speed
    vtest = list(np.arange(-0.05, ws_rated + 0.05, 0.1)) + list(np.arange(ws_rated + 0.15, ws_cut_out + 0.5, 0.5))
    # group data according to constructed vector
    grouped = group_data.groupby(pd.cut(group_data["windspeed [m/s]"], vtest), observed=False)

    # evaluate and save quantiles for each interval of wind speeds
    quantiles = []
    index = []
    relevant = 0
    for key, df in grouped:
        if not df["power [kW]"].dropna().empty:
            index.append((key.left + key.right)/2)
            this_quantile = df["power [kW]"].dropna().quantile(np.linspace(0.0, 0.5, number_of_quantiles))
            quantiles.append(this_quantile)

    # put quantiles of each wind speed interval in dataframe and relabel index by corresponding wind speed.
    df_quantiles = pd.DataFrame(quantiles)
    df_quantiles.index = index

    # introduce new column, where the value of the chosen quantile is saved for each windspeed interval
    df_quantiles["mask"] = -0.05 * nominal_power

    # loop over quantiles and find index where derivative is lower than -1
    for column in df_quantiles.columns:
        # numerical derivative
        quantile_test = df_quantiles[column].diff()
        # check if line moves after cut in speed -> if derivative is on average smaller than 1 in that range, don't consider the line
        if np.mean(abs(quantile_test[((df_quantiles.index <= ws_cut_in * 2) & (df_quantiles.index >= ws_cut_in * 1))])) <= 1:
            continue
        if relevant > ws_rated * 1.5:
            break
        # look at derivative after index, where limit is already determined
        quantile_test = quantile_test[df_quantiles.index >= relevant]
        # extract values where derivative is below indicated threshol
        quantile_test = quantile_test <= threshold_fall

        # save the relevant quantile value to columns ["mask"]
        if sum(quantile_test) >= 1:
            # check if new quantile is suited for higher wind speeds (in comparison to previous one)
            if quantile_test[quantile_test].index[0] - relevant >= 0:
                # change quantile values in column mask and save index
                df_quantiles.loc[relevant:, ["mask"]] = df_quantiles.loc[relevant:, [column]].values
                relevant = quantile_test[quantile_test].index[0]

    # plot quantiles and chosen thresholdline to explain methodology
    if plot:
        fig, ax = plt.subplots(dpi=150)
        ax.scatter(data["windspeed [m/s]"], data["power [kW]"], color='k', s=1e-2)
        ax.plot(df_quantiles.index, df_quantiles["mask"], color="red",
                label="quantile based mask", linewidth=1)
        ax.legend()
        df_quantiles.plot(cmap='viridis', legend=False, ax=ax, linewidth=0.2)
        ax.plot(df_quantiles.index, df_quantiles["mask"], color="red", linewidth=1)
        ax.set_xlabel('Wind Speed (m/s)')
        ax.set_ylabel('Power Output (kW)')
        plt.show()

    # removing all data with x > speed von max und y < power von max
    if (type(cut_threshold) == np.float64) and np.isnan(cut_threshold) == False:
        for windspeed in df_quantiles.index:
            max_power = df_quantiles.loc[windspeed, "mask"]
            condition = (
                (data["windspeed [m/s]"] > windspeed) & 
                (data["power [kW]"] < max_power) & 
                (data["Flag Computed"] == 0) & 
                (data["windspeed [m/s]"] <= cut_threshold)
            )
            data.loc[condition, "Flag Computed"] = 2
    else:
        for windspeed in df_quantiles.index:
            max_power = df_quantiles.loc[windspeed, "mask"]
            condition = (
                (data["windspeed [m/s]"] > windspeed) & 
                (data["power [kW]"] < max_power) & 
                (data["Flag Computed"] == 0)
            )
            data.loc[condition, "Flag Computed"] = 2
        

    return data

# BOOKMARK: Sigma Clipping
def compute_flag_sigma(data: pd.DataFrame, nominal_power: float, plot: bool,sc_mult: float = 3, cut_threshold: float = np.nan):
    """Flag wind turbine data with n-sigma-clipping.

    Data is binned. For each bin, the median and variance of the power are calculated.
    Data points outside a 3 sigma range of the median are flagged.

    Variables
    ----------
    data_raw : DataFrame
        Raw data to be cleaned.
    nominal_power : float
        Nominal power of the wind turbine
    plot: bool
        If True the choice of quantiles is plot.
    sc_mult : float, default=3
        Multiplier of standard deviation to define outlier threshold.
    cut_threshold: float, default = None
        Windspeed threshold under which sigma clipping is applied. If the wind 
        speed exceeds this threshold, the data is not flagged by the sigma 
        clipping routine. If it is set to anything other than a number, sigma clipping
        is applied to all data points.

    Returns
    ----------
    data : pandas.DataFrame
        Cleaned data.

    Notes
    -----
    Wind speed values are in m/s.
    Power values are normalised by rated power.
    """
    # = = = = = OPTIONS = = = = =


    # Set step size for power binning
    step_size_binning = 1

    # Set smoothing options for Savitzky-Golay filter
    sg_window_size_kW = nominal_power/3
    sg_poly_order = 3
    sg_window_size = int(sg_window_size_kW / step_size_binning)

    # Smoothing options for median filter
    medfilt_window_size = int(sg_window_size/5)
    # Ensure this is a positive odd integer
    if medfilt_window_size % 2 == 0:
        medfilt_window_size += 1

    ###########################################################################

    # Set sigma multiplier for outlier detection
    sigma_mult = sc_mult

    # extract only unflagged data
    group_data = data[data["Flag Computed"] == 0]

    # Extract only data below cut threshold
    if (type(cut_threshold) in [np.float64, np.int64]) and np.isnan(cut_threshold) == False:
        group_data = group_data[group_data["windspeed [m/s]"] <= cut_threshold]

    # SIGMA CLIPPING
    # create bins for power
    bins_power = list(np.arange(group_data["power [kW]"].min() - 0.5 * step_size_binning, group_data["power [kW]"].max() + 0.5 * step_size_binning, step_size_binning))
    # group data according to constructed vector
    grouped = group_data.groupby(pd.cut(group_data["power [kW]"], bins_power), observed=False)

    bins_power_cv = calc_central_values(grouped)

    # evaluate and save median and std. dev. for each interval
    medians = {}
    std_devs = {}
    int_upper = {}
    int_lower = {}
    int_lower_medfilt = {}
    int_upper_medfilt = {}

    for key, df in grouped:
        if (not df["windspeed [m/s]"].dropna().empty) & (not df.count()["windspeed [m/s]"] == 1):
                medians[key] = df.median()["windspeed [m/s]"]
                std_devs[key] = df.std()["windspeed [m/s]"]
        else:
            medians[key] = np.nan
            std_devs[key] = np.nan

    # Fill NaNs with nearest values
    medians = dict(zip(medians.keys(),pd.Series(medians.values()).ffill()))
    std_devs = dict(zip(medians.keys(),pd.Series(std_devs.values()).ffill()))
    medians = dict(zip(medians.keys(),pd.Series(medians.values()).bfill()))
    std_devs = dict(zip(medians.keys(),pd.Series(std_devs.values()).bfill()))

    # Smooth medians and std. devs. with Savitzky-Golay filter while preserving dictionary keys
    try:
        medians_sg = dict(zip(medians.keys(), signal.savgol_filter(list(medians.values()), sg_window_size, sg_poly_order)))
    except:
        print("Error in Savitzky-Golay filter. Trying again with different padding...")
        medians_sg = dict(zip(medians.keys(), signal.savgol_filter(list(medians.values()), sg_window_size, sg_poly_order, mode="nearest")))

    for key, df in grouped:
        int_lower[key] = medians_sg[key] - sigma_mult * std_devs[key]
        int_upper[key] = medians_sg[key] + sigma_mult * std_devs[key]

    int_lower_medfilt = dict(zip(int_lower.keys(), ndimage.median_filter(list(int_lower.values()), size=medfilt_window_size, mode="constant")))
    int_upper_medfilt = dict(zip(int_upper.keys(), ndimage.median_filter(list(int_upper.values()), size=medfilt_window_size, mode="constant")))

    for key, df in grouped:
        if bins_power_cv[key] > nominal_power * 0.96:
            # Flag only lower interval, leaving upper interval intact
            ind_flag = df[(df["windspeed [m/s]"] <= int_lower_medfilt[key])].index
            data.loc[ind_flag, "Flag Computed"] = 5        
        elif bins_power_cv[key] > nominal_power * 0.01:
            # Flag both intervals
            ind_flag = df[(df["windspeed [m/s]"] <= int_lower_medfilt[key]) | (df["windspeed [m/s]"] >= int_upper_medfilt[key])].index
            data.loc[ind_flag, "Flag Computed"] = 5
        else:
            # Flag only upper interval, leaving lower interval intact
            ind_flag = df[df["windspeed [m/s]"] >= int_upper_medfilt[key]].index
            data.loc[ind_flag, "Flag Computed"] = 5

    # Diagnostic plots for tuning the filter. Set to False for production runs.
    diag_plot = False
    if diag_plot:
        plot_sg_diagnostics(bins_power_cv, medians_sg, int_lower, int_upper, int_lower_medfilt, int_upper_medfilt)

    # plots to explain methodology
    if plot:
        create_static_plot(data, sigma_mult, medians_sg, bins_power_cv, int_lower_medfilt, int_upper_medfilt)

    return data

# BOOKMARK: PC shift
def compute_flag_pc_shift(data: pd.DataFrame, shift_ws: float, shift_pw: float,
                          power_curve: pd.DataFrame, ws_rated: float, 
                          ws_cut_out: float, nominal_power: float, plot: bool, 
                          cut_threshold: float = np.nan):
    """Flag wind turbine data by shifting the power curve to the right and left.

    The manufacturer power curve is shifted to the left and right by a certain
    amount, given by the "shift" parameter

    Variables
    ----------
    data: DataFrame
        Raw data to be cleaned.
    shift_ws: float
        Left/right shift of the power curve in % of rated wind speed. 
        Larger values produce more conservative results.
    shift_pw: float
        Up/down shift of the power curve in % of nominal power. 
        Larger values produce more conservative results.
    power_curve: DataFrame
        Power curve of the turbine, generated by get_turbine_power_curve_from_type.
    ws_rated : float
        Rated wind speed in m/s.
    ws_cut_out : float
        Cut-out wind speed in m/s.
    nominal_power : float
        Nominal power of the wind turbine
    plot: bool
        If True the choice of quantiles is plot.
    cut_threshold: float, default = nan
        Windspeed threshold under which pc shifting is applied. If the wind
        speed exceeds this threshold, the data is not flagged by the pc shifting.
        If it is set to anything other than a number, pc shifting is applied to
        all data points.

    Returns
    ----------
    data : pandas.DataFrame
        Cleaned data.

    Notes
    -----
    Wind speed values are in m/s.
    Power values are normalised by rated power.
    """

    # extract only unflagged data to be processed
    data_tbp = data[data["Flag Computed"] == 0]

    # Extract only data below cut threshold
    if (type(cut_threshold) == np.float64) and np.isnan(cut_threshold) == False:
        data_tbp = data_tbp[data_tbp["windspeed [m/s]"] <= cut_threshold]

    # create right shifted power curve
    power_curve_shifted = power_curve.copy()
    power_curve_shifted["value"] = power_curve_shifted["value"] * 1e-3
    power_curve_shifted["wind_speed_rs"] = power_curve_shifted["wind_speed"] + shift_ws/100 * ws_rated
    power_curve_shifted["wind_speed_ls"] = power_curve_shifted["wind_speed"] - shift_ws/100 * ws_rated
    power_curve_shifted["value_us"] = power_curve_shifted["value"] + shift_pw/100 * nominal_power
    power_curve_shifted["value_ds"] = power_curve_shifted["value"] - shift_pw/100 * nominal_power

    # Filter data where the wind speed exceeds the valid mask area (to the right of cut-out of left-shifted curve)
    threshold = ws_cut_out - (shift_ws+1)/100 * ws_rated
    data_tbp = data_tbp[data_tbp["windspeed [m/s]"] < threshold]

    # Flag data points where the power is above the left-shifted power curve
    data_tbp.loc[data_tbp["power [kW]"] > np.interp(data_tbp["windspeed [m/s]"], power_curve_shifted["wind_speed_ls"], power_curve_shifted["value_us"]), "Flag Computed"] = 6
    # Flag data points where the power is below the right-shifted power curve
    data_tbp.loc[data_tbp["power [kW]"] < np.interp(data_tbp["windspeed [m/s]"], power_curve_shifted["wind_speed_rs"], power_curve_shifted["value_ds"]), "Flag Computed"] = 6
    

    # PLOTTING
    if plot:
        ax, fig = plt.subplots(dpi=150)
        plt.scatter(data_tbp[data_tbp["Flag Computed"] != 6]["windspeed [m/s]"], data_tbp[data_tbp["Flag Computed"] != 6]["power [kW]"], color='k', s=1e-2, label="unflagged data")
        plt.scatter(data_tbp[data_tbp["Flag Computed"] == 6]["windspeed [m/s]"], data_tbp[data_tbp["Flag Computed"] == 6]["power [kW]"], color='red', s=1e-2, label="flagged data")
        plt.plot(power_curve_shifted["wind_speed"], power_curve_shifted["value"], color="red", label="original power curve", linewidth=1)
        plt.plot(power_curve_shifted["wind_speed_rs"], power_curve_shifted["value_ds"], color="blue", label="right-shifted power curve", linewidth=1)
        plt.plot(power_curve_shifted["wind_speed_ls"], power_curve_shifted["value_us"], color="blue", label="left-shifted power curve", linewidth=1)
        ax.legend(loc='lower right')
        plt.xlabel('Wind Speed (m/s)')
        plt.ylabel('Power Output (kW)')
        plt.tight_layout()
        plt.show()

    # Write new flags from data_tbp to data
    data.loc[data_tbp.index, "Flag Computed"] = data_tbp["Flag Computed"]    
    return data

def plot_turbine_power_curves(path: str):
    fig, ax = plt.subplots(figsize=(16, 9))
    # [
    #     "3.2M114", "3.2M122", "D6", "D8", "E101", "E126", "E40", "E66", "E70", "E82_2MW",
    #     "E82_3MW", "E92", "E92_2MW", "E92_restr", "M1800-600/150", "MM100", "MM82", "MM92", "NM48/750", "V100",
    #     "V112", "V126", "V136", "V150", "V29", "V44", "V47", "V52", "V80", "V90"
    #     ]
    for t_id in [
        "D8",
        "E66", "E82_3MW", "E101",
        "V100", "V150", "V162-7.2MW"]:
        powercurve_df = get_turbine_power_curve_from_type(path=path, turbine_type=t_id)
        ax.plot(powercurve_df["wind_speed"], powercurve_df["value"], label=t_id)
    ax.set_xlabel("windspeed [m/s]")
    ax.set_ylabel("power [kW]")
    ax.legend()
    plt.show()

# BOOKMARK: Compute power and flag
def compute_power_and_flag(filepath: str, dc_config: pd.DataFrame, plotting: bool) -> Tuple:
    """Evaluate wind power production from turbine curve and compare to real data."""

    data = pd.read_table(filepath, na_values=-99.99)
    data["TimeStamp"] = pd.to_datetime(data["TimeStamp"], format="%d/%m/%Y %H:%M")

    # extract and filter relevant data, complete time series to 10 min frequency, 
    # convert to UTC (Austria)
    data = tsc.prepare_timeseries_data(data)
    turbine_id, turbine, turbine_spec = initialize_turbine(filepath=filepath)

    if turbine is None or turbine_spec is None:
        return (turbine_id, np.NAN, np.NAN, np.NAN, np.NAN)
    # method to calculate power output
    power_computed = wpl.power_output.power_curve(wind_speed=data["windspeed [m/s]"], power_curve_wind_speeds=turbine.power_curve["wind_speed"],
                                                  power_curve_values=turbine.power_curve["value"])
    # save computed power to dataframe
    data["Power computed [kW]"] = power_computed * 1e-3

    # Read corresponding line from dc_config
    turbine_type = turbine_spec[3]
    turbine_data = dc_config[dc_config["turbine_type"] == turbine_type]

    # ACTUAL DATA CLEANING AND FLAGGING #######################################
    print(f"Processing turbine {turbine_id} ...")

    # First step: Flag invalid data
    data = flag_invalid_data(data, nominal_power=turbine.nominal_power * 1e-3, ws_cut_in=turbine_spec[0])
    # Second step: Apply pc shifting
    if turbine_data["perform_pc_shift"].values[0]:
        shift_ws = turbine_data["pc_shift_ws"].values[0]
        shift_pw = turbine_data["pc_shift_pw"].values[0]
        cut_thresh_pc = turbine_data["cut_thresh_pc"].values[0]
        data = compute_flag_pc_shift(data, shift_ws=shift_ws, shift_pw=shift_pw, power_curve=turbine.power_curve, 
                                    ws_rated=turbine_spec[2], ws_cut_out=turbine_spec[1], 
                                    nominal_power=turbine.nominal_power * 1e-3, plot=plotting, cut_threshold=cut_thresh_pc)
    # Third step: Apply sigma clipping
    if turbine_data["perform_sigma_clipping"].values[0]:
        sc_mult = turbine_data["sc_mult"].values[0]
        cut_thresh_sc = turbine_data["cut_thresh_sc"].values[0]
        data = compute_flag_sigma(data=data, nominal_power=turbine.nominal_power * 1e-3, plot=plotting, sc_mult=sc_mult, cut_threshold=cut_thresh_sc)
    # Fourth step: Apply quantile based flagging
    if turbine_data["perform_quantile_mapping"].values[0]:
        num_quantiles = turbine_data["num_quantiles"].values[0]
        thresh_fall = turbine_data["thresh_fall"].values[0]
        cut_thresh_qm = turbine_data["cut_thresh_qm"].values[0]
        data = compute_flag_quantiles(data=data, ws_cut_in=turbine_spec[0], ws_rated=turbine_spec[2], ws_cut_out=turbine_spec[1],
                                    nominal_power=turbine.nominal_power * 1e-3, plot=plotting,
                                    number_of_quantiles=num_quantiles, threshold_fall=thresh_fall, cut_threshold=cut_thresh_qm)
    filename = os.path.basename(filepath)
    filename = filename.replace(".dat", ".csv")
    save_path = os.path.join(FolderPaths.flagged_dir, filename)
    data.to_csv(save_path, encoding="utf-8")

    # evaluate flagging and model performance
    flagged_values = int(sum(data["Flag Computed"].loc[data["Flag Computed"] > 0]))
    core = data["power [kW]"].loc[data["Flag Computed"] == 0] - data["Power computed [kW]"].loc[data["Flag Computed"] == 0]
    RMSE = ((core) ** 2).mean() ** .5 / (turbine.nominal_power * 1e-3)
    MSE = abs(core).mean() / (turbine.nominal_power * 1e-3)
    BIAS = core.mean() / (turbine.nominal_power * 1e-3)
    return (turbine_id, BIAS, MSE, RMSE, flagged_values)

def process_file(this_file, dc_config, folder_path, plotting):
    """Wrapper for multiprocessing"""
    power_evaluation_line = compute_power_and_flag(this_file, dc_config, plotting)
    put_power_evaluation_line_into_database(path=folder_path, power_evaluation_line=power_evaluation_line)

# BOOKMARK: main function
def main(plotting: bool = False):
    '''Main function to run the data cleaning process.'''
    # Read in files and initialize folders ####################################
    input_files_list = os.listdir(FolderPaths.input_dir)
    num_input_files = len(input_files_list)

    # Read data cleaning parameters from config file
    dc_config_path = os.path.join(FolderPaths.metadata_dir, "data_cleaning_config.xlsx")
    dc_config = pd.read_excel(dc_config_path)

    os.makedirs(FolderPaths.flagged_dir, exist_ok=True)

    if os.listdir(FolderPaths.flagged_dir):
        [os.remove(os.path.join(FolderPaths.flagged_dir, f)) for f in os.listdir(FolderPaths.flagged_dir)]

    # Add full path to input files
    input_files_list = [os.path.join(FolderPaths.input_dir,f) for f in input_files_list]

    # Run data cleaning in parallel ###########################################
    # Generator object providing a full config dataframe per iteration
    # This is used to pass the config to the process_file function
    config_gen = (dc_config for f in input_files_list)
    plotting_gen = (plotting for f in input_files_list)

    # XXX
    # process_file(input_files_list[0], FolderPaths.flagged_dir, dc_config, plotting)

    # Create a ProcessPoolExecutor for concurrent execution
    print_green(f"Flagging in a pool of {ParallelProcessing.num_conc_proc} processes")
    print("This may take a while ...")
    with concurrent.futures.ProcessPoolExecutor(max_workers=ParallelProcessing.num_conc_proc) as executor:
        executor.map(process_file, 
                     input_files_list, 
                     config_gen, 
                     [FolderPaths.flagged_dir]*len(input_files_list), 
                     plotting_gen)
        
    # Check if all files were processed #######################################
    num_files_prepared = len(os.listdir(FolderPaths.flagged_dir))
    delta = num_input_files - num_files_prepared
    if delta != 0:
        prepared_files = os.listdir(FolderPaths.flagged_dir)
        diff_files = list(set(input_files_list) - set(prepared_files))
        print_red(f"WARNING: {delta} files were not processed:")
        print_red(f"{diff_files}")