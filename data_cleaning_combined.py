"""Methods for cleaning power production data."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def compute_flag_quantiles(data: pd.DataFrame, ws_cut_in: float, ws_rated: float, ws_cut_out: float, nominal_power: float, threshold: float, stepsize: float, plot: bool):
    """Flag wind turbine data based on quantiles.

    Here quantiles are used to flag data. To find the suitable quantile, many are computed and followed (starting with the lowest) as long as their
    derivation is positive. When the quantile oscillates down (negative derivation), the next higher quantile is chosen (from lower wind speeds to higher wind speeds).


    The code is inspired by:
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
    threshold : float
        Threshold value for maxmimal allowed derivative
    stepsize : float
        interval size, where quantiles are evaluated. The more datapoints are given, the greater the step size should be
    plot: bool
        If True the choice of quantiles is plot.

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

    # Power at zero wind speed
    data.loc[((data["power [kW]"] > 0) & (data["windspeed [m/s]"] == 0 )), "Flag Computed"] = 1

    # extract only unflagged data
    group_data = data[data["Flag Computed"] == 0]

    # FLAGGING BASED ON QUANTILES ONLY
    # create wind speed vector with step size 0.1 until rated wind speed and 1 between rated wind speed and cut-out wind speed
    vtest = list(np.arange(-stepsize / 2, ws_rated + stepsize / 2, stepsize)) + list(np.arange(ws_rated + 0.15, ws_cut_out + 0.5, 0.5))
    # group data according to constructed vector
    grouped = group_data.groupby(pd.cut(group_data["windspeed [m/s]"], vtest))

    # evaluate and save quantiles for each interval of wind speeds
    quantiles = []
    index = []
    relevant = 0
    for key, df in grouped:
        if not df["power [kW]"].dropna().empty:
            index.append((key.left + key.right)/2)
            this_quantile = df["power [kW]"].dropna().quantile(np.linspace(0.0, 0.5, 200))
            quantiles.append(this_quantile)

    # put quantiles of each wind speed interval in dataframe and relabel index by corresponding wind speed.
    df_quantiles = pd.DataFrame(quantiles)
    df_quantiles.index = index

    # introduce new column, where the value of the chosen quantile is saved for each windspeed interval
    df_quantiles["mask"] = -0.05 * nominal_power

    # loop over quantiles and find index where derivative is lower than given threshold
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
        quantile_test = quantile_test <= threshold

        # save the relevant quantile value to columns ["mask"]
        if sum(quantile_test) >= 1:
            # check if derivative meets criteria at higher wind speed (compared to previous quantiles)
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
    for windspeed in df_quantiles.index:
        max_power = df_quantiles.loc[windspeed, ["mask"]].values[0]
        data["Flag Computed"][((data["windspeed [m/s]"] > windspeed) & (data["power [kW]"] < max_power))] = 2

    return data

def compute_flag_goretti(data: pd.DataFrame, ws_cut_in: float, ws_rated: float, ws_cut_out: float, nominal_power: float, k_up=3, k_low=3):
    """Flag wind turbine data based on method provided by goretti.

    Here power values of certain wind speed intervals are cleaned based on threshold values. The interval between cut-in and rated speed is cleaned
    based on quantiles with the quantile value provided as input. In contrast to the original method, here the bin size
    (wind speed interval in which quantile is evaluated) varies with the amount of data provided - to have outcomes not changind with the amount of data provided.

    inspiration: https://github.com/ggoretti/data_cleaning

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

    # Power at zero wind speed
    data.loc[((data["power [kW]"] > 0) & (data["windspeed [m/s]"] == 0 )), "Flag Computed"] = 1

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