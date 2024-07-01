"""01.07.2024, Version 1.0.

This module contains functions to clean time series data.

@author: Johanna Ganglbauer
"""

import numpy as np
import pandas as pd
import datetime as dt

def last_sunday(year: int, month: int):
    """Find last sunday of a given month in a given year."""
    if month == 12:
        year += 1
        month = 1
    else:
        month += 1
    last_sunday_date = dt.datetime(year, month, 1)
    # erster Sonntag
    last_sunday_date += dt.timedelta(days=6-last_sunday_date.weekday())
    last_sunday_date -= dt.timedelta(days=7)  # last sunday
    return last_sunday_date

def convert_to_utc_austria(timeseries_data: pd.DataFrame) -> pd.DataFrame:
    """Detects austrian time shift, if contained in Time Series, and converts data to UTC format. Makes only sense for Austrian data!

    :param timeseries_data: dataframe with index in dt.timestamp format describing time stamp of corresponding measurement. 
    :type timeseries_data: pd.DataFrame
    :return: dataframe with index in dt.timestamp format converted to UTC if time shift was detected (Austria)
    :rtype: pd.DataFrame
    """    
    # Extract the years from the index
    years = timeseries_data.index.year.unique()

    # Loop over the years
    # Create a range of dates that includes all the last Sundays in October
    # that is when the time shift happens in Austria
    last_sunday_dates_october = [last_sunday(int(year), 10) for year in years]

    # Loop over the dates and detect timeshift
    time_in_UTC = True
    for date in last_sunday_dates_october:
        # Check if there are duplicate timestamps between 2 and 3am on this date
        # where the time shift would happen
        date_1 = date + dt.timedelta(seconds=2*3600)
        date_2 = date + dt.timedelta(seconds=3*3600)
        duplicates = timeseries_data.loc[(timeseries_data.index >= date_1) & (timeseries_data.index < date_2)].index.duplicated()
        if duplicates.any():
            time_in_UTC = False

    # Loop over the years
    if not time_in_UTC:
        # find first half of duplicated timestamp
        duplicated_mask = ~timeseries_data.index.duplicated()
        for year in years:
            # Find the last Sunday in March and October for the current year
            last_sunday_october_previous_year = last_sunday(year - 1, 10)
            last_sunday_march = last_sunday(year, 3)
            last_sunday_october = last_sunday(year, 10)

            # Subtract one hour from the timestamps outside of the wintertime period in the beginning of the year
            timeseries_data.loc[(
                (timeseries_data.index < last_sunday_march + dt.timedelta(seconds=2 * 3600)) |
                (timeseries_data.index >= last_sunday_october_previous_year + dt.timedelta(hours=2 * 3600))
                )].shift(periods=-1, freq="H")

            # Subtract two hours from the timestamps between last Sunday in March 2am
            # and last Sunday in October 3am
            # summertime to UTC
            timeseries_data.loc[(timeseries_data.index >= last_sunday_march + dt.timedelta(seconds=2 * 3600)) &
                                (timeseries_data.index < last_sunday_october + dt.timedelta(seconds=2 * 3600))].shift(periods=-2, freq="H")
            # also shift the first half of the duplicated timestamp in October 02:00-03:00 by 2 hours
            timeseries_data.loc[(
                (timeseries_data.index >= last_sunday_october + dt.timedelta(seconds=2 * 3600)) &
                (timeseries_data.index < last_sunday_october + dt.timedelta(seconds=3 * 3600)) &
                duplicated_mask
                )].shift(periods=-2, freq="H")

        # Subtract one hour from the timestamps outside of this period
        # wintertime to UTC
        timeseries_data.loc[(timeseries_data.index < last_sunday_march + dt.timedelta(seconds=2 * 3600)) |
                            (timeseries_data.index >= last_sunday_october + dt.timedelta(hours=3 * 3600))].shift(periods=-1, freq="H")
    else:
        pass

    return timeseries_data


def prepare_timeseries_data(data: pd.DataFrame) -> pd.DataFrame:
    """Extract relevant data, resample time series to 10 min frequency and set invalid values to NaN.

    :param data: data frame containing wind speed and wind power data
    :type data: pd.DataFrame
    :return: data frame containing resampled wind speed and wind power data
    :rtype: pd.dataframe]
    """

    """Select needed data from input"""
    timeseries_data = pd.DataFrame()

    # possible column names for timestamp, power, windspeed, temperature, winddirection and gondelposition
    # should be completed for new data formats.
    timestamp_categories = ["timestamp", "TimeStamp", "time", "date"]
    power_categories = ["poweravg", "power [kW]", "power_avg", "Turbine Power"]
    windspeed_categories = ["windspeedavg", "windspeed [m/s]", "windspeed_avg", "Turbine Wind Speed Mean"]
    temperature_categories = ["ambienttempavg", "temperature [C]", "temperature_out"]
    winddirection_categories = ["winddir", "wind direction [deg]", "winddirection"]
    gondelposition_categories = ["gondelpos", "gondel position [deg]"]

    # copy data to new data frame
    timeseries_data["power [kW]"] = data[data.columns.intersection(power_categories)]
    timeseries_data["timestamp"] = data[data.columns.intersection(timestamp_categories)]
    timeseries_data["windspeed [m/s]"] = data[data.columns.intersection(windspeed_categories)]
    if data.columns.intersection(temperature_categories).empty:
        timeseries_data["temperature [C]"] = np.nan
    else:
        timeseries_data["temperature [C]"] = data[data.columns.intersection(temperature_categories)]
    if data.columns.intersection(winddirection_categories).empty:
        timeseries_data["wind direction [deg]"] = np.nan
    else:
        timeseries_data["wind direction [deg]"] = data[data.columns.intersection(winddirection_categories)]
    if data.columns.intersection(gondelposition_categories).empty:
        timeseries_data["gondel position [deg]"] = np.nan
    else:
        timeseries_data["gondel position [deg]"] = data[data.columns.intersection(gondelposition_categories)]
    timeseries_data.index = pd.to_datetime(timeseries_data["timestamp"], format="ISO8601")

    timeseries_data = convert_to_utc_austria(timeseries_data=timeseries_data)

    timeseries_data.sort_index()  # make sure the data is given in the right order
    timeseries_data.drop(columns=["timestamp"], inplace=True)

    # filter data according to accepted range and make sure degrees are not higher than 260
    timeseries_data = timeseries_data.clip(lower=[-1e3, 0, -30, None, None], upper=[1e4, 2e2, 50, None, None])
    timeseries_data[["wind direction [deg]", "gondel position [deg]"]] = timeseries_data[["wind direction [deg]", "gondel position [deg]"]].mod(360)

    # write resampled data timeseries to 10 min and take closest value
    timeseries_data = timeseries_data.resample("10min").first()

    number_of_invalids=timeseries_data[["power [kW]", "windspeed [m/s]"]].isna().count().sum()
    start = dt.datetime.strftime(timeseries_data.index.to_list()[0], "%Y-%m-%d %H:%M")
    end = dt.datetime.strftime(timeseries_data.index.to_list()[-1], "%Y-%m-%d %H:%M")

    return timeseries_data
