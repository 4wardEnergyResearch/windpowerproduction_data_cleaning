import matplotlib.pyplot as plt
import pandas as pd

from data_cleaning_combined import compute_flag_goretti, compute_flag_quantiles
from timeseries_cleaning import prepare_timeseries_data


# flag invalid values (curtailment, incing, maintenance, etc.)
def clean_and_plot(data: pd.DataFrame, ws_cut_in: float, ws_rated: float, ws_cut_out: float, nominal_power: float,
                   threshold: float, stepsize: float, method:str) -> None:
    if method == "quantiles":
        data = compute_flag_quantiles(
            data=data, ws_cut_in=ws_cut_in, ws_rated=ws_rated, ws_cut_out=ws_cut_out,
            nominal_power=nominal_power, threshold=threshold, stepsize=stepsize,plot=True
            )
    elif method == "goretti":
        data = compute_flag_goretti(
            data=data, ws_cut_in=ws_cut_in, ws_rated=ws_rated, ws_cut_out=ws_cut_out,
            nominal_power=nominal_power
            )
        
    grouped_data = data.groupby("Flag Computed")  # group data by flag

    # print power curves with flag
    _, ax = plt.subplots()
    for name, group in grouped_data:
        if name == 0:
            labelname = "valid data"
        elif name == 1:
            labelname = "value out of sensible range"
        elif name == 2 and method == "quantiles":
            labelname = "wind speed out of quantile (with not-negative derivative)"
        elif name == 2 and method == "goretti":
            labelname = "wind speed higher than cut out - power above zero \nwind speed lower than cut in - power above 4 %"
        elif name == 3 and method == "goretti":
            labelname = "wind speed between cut in and rated - power below 4 % \nwind speed between rated and cut out -power below 95 %"
        elif name == 4 and method == "goretti":
            labelname = "power out of 3rd-quantile of wind speed bin."
        ax.plot(group["windspeed [m/s]"], group["power [kW]"], marker=".", linestyle="", markersize=0.1, label=labelname)
    ax.set_xlabel("wind speed [m/s]")
    ax.set_ylabel("power [kW]")
    pos = ax.get_position()
    ax.set_position([pos.x0, pos.y0, pos.width, pos.height * 0.7])
    ax.legend(bbox_to_anchor=(0.5, 1.6), loc="upper center", markerscale=50)
    plt.show()

if __name__ == "__main__":

    # parameters
    ws_cut_in = 3  # m/s
    ws_rated = 12  # m/s
    ws_cut_out = 25  # m/s
    nominal_power_in_kW = 2e3  #  kW
    threshold = -10
    stepsize = 0.5
    method="quantiles"  # "goretti"

    # read in example data
    data = pd.read_table("example_data//Dataset 1 (with actual power).dat", na_values=-99.99)

    # convert TimeStamp to right format
    data["TimeStamp"] = pd.to_datetime(data["TimeStamp"], format="%d/%m/%Y %H:%M")

    # extract and filter relevant data, complete time series to 10 min frequency, convert to UTC time (only works in Austrian case)
    data = prepare_timeseries_data(data)
    clean_and_plot(data=data, ws_cut_in=ws_cut_in, ws_rated=ws_rated, ws_cut_out=ws_cut_out, nominal_power=nominal_power_in_kW,
                   threshold=threshold, stepsize=stepsize, method=method)