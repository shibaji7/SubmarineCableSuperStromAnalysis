import datetime as dt
import glob
import sys
import numpy as np

import matplotlib.dates as mdates
import pandas as pd

sys.path.append("py/")
from ts_plots import TimeSeriesPlot


def read_files_as_pandas(folder):
    df = []
    print(f"Folder: {folder}")
    files = glob.glob(folder + "/*.csv")
    files.sort()
    for f in files:
        print(f"\tFile: {f}")
        o = pd.read_csv(f, parse_dates=["Timestamp"])
        df.append(o)
    df = pd.concat(df)
    return df


def plot_datasets_instack(
    dfs,
    dates=[dt.datetime(2024, 5, 10, 12), dt.datetime(2024, 5, 12)],
    fig_title="",
    volt_key="V_(S1)_PAD (1S) (V)",
    curr_key="I_(S1)_PAD (1S) (mA)",
    fname="figures/AJC.png",
    remove_base=True,
):
    # omni = _load_omni_(dates)
    ts = TimeSeriesPlot(
        dates,
        fig_title,
        num_subplots=len(dfs),
        text_size=15,
        major_locator=mdates.HourLocator(byhour=range(0, 24, 4)),
    )
    for _, df in enumerate(dfs):
        df.Timestamp = df.Timestamp + dt.timedelta(minutes=15)
        ax = ts._add_axis()
        val = np.array(df[volt_key])
        if remove_base:
            val = val - np.nanmedian(val[:60])
        ax.plot(df.Timestamp, val, "b.", ls="None", ms=0.5, alpha=0.7)
        ax.set_xlabel("Time, UT")
        ax.set_ylabel("Voltage, V", fontdict=dict(color="b"))
        # ax = ax.twinx()
        # ax.set_ylabel("IMF [B], nT", fontdict=dict(color="r"))
        # ax.plot(omni.time, omni.B, "r.", ls="None", ms=0.5, alpha=1.0)
    if fname:
        ts.save(fname)
    return


if __name__ == "__main__":
    df = read_files_as_pandas("data/2024/AJC/AJC PFE DATA/Tumon Bay (GUAM)")
    stn, station_code, data_cad = "tmb", 4, 1
    plot_datasets_instack(
        [df],
        fig_title=f"Date: 10-12 May 2024; Stn: {stn.lower()}/guam",
        volt_key=f"S_4_V (V)",
        curr_key=f"S_4_I (mA)",
        fname=f"figures/ajc_{stn.lower()}.png",
    )
