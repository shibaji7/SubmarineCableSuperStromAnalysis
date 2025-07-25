import datetime as dt
import os

import numpy as np
import pandas as pd
from loguru import logger
from utils import StackPlots
import matplotlib.dates as mdates

os.makedirs("figures/2025/", exist_ok=True)


def read_plot_Meta_datasets():
    o = pd.read_csv("data/2025/meta-data-03June2025.csv", parse_dates=["UTC"])
    o.set_index("UTC", inplace=True)
    sp = StackPlots(nrows=2, ncols=1, datetime=True, figsize=(8, 3), text_size=12)

    ax = sp.axes[0]
    ax.set_ylabel("Voltages, V", color="r")
    ax.set_title("Submarine Cable Voltages on 03 June 2025")
    ax.text(
        0.05,
        0.95,
        "(a) Voltages on EU end",
        fontsize=14,
        ha="left",
        transform=ax.transAxes,
    )
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=6))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H"))
    ax.fill_between(
        o.index,
        o.VEUMin,
        o.VEUMax,
        color="red",
        alpha=0.5,
    )
    ax.set_xlim(
        dt.datetime(2025, 6, 3, 0, 0),
        dt.datetime(2025, 6, 5, 0, 0),
    )
    ax.set_ylim(-13600, -13300)
    ax = ax.twinx()
    ax.set_ylabel("Current, mA", color="b")
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=6))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H"))
    ax.fill_between(
        o.index,
        o.LCUSMin,
        o.LCUSMax,
        color="b",
        alpha=0.5,
    )
    ax.set_ylim(1100, 1300)

    ax = sp.axes[1]
    ax.set_xlabel("Time, UTC")
    ax.set_ylabel("Voltages, V", color="r")
    ax.text(
        0.05,
        0.95,
        "(b) Voltages on US end",
        fontsize=14,
        ha="left",
        transform=ax.transAxes,
    )
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=6))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H"))
    ax.set_xlim(
        dt.datetime(2025, 6, 3, 0, 0),
        dt.datetime(2025, 6, 5, 0, 0),
    )
    ax.fill_between(
        o.index,
        o.VUSMin,
        o.VUSMax,
        color="r",
        alpha=0.5,
    )
    ax.set_ylim(700, 1100)
    ax = ax.twinx()
    ax.set_ylabel("Current, mA", color="b")
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=6))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H"))
    ax.fill_between(
        o.index,
        o.LCUSMin,
        o.LCUSMax,
        color="b",
        alpha=0.5,
    )
    ax.set_ylim(1100, 1300)

    sp.save_fig("figures/2025/03June2025.data.png")
    sp.close()
    return


if __name__ == "__main__":
    """
    Main function to execute the analysis and plotting.
    """
    logger.info("Starting the Submarine Cable Super Storm Analysis for June 2025.")
    
    # Read and plot metadata datasets
    read_plot_Meta_datasets()
    
    # Additional analysis and plotting can be added here
    logger.info("Analysis completed. Check figures/2025/ for results.")