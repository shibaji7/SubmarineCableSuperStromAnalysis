import datetime as dt
import os

import sys
sys.path.extend(["py/", "py/tat1/"])

import numpy as np
import pandas as pd  # type: ignore
from bathymetry import BathymetryAnalysis
from cable import SCUBASModel
from loguru import logger  # type: ignore
from utils import StackPlots, create_from_lat_lon, read_iaga

from mpl_toolkits.axisartist.grid_finder import DictFormatter, FixedLocator
import matplotlib.dates as mdates
import datetime as dt

os.makedirs("figures/1958/", exist_ok=True)


def read_dataset(base_path: str = "data/1958/scaled_data/") -> pd.DataFrame:
    """
    Reads and processes geomagnetic data for the February 1958 superstorm.

    Parameters:
    -----------
    base_path : str
        File path template for geomagnetic data (D, H, Z components).

    Returns:
    --------
    pd.DataFrame
        Processed geomagnetic data with interpolated values and derived fields (X, Y, Z, F).
    """
    import glob

    stns, coords = ["ESK", "FRD"], ["HDZ", "HDZ"]
    frames = {}
    for stn, coord in zip(stns, coords):
        files = glob.glob(base_path + f"{stn}*.dat")
        files.sort()
        frames[stn] = pd.concat([read_iaga(f) for f in files])
    xlim=[dt.datetime(1958, 2, 11,), dt.datetime(1958, 2, 11, 5)]
    # Plot processed data
    for stn, coord in zip(stns, coords):
        sp = StackPlots(nrows=1, ncols=1, datetime=True, figsize=(6, 4), text_size=12)
        data = frames[stn]
        data.drop_duplicates().sort_index(inplace=True)
        _, ax = sp.plot_stack_plots(
            data.index,
            data.X - np.median(data.X.iloc[:60]),
            ylim=[-1500, 1500],
            label=r"$B_x$",
            interval=6,
        )
        sp.plot_stack_plots(
            data.index,
            data.Y - np.median(data.Y.iloc[:60]),
            ylim=[-1500, 1500],
            label=r"$B_y$",
            color="r",
            ax=ax,
            interval=6,
        )
        sp.plot_stack_plots(
            data.index,
            data.Z.shift(periods=10) - np.median(data.Z.iloc[:60]),
            ylim=[-1500, 1500],
            label=r"$B_z$",
            xlabel="Time, UT since 0 UT on 11 Feb 1958",
            color="k",
            ylabel=f"$B[{stn.lower()}]$, nT",
            xlim=xlim,
            ax=ax,
            interval=1,
        )
        ax.legend(loc=2, fontsize=12)
        data = data[(data.index >= xlim[0]) & (data.index < xlim[1])]
        data.index = data.index - dt.timedelta(minutes=2)
        k = 1.
        data.Z, data.X, data.Y = data.Z*k, data.X*k, data.Y*k
        # data.to_csv(f"data/1958/{stn}_scaled.csv", header=True, index=True, float_format="%g")
        sp.save_fig(f"figures/1958/1958.data_{stn}.png")
        sp.close()
    return

def plot_e_fields():
    model_out = pd.read_csv("data/1958/TAT1SimVolt.csv", parse_dates=["Time"])
    # from tat1958Scaled import compile_1958
    # model_out = compile_1958(gplot=False, scale=2).cable.tot_params.copy().reset_index()
    xlim=[dt.datetime(1958, 2, 11,), dt.datetime(1958, 2, 11, 5)]
    sp = StackPlots(
        nrows=2, ncols=1, datetime=True, 
        figsize=(6, 4), text_size=12, 
        gridspec_kw={
            "height_ratios": [4, 1],
            "wspace": 0.05,
            "hspace": 0.05,
        }   
    )
    ax, tax = sp.axes[0], sp.axes[0].twinx()

    ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H"))
    ix = 0
    for j in np.arange(8,-1,-1):
        ax.plot(model_out.Time, 2000*ix + model_out[f"E.X.0{j}"], color="b", ls="-", lw=0.6)
        ax.plot(model_out.Time, 2000*ix + model_out[f"E.Y.0{j}"] - 500, color="m", ls="-", lw=0.6)
        ix+=1
    ax.axvline(dt.datetime(1958, 2, 11, 0, 30), ymin=0.43, ymax=0.52, color="g", ls="-", lw=1.5)
    ax.text(dt.datetime(1958, 2, 11, 0, 32), 2000*3 + 500, "2 V/km", color="g", fontsize=10)
    ax.text(0.05, 0.95, "(A)", color="k", transform=ax.transAxes, fontsize=12)
    ax.set_yticklabels([])
    tax.set_yticklabels([])
    for j, name in enumerate(["CS-E", "DO-5","MAR", "DO-4",  "RDG-1", "DO-3","DO-2", "DO-1","CS-W"]):
        ax.text(
            xlim[1] + dt.timedelta(minutes=5),
            2000*j - 300,
            name,
            color="k",
            fontsize=6,
            rotation=90,
            va="center",
            ha="center",
        )
    ax.set_xlabel("")
    ax.set_ylabel("$E_x$, mv/km", color="b")
    tax.set_ylabel("$E_y$, mv/km", color="m")
    ax.set_xlim(xlim)

    ax = sp.axes[1]
    ax.text(0.05, 0.95, "(B)", color="k", transform=ax.transAxes, fontsize=12)
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
    ax.plot(model_out.Time, model_out["U0"], color="k", ls="-", lw=0.6, label="$U_E$")
    ax.plot(model_out.Time, model_out["U1"], color="r", ls="-", lw=0.6, label="$U_W$")
    ax.legend(loc=2, fontsize=10)
    ax.set_ylabel("Voltage, V")
    ax.set_ylim(-300, 300)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H"))
    ax.set_xlabel("Time, UT (11 Feb 1958)")
    sp.save_fig(f"figures/tat1/1958.Efield.png")
    sp.close()

    # model_out = compile_1958(gplot=False, scale=2).cable.tot_params.copy().reset_index()
    xlim=[dt.datetime(1958, 2, 11,), dt.datetime(1958, 2, 11, 5)]
    sp = StackPlots(
        nrows=1, ncols=1, datetime=True, 
        figsize=(6, 4), text_size=12,   
    )
    ax = sp.axes[0]
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H"))
    ax.set_ylabel("Voltage, V")
    ax.set_xlabel("Time, UT (11 Feb 1958)")
    ax.set_yticklabels([])
    ax.set_xlim(xlim)
    ix = 8
    for j, name in enumerate(["CS-E", "DO-5","MAR", "DO-4",  "RDG-1", "DO-3","DO-2", "DO-1","CS-W"]):
        ax.plot(
            model_out.Time,
            400* j + model_out[f"V(v).0{ix}"],
            color="k",
            ls="-",
            lw=0.6,
        )
        ax.text(
            xlim[1] + dt.timedelta(minutes=5),
            400*j + model_out[f"V(v).0{ix}"].iloc[-1],
            name,
            color="k",
            fontsize=6,
            rotation=90,
            va="center",
            ha="center",
        )
        ix -= 1
    ax.axvline(dt.datetime(1958, 2, 11, 0, 30), ymin=0.495, ymax=0.575, color="g", ls="-", lw=1.5)
    ax.text(dt.datetime(1958, 2, 11, 0, 32), 400*3 + 500, "0.2 kV", color="g", fontsize=10)
    sp.save_fig(f"figures/tat1/1958.Vs.png")
    sp.close()
    return

def toGeoMag_Domain():
    import pyIGRF
    from tat1958Scaled import read_dataset

    def _apply_Hmag_(row, D):
        XYgeo = np.array([[row["X"]], [row["Y"]]])
        R = np.array([
            [np.cos(D), np.sin(D)], [-np.sin(D), np.cos(D)]
        ])
        XYm = np.matmul(R, XYgeo)
        row["Xm"], row["Ym"] = XYm[0], XYgeo[1]
        return row

    def _apply_Hgeo_(row, D):
        R = np.array([
            [np.cos(D), np.sin(D)], [-np.sin(D), np.cos(D)]
        ])
        XYm = np.array([row["Xm"], row["Ym"]])
        XYgeo = np.matmul(np.linalg.inv(R), XYm)
        row["X"], row["Y"] = XYgeo[0], XYgeo[1]
        return row
    
    data = read_dataset()["ESK"]
    lat, lon, alt_km, date = (
        55.2678, -3.1757, 0.0, dt.datetime(1958, 2, 11, 0)
    )
    D, I, H, X, Y, Z, F = pyIGRF.igrf_value(lat, lon, alt_km, date.year)
    D = np.deg2rad(D)
    logger.info(f"Declination (deg): {float(np.rad2deg(D))}")
    data = data.apply(lambda x: _apply_Hmag_(x, D), axis=1)
    data.index = data.index - dt.timedelta(minutes=2)
    data_recreate = data.copy()
    data_recreate = data_recreate.apply(lambda x: _apply_Hgeo_(x, D), axis=1)


    D = np.deg2rad(-28.76)
    # D = np.deg2rad(-40)
    logger.info(f"Declination (deg): {float(np.rad2deg(D))}")
    data_new = data.copy()
    data_new = data_new.apply(lambda x: _apply_Hgeo_(x, D), axis=1)


    xlim=[dt.datetime(1958, 2, 11,), dt.datetime(1958, 2, 11, 5)]
    sp = StackPlots(
        nrows=3, ncols=1, datetime=True, 
        figsize=(8, 3), text_size=12,    
    )
    ax = sp.axes[0]
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H"))
    ax.set_ylabel("$\widetilde{B}_{GEO}$, nT")
    # ax.set_xlabel("Time, UT (11 Feb 1958)")
    ax.plot(data.index, data.X - np.median(data.X.iloc[:60]), color="r", ls="-", label="$B_x$", lw=2)
    ax.plot(data.index, data.Y - np.median(data.Y.iloc[:60]), color="k", ls="-", label="$B_y$", lw=2.)

    ax.plot(data_recreate.index, data_recreate.X - np.median(data_recreate.X.iloc[:60]), color="m", ls="--", lw=1.)
    ax.plot(data_recreate.index, data_recreate.Y - np.median(data_recreate.Y.iloc[:60]), color="b", ls="--", lw=1.)
    ax.legend(loc=2, fontsize=10)
    ax.set_ylim(-1000, 1000)
    ax.set_xlim(xlim)

    ax = sp.axes[1]
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H"))
    ax.set_ylabel("$\widetilde{B}_{MAG}$, nT")
    # ax.set_xlabel("Time, UT (11 Feb 1958)")
    ax.plot(data.index, data.Xm - np.median(data.Xm.iloc[:60]), color="r", ls="-")
    ax.plot(data.index, data.Ym - np.median(data.Ym.iloc[:60]), color="k", ls="-")
    ax.set_xlim(xlim)
    ax.set_ylim(-1000, 1000)

    ax = sp.axes[2]
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H"))
    ax.set_ylabel("$\widetilde{B}_{GEO} [W]$, nT")
    ax.set_xlabel("Time, UT (11 Feb 1958)")
    ax.plot(data_new.index, data_new.X - np.median(data_new.X.iloc[:60]), color="r", ls="-", label="$B_x$")
    ax.plot(data_new.index, data_new.Y - np.median(data_new.Y.iloc[:60]), color="k", ls="-", label="$B_y$")
    ax.legend(loc=2, fontsize=10)
    ax.set_ylim(-1000, 1000)
    ax.set_xlim(xlim)
    sp.save_fig(f"figures/tat1/1958.Data.png")
    sp.close()
    return

if __name__ == "__main__":
    plot_e_fields()
    toGeoMag_Domain()