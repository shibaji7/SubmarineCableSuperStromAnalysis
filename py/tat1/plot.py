import datetime as dt
import os

import sys
sys.path.extend(["py/", "py/tat1/"])

import numpy as np
import pandas as pd  # type: ignore
from bathymetry import BathymetryAnalysis
from cable import SCUBASModel
from loguru import logger  # type: ignore
from utils import StackPlots, NatureStackPlots, create_from_lat_lon, read_iaga

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
    xlim=[dt.datetime(1958, 2, 11,), dt.datetime(1958, 2, 11, 4)]
    for stn, coord in zip(stns, coords):
        sp = NatureStackPlots(nrows=1, ncols=1, datetime=True, figsize=(3.5, 2.5), text_size=7, column='single')
        data = frames[stn]
        data.drop_duplicates().sort_index(inplace=True)
        _, ax = sp.plot_stack_plots(
            data.index,
            data.X - np.median(data.X.iloc[:60]),
            ylim=[-1000, 700],
            label=r"$B_x$",
            interval=6,
        )
        sp.plot_stack_plots(
            data.index,
            data.Y - np.median(data.Y.iloc[:60]),
            ylim=[-1000, 700],
            label=r"$B_y$",
            color="#D55E00",
            ax=ax,
            interval=6,
        )
        sp.plot_stack_plots(
            data.index,
            data.Z.shift(periods=10) - np.median(data.Z.iloc[:60]),
            ylim=[-1000, 700],
            label=r"$B_z$",
            xlabel="Time, UT (11 Feb 1958)",
            color="#009E73",
            ylabel=f"$B[{stn.lower()}]$, nT",
            xlim=xlim,
            ax=ax,
            interval=1,
        )
        ax.legend(loc=2, fontsize=6)
        data = data[(data.index >= xlim[0]) & (data.index < xlim[1])]
        data.index = data.index - dt.timedelta(minutes=2)
        k = 1.
        data.Z, data.X, data.Y = data.Z*k, data.X*k, data.Y*k
        sp.save_fig(f"figures/1958/1958.data_{stn}.png")
        sp.save_fig(f"figures/1958/1958.data_{stn}.pdf")
        sp.close()
    return

def plot_e_fields():
    model_out = pd.read_csv("data/1958/TAT1SimVolt.csv", parse_dates=["Time"])
    print("?????????", model_out["Vt(v)"].min(),model_out["Vt(v)"].max())
    xlim=[dt.datetime(1958, 2, 11,), dt.datetime(1958, 2, 11, 5)]
    import matplotlib.pyplot as plt
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial']
    plt.rcParams['font.size'] = 7
    gs = plt.GridSpec(2, 1, height_ratios=[12, 3], hspace=0.25)
    fig = plt.figure(figsize=(3.5, 3), dpi=1000)
    ax = fig.add_subplot(gs[0])
    tax = ax.twinx()

    ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H"))
    ix = 0
    for j in np.arange(8,-1,-1):
        ax.plot(model_out.Time, 2000*ix + model_out[f"E.X.0{j}"], color="#0072B2", ls="-", lw=1.0)
        ax.plot(model_out.Time, 2000*ix + model_out[f"E.Y.0{j}"] - 500, color="#CC79A7", ls="-", lw=1.0)
        ix+=1
    ax.axvline(dt.datetime(1958, 2, 11, 0, 30), ymin=0.43, ymax=0.52, color="#009E73", ls="-", lw=1.5)
    ax.text(dt.datetime(1958, 2, 11, 0, 32), 2000*3 + 500, "2 V/km", color="#009E73", fontsize=6)
    ax.text(0.02, 0.98, "(a)", transform=ax.transAxes, fontdict=dict(size=10, weight='bold'), ha='left', va='top')
    ax.set_yticklabels([])
    tax.set_yticklabels([])
    y_positions = [18000, 16000, 14000, 12000, 10000, 8000, 6000, 4000, 2000]
    for j, (name, ypos) in enumerate(zip(["CS-E", "DO-5","MAR", "DO-4", "RDG-1", "DO-3","DO-2", "DO-1","CS-W"], y_positions)):
        ax.text(
            xlim[1] + dt.timedelta(minutes=5),
            ypos,
            name,
            color="#0072B2",
            fontsize=4,
            rotation=90,
            va="center",
            ha="center",
        )
    ax.set_xlabel("")
    ax.set_ylabel("$E_x$, mv/km", color="#0072B2")
    tax.set_ylabel("$E_y$, mv/km", color="#CC79A7")
    ax.set_xlim(xlim)

    ax = fig.add_subplot(gs[1])
    ax.text(0.02, 0.98, "(b)", transform=ax.transAxes, fontdict=dict(size=10, weight='bold'), ha='left', va='top')
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
    ax.plot(model_out.Time, model_out["U0"], color="#0072B2", ls="-", lw=1.0, label="$U_E$")
    ax.plot(model_out.Time, model_out["U1"], color="#D55E00", ls="-", lw=1.0, label="$U_W$")
    ax.legend(loc=2, fontsize=6)
    ax.set_ylabel("Voltage, V")
    ax.set_ylim(-300, 300)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H"))
    ax.set_xlabel("Time, UT (11 Feb 1958)")
    fig.savefig(f"figures/tat1/1958.Efield.png", bbox_inches='tight')
    fig.savefig(f"figures/tat1/1958.Efield.pdf", bbox_inches='tight')
    plt.close()

    xlim=[dt.datetime(1958, 2, 11,), dt.datetime(1958, 2, 11, 5)]
    fig, ax = plt.subplots(1, 1, figsize=(3.5, 2.5), dpi=1000)
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
            color="#0072B2",
            ls="-",
            lw=1.0,
        )
        ax.text(
            xlim[1] + dt.timedelta(minutes=5),
            400*j + model_out[f"V(v).0{ix}"].iloc[-1],
            name,
            color="#0072B2",
            fontsize=5,
            rotation=90,
            va="center",
            ha="center",
        )
        ix -= 1
    ax.axvline(dt.datetime(1958, 2, 11, 0, 30), ymin=0.495, ymax=0.575, color="#009E73", ls="-", lw=1.5)
    ax.text(dt.datetime(1958, 2, 11, 0, 32), 400*3 + 500, "0.2 kV", color="#009E73", fontsize=6)
    ax.text(0.02, 0.98, "(a)", transform=ax.transAxes, fontdict=dict(size=10, weight='bold'), ha='left', va='top')
    fig.savefig(f"figures/tat1/1958.Vs.png", bbox_inches='tight')
    fig.savefig(f"figures/tat1/1958.Vs.pdf", bbox_inches='tight')
    plt.close()
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
        row["Xm"], row["Ym"] = XYm[0,0], XYgeo[1,0]
        return row

    def _apply_Hgeo_(row, D):
        R = np.array([
            [np.cos(D), np.sin(D)], [-np.sin(D), np.cos(D)]
        ])
        XYm = np.array([[row["Xm"]], [row["Ym"]]])
        XYgeo = np.matmul(np.linalg.inv(R), XYm)
        row["X"], row["Y"] = XYgeo[0,0], XYgeo[1,0]
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
    sp = NatureStackPlots(
        nrows=3, ncols=1, datetime=True, 
        figsize=(3.5, 4), text_size=7, column='single',
    )
    ax = sp.axes[0]
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H"))
    ax.set_ylabel("$\widetilde{B}_{GEO}$, nT")
    ax.plot(data.index, data.X - np.median(data.X.iloc[:60]), color="#D55E00", ls="-", label="$B_x$", lw=1.0)
    ax.plot(data.index, data.Y - np.median(data.Y.iloc[:60]), color="#0072B2", ls="-", label="$B_y$", lw=1.0)

    ax.plot(data_recreate.index, data_recreate.X - np.median(data_recreate.X.iloc[:60]), color="#CC79A7", ls="--", lw=0.8)
    ax.plot(data_recreate.index, data_recreate.Y - np.median(data_recreate.Y.iloc[:60]), color="#56B4E9", ls="--", lw=0.8)
    ax.legend(loc=2, fontsize=6)
    sp.add_panel_label(ax)
    ax.set_ylim(-1000, 1000)
    ax.set_xlim(xlim)

    ax = sp.axes[1]
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H"))
    ax.set_ylabel("$\widetilde{B}_{MAG}$, nT")
    ax.plot(data.index, data.Xm - np.median(data.Xm.iloc[:60]), color="#D55E00", ls="-")
    ax.plot(data.index, data.Ym - np.median(data.Ym.iloc[:60]), color="#0072B2", ls="-")
    sp.add_panel_label(ax)
    ax.set_xlim(xlim)
    ax.set_ylim(-1000, 1000)

    ax = sp.axes[2]
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H"))
    ax.set_ylabel("$\widetilde{B}_{GEO} [W]$, nT")
    ax.set_xlabel("Time, UT (11 Feb 1958)")
    ax.plot(data_new.index, data_new.X - np.median(data_new.X.iloc[:60]), color="#D55E00", ls="-", label="$B_x$")
    ax.plot(data_new.index, data_new.Y - np.median(data_new.Y.iloc[:60]), color="#0072B2", ls="-", label="$B_y$")
    ax.legend(loc=2, fontsize=6)
    sp.add_panel_label(ax)
    ax.set_ylim(-1000, 1000)
    ax.set_xlim(xlim)
    sp.save_fig(f"figures/tat1/1958.Data.png")
    sp.save_fig(f"figures/tat1/1958.Data.pdf")
    sp.close()
    return

def plot_e_fields_edge():
    model_out = pd.read_csv("data/1958/TAT1SimVolt.csv", parse_dates=["Time"])
    xlim=[dt.datetime(1958, 2, 11,), dt.datetime(1958, 2, 11, 4)]
    sp = NatureStackPlots(
        nrows=1, ncols=1, datetime=True, 
        figsize=(3.5, 2.5), text_size=7, column='single',
    )
    ax = sp.axes[0]

    ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H"))
    ax.plot(model_out.Time, 6000 + model_out[f"E.X.00.left"], color="#0072B2", ls="-", lw=1.0, label="$E_x$")
    ax.plot(model_out.Time, 4000 + model_out[f"E.Y.00.left"], color="#D55E00", ls="-", lw=1.0, label="$E_y$")
    ax.legend(loc=3, fontsize=6)
    ax.text(0.02, 0.98, "(a)", transform=ax.transAxes, fontdict=dict(size=10, weight='bold'), ha='left', va='top')
    ax.text(
        xlim[1] + dt.timedelta(minutes=5),
        5000+model_out[f"E.X.00.left"].iloc[-1],
        "CS-W",
        color="#0072B2",
        fontsize=6,
        rotation=90,
        va="center",
        ha="center",
    )

    ax.plot(model_out.Time, -4000 + model_out[f"E.X.08.right"], color="#0072B2", ls="-", lw=1.0)
    ax.plot(model_out.Time, -6000 + model_out[f"E.Y.08.right"], color="#D55E00", ls="-", lw=1.0)
    ax.text(
        xlim[1] + dt.timedelta(minutes=5),
        -5000+model_out[f"E.X.08.right"].iloc[-1],
        "CS-E",
        color="#0072B2",
        fontsize=6,
        rotation=90,
        va="center",
        ha="center",
    )
    
    ax.axvline(dt.datetime(1958, 2, 11, 0, 30), ymin=12500/17000, ymax=14500/17000, color="#009E73", ls="-", lw=1.5)
    ax.text(dt.datetime(1958, 2, 11, 0, 32), 4500, "2 V/km", color="#009E73", fontsize=6)

    ax.set_ylim(-8500, 8500)
    ax.set_xlim(xlim)
    ax.set_ylabel("E fields, V/km")
    ax.set_xlabel("Time, UT (11 Feb 1958)")
    ax.set_yticklabels([])
    sp.save_fig(f"figures/tat1/1958.EdgeEFields.png")
    sp.save_fig(f"figures/tat1/1958.EdgeEFields.pdf")
    sp.close()
    return

if __name__ == "__main__":
    plot_e_fields()
    plot_e_fields_edge()
    # toGeoMag_Domain()
    read_dataset()