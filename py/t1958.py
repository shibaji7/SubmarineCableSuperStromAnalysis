"""
SCUBAS Model Analysis for February 1958 Superstorm
==================================================
This script analyzes the impact of the February 1958 superstorm on the TAT-1 submarine cable, which observed voltage fluctuations of up to 2700 V. The analysis uses the SCUBAS model to simulate the storm's effects on submarine cables and underwater infrastructure.

Key Features:
--------------
1. Reads and processes geomagnetic data for the storm period.
2. Analyzes bathymetry data to segment the cable path.
3. Computes conductivity profiles for each cable segment.
4. Simulates the electromagnetic response of the TAT-1 cable using the SCUBAS model.
5. Generates visualizations of the processed data, bathymetry, and simulation results.

Dependencies:
--------------
- Python libraries: numpy, pandas, loguru, matplotlib
- Custom modules: bathymetry, utils, cable, scubas.conductivity
- Data files: Geomagnetic data, bathymetry data

This project is part of the Submarine Cable Super Storm Analysis initiative, funded by the National Science Foundation (NSF): Geospace Environment Modeling.

Author: Shibaji Chakraborty
Date: [Update Date]
"""

import datetime as dt
import os

import numpy as np
import pandas as pd  # type: ignore
from bathymetry import BathymetryAnalysis
from cable import SCUBASModel
from loguru import logger  # type: ignore
from utils import StackPlots, create_from_lat_lon

os.makedirs("figures/1958/", exist_ok=True)


def read_dataset(
    base_path: str = "data/1958/{x}[Eskdalemuir]-rescale-HR.csv",
) -> pd.DataFrame:
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
    D, H, Z = (
        pd.read_csv(base_path.replace("{x}", "D"), parse_dates=["Time"]),
        pd.read_csv(base_path.replace("{x}", "H"), parse_dates=["Time"]),
        pd.read_csv(base_path.replace("{x}", "Z"), parse_dates=["Time"]),
    )
    # Filter data for the storm period
    D = D[
        (D.Time >= dt.datetime(1958, 2, 10, 11))
        & (D.Time < dt.datetime(1958, 2, 11, 8))
    ]
    H = H[
        (H.Time >= dt.datetime(1958, 2, 10, 11))
        & (H.Time < dt.datetime(1958, 2, 11, 8))
    ]
    Z = Z[
        (Z.Time >= dt.datetime(1958, 2, 10, 11))
        & (Z.Time < dt.datetime(1958, 2, 11, 8))
    ]

    # Combine and process data
    data = pd.DataFrame()
    data["Time"], data["H"], data["D"], data["Z"] = (
        D.Time,
        np.copy(H.H),
        np.copy(D.D),
        np.copy(Z.Z),
    )
    data["X"], data["Y"] = (
        data.H * np.cos(np.deg2rad(data.D)),
        data.H * np.sin(np.deg2rad(data.D)),
    )
    data = (
        data.set_index("Time")
        .resample("10s")
        .asfreq()
        .interpolate(method="cubic")
        .reset_index()
    )
    data["F"] = np.sqrt(data.X**2 + data.Y**2 + data.Z**2)
    data.Time = data.Time - dt.timedelta(minutes=6)
    data = data[data.Time < data.Time.iloc[-1] - dt.timedelta(minutes=2)]

    # Save processed data
    data = data.rename(columns=dict(Time="Date"))
    data[["Date", "X", "Y", "Z", "F"]].to_csv(
        "data/1958/compiled.csv", float_format="%g", index=False
    )
    logger.info("Data saved to data/1958/compiled.csv")

    # Plot processed data
    sp = StackPlots(nrows=1, ncols=1, datetime=True, figsize=(6, 4))
    _, ax = sp.plot_stack_plots(
        data.Date,
        data.X - np.median(data.X.iloc[:60]),
        ylim=[-1500, 1500],
        label=r"$B_x$",
        xlim=[dt.datetime(1958, 2, 11, 1), dt.datetime(1958, 2, 11, 5)],
    )
    sp.plot_stack_plots(
        data.Date,
        data.Y - np.median(data.Y.iloc[:60]),
        ylim=[-1500, 1500],
        label=r"$B_y$",
        color="r",
        xlim=[dt.datetime(1958, 2, 11, 1), dt.datetime(1958, 2, 11, 5)],
        interval=1,
        ax=ax,
    )
    sp.plot_stack_plots(
        data.Date,
        data.Z - np.median(data.Z.iloc[:60]),
        ylim=[-1500, 1500],
        label=r"$B_z$",
        xlabel="Time, UT since 16 UT on 10 Feb 1958",
        color="k",
        ylabel=r"$B_{esk}$, nT",
        xlim=[dt.datetime(1958, 2, 11, 1), dt.datetime(1958, 2, 11, 5)],
        interval=1,
        ax=ax,
    )
    ax.legend(loc=2, fontsize=12)
    sp.save_fig("figures/1958/1958.data.png")
    sp.close()
    return data


def get_bathymetry(names, file_path: str = "data/1958/lat_long_bathymetry.csv") -> None:
    """
    Analyzes bathymetry data to segment the cable path.

    Parameters:
    -----------
    file_path : str
        File path for bathymetry data.

    Returns:
    --------
    tuple
        Bathymetry analysis object, segment coordinates, and segment definitions.
    """
    segments = [
        (0, 32),
        (32, 50),
        (50, 60),
        (60, 170),
        (170, 330),
        (330, 410),
        (410, 435),
        (435, -1),
    ]
    colors = [
        "tab:blue",
        "tab:orange",
        "tab:green",
        "tab:red",
        "tab:purple",
        "tab:brown",
        "tab:pink",
        "tab:gray",
        "tab:olive",
        "tab:cyan",
        "gold",
        "limegreen",
        "darkviolet",
        "crimson",
        "teal",
        "peru",
        "orchid",
        "slategray",
        "salmon",
        "darkkhaki",
    ]

    # Initialize and use the BathymetryAnalysis class
    bathymetry = BathymetryAnalysis(file_path, segments, colors)
    bathymetry.load_data()
    bathymetry.plot_bathymetry("figures/1958/bathymetry_TAT-1.png", names=names)
    segment_coordinates = bathymetry.get_segment_coordinates()
    print("Segment Coordinates:", segment_coordinates)
    return bathymetry, segment_coordinates, segments


def get_conductivity_profile(dSegments, segments, bth):
    """
    Computes conductivity profiles for each cable segment.

    Parameters:
    -----------
    dSegments : list
        Segment coordinates.
    segments : list
        Segment definitions.
    bth : pd.DataFrame
        Bathymetry data.

    Returns:
    --------
    list
        Conductivity profiles for each segment.
    """
    from scubas.conductivity import ConductivityProfile  # type: ignore

    profiles = ConductivityProfile.compile_bined_profiles(np.array(dSegments))
    for p, seg in zip(profiles, segments):
        o = bth.iloc[seg[0] : seg[1]]
        depth = np.median(o["bathymetry.meters"])
        p.layers[0].thickness = depth / 1e3  # in meters
    return profiles


def load_extracted_voltage(fname="data/1958/Voltage/TAT1Volt-rescale.csv"):
    # TAT1Volt-rescale.csv
    data = pd.read_csv(fname, parse_dates=["Time"])
    return data


def compile_1958(datafile=["data/1958/compiled.csv"]):
    """
    Main function to run the SCUBAS model for the 1958 superstorm.

    Parameters:
    -----------
    datafile : list
        List of data files for each cable segment.

    Returns:
    --------
    None
    """
    names = ["CS-W", "DO-1", "DO-2", "DO-3", "DO-4", "MAR", "DO-5", "CS-E"]
    _ = read_dataset()
    bathymetry, segment_coordinates, segments = get_bathymetry(names)
    segment_files = [datafile] * len(segment_coordinates)
    profiles = get_conductivity_profile(
        segment_coordinates, segments, bathymetry.bathymetry_data
    )
    cable = create_from_lat_lon(
        segment_coordinates,
        profiles,
        names=names,
    )
    model = SCUBASModel(
        cable_name="TAT-1",
        cable_structure=cable,
        segment_files=segment_files,
    )
    model.read_stations(["ESK"], [datafile])
    model.initialize_TL()
    model.run_cable_segment()

    # Generate plots
    model.plot_TS_with_others(
        fname="figures/1958/1958.Scubas.png",
        date_lim=[dt.datetime(1958, 2, 10, 16), dt.datetime(1958, 2, 11, 8)],
        fig_title="SCUBAS (Esk) / Time: UT since 16 UT on 10 Feb 1958",
        text_size=10,
    )
    model.plot_profiles(
        fname="figures/1958/1958.Profiles.png",
        xlim=[1e-6, 1e-2],
        tylim=[-90, 90],
        tyticks=[-90, -45, 0, 45, 90],
        aylim=[1e-3, 1e0],
        t_mul=1.0,
    )
    model.plot_e_fields(
        fname="figures/1958/1958.Scubas.Exfield.png",
        date_lim=[dt.datetime(1958, 2, 10, 16), dt.datetime(1958, 2, 11, 8)],
        fig_title=r"$E_x$-field / Time: UT since 16 UT on 10 Feb 1958",
        text_size=15,
        ylim=[-1000, 1000],
        component="X",
        groups=[[0, 1, 2], [3, 4, 5], [6, 7]],
    )
    model.plot_e_fields(
        fname="figures/1958/1958.Scubas.Eyfield.png",
        date_lim=[dt.datetime(1958, 2, 10, 16), dt.datetime(1958, 2, 11, 8)],
        fig_title=r"$E_y$-field / Time: UT since 16 UT on 10 Feb 1958",
        text_size=15,
        ylim=[-1000, 1000],
        component="Y",
        groups=[[0, 1, 2], [3, 4, 5], [6, 7]],
    )
    obs = load_extracted_voltage()
    model.plot_zoomedin_analysis(
        fname="figures/1958/1958.Scubas.Compare.png",
        inputs=obs,
        date_lims=[dt.datetime(1958, 2, 11, 1), dt.datetime(1958, 2, 11, 4)],
        ylim=[-3000, 3000],
        interval=30,
        mult=1,
    )
    run_detailed_error_analysis(
        inputs=obs,
        cable=model.cable,
        date_lims=[dt.datetime(1958, 2, 11, 1), dt.datetime(1958, 2, 11, 4)],
        fnames=[
            "figures/1958/1958.Error.qq.png",
        ],
    )
    return model, cable


def run_detailed_error_analysis(
    inputs,
    cable,
    date_lims=[],
    fnames=[
        "figures/1958/1958.Error.qq.png",
    ],
):
    # Case special
    x = np.array(inputs.Voltage)
    o = cable.tot_params.copy()
    o = o[
        (o.index >= date_lims[0] - dt.timedelta(minutes=10))
        & (o.index <= date_lims[1] + dt.timedelta(minutes=10))
    ]["Vt(v)"]
    dT = np.array((o.index - o.index[0]).total_seconds())
    inputs["newdT"] = inputs.Time.apply(lambda j: (j - o.index[0]).total_seconds())
    y = np.interp(inputs.newdT, dT, -np.array(o))
    e = y - x  # Error Pred - Obs

    sp = StackPlots(nrows=2, ncols=2, figsize=(4, 2.5), sharex=False, text_size=12)
    ax = sp.axes[0]
    ax.hist(e, 50, color="b", histtype="step")
    ax.set_xlabel("Error, V", fontsize=12)
    ax.set_ylabel("Counts", fontsize=12)
    ax.tick_params(axis="x", labelsize=12)
    ax.set_xlim(-3000, 3000)
    ax.tick_params(axis="y", labelsize=12)
    ax.text(
        0.05,
        0.9,
        "(A)",
        ha="left",
        va="center",
        transform=ax.transAxes,
        fontsize=12,
    )

    ax = sp.axes[1]
    ax.set_xlim([-3000, 3000])
    ax.set_ylim([-3000, 3000])
    from verify.plot import qqPlot

    qqPlot(
        y,
        x,
        modelName="SCUBAS",
        addTo=sp.axes[1],
        plot_kwargs=dict(
            c="b",
            marker="s",
            s=4,
        ),
    )
    ax.set_title("")
    ax.text(
        0.05,
        0.9,
        "(B) QQ Plot",
        ha="left",
        va="center",
        transform=ax.transAxes,
        fontsize=12,
    )
    ax.set_xlabel("Predicted, V", fontsize=12)
    ax.set_ylabel("Observed, V", fontsize=12)
    ax.tick_params(axis="x", labelsize=12)
    ax.tick_params(axis="y", labelsize=12)

    ax = sp.axes[2]
    ax.scatter(
        x,
        e,
        c="b",
        marker="s",
        s=4,
    )
    ax.set_xlabel("Observed, V", fontsize=12)
    ax.set_xlim(-3000, 3000)
    ax.set_ylim(-3000, 3000)
    ax.set_ylabel("Error, V", fontsize=12)
    ax.text(
        0.05,
        0.9,
        "(C) Residue",
        ha="left",
        va="center",
        transform=ax.transAxes,
        fontsize=12,
    )
    ax.axhline(0, color="k", lw=0.8, ls="--")

    # Compute Scores (huber, quantile, expctile) and Isotonic fits
    from scores.processing.isoreg_impl import isotonic_fit

    iso_fit_result = isotonic_fit(
        fcst=y, obs=x, functional="mean", bootstraps=100, confidence_level=0.95
    )
    # Data
    x_sorted = iso_fit_result["fcst_sorted"]
    y_lower = iso_fit_result["confidence_band_lower_values"]
    y_upper = iso_fit_result["confidence_band_upper_values"]
    y_reg = iso_fit_result["regression_values"]
    weights = iso_fit_result["fcst_counts"]

    # Bounds
    total_min = min(np.min(x_sorted), np.min(y_lower))
    total_max = max(np.max(x_sorted), np.max(y_upper))

    # Histogram data
    bins = np.linspace(np.min(x_sorted), np.max(x_sorted), 11)

    ax = sp.axes[3]
    # Confidence band (shaded region)
    ax.fill_between(
        x_sorted,
        y_lower,
        y_upper,
        color="lightblue",
        alpha=0.5,
        label="95% confidence band",
    )

    # Diagonal reference line
    ax.plot([total_min, total_max], [total_min, total_max], "k--")

    # Regression line
    ax.plot(x_sorted, y_reg, color="b")

    # Histogram (on secondary y-axis)
    ax_hist = ax.twinx()
    ax_hist.set_xlim(-3000, 3000)
    ax_hist.set_ylim(0, 100)
    ax_hist.hist(x_sorted, color="purple", histtype="step")
    ax_hist.set_ylabel("Counts", color="purple")

    # Annotations
    ax.set_xlim(-3000, 3000)
    ax.set_ylim(-3000, 3000)
    ax.set_xlabel("Predicted, V")
    ax.set_ylabel("Observed, V", color="b")
    ax.text(
        0.05,
        0.95,
        "underprediction",
        ha="left",
        va="top",
        transform=ax.transAxes,
        fontsize=8,
        rotation=90,
        color="r",
    )
    ax.text(
        0.95,
        0.05,
        "overprediction",
        ha="right",
        va="bottom",
        transform=ax.transAxes,
        fontsize=8,
        rotation=90,
        color="r",
    )
    ax.text(0.3, 0.9, "(D)", ha="left", va="center", transform=ax.transAxes)
    sp.save_fig(fnames[0])
    sp.close()
    return


if __name__ == "__main__":
    compile_1958()
