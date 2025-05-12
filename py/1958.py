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

import numpy as np
import pandas as pd  # type: ignore
from bathymetry import BathymetryAnalysis
from cable import SCUBASModel
from loguru import logger  # type: ignore
from utils import StackPlots, create_from_lat_lon


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
        .resample("1s")
        .asfreq()
        .interpolate(method="cubic")
        .reset_index()
    )
    data["F"] = np.sqrt(data.X**2 + data.Y**2 + data.Z**2)
    data = data[data.Time < dt.datetime(1958, 2, 11, 7, 59)]

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
        xlim=[dt.datetime(1958, 2, 10, 11), dt.datetime(1958, 2, 11, 8)],
    )
    sp.plot_stack_plots(
        data.Date,
        data.Y - np.median(data.Y.iloc[:60]),
        ylim=[-1500, 1500],
        label=r"$B_y$",
        color="r",
        xlim=[dt.datetime(1958, 2, 10, 11), dt.datetime(1958, 2, 11, 8)],
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
        xlim=[dt.datetime(1958, 2, 10, 16), dt.datetime(1958, 2, 11, 8)],
        ax=ax,
    )
    ax.legend(loc=2, fontsize=12)
    sp.save_fig("figures/1958.data.png")
    sp.close()
    return data


def get_bathymetry(file_path: str = "data/1958/lat_long_bathymetry.csv") -> None:
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
        (410, -1),
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
    bathymetry.plot_bathymetry("figures/bathymetry.png")
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
        p.layers[0].thickness = depth / 1e3
    return profiles


def compile(datafile=["data/1958/compiled.csv"]):
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
    _ = read_dataset()
    bathymetry, segment_coordinates, segments = get_bathymetry()
    segment_files = [datafile] * len(segment_coordinates)
    profiles = get_conductivity_profile(
        segment_coordinates, segments, bathymetry.bathymetry_data
    )
    cable = create_from_lat_lon(
        segment_coordinates,
        profiles,
        names=["CS-W", "DO-1", "DO-2", "MAR", "DO-3", "DO-4", "CS-E"],
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
        fname="figures/1958.Scubas.png",
        date_lim=[dt.datetime(1958, 2, 10, 16), dt.datetime(1958, 2, 11, 8)],
        fig_title="SCUBAS (Esk) / Time: UT since 16 UT on 10 Feb 1958",
        text_size=10,
    )
    model.plot_profiles(
        fname="figures/1958.Profiles.png",
    )
    return


if __name__ == "__main__":
    compile()
