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
from scubas.datasets import Site, PROFILES
from utils import StackPlots, create_from_lat_lon, get_cable_informations, read_iaga

station_maps = dict(
    CNB=[
        "data/2024/AJC/CNB/cnb20240510qmin.min",
        "data/2024/AJC/CNB/cnb20240511qmin.min",
    ],
    CTA=[
        "data/2024/AJC/CTA/cta20240510qmin.min",
        "data/2024/AJC/CTA/cta20240511qmin.min",
    ],
    GUA=[
        "data/2024/AJC/GUA/gua20240510qmin.min",
        "data/2024/AJC/GUA/gua20240511qmin.min",
    ],
    KAK=[
        "data/2024/AJC/KAK/kak20240510qmin.min",
        "data/2024/AJC/KAK/kak20240511qmin.min",
    ],
)
dSegmented_files_map = dict(
    KAK=[f"figures/2024/AJC/KAK.csv"],
    GUA=[f"figures/2024/AJC/GUA.csv"],
    CTA=[f"figures/2024/AJC/CTA.csv"],
    CNB=[f"figures/2024/AJC/CNB.csv"],
)
import os

os.makedirs("figures/2024/AJC/", exist_ok=True)


def read_dataset() -> pd.DataFrame:
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

    stns = ["CNB", "CTA", "GUA", "KAK"]
    frames = {}
    for stn in stns:
        files = station_maps[stn]
        files.sort()
        frames[stn] = pd.concat([read_iaga(f) for f in files])

    # Plot processed data
    sp = StackPlots(nrows=4, ncols=1, datetime=True, figsize=(8, 3), text_size=12)
    for stn in stns:
        data = frames[stn]
        data.drop_duplicates().sort_index(inplace=True)
        _, ax = sp.plot_stack_plots(
            data.index,
            data.X - np.median(data.X.iloc[:60]),
            ylim=[-500, 500],
            label=r"$B_x$",
            xlim=[dt.datetime(2024, 5, 10, 12), dt.datetime(2024, 5, 12)],
            interval=6,
        )
        sp.plot_stack_plots(
            data.index,
            data.Y - np.median(data.Y.iloc[:60]),
            ylim=[-500, 500],
            label=r"$B_y$",
            color="r",
            xlim=[dt.datetime(2024, 5, 10, 12), dt.datetime(2024, 5, 12)],
            ax=ax,
            interval=6,
        )
        sp.plot_stack_plots(
            data.index,
            data.Z - np.median(data.Z.iloc[:60]),
            ylim=[-500, 500],
            label=r"$B_z$",
            xlabel="Time, UT",
            color="k",
            ylabel=f"$B[{stn.lower()}]$, nT",
            xlim=[dt.datetime(2024, 5, 10, 12), dt.datetime(2024, 5, 12)],
            ax=ax,
            interval=6,
        )
        data.reset_index(inplace=True)
        data.to_csv(f"figures/2024/AJC/{stn}.csv", index=False, header=True)
        ax.legend(loc=2, fontsize=12)
        sp.save_fig("figures/2024/AJC/2024.data.png")
        sp.close()
    return


def get_bathymetry(
    names, file_path: str = "data/2024/AJC/lat_long_bathymetry.csv"
) -> None:
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
        (0, 10),
        (10, 160),
        (160, 220),
        (220, 300),
        (300, 350),
        (350, -1),
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
    # bathymetry = BathymetryAnalysis(file_path, segments, colors)
    # bathymetry.load_data()
    # bathymetry.plot_bathymetry("figures/2024/AJC/bathymetry_AJC.png", names=names)
    # segment_coordinates = bathymetry.get_segment_coordinates()
    # print("Segment Coordinates:", segment_coordinates)
    bathymetry = BathymetryAnalysis(file_path, segments, colors)
    bathymetry.load_data()
    bathymetry.plot_bathymetry(
        "figures/2024/AJC/bathymetry_AJC.png",
        names=names,
        xticks=[0, 500, 1000, 2000, 3000, 8000],
        xlim=[0, bathymetry.bathymetry_data.distance.iloc[-1] / 1e3],
        ylim=[-8, 0.5],
        yticks=[-8, -6, -4, -2, -1, -0.5, 0],
        yticklabels=[8, 6, 4, 2, 1, 0.5, 0],
    )
    segment_coordinates = np.array(bathymetry.get_segment_coordinates())
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
    for j, p, seg in zip(range(len(profiles)), profiles, segments):
        o = bth.iloc[seg[0] : seg[1]]
        depth = np.median(o["bathymetry.meters"])
        print("in meters>>>>>>>>", depth)
        p.layers[0].thickness = depth  # in meters
        # All layers in meters
    return profiles


def load_extracted_voltage(
    time, data,
    fname="data/2024/AJC/AJC PFE DATA/Tumon Bay (GUAM)",
    key="S_4_V (V)", figfname="", 
    fig_title=f"Date: 10-12 May 2024; Stn: tmb/guam",
):
    from fetch_operators_dataset import read_files_as_pandas
    from ts_plots import TimeSeriesPlot
    import matplotlib.dates as mdates

    df = read_files_as_pandas(fname)
    df[key] = df[key] - np.nanmedian(df[key].tolist()[:60])
    ts = TimeSeriesPlot(
        dates=[dt.datetime(2024, 5, 10, 12), dt.datetime(2024, 5, 12)],
        fig_title=fig_title,
        num_subplots=1,
        text_size=15,
        major_locator=mdates.HourLocator(byhour=range(0, 24, 4)),
    )
    df.Timestamp = df.Timestamp + dt.timedelta(minutes=15)
    ax = ts._add_axis()
    ax.plot(df.Timestamp, df[key], "b.", ls="None", ms=0.5, alpha=0.7)
    ax.plot(time, data, ls="-", lw=0.8, color="red")
    ax.set_xlabel("Time, UT")
    ax.set_ylabel("Voltage, V", fontdict=dict(color="b"))
    ts.save(figfname)
    return df


def compile_2024_AJC():
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
    names = [
        "DO-1",
        "DO-2",
        "DO-3",
        "DO-4",
        "DO-5",
        "RDG-1",
    ]
    _ = read_dataset()
    bathymetry, segment_coordinates, segments = get_bathymetry(names)
    stns = [
        "KAK",
        "KAK",
        "KAK",
        "GUA",
        "GUA",
        "GUA",
    ]
    segment_files = [dSegmented_files_map[s] for s in stns]
    profiles = get_conductivity_profile(
        segment_coordinates, segments, bathymetry.bathymetry_data
    )
    logger.info(f"Profile len: {len(profiles)}")
    cable = create_from_lat_lon(
        segment_coordinates,
        profiles,
        names=names,
        # left_active_termination=PROFILES.LD,
        right_active_termination=PROFILES.LD,
    )
    model = SCUBASModel(
        cable_name="AJC",
        cable_structure=cable,
        segment_files=segment_files,
    )
    model.read_stations(
        ["KAK", "GUA"],
        [
            dSegmented_files_map["KAK"],
            dSegmented_files_map["GUA"],
        ],
        clean=False,
    )
    model.initialize_TL()
    model.run_cable_segment()

    # # # Generate plots
    model.plot_TS_with_others(
        fname="figures/2024/AJC/2024.Scubas.png",
        date_lim=[dt.datetime(2024, 5, 10, 12), dt.datetime(2024, 5, 12)],
        fig_title="SCUBAS / Time, UT since 12 UT on 10 May 2024",
        text_size=10,
        ylim=[-15, 15],
    )
    # model.plot_profiles(
    #     fname="figures/2024/AJC/2024.Profiles.png",
    #     xlim=[1e-6, 1e-2],
    #     tylim=[-90, 90],
    #     tyticks=[-90, -45, 0, 45, 90],
    #     aylim=[1e-3, 1e0],
    #     t_mul=1e-3,
    #     nrows=3,
    #     ncols=3,
    # )
    # model.plot_e_fields(
    #     fname="figures/2024/AJC/2024.Scubas.Exfield.png",
    #     date_lim=[dt.datetime(2024, 5, 10, 12), dt.datetime(2024, 5, 12)],
    #     fig_title=r"$E_x$-field / Time: UT since 12 UT on 10 May 2024",
    #     text_size=15,
    #     ylim=[-100, 100],
    #     nrows=3,
    #     component="X",
    #     groups=[[0, 1, 2], [3, 4, 5], [6, 7, 8]],
    # )
    # model.plot_e_fields(
    #     fname="figures/2024/AJC/2024.Scubas.Eyfield.png",
    #     date_lim=[dt.datetime(2024, 5, 10, 12), dt.datetime(2024, 5, 12)],
    #     fig_title=r"$E_y$-field / Time: UT since 12 UT on 10 May 2024",
    #     text_size=15,
    #     nrows=3,
    #     ylim=[-100, 100],
    #     component="Y",
    #     groups=[[0, 1, 2], [3, 4, 5], [6, 7, 8]],
    # )
    obs = load_extracted_voltage(
        model.cable.tot_params.index,
        -model.cable.tot_params["Vt(v)"],
        figfname="figures/2024/AJC/Compare-Stn.png"
    )
    # model.plot_zoomedin_analysis(
    #     fname="figures/1958/1958.Scubas.Compare.png",
    #     inputs=obs,
    #     date_lims=[dt.datetime(1958, 2, 11, 1), dt.datetime(1958, 2, 11, 4)],
    #     ylim=[-3000, 3000],
    #     interval=30,
    #     mult=1,
    # )
    # return model, cable


if __name__ == "__main__":
    compile_2024_AJC()
