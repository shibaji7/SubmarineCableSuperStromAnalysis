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
from utils import StackPlots, get_cable_informations, read_iaga

station_maps = dict(
    FRD=[
        "data/1989/FRD.csv",
        # "data/1989/FRD_19890313_XYZ.txt",
        # "data/1989/FRD_19890314_XYZ.txt"
    ],
    STJ=[
        "data/1989/STJ.csv",
        # "data/1989/STJ_19890313_XYZ.txt",
        # "data/1989/STJ_19890314_XYZ.txt"
    ],
    HAD=[
        "data/1989/HAD.csv",
        # "data/1989/HAD_19890313_HDZ.txt",
        # "data/1989/HAD_19890314_HDZ.txt",
    ],
)


def read_dataset(base_path: str = "data/1989/") -> pd.DataFrame:
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

    stns, coords = ["FRD", "STJ", "HAD"], ["XYZ", "XYZ", "HDZ"]
    frames = {}
    for stn, coord in zip(stns, coords):
        files = glob.glob(base_path + f"{stn}*{coord}*.txt")
        files.sort()
        frames[stn] = pd.concat([read_iaga(f) for f in files])

    # Plot processed data
    sp = StackPlots(nrows=3, ncols=1, datetime=True, figsize=(6, 4), text_size=12)
    for stn, coord in zip(stns, coords):
        data = frames[stn]
        data.drop_duplicates().sort_index(inplace=True)
        print(len(data), 24 * 60 * 3)
        _, ax = sp.plot_stack_plots(
            data.index,
            data.X - np.median(data.X.iloc[:60]),
            ylim=[-1500, 1500],
            label=r"$B_x$",
            xlim=[dt.datetime(1989, 3, 12, 12), dt.datetime(1989, 3, 14, 12)],
            interval=6,
        )
        sp.plot_stack_plots(
            data.index,
            data.Y - np.median(data.Y.iloc[:60]),
            ylim=[-1500, 1500],
            label=r"$B_y$",
            color="r",
            xlim=[dt.datetime(1989, 3, 12, 12), dt.datetime(1989, 3, 14, 12)],
            ax=ax,
            interval=6,
        )
        sp.plot_stack_plots(
            data.index,
            data.Z - np.median(data.Z.iloc[:60]),
            ylim=[-1500, 1500],
            label=r"$B_z$",
            xlabel="Time, UT since 12 UT on 12 March 1989",
            color="k",
            ylabel=f"$B[{stn.lower()}]$, nT",
            xlim=[dt.datetime(1989, 3, 12, 12), dt.datetime(1989, 3, 14, 15)],
            ax=ax,
            interval=6,
        )
        ax.legend(loc=2, fontsize=12)
        data.to_csv(f"data/1989/{stn}.csv", header=True, index=True, float_format="%g")
        sp.save_fig("figures/1989.data.png")
        sp.close()
    return


def compile():
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
    read_dataset()
    stns = ["FRD", "FRD", "STJ", "STJ", "STJ", "STJ", "STJ", "HAD", "HAD"]
    segment_files = [station_maps[s] for s in stns]
    cable = get_cable_informations()
    model = SCUBASModel(
        cable_name="TAT-8",
        cable_structure=cable,
        segment_files=segment_files,
    )
    model.read_stations(stns, segment_files, False)
    model.initialize_TL()
    model.run_cable_segment()

    # Generate plots
    model.plot_TS_with_others(
        fname="figures/1989.Scubas.png",
        date_lim=[dt.datetime(1989, 3, 12, 12), dt.datetime(1989, 3, 14, 12)],
        fig_title="SCUBAS / Time: UT since 12 UT on 12 March 1989",
        text_size=10,
        ylim=[-800, 800],
        interval=6,
    )
    model.plot_profiles(
        nrows=3,
        ncols=3,
        figsize=(3.2, 3),
        fname="figures/1989.Profiles.png",
        xlim=[1e-6, 1e-2],
        tylim=[-90, 90],
        tyticks=[-90, -45, 0, 45, 90],
        aylim=[1e-3, 1e0],
        tag0_loc=[0, 3, 6],
        tag1_loc=[6, 7, 8],
        tag2_loc=[2, 5, 8],
    )
    return


if __name__ == "__main__":
    compile()
