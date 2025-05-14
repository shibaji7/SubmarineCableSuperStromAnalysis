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
from scubas.datasets import Site
from utils import StackPlots, get_cable_informations, read_iaga

station_maps = dict(
    FRD=[
        "data/1989/FRD.csv",
    ],
    STJ=[
        "data/1989/STJ.csv",
    ],
    HAD=[
        "data/1989/HAD.csv",
    ],
)
scale_stj = 1

import os
os.makedirs("figures/1989/", exist_ok=True)


def get_conductivity_profile(dSegments, names):
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

    np.random.seed(0)
    cp = ConductivityProfile()
    pfls, profiles = [], []
    for seg in dSegments:
        p = cp._compile_profile_(seg)
        p.fillna(0, inplace=True)
        pfls.append(p)
    for i in range(len(pfls) - 1):
        p = pfls[i].copy()
        p.thickness = (
            np.random.uniform(
                np.array(pfls[i].thickness),
                np.array(pfls[i + 1].thickness),
            )
            * 1e3
        )
        p = Site.init(
            1.0 / p["resistivity"].to_numpy(dtype=float),
            p["thickness"].to_numpy(dtype=float),
            p["name"],
            "",
            names[i],
        )
        profiles.append(p)
    return profiles


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
        if stn == "STJ":
            data.X, data.Y, data.Z, data.F = (
                data.X * scale_stj,
                data.Y * scale_stj,
                data.Z * scale_stj,
                data.F * scale_stj,
            )
        data.to_csv(f"data/1989/{stn}.csv", header=True, index=True, float_format="%g")
        sp.save_fig("figures/1989/1989.data.png")
        sp.close()
    return


def load_extracted_voltage(fname="data/1989/Voltage/SSC-rescale.csv"):
    # TAT8Volt-rescale.csv
    data = pd.read_csv(fname, parse_dates=["Time"])
    return data


def compile_1989(gplot=False):
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
    cable = get_cable_informations(wtr_depths=dict(MAR=1500, CS_W=50, DO_1=2000))
    model = SCUBASModel(
        cable_name="TAT-8",
        cable_structure=cable,
        segment_files=segment_files,
    )
    model.read_stations(stns, segment_files, False)
    model.initialize_TL()
    model.run_cable_segment()

    # # Generate plots
    if gplot:
        model.plot_TS_with_others(
            fname="figures/1989/1989.Scubas.png",
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
            fname="figures/1989/1989.Profiles.png",
            xlim=[1e-6, 1e-2],
            tylim=[-90, 90],
            tyticks=[-90, -45, 0, 45, 90],
            aylim=[1e-3, 1e0],
            tag0_loc=[0, 3, 6],
            tag1_loc=[6, 7, 8],
            tag2_loc=[2, 5, 8],
            t_mul=1e-3,
        )
        model.plot_e_fields(
            fname="figures/1989/1989.Scubas.Exfield.png",
            date_lim=[dt.datetime(1989, 3, 12, 12), dt.datetime(1989, 3, 14, 12)],
            fig_title=r"$E_x$-field / Time: UT since 12 UT on 12 March 1989",
            text_size=15,
            ylim=[-300, 300],
            interval=6,
            component="X",
            groups=[[0, 1, 2], [3, 4, 5], [6, 7, 8]],
        )
        model.plot_e_fields(
            fname="figures/1989/1989.Scubas.Eyfield.png",
            date_lim=[dt.datetime(1989, 3, 12, 12), dt.datetime(1989, 3, 14, 12)],
            fig_title=r"$E_y$-field / Time: UT since 12 UT on 12 March 1989",
            text_size=15,
            ylim=[-300, 300],
            interval=6,
            component="Y",
            groups=[[0, 1, 2], [3, 4, 5], [6, 7, 8]],
        )
    ssc = load_extracted_voltage()
    model.plot_zoomedin_analysis(
        fname="figures/1989/1989.Scubas.Compare.SSC.png",
        inputs=ssc,
        date_lims=[dt.datetime(1989, 3, 13, 1), dt.datetime(1989, 3, 13, 2)],
        ylim=[-20, 100],
    )
    obs = load_extracted_voltage("data/1989/Voltage/TAT8Volt-rescale.csv")
    model.plot_zoomedin_analysis(
        fname="figures/1989/1989.Scubas.Compare.png",
        inputs=obs,
        date_lims=[dt.datetime(1989, 3, 13, 12), dt.datetime(1989, 3, 14, 12)],
        ylim=[-700, 700],
        interval=120 * 2,
    )
    model.run_detailed_error_analysis(
        inputs=obs,
        date_lims=[dt.datetime(1989, 3, 13, 12), dt.datetime(1989, 3, 14, 12)],
    )
    return model, cable


if __name__ == "__main__":
    compile_1989(True)
