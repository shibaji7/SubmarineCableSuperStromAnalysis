import datetime as dt

import numpy as np
import pandas as pd  # type: ignore
from bathymetry import BathymetryAnalysis
from cable import SCUBASModel
from loguru import logger  # type: ignore
from scubas.datasets import Site
from utils import StackPlots, get_cable_informations, create_from_lat_lon

import os
os.makedirs("figures/2024/MAGE/", exist_ok=True)

station_maps = dict(
    FRD=[
        "data/2024/MAGE/frd_May2024_MAGE.csv",
    ],
    STJ=[
        "data/2024/MAGE/stj_May2024_MAGE.csv",
    ],
    HAD=[
        "data/2024/MAGE/had_May2024_MAGE.csv"
    ],
)
scale_stj = 1


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

    stns = ["FRD", "STJ", "HAD"]
    frames = {}
    for stn in stns:
        files = station_maps[stn]
        files.sort()
        frames[stn] = pd.concat([pd.read_csv(f, parse_dates=["Date"]) for f in files])

    # Plot processed data
    sp = StackPlots(nrows=3, ncols=1, datetime=True, figsize=(6, 3), text_size=12)
    for stn in stns:
        data = frames[stn].set_index("Date")
        data.drop_duplicates().sort_index(inplace=True)
        _, ax = sp.plot_stack_plots(
            data.index,
            data.X - np.median(data.X.iloc[:60]),
            ylim=[-2000, 2000],
            label=r"$B_x$",
            xlim=[dt.datetime(2024, 5, 10, 12), dt.datetime(2024, 5, 12)],
            interval=6,
        )
        sp.plot_stack_plots(
            data.index,
            data.Y - np.median(data.Y.iloc[:60]),
            ylim=[-2000, 2000],
            label=r"$B_y$",
            color="r",
            xlim=[dt.datetime(2024, 5, 10, 12), dt.datetime(2024, 5, 12)],
            ax=ax,
            interval=6,
        )
        sp.plot_stack_plots(
            data.index,
            data.Z - np.median(data.Z.iloc[:60]),
            ylim=[-2000, 2000],
            label=r"$B_z$",
            xlabel="Time, UT since 12 UT on 10 May 2024",
            color="k",
            ylabel=f"$B[{stn.lower()}]$, nT",
            xlim=[dt.datetime(2024, 5, 10, 12), dt.datetime(2024, 5, 12)],
            ax=ax,
            interval=6,
        )
        ax.legend(loc=2, fontsize=12)
        if stn == "STJ":
            data.X, data.Y, data.Z = (
                data.X * scale_stj,
                data.Y * scale_stj,
                data.Z * scale_stj,
                # data.F * scale_stj,
            )
        sp.save_fig("figures/2024/MAGE/2024.MAGE.data.png")
        sp.close()
    return


def compile_2024_MAGE_TAT8(gplot=False):
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

    # # Generate plots
    if gplot:
        model.plot_TS_with_others(
            fname="figures/2024/MAGE/2024.Scubas.MAGE-TAT8.png",
            date_lim=[dt.datetime(2024, 5, 10, 12), dt.datetime(2024, 5, 12)],
            fig_title="SCUBAS / Time, UT since 12 UT on 10 May 2024",
            text_size=10,
            ylim=[-800, 800],
            interval=6,
        )
        model.plot_e_fields(
            fname="figures/2024/MAGE/2024.Scubas.MAGE.Exfield-TAT8.png",
            date_lim=[dt.datetime(2024, 5, 10, 12), dt.datetime(2024, 5, 12)],
            fig_title=r"$E_x$-field / Time: UT since 12 UT on 10 May 2024",
            text_size=15,
            ylim=[-300, 300],
            interval=6,
            component="X",
            groups=[[0, 1, 2], [3, 4, 5], [6, 7, 8]],
        )
        model.plot_e_fields(
            fname="figures/2024/MAGE/2024.Scubas.MAGE.Eyfield-TAT8.png",
            date_lim=[dt.datetime(2024, 5, 10, 12), dt.datetime(2024, 5, 12)],
            fig_title=r"$E_y$-field / Time: UT since 12 UT on 10 May 2024",
            text_size=15,
            ylim=[-300, 300],
            interval=6,
            component="Y",
            groups=[[0, 1, 2], [3, 4, 5], [6, 7, 8]],
        )
    return model, cable


def compile_2024_MAGE_TAT1(gplot=False):
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
    
    import sys
    sys.path.append("py/")
    from t1958 import get_bathymetry, get_conductivity_profile

    read_dataset()
    stns = ["FRD", "FRD", "STJ", "STJ", "STJ", "STJ", "HAD", "HAD"]
    segment_files = [station_maps[s] for s in stns]
    names = ["CS-W", "DO-1", "DO-2", "DO-3", "DO-4", "MAR", "DO-5", "CS-E"]
    bathymetry, segment_coordinates, segments = get_bathymetry(names)
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
    model.read_stations(stns, segment_files, False)
    model.initialize_TL()
    model.run_cable_segment()

    # # Generate plots
    if gplot:
        model.plot_TS_with_others(
            fname="figures/2024/MAGE/2024.Scubas.MAGE-TAT1.png",
            date_lim=[dt.datetime(2024, 5, 10, 12), dt.datetime(2024, 5, 12)],
            fig_title="SCUBAS / Time, UT since 12 UT on 10 May 2024",
            text_size=10,
            ylim=[-2000, 2000],
            interval=6,
        )
        model.plot_e_fields(
            fname="figures/2024/MAGE/2024.Scubas.MAGE.Exfield-TAT1.png",
            date_lim=[dt.datetime(2024, 5, 10, 12), dt.datetime(2024, 5, 12)],
            fig_title=r"$E_x$-field / Time: UT since 12 UT on 10 May 2024",
            text_size=15,
            ylim=[-3000, 3000],
            interval=6,
            component="X",
            groups=[[0, 1, 2], [3, 4, 5], [6, 7]],
        )
        model.plot_e_fields(
            fname="figures/2024/MAGE/2024.Scubas.MAGE.Eyfield-TAT1.png",
            date_lim=[dt.datetime(2024, 5, 10, 12), dt.datetime(2024, 5, 12)],
            fig_title=r"$E_y$-field / Time: UT since 12 UT on 10 May 2024",
            text_size=15,
            ylim=[-3000, 3000],
            interval=6,
            component="Y",
            groups=[[0, 1, 2], [3, 4, 5], [6, 7]],
        )
    return model, cable


if __name__ == "__main__":
    compile_2024_MAGE_TAT8(True)
    compile_2024_MAGE_TAT1(True)
