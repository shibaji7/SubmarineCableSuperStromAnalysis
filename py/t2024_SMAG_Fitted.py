import datetime as dt

import numpy as np
import pandas as pd  # type: ignore
from bathymetry import BathymetryAnalysis
from cable import SCUBASModel
from loguru import logger  # type: ignore
from scubas.datasets import Site
from utils import StackPlots, get_cable_informations, read_iaga

station_maps = dict(
    SEG0=[
        "data/2024/SMAG/cable_segment_0.csv",
    ],
    SEG1=[
        "data/2024/SMAG/cable_segment_1.csv",
    ],
    SEG2=[
        "data/2024/SMAG/cable_segment_2.csv",
    ],
    SEG3=[
        "data/2024/SMAG/cable_segment_3.csv",
    ],
    SEG4=[
        "data/2024/SMAG/cable_segment_4.csv",
    ],
    SEG5=[
        "data/2024/SMAG/cable_segment_5.csv",
    ],
    SEG6=[
        "data/2024/SMAG/cable_segment_6.csv",
    ],
    SEG7=[
        "data/2024/SMAG/cable_segment_7.csv",
    ],
    SEG8=[
        "data/2024/SMAG/cable_segment_8.csv",
    ],
)
import os
os.makedirs("figures/2024/SMAG-fit/", exist_ok=True)

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

    stns = [f"SEG{i}" for i in range(9)]
    frames = {}
    for stn in stns:
        files = station_maps[stn]
        files.sort()
        frames[stn] = pd.concat([pd.read_csv(f, parse_dates=["Date"]) for f in files])

    # Plot processed data
    sp = StackPlots(nrows=3, ncols=1, datetime=True, figsize=(6, 3), text_size=12)
    for stn in [stns[0], stns[4], stns[8]]:
        data = frames[stn].set_index("Date")
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
        ax.legend(loc=2, fontsize=12)
        sp.save_fig("figures/2024/SMAG-fit/2024.SMAG-fit.data.png")
        sp.close()
    return


def compile(gplot=False):
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
    stns = [f"SEG{i}" for i in range(9)]
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

    # # # Generate plots
    if gplot:
        model.plot_TS_with_others(
            fname="figures/2024/SMAG-fit/2024.Scubas.SMAG-fit.png",
            date_lim=[dt.datetime(2024, 5, 10, 12), dt.datetime(2024, 5, 12)],
            fig_title="SCUBAS / Time, UT since 12 UT on 10 May 2024",
            text_size=10,
            ylim=[-800, 800],
            interval=6,
        )
        model.plot_e_fields(
            fname="figures/2024/SMAG-fit/2024.Scubas.SMAG-fit.Exfield.png",
            date_lim=[dt.datetime(2024, 5, 10, 12), dt.datetime(2024, 5, 12)],
            fig_title=r"$E_x$-field / Time: UT since 12 UT on 10 May 2024",
            text_size=15,
            ylim=[-300, 300],
            interval=6,
            component="X",
            groups=[[0, 1, 2], [3, 4, 5], [6, 7, 8]],
        )
        model.plot_e_fields(
            fname="figures/2024/SMAG-fit/2024.Scubas.SMAG-fit.Eyfield.png",
            date_lim=[dt.datetime(2024, 5, 10, 12), dt.datetime(2024, 5, 12)],
            fig_title=r"$E_y$-field / Time: UT since 12 UT on 10 May 2024",
            text_size=15,
            ylim=[-300, 300],
            interval=6,
            component="Y",
            groups=[[0, 1, 2], [3, 4, 5], [6, 7, 8]],
        )
    return


if __name__ == "__main__":
    compile(True)
