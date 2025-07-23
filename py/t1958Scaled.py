import datetime as dt
import os

import numpy as np
import pandas as pd  # type: ignore
from bathymetry import BathymetryAnalysis
from cable import SCUBASModel
from loguru import logger  # type: ignore
from utils import StackPlots, create_from_lat_lon, read_iaga

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

    stns, coords = ["FRD", "ESK", "HAD", "LER"], ["HDZ", "HDZ", "HDZ", "HDZ"]
    frames = {}
    for stn, coord in zip(stns, coords):
        files = glob.glob(base_path + f"{stn}*.dat")
        files.sort()
        frames[stn] = pd.concat([read_iaga(f) for f in files])

    # Plot processed data
    sp = StackPlots(nrows=4, ncols=1, datetime=True, figsize=(6, 4), text_size=12)
    for stn, coord in zip(stns, coords):
        data = frames[stn]
        data.drop_duplicates().sort_index(inplace=True)
        _, ax = sp.plot_stack_plots(
            data.index,
            data.X - np.median(data.X.iloc[:60]),
            ylim=[-1500, 1500],
            label=r"$B_x$",
            # xlim=[dt.datetime(1989, 3, 12, 12), dt.datetime(1989, 3, 14, 12)],
            interval=6,
        )
        sp.plot_stack_plots(
            data.index,
            data.Y - np.median(data.Y.iloc[:60]),
            ylim=[-1500, 1500],
            label=r"$B_y$",
            color="r",
            # xlim=[dt.datetime(1989, 3, 12, 12), dt.datetime(1989, 3, 14, 12)],
            ax=ax,
            interval=6,
        )
        sp.plot_stack_plots(
            data.index,
            data.Z - np.median(data.Z.iloc[:60]),
            ylim=[-1500, 1500],
            label=r"$B_z$",
            xlabel="Time, UT since 12 UT on 10 Feb 1989",
            color="k",
            ylabel=f"$B[{stn.lower()}]$, nT",
            xlim=[dt.datetime(1958, 2, 10, 12), dt.datetime(1958, 2, 12, 6)],
            ax=ax,
            interval=6,
        )
        ax.legend(loc=2, fontsize=12)
        data.to_csv(f"data/1958/{stn}_scaled.csv", header=True, index=True, float_format="%g")
        sp.save_fig("figures/1958/1958.data.png")
        sp.close()
    return

def compile_1958(gplot=False):
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
    return

if __name__ == "__main__":
    compile_1958(True)