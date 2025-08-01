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

    stns, coords = ["ESK", "FRD"], ["HDZ", "HDZ", "HDZ", "HDZ"]
    frames = {}
    for stn, coord in zip(stns, coords):
        files = glob.glob(base_path + f"{stn}*.dat")
        files.sort()
        frames[stn] = pd.concat([read_iaga(f) for f in files])
    xlim=[dt.datetime(1958, 2, 11,), dt.datetime(1958, 2, 11, 5)]
    # Plot processed data
    sp = StackPlots(nrows=len(stns), ncols=1, datetime=True, figsize=(6, 4), text_size=12)
    for stn, coord in zip(stns, coords):
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
            data.Z - np.median(data.Z.iloc[:60]),
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
        data.index = data.index - dt.timedelta(minutes=1)
        data.to_csv(f"data/1958/{stn}_scaled.csv", header=True, index=True, float_format="%g")
        sp.save_fig("figures/1958/1958.data.png")
        sp.close()
    return

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
        p.layers[0].thickness = depth / 1e2  # in meters
    return profiles

def load_extracted_voltage(fname="data/1958/Voltage/TAT1Volt-rescale.csv"):
    # TAT1Volt-rescale.csv
    data = pd.read_csv(fname, parse_dates=["Time"])
    return data

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
    names = ["CS-W", "DO-1", "DO-2", "DO-3", "DO-4", "MAR", "DO-5", "CS-E"]
    _ = read_dataset()
    bathymetry, segment_coordinates, segments = get_bathymetry(names)
    segment_names = ["FRD", "FRD", "FRD", "FRD", "HAD", "HAD", "HAD", "HAD"]
    segment_names = ["ESK", "ESK", "ESK", "ESK", "ESK", "ESK", "ESK", "ESK"]
    segment_names = ["FRD", "FRD", "FRD", "FRD", "ESK", "ESK", "ESK", "ESK"]
    segment_files = [
        [f"data/1958/{name}_scaled.csv"] for name in segment_names
    ]
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

    model.read_stations(segment_names, segment_files)
    model.initialize_TL()

    model.run_cable_segment("data/1958/TAT1SimVolt.csv")

    xlim=[dt.datetime(1958, 2, 11,), dt.datetime(1958, 2, 11, 5)]
    model.plot_TS_with_others(
        fname="figures/1958/1958.Scubas.png",
        date_lim=xlim,
        fig_title="SCUBAS, Time: UT since 0 UT on 11 Feb 1958",
        text_size=10, ylim=[-3000, 3000]
    )

    model.plot_profiles(
        fname="figures/1958/1958.Profiles.png",
        xlim=[1e-6, 1e-2],
        tylim=[-90, 90],
        tyticks=[-90, -45, 0, 45, 90],
        aylim=[1e-3, 1e0],
        t_mul=1.0,
        nrows=2,
        ncols=4,
        text_size=15,
        tag0_loc=[0, 4],
        tag1_loc=[4, 5, 6, 7],
        tag2_loc=[3, 7],
        figsize=(4, 4),
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
        mult=-1,
    )
    return

if __name__ == "__main__":
    compile_1958(True)