"""
Run SCUBAS model for February 1958 storm
This script sets up the parameters for the SCUBAS model to analyze the impact of a storm that occurred in February 1958.
The script defines the storm parameters, including the date, location, and intensity of the storm. It also specifies the model parameters, such as the grid size and time step for the simulation. The script then runs the SCUBAS model with these parameters and outputs the results.
The script is designed to be run in a Python environment with the necessary libraries installed. It is intended for use by researchers and engineers studying the impact of storms on submarine cables and other underwater infrastructure.
This script is part of the Submarine Cable Super Storm Analysis project, which aims to improve the understanding of the impact of extreme weather events on submarine cables and other underwater infrastructure.
This project is funded by the National Science Foundation (NSF): Geospace Environment Modeling.
"""

import datetime as dt
import numpy as np
import pandas as pd
from loguru import logger

from bathymetry import BathymetryAnalysis
from utils import create_from_lat_lon
from utils import StackPlots
from cable import SCUBASModel

def read_dataset(base_path: str="data/1958/{x}[Eskdalemuir]-rescale-HR.csv") -> pd.DataFrame:
    """
    Read the dataset from the specified file path and return it as a pandas DataFrame.
    """
    D, H, Z = (
        pd.read_csv(base_path.replace("{x}", "D"), parse_dates=["Time"]),
        pd.read_csv(base_path.replace("{x}", "H"), parse_dates=["Time"]),
        pd.read_csv(base_path.replace("{x}", "Z"), parse_dates=["Time"])
    )
    D = D[(D.Time>=dt.datetime(1958, 2, 10, 11)) & (D.Time<dt.datetime(1958, 2, 11, 8))]
    H = H[(H.Time>=dt.datetime(1958, 2, 10, 11)) & (H.Time<dt.datetime(1958, 2, 11, 8))]
    Z = Z[(Z.Time>=dt.datetime(1958, 2, 10, 11)) & (Z.Time<dt.datetime(1958, 2, 11, 8))]
    data = pd.DataFrame()
    data["Time"], data["H"], data["D"], data["Z"] = D.Time, np.copy(H.H), np.copy(D.D), np.copy(Z.Z)
    data["X"], data["Y"] = (
        data.H * np.cos(np.deg2rad(data.D)),
        data.H * np.sin(np.deg2rad(data.D))
    )
    data = data.set_index("Time").resample("1s").asfreq().interpolate(method="cubic").reset_index()
    data["F"] = np.sqrt(data.X**2+data.Y**2+data.Z**2)
    data = data[data.Time<dt.datetime(1958, 2, 11, 7, 59)]

    data = data.rename(columns=dict(Time="Date"))
    data[["Date", "X", "Y", "Z", "F"]].to_csv("data/1958/compiled.csv", float_format="%g", index=False)
    logger.info("Data saved to data/1958/compiled.csv")

    sp = StackPlots(nrows=1, ncols=1, datetime=True, figsize=(6, 4))
    _, ax = sp.plot_stack_plots(
        data.Date, data.X-np.median(data.X.iloc[:60]), 
        ylim=[-1500, 1500], label=r"$B_x$", 
        xlim=[dt.datetime(1958,2,10,11), dt.datetime(1958,2,11,8)]
    )
    sp.plot_stack_plots(
        data.Date, data.Y-np.median(data.Y.iloc[:60]), 
        ylim=[-1500, 1500], label=r"$B_y$", color="r",
        xlim=[dt.datetime(1958,2,10,11), dt.datetime(1958,2,11,8)], ax=ax,
    )
    sp.plot_stack_plots(
        data.Date, data.Z-np.median(data.Z.iloc[:60]), 
        ylim=[-1500, 1500], label=r"$B_z$", xlabel="Time, UT since 16 UT on 10 Feb 1958", color="k", ylabel=r"$B_{esk}$, nT",
        xlim=[dt.datetime(1958,2,10,16), dt.datetime(1958,2,11,8)], ax=ax,
    )
    ax.legend(loc=2, fontsize=12)
    sp.save_fig("figures/1958.data.png")
    sp.close()
    return data

def get_bathymetry(file_path: str="data/1958/lat_long_bathymetry.csv") -> None:
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
    from scubas.conductivity import ConductivityProfile
    profiles = ConductivityProfile.compile_bined_profiles(np.array(dSegments))
    for p, seg in zip(profiles, segments):
        o = bth.iloc[seg[0]:seg[1]]
        depth = o["bathymetry.meters"].max()
        p.layers[0].thickness = depth/1e3
    return profiles

def compile(datafile = ["data/1958/compiled.csv"]):
    _ = read_dataset()
    bathymetry, segment_coordinates, segments = get_bathymetry()
    segment_files = [datafile]*len(segment_coordinates)
    profiles = get_conductivity_profile(
        segment_coordinates, segments, 
        bathymetry.bathymetry_data
    )
    cable = create_from_lat_lon(
        segment_coordinates, 
        profiles, 
    )
    model = SCUBASModel(
        cable_name="TAT-1",
        cable_structure=cable,
        segment_files=segment_files,
    )
    model.read_stations(["ESK"], [datafile])
    model.initialize_TL()
    model.run_cable_segment()

    model.plot_TS_with_others(
        fname="figures/1958.Scubas.png", 
        date_lim=[dt.datetime(1958,2,10,16), dt.datetime(1958,2,11,8)],
        fig_title="SCUBAS (Esk) / Time: UT since 16 UT on 10 Feb 1958",
        text_size=10
    )
    return

if __name__ == "__main__":
    compile()