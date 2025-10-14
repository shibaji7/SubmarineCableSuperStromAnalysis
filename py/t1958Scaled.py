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
        data.index = data.index - dt.timedelta(minutes=2)
        k = 1.3
        data.Z, data.X, data.Y = data.Z*k, data.X*k, data.Y*k
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
    run_detailed_error_analysis(
        inputs=obs,
        cable=model.cable,
        date_lims=[dt.datetime(1958, 2, 11, 1), dt.datetime(1958, 2, 11, 4)],
        fnames=[
            "figures/1958/1958.Error.qq.png",
        ],
    )
    Dst1958 = pd.read_csv(
        "data/1958/Dst.csv",
        skiprows=17,                # Skip metadata/header lines
        dtype={"DATE": str, "TIME": str, "DOY": int, "DST": float},
        sep="\\s+",              # Use regex to split on whitespace
    )
    Dst1958["DATETIME"] = pd.to_datetime(Dst1958["DATE"] + " " + Dst1958["TIME"])
    Dst1958.drop(columns=["DATE", "TIME", "|"], inplace=True)
    print(Dst1958.set_index("DATETIME").resample('1min').interpolate().head())
    Dst1958 = Dst1958.set_index("DATETIME").resample('1min').interpolate().reset_index()
    Dst1958 = Dst1958.rename(columns={"DATETIME": "time", "DST": "SymH"})
    model.run_detailed_error_analysis(
        inputs=obs,
        date_lims=[dt.datetime(1958, 2, 11, 1), dt.datetime(1958, 2, 11, 4)],
        fnames=[
            "figures/1958/1958.Errors.qq.png",
            "figures/1958/1958.Scores.png",
        ],
        omni=Dst1958,
        lims=[-3000, 3000],
    )
    return


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
    compile_1958(True)