"""Build scaffold for TAT1 paper figures.

This file is the single place where the final paper figure builders will live.
The individual figure mappings are intentionally left unassigned until each
manuscript figure is dictated.
"""

from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp") / "matplotlib-cache"))

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.ticker as mticker
from matplotlib.ticker import MultipleLocator

try:
    from .tat1_style import PNG_DIR, apply_tat1_style, finish_axis
except ImportError:
    from tat1_style import PNG_DIR, apply_tat1_style, finish_axis


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TAT1_CODE_DIR = Path(__file__).resolve().parent
TAT1_FIGURE_DIR = PROJECT_ROOT / "figures" / "tat1"
TAT1_PNG_DIR = TAT1_FIGURE_DIR / "pngfiles"

for path in (PROJECT_ROOT / "py", TAT1_CODE_DIR):
    path_text = str(path)
    if path_text not in sys.path:
        sys.path.insert(0, path_text)


SOURCE_CANDIDATES = {
    "bathymetry": {
        "script": "plot_bathymetry.py",
        "outputs": ["bathymetry_TAT-1.png", "bathymetry_TAT-1.pdf"],
    },
    "esk_b_fields": {
        "script": "plot_esk.py",
        "outputs": ["1958.ESK.png", "1958.ESK.pdf"],
    },
    "b_and_e": {
        "script": "plot_b_e_combined.py",
        "outputs": ["1958.B_and_E.png", "1958.B_and_E.pdf"],
    },
    "e_along_cable": {
        "script": "plot_e_along.py",
        "outputs": ["1958.E_along_Cable.png", "1958.E_along_Cable.pdf"],
    },
    "scubas_compare": {
        "script": "plot_scubas_compare.py",
        "outputs": ["1958.Scubas.Compare.png", "1958.Scubas.Compare.pdf"],
    },
    "scaling_optimization": {
        "script": "find_optimal_scale.py",
        "outputs": ["1958.ScalingOptimization.png", "1958.ScalingOptimization.pdf"],
    },
    "peak_timing": {
        "script": "analyze_peak_timing.py",
        "outputs": ["1958.PeakTiming.png", "1958.PeakTiming.pdf"],
    },
    "peak_error_analysis": {
        "script": "analyze_peaks.py",
        "outputs": ["1958.ErrorAnalysis.png", "1958.ErrorAnalysis.pdf"],
    },
}


@dataclass(frozen=True)
class FigureSpec:
    """One manuscript figure entry."""

    figure_id: str
    kind: str
    builder: Callable[[], list[Path]]
    description: str = "Unassigned"


def prepare_environment(font_size: int = 12) -> None:
    """Set project paths, output folders, and shared plotting style."""
    TAT1_PNG_DIR.mkdir(parents=True, exist_ok=True)
    apply_tat1_style(font_size=font_size)


def save_current_figure(name: str, *, dpi: int = 300) -> Path:
    """Save the active Matplotlib figure into figures/tat1/pngfiles."""
    TAT1_PNG_DIR.mkdir(parents=True, exist_ok=True)
    output = TAT1_PNG_DIR / f"{Path(name).stem}.png"
    plt.gcf().savefig(output, dpi=dpi, bbox_inches="tight", facecolor="white")
    return output


def align_axis_to_top_map_frame(top_ax: plt.Axes, target_ax: plt.Axes) -> None:
    """Match a lower panel to the visible map frame inside the top raster."""
    top_pos = top_ax.get_position()
    target_pos = target_ax.get_position()
    map_left_frac = 0.122
    map_right_frac = 0.883
    left = top_pos.x0 + top_pos.width * map_left_frac
    right = top_pos.x0 + top_pos.width * map_right_frac
    target_ax.set_position([left, target_pos.y0, right - left, target_pos.height])


def build_tat1_map_image() -> Path:
    """Create a temporary GEBCO bathymetry map with only the TAT-1 cable."""
    import geometry
    from bathymetry import get_TAT1_segments

    temp_path = Path(tempfile.gettempdir()) / "tat1_figure02_map.png"
    fig, ax = geometry.create_new_pane(
        dt.datetime(1989, 3, 12),
        central_longitude=-30,
        central_latitude=50,
        extent=[-80, 10, 29, 71],
        darray=20,
        cx=[0.92, 0.2, 0.03, 0.5],
    )
    overlay_tat1_new_segments(ax, geometry, get_TAT1_segments)
    fig.savefig(temp_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return temp_path


def overlay_tat1_new_segments(ax, geometry_module, get_tat1_segments) -> None:
    """Overlay the new 9-segment TAT-1 route on a Cartopy axis."""
    lats = np.asarray(get_tat1_segments("lat", which="new"), dtype=float)
    lons = np.asarray(get_tat1_segments("lon", which="new"), dtype=float)
    transform = geometry_module.ccrs.PlateCarree()

    ax.plot(
        lons,
        lats,
        ls="-",
        lw=1.2,
        color="k",
        transform=transform,
        zorder=6,
    )
    ax.scatter(
        lons,
        lats,
        marker="s",
        s=5,
        c="m",
        transform=transform,
        zorder=7,
    )
    for j in range(len(lons) - 1):
        ax.text(
            (lons[j] + lons[j + 1]) / 2,
            1 + ((lats[j] + lats[j + 1]) / 2),
            j + 1,
            ha="center",
            va="center",
            transform=transform,
            fontsize=8,
            fontdict={"weight": "bold", "color": "m"},
            zorder=8,
        )

    station_lons = [-77.4588, -52.7453, 355.516, -3.1757]
    station_lats = [38.3004, 47.5556, 50.995, 55.2678]
    ax.scatter(station_lons, station_lats, marker="D", s=5, c="r", transform=transform, zorder=8)
    station_labels = [
        ("FRD", -77.4588, 39.3004, 90),
        ("ESK", 356.516, 57.2678, 0),
    ]
    for label, lon, lat, rotation in station_labels:
        ax.text(
            lon,
            lat,
            label,
            ha="center",
            va="bottom",
            transform=transform,
            fontsize=10,
            fontdict={"color": "r"},
            rotation=rotation,
            zorder=9,
        )


def read_esk_data(base_path: Path | None = None) -> pd.DataFrame:
    """Read Eskdalemuir IAGA data used by the B/E combined figure."""
    if base_path is None:
        base_path = PROJECT_ROOT / "data" / "1958" / "scaled_data"

    files = sorted(glob.glob(str(base_path / "ESK*.dat")))
    if not files:
        raise FileNotFoundError(f"No ESK*.dat files found under {base_path}")
    data = pd.concat([read_iaga_file(Path(path)) for path in files])
    data.drop_duplicates().sort_index(inplace=True)
    return data


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in km between two geographic points."""
    radius_km = 6371.0
    phi1, phi2 = np.deg2rad(lat1), np.deg2rad(lat2)
    dphi = np.deg2rad(lat2 - lat1)
    dlambda = np.deg2rad(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    return float(2 * radius_km * np.arctan2(np.sqrt(a), np.sqrt(1 - a)))


def plot_tat1_bathymetry_panel(ax: plt.Axes) -> None:
    """Draw the TAT-1 along-cable bathymetry profile with legible text."""
    names = ["CS-W", "DO-1", "DO-2", "DO-3", "MAR", "DO-4", "RDG-1", "DO-5", "CS-E"]
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
    ]
    segments = [
        (0, 32),
        (32, 50),
        (50, 60),
        (60, 170),
        (170, 210),
        (210, 335),
        (335, 390),
        (390, 442),
        (445, 486),
    ]
    methods = [
        lambda x: x.quantile(0.25),
        *([np.mean] * 7),
        lambda x: x.quantile(0.40),
    ]

    bathy = pd.read_csv(PROJECT_ROOT / "data" / "1958" / "lat_long_bathymetry-modified.csv")
    bathy["distance"] = 0.0
    for i in range(1, len(bathy)):
        bathy.loc[i, "distance"] = bathy.loc[i - 1, "distance"] + haversine_km(
            bathy["lat"].iloc[i - 1],
            bathy["lon"].iloc[i - 1],
            bathy["lat"].iloc[i],
            bathy["lon"].iloc[i],
        ) * 1e3

    ax.plot(
        bathy["distance"] / 1e3,
        -1 * bathy["bathymetry.meters"] / 1e3,
        color="k",
        lw=0.7,
    )

    dist, depth = [], []
    for i, seg in enumerate(segments):
        segment_data = bathy.iloc[seg[0] : seg[1]]
        dist.append(segment_data["distance"].iloc[0] / 1e3)
        depth.append(methods[i](segment_data["bathymetry.meters"]) / 1e3)
        ax.text(
            segment_data["distance"].mean() / 1e3,
            -(segment_data["bathymetry.meters"].mean() / 1e3) - 0.1,
            names[i],
            ha="center",
            va="top",
            rotation=90,
            fontsize=12,
            color="red",
        )

    dist.append(bathy["distance"].iloc[-1] / 1e3)
    depth.append(bathy["bathymetry.meters"].iloc[-1] / 1e3)
    depth = np.array(depth)
    depth[depth > 0] = depth[depth > 0] * -1
    ax.step(dist, depth, where="post", ls="-", lw=2.0, color="#0072B2")

    ax.set_xticks([0, 500, 2000, 4000])
    ax.set_xlim([0, 3900])
    ax.axhline(0, ls="--", lw=0.6, color="b", alpha=0.7)
    ax.set_ylim([-6, 0.5])
    ax.set_yticks([-6, -5, -4, -3, -2, -1, -0.5])
    ax.set_yticklabels([6, 5, 4, 3, 2, 1, 0.5])
    ax.set_ylabel("Depths, km", fontsize=13)
    ax.set_xlabel("Distance, km", fontsize=13)
    ax.tick_params(axis="both", labelsize=12)


def read_iaga_file(path: Path) -> pd.DataFrame:
    """Read the subset of IAGA-2002 needed for the TAT1 B-field panels."""
    header_records = {"header_length": 0}
    with path.open("r") as openfile:
        for line in openfile:
            if line[0] != " ":
                continue
            header_records["header_length"] += 1
            label = line[1:24].strip()
            description = line[24:-2].strip()
            header_records[label.lower()] = description

    reported = header_records["reported"]
    if len(reported) % 4 != 0:
        raise ValueError(f"IAGA reported record is not divisible by 4: {reported}")

    record_length = len(reported) // 4
    column_names = [x for x in reported[record_length - 1 :: record_length]]
    seen_count: dict[str, int] = {}
    for i, col in enumerate(column_names):
        if col in seen_count:
            column_names[i] += str(seen_count[col])
            seen_count[col] += 1
        else:
            seen_count[col] = 1

    df = pd.read_csv(
        path,
        header=header_records["header_length"],
        sep=r"\s+",
        parse_dates=[[0, 1]],
        index_col=0,
        usecols=[0, 1, 3, 4, 5, 6],
        na_values=[99999.90, 99999.0, 88888.80, 88888.00],
        names=["Date", "Time"] + column_names,
    )
    df.index.name = "Date"
    if "X" not in column_names and "Y" not in column_names:
        if "H" not in column_names or "D" not in column_names:
            raise ValueError(f"Only HDZF-to-XYZF conversion is supported for {path}")
        df["X"] = df["H"] * np.cos(np.deg2rad(df["D"] / 60.0))
        df["Y"] = df["H"] * np.sin(np.deg2rad(df["D"] / 60.0))
        del df["H"], df["D"]
    return df


def read_dst_data(year: int) -> pd.DataFrame:
    """Read Kyoto-style Dst data for a given event year."""
    dst = pd.read_csv(
        PROJECT_ROOT / "data" / str(year) / "Dst.csv",
        skiprows=18,
        names=["DATE", "TIME", "DOY", "DST"],
        dtype={"DATE": str, "TIME": str, "DOY": int, "DST": float},
        sep=r"\s+",
    )
    dst["DATETIME"] = pd.to_datetime(dst["DATE"] + " " + dst["TIME"])
    drop_cols = [col for col in ["DATE", "TIME", "|"] if col in dst.columns]
    dst.drop(columns=drop_cols, inplace=True)
    return dst


def read_ae_data(year: int) -> pd.DataFrame:
    """Read AE index data for a given event year."""
    ae = pd.read_csv(
        PROJECT_ROOT / "data" / str(year) / "AE.csv",
        skiprows=15,
        names=["DATE", "TIME", "DOY", "AE", "AU", "AL", "AO"],
        dtype={
            "DATE": str,
            "TIME": str,
            "DOY": int,
            "AE": float,
            "AU": float,
            "AL": float,
            "AO": float,
        },
        sep=r"\s+",
    )
    ae["DATETIME"] = pd.to_datetime(ae["DATE"] + " " + ae["TIME"])
    drop_cols = [col for col in ["DATE", "TIME", "DOY", "|"] if col in ae.columns]
    ae.drop(columns=drop_cols, inplace=True)
    return ae


def read_iaga_h_component(path: Path) -> pd.DataFrame:
    """Read H from an IAGA-2002 file that reports HDZF records."""
    header_line = None
    with path.open("r") as fp:
        for line_no, line in enumerate(fp):
            if line.strip().startswith("DATE"):
                header_line = line_no
                break
    if header_line is None:
        raise ValueError(f"Could not find DATE header in {path}")

    data = pd.read_csv(
        path,
        skiprows=header_line + 1,
        sep=r"\s+",
        names=["date", "time", "doy", "H", "D", "Z", "F"],
        usecols=["date", "time", "H"],
        na_values=[99999.90, 99999.0, 99999.99, 88888.80, 88888.00],
    )
    data["DATETIME"] = pd.to_datetime(data["date"] + " " + data["time"])
    data.drop(columns=["date", "time"], inplace=True)
    return data


def read_digitized_hour_series(path: Path) -> pd.DataFrame:
    """Read digitized TAT1 trace CSVs with Date as decimal UT hour."""
    data = pd.read_csv(path)
    required = {"Date", "H"}
    if required - set(data.columns):
        raise ValueError(f"{path} must contain columns {sorted(required)}")
    data = data.rename(columns={"Date": "hour", "H": "value"})
    data = data.sort_values("hour").drop_duplicates("hour")
    return data


def baseline_removed(values: pd.Series, hours: pd.Series | None = None) -> pd.Series:
    """Remove a first-hour median baseline from a trace."""
    if hours is None:
        baseline = values.iloc[: min(60, len(values))].median()
    else:
        early = values[(hours >= 0) & (hours <= 1)]
        baseline = early.median() if not early.empty else values.iloc[: min(60, len(values))].median()
    return values - baseline


def read_kpap_txt(year: int = 1958) -> tuple[list[float], pd.DataFrame]:
    """Read 3-hourly ap values from the fixed-width KpAp text file."""
    kp: list[float] = []
    ap_records = []
    with (PROJECT_ROOT / "data" / str(year) / "KpAp.txt").open("r") as f:
        lines = f.readlines()

    for line in lines[1:]:
        text = line.rstrip("\n")
        date_text = text[:8]
        ap_text = text[28:]
        ap_values = [float(ap_text[x : x + 3].strip()) for x in range(0, 24, 3)]
        ap_records.extend(
            {
                "date": dt.datetime.strptime(date_text + f"{hour:02d}", "%Y%m%d%H"),
                "Ap": ap_values[hour // 3],
            }
            for hour in range(0, 24, 3)
        )
    return kp, pd.DataFrame.from_records(ap_records)


def unassigned_figure(figure_id: str) -> list[Path]:
    """Placeholder used until a manuscript figure is assigned."""
    raise NotImplementedError(
        f"{figure_id} is not assigned yet. Tell me which existing figure or "
        "script should be recreated for this manuscript figure."
    )


def figure01() -> list[Path]:
    return unassigned_figure("figure01")


def figure02() -> list[Path]:
    """Figure 02: TAT-1 map and along-cable bathymetry profile."""
    prepare_environment(font_size=9)

    map_image = build_tat1_map_image()

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(7.2, 6.4),
        dpi=300,
        gridspec_kw={"height_ratios": [1.2, 1.0], "hspace": 0.03},
    )

    axes[0].imshow(plt.imread(map_image))
    axes[0].set_axis_off()
    axes[0].text(
        0.015,
        0.975,
        "(a)",
        transform=axes[0].transAxes,
        ha="left",
        va="top",
        fontsize=12,
        fontweight="bold",
        color="black",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=1.5),
    )

    plot_tat1_bathymetry_panel(axes[1])
    align_axis_to_top_map_frame(axes[0], axes[1])
    axes[1].text(
        -0.015,
        1.055,
        "(b)",
        transform=axes[1].transAxes,
        ha="right",
        va="bottom",
        fontsize=12,
        fontweight="bold",
        color="black",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=1.5),
    )

    output = TAT1_PNG_DIR / "figure02.png"
    fig.savefig(output, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return [output]


def figure03() -> list[Path]:
    """Figure 03: TAT-1 segment conductivity profiles."""
    prepare_environment(font_size=9)
    os.environ.setdefault("SPACEPY", str(Path("/tmp") / "spacepy"))

    from cable import SCUBASModel
    from scubas.datasets import PROFILES
    from tat1958Scaled import get_bathymetry, get_conductivity_profile
    from utils import create_from_lat_lon

    names = ["CS-W", "DO-1", "DO-2", "DO-3", "MAR", "DO-4", "RDG-1", "DO-5", "CS-E"]
    segment_names = ["CS-W", "DO-1", "DO-2", "DO-3", "RDG-1", "DO-4", "MAR", "DO-5", "CS-E"]
    segment_files = [[f"data/1958/{name}_scaled.csv"] for name in segment_names]
    missing = [file for files in segment_files for file in files if not (PROJECT_ROOT / file).exists()]
    if missing:
        raise FileNotFoundError(f"Missing segment input files for Figure 03: {missing}")

    bathymetry, segment_coordinates, segments = get_bathymetry(names)
    profiles = get_conductivity_profile(
        segment_coordinates,
        segments,
        bathymetry.bathymetry_data,
    )
    cable = create_from_lat_lon(
        segment_coordinates,
        profiles,
        names=names,
        left_active_termination=PROFILES.LD0,
        right_active_termination=PROFILES.LD0,
    )
    model = SCUBASModel(
        cable_name="TAT-1",
        cable_structure=cable,
        segment_files=segment_files,
    )
    model.initialize_TL()

    output = TAT1_PNG_DIR / "figure03.png"
    model.plot_profiles(
        fname=str(output),
        xlim=[1e-6, 1e-2],
        tylim=[-90, 90],
        tyticks=[-90, -45, 0, 45, 90],
        aylim=[1e-3, 1e0],
        t_mul=1.0,
        nrows=3,
        ncols=3,
        text_size=15,
        tag0_loc=[0, 3, 6],
        tag1_loc=[6, 7, 8],
        tag2_loc=[2, 5, 8],
        figsize=(3.5, 3),
    )
    return [output]


def figure04() -> list[Path]:
    """Figure 04: H-component station stack plot with AU/AL."""
    prepare_environment(font_size=10)

    event_day = dt.datetime(1958, 2, 11)
    extract_dir = TAT1_CODE_DIR / "tat1_data_extract" / "1min"
    station_specs = [
        ("ESK", None),
        ("Byrd", extract_dir / "BYRD_1min.csv"),
        ("Halley Bay", extract_dir / "HalleyBay_1min.csv"),
        ("Sitka", extract_dir / "SITKA_1min.csv"),
    ]

    fig, axes = plt.subplots(
        5,
        1,
        figsize=(7.2, 6.2),
        dpi=300,
        sharex=True,
        gridspec_kw={"hspace": 0.14},
    )

    for ax, (label, path) in zip(axes[:4], station_specs, strict=True):
        if label == "ESK":
            esk = read_iaga_h_component(
                PROJECT_ROOT / "data" / "1958" / "scaled_data" / "ESK_Feb1958min.dat"
            )
            esk["hour"] = (esk["DATETIME"] - event_day).dt.total_seconds() / 3600.0
            esk = esk[(esk["hour"] >= 0) & (esk["hour"] <= 24)]
            hours = esk["hour"]
            values = baseline_removed(esk["H"], hours)
        else:
            trace = read_digitized_hour_series(path)
            hours = trace["hour"]
            values = baseline_removed(trace["value"], hours)

        ax.plot(hours, values, color="black", lw=1.2)
        ax.set_ylabel(f"{label}\n$\\Delta H$ (nT)", fontsize=10)
        ax.set_xlim(0, 24)
        ax.set_ylim(-2000, 2000)
        ax.set_yticks([-2000, 0, 2000])
        ax.tick_params(axis="both", labelsize=9)
        finish_axis(ax, grid=True, zero_line=True)

    au = read_digitized_hour_series(extract_dir / "AU_1min.csv")
    al = read_digitized_hour_series(extract_dir / "AL_1min.csv")
    axes[4].plot(
        au["hour"],
        baseline_removed(au["value"], au["hour"]),
        color="black",
        lw=1.2,
        label="AU",
    )
    axes[4].plot(
        al["hour"],
        baseline_removed(al["value"], al["hour"]),
        color="black",
        lw=1.2,
        ls="--",
        label="AL",
    )
    axes[4].set_ylabel("AU / AL\n(nT)", fontsize=10)
    axes[4].set_xlabel("Time, UT on 11 Feb 1958", fontsize=11)
    axes[4].set_xlim(0, 24)
    axes[4].set_ylim(-2000, 1000)
    axes[4].set_yticks([-2000, -1000, 0, 1000])
    axes[4].xaxis.set_major_locator(MultipleLocator(3))
    axes[4].xaxis.set_minor_locator(MultipleLocator(1))
    axes[4].tick_params(axis="both", labelsize=9)
    finish_axis(axes[4], grid=True, zero_line=True, legend=True, legend_loc="upper right", legend_ncol=2)

    axes[0].set_title("H-component magnetic perturbations and auroral indices", fontsize=12)

    output = TAT1_PNG_DIR / "figure04.png"
    fig.savefig(output, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return [output]


def figure05() -> list[Path]:
    """Figure 05: TAT-1/TAT-8 electrojet map and TAT-1 declination map."""
    prepare_environment(font_size=10)

    map_path = PROJECT_ROOT / "figures" / "GEBCO_2024_Bathymetry_TAT1,8_Electrojet_60MLAT.png"
    declination_pdf = TAT1_CODE_DIR / "TAT1_declination_geoplot.pdf"
    declination_png = TAT1_CODE_DIR / "TAT1_declination_geoplot.png"
    declination_tiff = TAT1_CODE_DIR / "TAT1_declination_geoplot.tiff"
    if not map_path.exists():
        raise FileNotFoundError(f"Missing Figure 05 source image: {map_path}")

    map_image = plt.imread(map_path)
    if declination_pdf.exists():
        declination_image = read_pdf_page_image(declination_pdf, dpi=300)
    elif declination_png.exists():
        declination_image = plt.imread(declination_png)
    elif declination_tiff.exists():
        declination_image = plt.imread(declination_tiff)
    else:
        raise FileNotFoundError(
            "Missing Figure 05 declination source: "
            f"{declination_pdf}, {declination_png}, or {declination_tiff}"
        )

    map_aspect = map_image.shape[1] / map_image.shape[0]
    declination_aspect = declination_image.shape[1] / declination_image.shape[0]
    height_ratios = [1.0 / map_aspect, 1.0 / declination_aspect]

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(7.2, 7.9),
        dpi=300,
        gridspec_kw={"height_ratios": height_ratios, "hspace": 0.025},
    )
    for ax, image, label in zip(
        axes,
        [map_image, declination_image],
        ["(a)", "(b)"],
        strict=True,
    ):
        ax.imshow(image)
        ax.set_axis_off()
        ax.text(
            0.012,
            0.975,
            label,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=12,
            fontweight="bold",
            color="black",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=1.5),
        )

    output = TAT1_PNG_DIR / "figure05.png"
    fig.savefig(output, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return [output]


def read_pdf_page_image(pdf_path: Path, dpi: int = 300) -> np.ndarray:
    """Render the first PDF page to an RGB image array for raster composites."""
    if shutil.which("pdftoppm") is None:
        raise RuntimeError("pdftoppm is required to render PDF source images")

    with tempfile.TemporaryDirectory() as temp_dir:
        prefix = Path(temp_dir) / "page"
        subprocess.run(
            ["pdftoppm", "-png", "-singlefile", "-r", str(dpi), str(pdf_path), str(prefix)],
            check=True,
        )
        return plt.imread(prefix.with_suffix(".png"))


def figure06() -> list[Path]:
    """Figure 06: Eskdalemuir B-field and modeled E_parallel along TAT-1."""
    prepare_environment(font_size=8)

    xlim = [dt.datetime(1958, 2, 11, 1), dt.datetime(1958, 2, 11, 4)]
    data = read_esk_data()
    model_out = pd.read_csv(
        PROJECT_ROOT / "data" / "1958" / "TAT1SimVolt_1.0.csv",
        parse_dates=["Time"],
    )

    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(3.5, 4.0),
        dpi=300,
        sharex=True,
        gridspec_kw={"hspace": 0.05},
    )

    ax1.xaxis.set_major_locator(mdates.HourLocator(interval=1))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%H"))

    data_z_shifted = data.Z.copy()
    data_z_shifted.index = data_z_shifted.index + dt.timedelta(minutes=10)

    ax1.plot(
        data.index,
        data.X - np.median(data.X.iloc[:60]),
        color="#0072B2",
        ls="-",
        lw=1.0,
        label="$B_x$",
    )
    ax1.plot(
        data.index,
        data.Y - np.median(data.Y.iloc[:60]),
        color="#D55E00",
        ls="-",
        lw=1.0,
        label="$B_y$",
    )
    ax1.plot(
        data_z_shifted.index,
        data_z_shifted - np.median(data.Z.iloc[:60]),
        color="#009E73",
        ls="-",
        lw=1.0,
        label="$B_z$",
    )

    ax1.legend(loc=2, fontsize=7)
    ax1.set_ylim(-1000, 1000)
    ax1.set_xlim(xlim)
    ax1.set_ylabel("$B$, nT")
    ax1.text(
        -0.10,
        0.98,
        "(a)",
        transform=ax1.transAxes,
        fontdict=dict(size=10, weight="bold"),
        ha="left",
        va="top",
    )
    ax1.grid(True, lw=0.3, alpha=0.3)

    ax2.xaxis.set_major_locator(mdates.HourLocator(interval=1))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%H"))
    ax2.set_ylabel("$E_{||}$, mV/km")
    ax2.set_xlabel("Time, UT (11 Feb 1958)")
    ax2.set_yticklabels([])
    ax2.set_xlim(xlim)

    names = ["CS-E", "DO-5", "RDG-1", "DO-4", "MAR", "DO-3", "DO-2", "DO-1", "CS-W"]
    names.reverse()
    y_positions = np.array([9000, 8000, 7000, 6000, 5000, 4000, 3000, 2000, 1000])
    label_x = xlim[1] + dt.timedelta(minutes=2)

    for j, name in enumerate(names):
        ex_col = f"E.X.0{j}"
        ey_col = f"E.Y.0{j}"

        if ex_col in model_out.columns and ey_col in model_out.columns:
            e_parallel = model_out[ex_col] * 0.5 + model_out[ey_col] * 0.5
            ax2.plot(
                model_out.Time,
                y_positions[j] + e_parallel - np.median(e_parallel),
                color="k",
                ls="-",
                lw=1.0,
            )
            ax2.text(
                label_x,
                y_positions[j],
                name,
                color="#0072B2",
                fontsize=5,
                va="center",
                ha="left",
            )

    ax2.set_ylim(-2000, 11000)
    ax2.axvline(
        dt.datetime(1958, 2, 11, 1, 30),
        ymin=6000 / 13000,
        ymax=7000 / 13000,
        color="#009E73",
        ls="-",
        lw=1.5,
    )
    ax2.text(
        dt.datetime(1958, 2, 11, 1, 32),
        4200,
        "1000 mV/km",
        color="#009E73",
        fontsize=6,
    )
    ax2.text(
        -0.10,
        0.98,
        "(b)",
        transform=ax2.transAxes,
        fontdict=dict(size=10, weight="bold"),
        ha="left",
        va="top",
    )
    ax2.grid(True, lw=0.3, alpha=0.3)

    output = TAT1_PNG_DIR / "figure06.png"
    fig.savefig(output, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return [output]


def figure07() -> list[Path]:
    """Figure 07: observed and modeled TAT-1 cable voltage comparison."""
    prepare_environment(font_size=9)

    obs = pd.read_csv(
        PROJECT_ROOT / "data" / "1958" / "Voltage" / "TAT1Volt-rescale.csv",
        parse_dates=["Time"],
    )
    sim = pd.read_csv(
        PROJECT_ROOT / "data" / "1958" / "TAT1SimVolt_1.0.csv",
        parse_dates=["Time"],
    )
    sim_scaled = pd.read_csv(
        PROJECT_ROOT / "data" / "1958" / "TAT1SimVolt_1.8.csv",
        parse_dates=["Time"],
    )

    xlim = [dt.datetime(1958, 2, 11, 1), dt.datetime(1958, 2, 11, 4)]
    fig, ax = plt.subplots(1, 1, figsize=(6.4, 2.7), dpi=300)

    ax.scatter(
        obs.Time,
        obs.Voltage,
        color="0.45",
        marker="s",
        s=7,
        label="Observations",
        alpha=0.85,
        linewidths=0,
        zorder=3,
    )
    ax.plot(
        sim.Time,
        -sim["Vt(v)"],
        color="#D55E00",
        ls="-",
        lw=1.4,
        label="SCUBAS",
        zorder=4,
    )
    ax.plot(
        sim_scaled.Time,
        -sim_scaled["Vt(v)"],
        color="#009E73",
        ls="-",
        lw=1.3,
        label=r"SCUBAS (~68% scaled $|B|$ at western edge)",
        zorder=5,
    )

    ax.set_xlim(xlim)
    ax.set_ylim(-3000, 3000)
    ax.set_ylabel("Voltage (V)")
    ax.set_xlabel("Time, UT (11 Feb 1958)")
    ax.xaxis.set_major_locator(mdates.MinuteLocator(byminute=range(0, 60, 30)))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.legend(loc="upper right", fontsize=7, frameon=False)
    finish_axis(ax)

    output = TAT1_PNG_DIR / "figure07.png"
    fig.savefig(output, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return [output]


def figure08() -> list[Path]:
    """Figure 08: MATLAB-generated Earth nightside field-line schematic."""
    source = TAT1_FIGURE_DIR / "earth_fieldlines_nightside_cropped.png"
    output = TAT1_PNG_DIR / "figure08.png"

    subprocess.run(
        ["matlab", "-batch", "addpath('py/tat1'); plot_figure6"],
        cwd=PROJECT_ROOT,
        check=True,
    )
    if not source.exists():
        raise FileNotFoundError(f"MATLAB did not create expected source image: {source}")

    shutil.copyfile(source, output)
    return [output]


OVAL_COEFF = {
    "poleward": {
        "A0": [-0.07, 24.54, -12.53, 2.15],
        "A1": [-10.06, 19.83, -9.33, 1.24],
        "A2": [-4.44, 7.47, -3.01, 0.25],
        "A3": [-3.77, 7.90, -4.73, 0.91],
        "a1": [-6.61, 10.17, -5.80, 1.19],
        "a2": [6.37, -1.10, 0.34, -0.38],
        "a3": [-4.48, 10.16, -5.87, 0.98],
    },
    "equatorward": {
        "A0": [1.61, 23.21, -10.97, 2.03],
        "A1": [-9.59, 17.78, -7.20, 0.96],
        "A2": [-12.07, 17.49, -7.96, 1.15],
        "A3": [-6.56, 11.44, -6.73, 1.31],
        "a1": [-2.22, 1.50, -0.58, 0.08],
        "a2": [-23.98, 42.79, -26.96, 5.56],
        "a3": [-20.07, 36.67, -24.20, 5.11],
    },
    "diffuse": {
        "A0": [3.44, 29.77, -16.38, 3.35],
        "A1": [-2.41, 7.89, -4.32, 0.87],
        "A2": [-0.74, 3.94, -3.09, 0.72],
        "A3": [-2.12, 3.24, -1.67, 0.31],
        "a1": [-1.68, -2.48, 1.58, -0.28],
        "a2": [8.69, -20.73, 13.03, -2.14],
        "a3": [8.61, -5.34, -1.36, 0.76],
    },
}


def oval_poly3(coefficients: list[float], value: float) -> float:
    b0, b1, b2, b3 = coefficients
    return b0 + b1 * value + b2 * value**2 + b3 * value**3


def starkov_oval_boundary(boundary_name: str, al_nt: float, mlt_hours: np.ndarray) -> np.ndarray:
    """Return Starkov/Sigernes auroral boundary colatitude in degrees."""
    log_al = math.log10(max(abs(al_nt), 1.0))
    coeff = OVAL_COEFF[boundary_name]
    a0 = oval_poly3(coeff["A0"], log_al)
    a1_amp = oval_poly3(coeff["A1"], log_al)
    a2_amp = oval_poly3(coeff["A2"], log_al)
    a3_amp = oval_poly3(coeff["A3"], log_al)
    phase1 = oval_poly3(coeff["a1"], log_al)
    phase2 = oval_poly3(coeff["a2"], log_al)
    phase3 = oval_poly3(coeff["a3"], log_al)
    mlt_hours = np.asarray(mlt_hours)
    return (
        a0
        + a1_amp * np.cos(np.radians(15 * (mlt_hours - phase1)))
        + a2_amp * np.cos(np.radians(30 * (mlt_hours - phase2)))
        + a3_amp * np.cos(np.radians(45 * (mlt_hours - phase3)))
    )


def slerp_route(p1: tuple[float, float], p2: tuple[float, float], n_points: int = 25) -> list[tuple[float, float]]:
    """Great-circle interpolation between two geographic lat/lon points."""
    lat1, lon1 = np.radians(p1)
    lat2, lon2 = np.radians(p2)
    x1 = np.array([math.cos(lat1) * math.cos(lon1), math.cos(lat1) * math.sin(lon1), math.sin(lat1)])
    x2 = np.array([math.cos(lat2) * math.cos(lon2), math.cos(lat2) * math.sin(lon2), math.sin(lat2)])
    omega = math.acos(np.clip(np.dot(x1, x2), -1, 1))
    points: list[tuple[float, float]] = []
    for fraction in np.linspace(0, 1, n_points):
        if omega < 1e-6:
            xv = x1
        else:
            xv = (
                math.sin((1 - fraction) * omega) * x1
                + math.sin(fraction * omega) * x2
            ) / math.sin(omega)
        lat = math.degrees(math.asin(np.clip(xv[2], -1, 1)))
        lon = math.degrees(math.atan2(xv[1], xv[0]))
        points.append((lat, lon))
    return points


def nearest_baseline_removed_value(data: pd.DataFrame, target_hour: float, baseline: float) -> tuple[float, float]:
    idx = int(np.argmin(np.abs(data["hour"].to_numpy(dtype=float) - target_hour)))
    return float(data["hour"].iloc[idx]), float(data["value"].iloc[idx] - baseline)


def figure09() -> list[Path]:
    """Figure 09: four auroral oval snapshots from the AL-driven Starkov model."""
    import aacgmv2
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    from cartopy.feature.nightshade import Nightshade
    from matplotlib.path import Path as MplPath

    prepare_environment(font_size=9)

    data_dir = TAT1_CODE_DIR / "tat1_data_extract" / "scaled"
    au = read_digitized_hour_series(data_dir / "AU.csv")
    al = read_digitized_hour_series(data_dir / "AL.csv")
    au0 = float(au["value"].iloc[:3].mean())
    al0 = float(al["value"].iloc[:3].mean())
    al_deviation = al["value"] - al0
    peak_al_value = float(al_deviation.min())

    snapshot_specs = [
        ("Pre-storm", 0.51),
        ("SSC onset", 1.42),
        ("Peak", 2.042),
        ("Recovery", 4.05),
    ]
    mlt_grid = np.linspace(0, 24, 361)
    cable_west = (48.15, -54.13)
    cable_east = (56.40, -5.47)
    route_geo = slerp_route(cable_west, cable_east, n_points=35)
    route_lats = np.array([point[0] for point in route_geo])
    route_lons = np.array([point[1] for point in route_geo])

    def boundary_to_geo(colatitude: np.ndarray, timestamp: dt.datetime) -> tuple[np.ndarray, np.ndarray]:
        aacgm_lat = 90 - np.asarray(colatitude, dtype=float)
        aacgm_lon = np.asarray(aacgmv2.convert_mlt(mlt_grid, timestamp, m2a=True), dtype=float)
        geo_lat, geo_lon, _ = aacgmv2.convert_latlon_arr(
            aacgm_lat,
            aacgm_lon,
            np.zeros_like(aacgm_lat),
            timestamp,
            method_code="A2G",
        )
        mask = np.isfinite(geo_lat) & np.isfinite(geo_lon)
        return np.asarray(geo_lon)[mask], np.asarray(geo_lat)[mask]

    def split_geographic_path(lon: np.ndarray, lat: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
        lon = np.asarray(lon, dtype=float)
        lat = np.asarray(lat, dtype=float)
        valid = np.isfinite(lon) & np.isfinite(lat) & (lat >= 30) & (lat <= 90)
        lon = lon[valid]
        lat = lat[valid]
        if len(lon) < 2:
            return []

        jumps = np.where(np.abs(np.diff(lon)) > 120)[0] + 1
        segments = []
        for segment_lon, segment_lat in zip(np.split(lon, jumps), np.split(lat, jumps), strict=True):
            if len(segment_lon) >= 2:
                segments.append((segment_lon, segment_lat))
        return segments

    snapshots: list[dict[str, object]] = []
    for label, target_hour in snapshot_specs:
        _, au_value = nearest_baseline_removed_value(au, target_hour, au0)
        hour, al_value = nearest_baseline_removed_value(al, target_hour, al0)
        display_hour = target_hour if label == "Peak" else hour
        model_al_value = peak_al_value if label == "Peak" else al_value
        hh = int(display_hour)
        mm = int(round((display_hour - hh) * 60))
        if mm == 60:
            hh += 1
            mm = 0
        timestamp = dt.datetime(1958, 2, 11, hh, mm)

        poleward = starkov_oval_boundary("poleward", model_al_value, mlt_grid)
        equatorward = starkov_oval_boundary("equatorward", model_al_value, mlt_grid)
        diffuse = starkov_oval_boundary("diffuse", model_al_value, mlt_grid)

        snapshots.append(
            {
                "label": label,
                "timestamp": timestamp,
                "hour": display_hour,
                "AU": au_value,
                "AL": model_al_value,
                "poleward_geo": boundary_to_geo(poleward, timestamp),
                "equatorward_geo": boundary_to_geo(equatorward, timestamp),
                "diffuse_geo": boundary_to_geo(diffuse, timestamp),
            }
        )

    colors = {
        "deep_blue": "#005AB5",
        "red": "#D55E00",
        "gold": "#F0E442",
        "grey": "0.45",
        "light_grey": "0.82",
    }
    boundary_styles = {
        "poleward": dict(color=colors["deep_blue"], ls="-", lw=1.5, label="Poleward boundary"),
        "equatorward": dict(color=colors["deep_blue"], ls="--", lw=1.5, label="Equatorward boundary"),
        "diffuse": dict(color=colors["grey"], ls=":", lw=1.2, label="Diffuse boundary"),
    }

    projection = ccrs.NorthPolarStereo(central_longitude=0)
    data_crs = ccrs.PlateCarree()
    fig, axes = plt.subplots(
        2,
        2,
        subplot_kw={"projection": projection},
        figsize=(7.4, 5.2),
        dpi=300,
    )
    fig.subplots_adjust(wspace=-0.28, hspace=0.34, bottom=0.15, top=0.93)
    circle_theta = np.linspace(0, 2 * np.pi, 160)
    circle = MplPath(
        np.column_stack(
            [0.5 + 0.5 * np.sin(circle_theta), 0.5 + 0.5 * np.cos(circle_theta)]
        )
    )

    for ax, snapshot, label in zip(axes.flat, snapshots, ["(a)", "(b)", "(c)", "(d)"], strict=True):
        ax.set_extent([-180, 180, 30, 90], crs=data_crs)
        ax.set_boundary(circle, transform=ax.transAxes)
        ax.add_feature(cfeature.OCEAN.with_scale("110m"), facecolor="#EEF4FA", zorder=0)
        ax.add_feature(cfeature.LAND.with_scale("110m"), facecolor="#F3F0E8", edgecolor="none", zorder=0)
        ax.add_feature(Nightshade(snapshot["timestamp"], alpha=0.24), zorder=1)
        ax.coastlines(resolution="110m", linewidth=0.45, color="0.35", zorder=2)
        gridlines = ax.gridlines(
            crs=data_crs,
            draw_labels=False,
            linewidth=0.35,
            color=colors["light_grey"],
            alpha=0.75,
            linestyle="-",
            zorder=2,
        )
        gridlines.xlocator = mticker.FixedLocator(np.arange(-180, 181, 30))
        gridlines.ylocator = mticker.FixedLocator(np.arange(30, 91, 10))

        pole_segments = split_geographic_path(*snapshot["poleward_geo"])
        eq_segments = split_geographic_path(*snapshot["equatorward_geo"])
        for (pole_lon, pole_lat), (eq_lon, eq_lat) in zip(pole_segments, eq_segments):
            if len(pole_lon) != len(eq_lon):
                continue
            ax.fill(
                np.concatenate([pole_lon, eq_lon[::-1]]),
                np.concatenate([pole_lat, eq_lat[::-1]]),
                color=colors["gold"],
                alpha=0.24,
                linewidth=0,
                transform=data_crs,
                zorder=3,
            )
        for boundary in ("diffuse", "equatorward", "poleward"):
            style = dict(boundary_styles[boundary])
            boundary_label = style.pop("label")
            lon, lat = snapshot[f"{boundary}_geo"]
            for i, (segment_lon, segment_lat) in enumerate(split_geographic_path(lon, lat)):
                ax.plot(
                    segment_lon,
                    segment_lat,
                    transform=data_crs,
                    zorder=4,
                    label=boundary_label if i == 0 else None,
                    **style,
                )

        ax.plot(
            route_lons,
            route_lats,
            color=colors["red"],
            lw=2.2,
            transform=data_crs,
            zorder=6,
            label="TAT-1 route",
        )
        ax.scatter(
            [cable_west[1]],
            [cable_west[0]],
            marker="s",
            s=24,
            color=colors["red"],
            transform=data_crs,
            zorder=7,
        )
        ax.scatter(
            [cable_east[1]],
            [cable_east[0]],
            marker="o",
            s=24,
            color=colors["red"],
            transform=data_crs,
            zorder=7,
        )
        hh = int(snapshot["hour"])
        mm = int(round((snapshot["hour"] - hh) * 60))
        if mm == 60:
            hh += 1
            mm = 0
        ax.set_title(
            f"{label} {snapshot['label']}\n{hh:02d}:{mm:02d} UT, AL={snapshot['AL']:.0f} nT",
            fontsize=8.5,
            fontweight="bold",
            pad=5,
        )

    handles, labels = axes.flat[0].get_legend_handles_labels()
    legend_items = dict(zip(labels, handles))
    fig.legend(
        legend_items.values(),
        legend_items.keys(),
        loc="lower center",
        ncol=2,
        frameon=False,
        fontsize=8,
        bbox_to_anchor=(0.5, 0.025),
    )

    output = TAT1_PNG_DIR / "figure09.png"
    fig.savefig(output, dpi=300, facecolor="white")
    plt.close(fig)
    return [output]


def figureS01() -> list[Path]:
    """Figure S01: digitized TAT1 trace overlay."""
    prepare_environment(font_size=9)

    data_dir = TAT1_CODE_DIR / "tat1_data_extract"
    config_path = data_dir / "tat1.json"
    with config_path.open() as fp:
        config = json.load(fp)

    image_path = data_dir / config["image"]
    image = plt.imread(image_path)
    height, width = image.shape[0], image.shape[1]

    fig = plt.figure(figsize=(width / 600.0, height / 600.0), dpi=600)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(image, extent=(0, width, height, 0), origin="upper", aspect="auto")

    for panel in config["panels"]:
        trace = read_digitized_hour_series(data_dir / panel["csv"])
        x = trace["hour"].to_numpy(dtype=float)
        y = trace["value"].to_numpy(dtype=float)
        x_min = float(np.nanmin(x))
        x_max = float(np.nanmax(x))
        if x_max == x_min:
            raise ValueError(f"{panel['name']} has zero x-range.")

        x_left = float(panel["x_left"])
        x_right = float(panel["x_right"])
        x_pixels = x_left + (x - x_min) / (x_max - x_min) * (x_right - x_left)

        center = float(np.nanmedian(y))
        q05, q95 = np.nanquantile(y, [0.05, 0.95])
        spread = max(center - q05, q95 - center, float(np.nanstd(y)), 1e-9)
        scale = 0.88 * float(panel["half_height"]) / spread
        target_center = float(panel["y_center"]) + float(panel.get("bias", 0.0)) * float(
            panel["half_height"]
        )
        y_pixels = target_center - (y - center) * scale

        ax.plot(
            x_pixels,
            y_pixels,
            color=panel["color"],
            lw=1.2,
            alpha=0.92,
            zorder=3,
            solid_joinstyle="round",
            solid_capstyle="round",
        )

    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.axis("off")

    output = TAT1_PNG_DIR / "figureS01.png"
    fig.savefig(output, dpi=600, bbox_inches=None, pad_inches=0)
    plt.close(fig)
    return [output]


def figureS03() -> list[Path]:
    """Figure S03: top 1958 Dst/AE/ap panel from Dst_StackPlots.png."""
    prepare_environment(font_size=9)

    dst = read_dst_data(1958)
    ae = read_ae_data(1958)
    _, ap = read_kpap_txt(1958)

    fig, ax = plt.subplots(1, 1, figsize=(7.2, 2.6), dpi=300)
    ax.plot(dst["DATETIME"], dst["DST"], color="blue", linewidth=1.2)
    ax.set_xlim(dt.datetime(1958, 2, 6), dt.datetime(1958, 2, 18))
    ax.set_ylim(-700, 100)
    ax.set_ylabel("Dst (nT)", color="blue")
    ax.set_xlabel("Day of Month (Feb 1958)")
    ax.tick_params(axis="y", labelcolor="blue")
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d"))
    ax.xaxis.set_minor_locator(mdates.HourLocator(interval=6))
    ax.yaxis.set_major_locator(MultipleLocator(200))
    ax.axhline(0, color="black", linestyle="--", linewidth=0.7)
    ax.hlines(
        dst["DST"].min(),
        dt.datetime(1958, 2, 11, 4),
        dt.datetime(1958, 2, 11, 20),
        color="red",
        linestyle=":",
        linewidth=1.4,
    )
    ax.vlines(
        dt.datetime(1958, 2, 11, 1),
        -200,
        100,
        color="m",
        linestyle=":",
        linewidth=1.4,
    )
    ax.text(
        dt.datetime(1958, 2, 11, 2),
        72,
        "1 UT (2/11)",
        ha="left",
        va="top",
        color="m",
    )
    ax.text(
        0.95,
        0.25,
        f"Dst$_m$={dst['DST'].min():.0f} nT",
        ha="right",
        va="bottom",
        transform=ax.transAxes,
    )
    ax.set_title("February 1958")

    ae_ax = ax.twinx()
    ae_ax.plot(ae["DATETIME"], ae["AE"], color="orange", linewidth=1.0)
    ae_ax.set_ylabel("AE (nT)", color="orange")
    ae_ax.tick_params(axis="y", labelcolor="orange")
    ae_ax.set_ylim(0, 3000)
    ae_ax.yaxis.set_major_locator(MultipleLocator(500))

    ap_ax = ax.twinx()
    ap_ax.spines["right"].set_position(("axes", 1.10))
    ap_ax.step(ap["date"], ap["Ap"], color="red", linewidth=1.0, where="post")
    ap_ax.set_ylabel("ap (nT)", color="red")
    ap_ax.tick_params(axis="y", labelcolor="red")
    ap_ax.set_ylim(0, 400)
    ap_ax.yaxis.set_major_locator(MultipleLocator(100))

    output = TAT1_PNG_DIR / "figureS03.png"
    fig.savefig(output, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return [output]

FIGURES: dict[str, FigureSpec] = {
    "figure01": FigureSpec("figure01", "main", figure01),
    "figure02": FigureSpec(
        "figure02",
        "main",
        figure02,
        "TAT-1-only GEBCO map over TAT-1 bathymetry profile",
    ),
    "figure03": FigureSpec(
        "figure03",
        "main",
        figure03,
        "Generates TAT-1 1958 segment conductivity profiles",
    ),
    "figure04": FigureSpec(
        "figure04",
        "main",
        figure04,
        "ESK, BYRD, Halley Bay, Sitka H-component stack plot with AU/AL",
    ),
    "figure05": FigureSpec(
        "figure05",
        "main",
        figure05,
        "GEBCO/electrojet map over TAT-1 declination geoplot",
    ),
    "figure06": FigureSpec(
        "figure06",
        "main",
        figure06,
        "Recreates 1958.B_and_E.png from plot_b_e_combined.py",
    ),
    "figure07": FigureSpec(
        "figure07",
        "main",
        figure07,
        "Recreates 1958.Scubas.Compare.pdf voltage comparison",
    ),
    "figure08": FigureSpec(
        "figure08",
        "main",
        figure08,
        "Invokes plot_figure6.m and copies earth_fieldlines_nightside_cropped.png",
    ),
    "figure09": FigureSpec(
        "figure09",
        "main",
        figure09,
        "Builds a 2x2 AL-driven auroral oval snapshot figure from build_oval.py logic",
    ),
    "figureS01": FigureSpec(
        "figureS01",
        "supplement",
        figureS01,
        "Regenerates TAT1 digitized trace overlay",
    ),
    "figureS03": FigureSpec(
        "fifureS03",
        "supplement",
        figureS03,
        "Top 1958 Dst/AE/ap panel from Dst_StackPlots.png",
    ),
}


def list_figures() -> None:
    """Print manuscript figure slots and known source candidates."""
    print("TAT1 manuscript figure slots:")
    for spec in FIGURES.values():
        print(f"  {spec.figure_id:9s} {spec.kind:11s} {spec.description}")

    print("\nKnown source candidates already found under py/tat1:")
    for key, value in SOURCE_CANDIDATES.items():
        outputs = ", ".join(value["outputs"])
        print(f"  {key:22s} {value['script']} -> {outputs}")

    print(f"\nPNG output directory: {TAT1_PNG_DIR}")


def run_figures(figure_ids: list[str], *, font_size: int = 12) -> list[Path]:
    """Run selected assigned figure builders."""
    prepare_environment(font_size=font_size)
    written: list[Path] = []
    for figure_id in figure_ids:
        if figure_id not in FIGURES:
            raise KeyError(f"Unknown figure id: {figure_id}")
        written.extend(FIGURES[figure_id].builder())
    return written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build TAT1 paper figures after figure slots are assigned."
    )
    parser.add_argument(
        "figures",
        nargs="*",
        help="Figure IDs to build, e.g. figure01 figureS01. Defaults to all.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List manuscript slots and known source candidates.",
    )
    parser.add_argument(
        "--font-size",
        type=int,
        default=12,
        help="Base font size passed to the shared TAT1 SciencePlots style.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.list:
        list_figures()
        return

    figure_ids = args.figures or list(FIGURES)
    written = run_figures(figure_ids, font_size=args.font_size)
    for path in written:
        print(path)


if __name__ == "__main__":
    main()
