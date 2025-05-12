import matplotlib.pyplot as plt
import scienceplots

plt.style.use(["science", "ieee"])
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Tahoma", "DejaVu Sans", "Lucida Grande", "Verdana"]
plt.rcParams["text.usetex"] = False
import mpl_toolkits.axisartist.floating_axes as floating_axes
import numpy as np
from matplotlib.projections import polar
from matplotlib.transforms import Affine2D
from mpl_toolkits.axisartist.grid_finder import DictFormatter, FixedLocator

import pandas as pd
import matplotlib as mpl

from typing import Optional, Sequence

import matplotlib.dates as mdates


class StackPlots:

    def __init__(
        self,
        nrows: int,
        ncols: int,
        dpi: int = 1000,
        datetime: bool = False,
        polar: bool = False,
        figsize: tuple = (8, 3),
        text_size=15,
    ):
        mpl.rc("font", size=text_size)
        self.nrows = nrows
        self.ncols = ncols
        self.fig, self.axes = plt.subplots(
            self.nrows,
            self.ncols,
            figsize=(self.ncols * figsize[0], self.nrows * figsize[1]),
            sharex=True,
            dpi=dpi,
            subplot_kw={"projection": "polar"} if polar else {},
        )
        self.axes = self.axes.flatten() if self.nrows > 1 else [self.axes]
        self.fig.subplots_adjust(hspace=0.5, wspace=0.3)
        self.plot_id = 0
        self.datetime = datetime
        pass

    def plot_stack_plots(
        self,
        time: Sequence,
        value: Sequence,
        title: Optional[str] = None,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        label: Optional[str] = None,
        text: Optional[str] = None,
        ylim: Optional[Sequence] = None,
        xlim: Optional[Sequence] = None,
        color: str = "blue",
        lw: float = 0.8,
        ls: str = "-",
        ax: Optional[plt.Axes] = None,
        ylabel_color: Optional[str] = "k",
    ) -> tuple:
        """
        Plot a stack of plots with the given time and value data.
        :param time: Time data for the x-axis
        :param value: Value data for the y-axis
        :param title: Title for the plot
        :param xlabel: Label for the x-axis
        :param ylabel: Label for the y-axis
        """
        if self.plot_id > len(self.axes):
            raise ValueError("No more axes available for plotting.")
        if ax is None:
            ax = self.axes[self.plot_id]
            self.plot_id += 1
        if title:
            ax.set_title(title, fontdict=dict(size=12))
        if xlabel:
            ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel, color=ylabel_color)
        if ylim:
            ax.set_ylim(ylim)
        else:
            ax.set_ylim([min(value), max(value)])
        if xlim:
            ax.set_xlim(xlim)
        else:
            ax.set_xlim([time[0], time[-1]])
        if self.datetime:
            ax.xaxis.set_major_locator(mdates.HourLocator(interval=3))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%H"))
        ax.plot(time, value, color=color, linewidth=lw, ls=ls, label=label)
        if text:
            ax.text(0.05, 1.05, text, ha="left", va="center", transform=ax.transAxes)

        plt.tight_layout()
        return self.fig, ax

    def save_fig(self, filename: str):
        """
        Save the figure to a file.
        :param filename: Filename to save the figure
        """
        self.fig.savefig(filename, bbox_inches="tight")
        return

    def close(self):
        """
        Close the figure.
        """
        plt.close(self.fig)
        return

    def plot_dirctional_plots(
        self,
        theta: Sequence,
        r: Sequence,
        title: Optional[str] = None,
        text: Optional[str] = None,
        color: str = "black",
        ax: Optional[plt.Axes] = None,
        rlims: Optional[Sequence] = [0, 1],
        rticks: Optional[Sequence] = [0, 0.5, 1.0],
        theta_ticks: Optional[Sequence] = [0, np.pi / 2, np.pi, 3 * np.pi / 2],
        cable_angle: Optional[float] = None,
    ):
        """
        Plot directional plots.
        """
        if self.plot_id > len(self.axes):
            raise ValueError("No more axes available for plotting.")
        if ax is None:
            ax = self.axes[self.plot_id]
            ax.set_theta_zero_location("N")
            ax.set_theta_direction(-1)
            self.plot_id += 1
            if cable_angle is not None:
                ax.plot(
                    np.deg2rad([cable_angle, cable_angle]),
                    [0, 1],
                    lw=1.2,
                    ls="-",
                    color="m",
                )
        if title:
            ax.set_title(title, fontdict=dict(size=12))

        ax.bar(
            np.deg2rad(theta),
            r,
            bottom=0.0,
            color=color,
            width=np.deg2rad(theta[1] - theta[0]),
            alpha=0.5,
        )
        ax.set_rticks(rticks)
        ax.set_xticks(theta_ticks)
        ax.set_rmax(rlims[1])
        ax.set_rmin(rlims[0])
        if text:
            ax.text(
                -0.1,
                1.05,
                text,
                ha="left",
                va="center",
                transform=ax.transAxes,
                color=color,
            )
        return self.fig, ax

def clean_B_fields(stns, stn_files):
    frames = dict()
    for stn, fs in zip(stns, stn_files):
        o = pd.DataFrame()
        o = pd.concat([o, read_Bfield_data(fs)])
        # Remove Datagaps
        print("Pre-Is there any issue / nan data? (X,Y,Z)", o.X.hasnans, o.Y.hasnans, o.Z.hasnans)
        o = o.replace(99999.0, np.nan)
        for key in ["X", "Y", "Z"]:                    
            o[key] = o[key].interpolate(method="from_derivatives")
            o[key] = o[key].ffill()
            o[key] = o[key].bfill()
        print("Post-Is there any issue / nan data? (X,Y,Z)", o.X.hasnans, o.Y.hasnans, o.Z.hasnans)
        fs[0] = fs[0].replace(".txt", ".csv")
        o.to_csv(fs[0], header=True, index=True, float_format="%g")
        frames[stn] = o
    return frames


from types import SimpleNamespace
from scubas.datasets import PROFILES


def create_from_lat_lon(
        dSegments, profiles, 
        width=1.0, flim=[1e-6, 1e0], 
        left_active_termination=None,
        right_active_termination=None,
    ):
    cable_seg = []
    for i in range(len(dSegments)-1):
        initial = dSegments[i]
        final = dSegments[i+1]
        cable_seg.append(
            dict(
                initial=dict(lat=initial[0], lon=initial[1]), 
                final=dict(lat=final[0], lon=final[1]),
                sec_id=f"Sec-{i}",
                site=profiles[0],
                active_termination=dict(
                    right=None,
                    left=None,
                ),
            )
        )
    if left_active_termination:
        cable_seg[0]["active_termination"] = dict(
            right=None,
            left=left_active_termination,
        )
    if right_active_termination:
        cable_seg[-1]["active_termination"] = dict(
            right=right_active_termination,
            left=None,
        )
    cable = SimpleNamespace(**dict(
        cable_seg = cable_seg
    ))
    for seg in cable.cable_seg:
        seg["center"] = dict(
            lat=0.5*(seg["initial"]["lat"]+seg["final"]["lat"]),
            lon=0.5*(seg["initial"]["lon"]+seg["final"]["lon"]),
        )
        seg["width"], seg["flim"] = width, flim
    return cable


def get_cable_informations(kind="TAT-8", width=1.0, flim=[1e-6, 1e0]):
    if kind == "TAT-8":
        land50 = PROFILES.CS_E
        land50.layers[0].thickness = 50
        cable = SimpleNamespace(**dict(
            cable_seg = [
                dict(
                    initial=dict(lat=39.6, lon=-74.33), 
                    final=dict(lat=38.79, lon=-72.62),
                    sec_id="CS-W",
                    site=PROFILES.CS_W,
                    active_termination=dict(
                        right=None,
                        left=PROFILES.LD,
                    ),
                ),
                dict(
                    initial=dict(lat=38.79, lon=-72.62), 
                    final=dict(lat=37.11, lon=-68.94),
                    sec_id="DO-1",
                    site=PROFILES.DO_1,
                    active_termination=dict(
                        right=None,
                        left=None,
                    ),
                ),
                dict(
                    initial=dict(lat=37.11, lon=-68.94), 
                    final=dict(lat=39.80, lon=-48.20),
                    sec_id="DO-2",
                    site=PROFILES.DO_2,
                    active_termination=dict(
                        right=None,
                        left=None,
                    ),
                ),
                dict(
                    initial=dict(lat=39.80, lon=-48.20), 
                    final=dict(lat=40.81, lon=-45.19),
                    sec_id="DO-3",
                    site=PROFILES.DO_3,
                    active_termination=dict(
                        right=None,
                        left=None,
                    ),
                ),
                dict(
                    initial=dict(lat=40.81, lon=-45.19), 
                    final=dict(lat=43.15, lon=-39.16),
                    sec_id="DO-4",
                    site=PROFILES.DO_4,
                    active_termination=dict(
                        right=None,
                        left=None,
                    ),
                ),
                dict(
                    initial=dict(lat=43.15, lon=-39.16), 
                    final=dict(lat=44.83, lon=-34.48),
                    sec_id="DO-5",
                    site=PROFILES.DO_5,
                    active_termination=dict(
                        right=None,
                        left=None,
                    ),
                ),
                dict(
                    initial=dict(lat=44.83, lon=-34.48), 
                    final=dict(lat=46.51, lon=-22.43),
                    sec_id="MAR",
                    site=PROFILES.MAR,
                    active_termination=dict(
                        right=None,
                        left=None,
                    ),
                ),
                dict(
                    initial=dict(lat=46.51, lon=-22.43), 
                    final=dict(lat=47.85, lon=-9.05),
                    sec_id="DO-6",
                    site=PROFILES.DO_6,
                    active_termination=dict(
                        right=None,
                        left=None,
                    ),
                ),
                dict(
                    initial=dict(lat=47.85, lon=-9.05), 
                    final=dict(lat=50.79, lon=-4.55),
                    sec_id="CS-E",
                    site=PROFILES.CS_E,
                    active_termination=dict(
                        right=land50,
                        left=None,
                    ),
                ),
            ]
        ))
    for seg in cable.cable_seg:
        seg["center"] = dict(
            lat=0.5*(seg["initial"]["lat"]+seg["final"]["lat"]),
            lon=0.5*(seg["initial"]["lon"]+seg["final"]["lon"]),
        )
        seg["width"], seg["flim"] = width, flim
    return cable

import os
import pyspedas
from loguru import logger
import pandas as pd
import datetime as dt
import numpy as np
os.environ["OMNIDATA_PATH"] = "/home/chakras4/OMNI/"

def _load_omni_(dates, res=1):
    import pyomnidata
    logger.info(f"OMNIDATA_PATH: {os.environ['OMNIDATA_PATH']}")
    pyomnidata.UpdateLocalData()
    omni = pd.DataFrame(
        pyomnidata.GetOMNI(dates[0].year,Res=res)
    )
    omni["time"] = omni.apply(
        lambda r: (
            dt.datetime(
                int(str(r.Date)[:4]), 
                int(str(r.Date)[4:6]),
                int(str(r.Date)[6:].replace(".0","")) 
            ) 
            + dt.timedelta(hours=r.ut)
        ), 
        axis=1
    )
    omni = omni[
        (omni.time>=dates[0])
        & (omni.time<=dates[1])
    ]
    return omni

def load_speadas(dates, probe="c"):
    time_range = [
        dates[0].strftime("%Y-%m-%d/%H:%M"),
        dates[1].strftime("%Y-%m-%d/%H:%M")
    ]
    data_fgm = pyspedas.themis.fgm(
        probe=probe, trange=time_range, 
        time_clip=True, no_update=False,notplot=True
    )
    data_mom = pyspedas.themis.mom(
        probe=probe, trange=time_range,
        notplot=True,no_update=False,time_clip=True
    )
    pdyn = {
        "x": data_mom["thc_peem_density"]["x"], 
        "y": data_mom["thc_peem_density"]["y"]*1.67*(10**(-6))*0.5*np.nansum(
            data_mom["thc_peim_velocity_gse"]["y"]**2, axis=1
        )
    }
    data_mom["pdyn"] = pdyn
    return data_fgm, data_mom


def read_iaga(file, return_xyzf=True, return_header=False):
    """
    Read IAGA profiles
    """

    # Read Headers
    header_records = {"header_length": 0}

    with open(file, "r") as openfile:
        for newline in openfile:
            if newline[0] == " ":
                header_records["header_length"] += 1
                label = newline[1:24].strip()
                description = newline[24:-2].strip()
                header_records[label.lower()] = description

    if len(header_records["reported"]) % 4 != 0:
        raise ValueError(
            "The header record does not contain 4 values: {0}".format(
                header_records["reported"]
            )
        )
    record_length = len(header_records["reported"]) // 4
    column_names = [
        x for x in header_records["reported"][record_length - 1 :: record_length]
    ]
    seen_count = {}
    for i, col in enumerate(column_names):
        if col in seen_count:
            column_names[i] += str(seen_count[col])
            seen_count[col] += 1
        else:
            seen_count[col] = 1
    df = pd.read_csv(
        file,
        header=header_records["header_length"],
        delim_whitespace=True,
        parse_dates=[[0, 1]],
        infer_datetime_format=True,
        index_col=0,
        usecols=[0, 1, 3, 4, 5, 6],
        na_values=[99999.90, 99999.0, 88888.80, 88888.00],
        names=["Date", "Time"] + column_names,
    )
    df.index.name = "Date"
    if return_xyzf and "X" not in column_names and "Y" not in column_names:
        # Convert the data to XYZF format
        # Only convert HD
        if "H" not in column_names or "D" not in column_names:
            raise ValueError(
                "Only have a converter for HDZF->XYZF\n"
                + "Input file is: "
                + header_records["reported"]
            )

        # IAGA-2002 D is reported in minutes of arc.
        df["X"] = df["H"] * np.cos(np.deg2rad(df["D"] / 60.0))
        df["Y"] = df["H"] * np.sin(np.deg2rad(df["D"] / 60.0))
        del df["H"], df["D"]
    if return_header:
        return df, header_records
    else:
        return df

def read_Bfield_data(files, return_xyzf=True, csv_file_date_name="Date"):
    """
    Read B-Files
    """
    Bfield = pd.DataFrame()
    for file in files:
        file_type = file.split(".")[-1]
        if file_type == "txt":
            o = read_iaga(file, return_xyzf, return_header=False)
        elif file_type == "csv":
            o = pd.read_csv(file, parse_dates=[csv_file_date_name])
            o = o.rename(columns={csv_file_date_name: "Date"})
            o = o.set_index("Date")
            o.index.name = "Date"
        Bfield = pd.concat([Bfield, o])
    return Bfield