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


import matplotlib as mpl

mpl.rc("font", size=15)
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
    ):
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
            ax.set_title(title)
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
