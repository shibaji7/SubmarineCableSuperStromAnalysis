import datetime as dt
import sys

import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from loguru import logger
from scubas.cables import Cable, TransmissionLine

sys.path.append("py/")

from utils import StackPlots, clean_B_fields, get_cable_informations


class SCUBASModel(object):
    def __init__(
        self,
        cable_name="TAT-8",
        cable_structure=get_cable_informations(),
        segment_files=[],
    ):
        self.cable_name = cable_name
        self.cable_structure = cable_structure
        self.segment_files = segment_files
        logger.info(f"Initialize {cable_name}")
        return

    def read_stations(
        self,
        stns=["FRD", "STJ", "HAD"],
        stn_files=[
            ["dataset/May2024/frd20240510psec.sec.txt"],
            ["dataset/May2024/had20240510psec.sec.txt"],
            ["dataset/May2024/stj20240510psec.sec.txt"],
        ],
        clean=True,
    ):
        if clean:
            self.frames = clean_B_fields(stns, stn_files)
        else:
            frames = {}
            for stn, files in zip(stns, stn_files):
                frames[stn] = pd.concat(
                    [pd.read_csv(f, parse_dates=["Date"]) for f in files]
                )
                frames[stn] = frames[stn].set_index("Date")
        return

    def initialize_TL(self):
        self.tlines = []
        for i, seg in enumerate(self.cable_structure.cable_seg):
            self.tlines.append(
                TransmissionLine(
                    sec_id=seg["sec_id"],
                    directed_length=dict(
                        edge_locations=dict(initial=seg["initial"], final=seg["final"])
                    ),
                    elec_params=dict(
                        site=seg["site"],
                        width=seg["width"],
                        flim=seg["flim"],
                    ),
                    active_termination=seg["active_termination"],
                ).compile_oml(self.segment_files[i]),
            )
        return

    def run_cable_segment(self):
        # Running the cable operation
        logger.info(f"Components: {self.tlines[0].components}")
        self.cable = Cable(self.tlines, self.tlines[0].components)
        return

    def plot_TS_with_others(
        self,
        fname,
        date_lim=[],
        fig_title="",
        vlines=[],
        major_locator=mdates.MinuteLocator(byminute=range(0, 60, 5)),
        minor_locator=mdates.MinuteLocator(byminute=range(0, 60, 1)),
        text_size=15,
        ylim=[-3000, 3000],
        interval=2,
    ):
        sp = StackPlots(
            nrows=1, ncols=1, datetime=True, figsize=(6, 4), text_size=text_size
        )
        _, ax = sp.plot_stack_plots(
            self.cable.tot_params.index,
            self.cable.tot_params["Vt(v)"],
            ylim=ylim,
            xlim=date_lim,
            title=fig_title,
            xlabel="Time, UT",
            ylabel="Cable Voltage, V",
            interval=interval,
        )
        sp.save_fig(fname)
        sp.close()
        return

    def plot_profiles(
        self,
        fname,
        text_size=10,
        nrows=4,
        ncols=2,
        figsize=(3, 3),
        ayticks=[1e-6, 1e-3, 1e0],
        tyticks=[0, 30, 45, 60, 90],
        xticks=[1e-6, 1e-3, 1e0],
        xlim=[1e-6, 1e0],
        aylim=[1e-4, 1e2],
        tylim=[0, 90],
        tag0_loc=[0, 2, 4, 6],
        tag1_loc=[6, 5],
        tag2_loc=[1, 3, 5],
    ):
        sp = StackPlots(
            nrows=nrows,
            ncols=ncols,
            datetime=False,
            figsize=figsize,
            text_size=text_size,
        )
        for i, tl, seg in zip(
            range(len(self.tlines)), self.tlines, self.cable_structure.cable_seg
        ):
            id = seg["sec_id"]
            tf = tl.model.get_TFs()
            ax, tax = sp.axes[i], sp.axes[i].twinx()
            ax.set_xticks(xticks)
            ax.set_yticks(ayticks)
            tax.set_yticks(tyticks)
            if i in tag0_loc:
                ax.set_ylabel("Amplitude, mV/km/nT", color="r")
                tax.tick_params(axis="y", labelright=False)
            if i in tag1_loc:
                ax.set_xlabel("Frequency, Hz")
            if i in tag2_loc:
                ax.tick_params(axis="y", labelleft=False)
                tax.set_ylabel("Phase, deg", color="b")
            ax.loglog(
                tf.freq,
                np.abs(tf.E2B),
                "r",
                lw=1.0,
                ls="-",
            )
            ax.set_ylim(aylim)
            ax.set_xlim(xlim)
            tax.plot(
                tf.freq,
                np.angle(tf.E2B, deg=True),
                "b",
                lw=1.0,
                ls="-",
            )
            tax.set_xlim(xlim)
            tax.set_ylim(tylim)
            ax.text(0.05, 0.95, id, ha="left", va="center", transform=ax.transAxes)
        if len(self.tlines) < len(sp.axes):
            sp.axes[-1].axis("off")
        sp.fig.subplots_adjust(wspace=0.1, hspace=0.1)
        sp.save_fig(fname)
        sp.close()
        return

    def plot_e_fields(self):
        return

    def conduct_error_analysis(self, inputs):
        return
