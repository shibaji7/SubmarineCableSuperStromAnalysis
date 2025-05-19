import datetime as dt
import sys

import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from loguru import logger
from scubas.cables import Cable, TransmissionLine

sys.path.append("py/")

from utils import (
    StackPlots,
    clean_B_fields,
    get_cable_informations,
    load_omni,
    quantile_loss,
)


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

    def initialize_TL(self, tmul=1):
        self.tlines = []
        for i, seg in enumerate(self.cable_structure.cable_seg):
            site = seg["site"]
            site.layers[0].thickness *= tmul
            self.tlines.append(
                TransmissionLine(
                    sec_id=seg["sec_id"],
                    directed_length=dict(
                        edge_locations=dict(initial=seg["initial"], final=seg["final"])
                    ),
                    elec_params=dict(
                        site=site,
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
        t_mul=1.,
    ):
        if t_mul == 1.:
            self.initialize_TL(1e3)
            t_mul = 1e-3
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
            id, water_depth = seg["sec_id"], seg["site"].layers[0].thickness
            # print(water_depth)
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
            ax.text(
                0.05,
                0.95,
                rf"{id}, $\tau_w$=%.2f km" % (water_depth*t_mul),
                ha="left",
                va="center",
                transform=ax.transAxes,
            )
        if len(self.tlines) < len(sp.axes):
            sp.axes[-1].axis("off")
        sp.fig.subplots_adjust(wspace=0.1, hspace=0.1)
        sp.save_fig(fname)
        sp.close()
        return

    def plot_e_fields(
        self,
        fname,
        text_size=10,
        nrows=3,
        ncols=1,
        figsize=(6, 3),
        ylim=[-100, 100],
        component="X",
        groups=[],
        date_lim=[],
        fig_title="",
        vlines=[],
        interval=2,
    ):
        sp = StackPlots(
            nrows=nrows,
            ncols=ncols,
            datetime=True,
            figsize=figsize,
            text_size=text_size,
        )
        cols = list(self.cable.tot_params.columns)
        efields = self.cable.tot_params[
            [f"E.{component}.%02d" % l for l in range(len(self.tlines))]
        ]
        for i, group in enumerate(groups):
            colors = ["r", "k", "b"]
            for j, g in enumerate(group):
                name = self.cable_structure.cable_seg[g]["sec_id"]
                if j == 0:
                    _, ax = sp.plot_stack_plots(
                        efields.index,
                        efields[f"E.{component}.%02d" % g],
                        ylim=ylim,
                        xlim=date_lim,
                        title=fig_title if i == 0 else "",
                        interval=interval,
                        color=colors[j],
                        label=rf"$E_x({name})$",
                        ylabel="E-field, mv/km",
                        xlabel="Time, UT" if i == len(sp.axes) - 1 else "",
                    )
                else:
                    sp.plot_stack_plots(
                        efields.index,
                        efields[f"E.{component}.%02d" % g],
                        ylim=ylim,
                        xlim=date_lim,
                        interval=interval,
                        ax=ax,
                        color=colors[j],
                        label=rf"$E_x({name})$",
                    )
            ax.legend(loc=2, fontsize=text_size - 3)
        sp.fig.subplots_adjust(wspace=0.1, hspace=0.1)
        sp.save_fig(fname)
        sp.close()
        return

    def plot_zoomedin_analysis(self, fname, inputs, ylim=[], date_lims=[], interval=15, mult=-1):
        # All in pyforcast tools or SCORES
        sp = StackPlots(
            nrows=1,
            ncols=1,
            datetime=True,
            figsize=(6, 2.5),
            text_size=12,
        )
        _, ax = sp.plot_stack_scatter(
            inputs.Time,
            inputs.Voltage,
            color="k",
            label=rf"Observations",
            ylim=ylim,
            ms=2,
            interval=interval,
            xlim=date_lims,
            xlabel="Time, UT",
            ylabel="Voltage, V",
        )
        sp.plot_stack_scatter(
            self.cable.tot_params.index,
            mult*self.cable.tot_params["Vt(v)"],
            color="r",
            ms=2,
            label=rf"SCUBAS",
            ylim=ylim,
            interval=interval,
            xlim=date_lims,
            ax=ax,
        )
        ax.legend(loc=2)
        sp.fig.subplots_adjust(wspace=0.1, hspace=0.1)
        sp.save_fig(fname)
        sp.close()
        return

    def run_detailed_error_analysis(
        self, inputs, date_lims=[], 
        fnames=[
            "figures/1989/1989.Error.qq.png",
            "figures/1989/1989.Scores.png",
        ]
    ):
        omni = load_omni(date_lims)
        omni = omni[(omni.time >= date_lims[0]) & (omni.time <= date_lims[1])]
        # Case special
        x = np.array(inputs.Voltage)
        o = self.cable.tot_params.copy()
        o = o[
            (o.index >= date_lims[0] - dt.timedelta(minutes=10))
            & (o.index <= date_lims[1] + dt.timedelta(minutes=10))
        ]["Vt(v)"]
        dT = np.array((o.index - o.index[0]).total_seconds())
        inputs["newdT"] = inputs.Time.apply(lambda j: (j - o.index[0]).total_seconds())
        y = np.interp(inputs.newdT, dT, -np.array(o))
        e = y - x  # Error Pred - Obs

        # Omni interpolation
        dOmniTime = omni.time.apply(lambda j: (j - o.index[0]).total_seconds())
        symhNew = np.interp(inputs.newdT, dOmniTime, omni.SymH)

        sp = StackPlots(nrows=2, ncols=2, figsize=(4, 2.5), sharex=False, text_size=12)
        ax = sp.axes[0]
        ax.hist(e, 50, color="b", histtype="step")
        ax.set_xlabel("Error, V", fontsize=12)
        ax.set_ylabel("Counts", fontsize=12)
        ax.tick_params(axis="x", labelsize=12)
        ax.set_xlim(-400, 400)
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
        ax.set_xlim([-400, 400])
        ax.set_ylim([-400, 400])
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
        ax.set_xlim(-400, 400)
        ax.set_ylim(-400, 400)
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

        ax = sp.axes[3]
        ax.scatter(
            symhNew,
            e,
            c="b",
            marker="s",
            s=4,
        )
        ax.set_xlabel("SymH, nT", fontsize=12)
        ax.set_xlim(-800, 0)
        ax.set_ylim(-400, 400)
        ax.axhline(0, color="k", lw=0.8, ls="--")
        ax.set_ylabel("Error, V", fontsize=12)
        ax.text(
            0.05,
            0.9,
            "(D)",
            ha="left",
            va="center",
            transform=ax.transAxes,
            fontsize=12,
        )
        # print(omni.SymH.min(), omni.SymH.max(), symhNew.min(), symhNew.max())
        sp.save_fig(fnames[0])
        sp.close()

        # Compute Scores (huber, quantile, expctile) and Isotonic fits
        from scores.processing.isoreg_impl import isotonic_fit

        sp = StackPlots(
            nrows=2, ncols=1, figsize=(4.5, 2.5), sharex=False, text_size=12
        )
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

        ax = sp.axes[0]
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
        ax_hist.set_xlim(-400, 400)
        ax_hist.set_ylim(0, 100)
        ax_hist.hist(x_sorted, color="purple", histtype="step")
        ax_hist.set_ylabel("Counts", color="purple")

        # Annotations
        ax.set_xlim(-400, 400)
        ax.set_ylim(-400, 400)
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
        ax.text(0.3, 0.9, "(A)", ha="left", va="center", transform=ax.transAxes)

        scores, quantile = [], np.arange(0, 1.01, 0.05)
        for q in quantile:
            scores.append(quantile_loss(x, y, q))
        scores = -np.log10(np.array(scores))
        ax = sp.axes[1]
        ax.scatter(quantile, scores, marker="s", s=3, color="b")
        ax.set_xlim(0, 1)
        ax.set_xlabel("Quantile")
        ax.set_ylabel("Q-Loss")
        ax.text(0.3, 0.9, "(B)", ha="left", va="center", transform=ax.transAxes)

        sp.save_fig(fnames[1])
        sp.close()
        return
