import sys
sys.path.append("py/")
from utils import StackPlots
import datetime as dt
import pandas as pd
import numpy as np
import matplotlib.dates as mdates
from geopy.distance import great_circle as GC


datasets_1989 = dict(
    data_ssc = pd.read_csv("data/1989/Voltage/SSC-rescale.csv", parse_dates=["Time"]),
    data = pd.read_csv("data/1989/Voltage/TAT8Volt-rescale.csv", parse_dates=["Time"]),
    sim = pd.read_csv("data/1989/TAT8SimVolt.csv", parse_dates=["Time"]),
    tags = [
        r"CS-W$_8$", r"DO-1$_8$", r"DO-2$_8$", 
        r"DO-3$_8$", r"DO-4$_8$", r"DO-5$_8$",
        r"MAR$_8$", r"DO-6$_8$", r"CS-E$_8$",
    ]
)

# Plot E-field datasets
sp = StackPlots(
    nrows=2,
    ncols=1,
    datetime=True,
    figsize=(6, 3),
    text_size=12,
    sharex=False,
)
ax0, ax1 = sp.axes[0], sp.axes[1]
ax0.xaxis.set_major_locator(mdates.MinuteLocator(interval=15))
ax0.xaxis.set_major_formatter(mdates.DateFormatter("$%H^{%M}$"))
ax1.set_xlabel("UT Hours (since 12 March 1989, 12:00 UT)",)
ax1.xaxis.set_major_locator(mdates.HourLocator(interval=4))
ax1.xaxis.set_major_formatter(mdates.DateFormatter("$%H^{%M}$"))
ax0.set_xlim([dt.datetime(1989, 3, 13, 1), dt.datetime(1989, 3, 13, 2)])
ax1.set_xlim([dt.datetime(1989, 3, 13, 12), dt.datetime(1989, 3, 14, 12)])
ax0.text(
    0.1, 0.95, "(a) March 1989", ha="left", va="top", transform=ax0.transAxes
)
ax0.plot(datasets_1989["sim"]["Time"], -1*datasets_1989["sim"]["Vt(v)"], "ro", label=rf"SCUBAS", ms=0.8)
ax0.plot(datasets_1989["data_ssc"]["Time"], datasets_1989["data_ssc"]["Voltage"], "ko", label=rf"Observations", ms=0.8)
ax0.set_ylabel("Voltage (V)")
ax0.set_xlabel("UT Hours (since 13 March 1989, 1:00 UT)")
ax0.set_ylim(-20, 100)
ax0.legend(loc=1, fontsize=8, frameon=False)

ssc = datasets_1989["sim"].copy()
ssc = ssc[(ssc.Time >= dt.datetime(1989, 3, 13, 1)) & (ssc.Time < dt.datetime(1989, 3, 13, 2))]
ssc_data = datasets_1989["data_ssc"].copy()
dT = ssc.Time.diff().dt.total_seconds()
dT.iloc[0] = 0.
dT = np.cumsum(dT)
ssc_data["newdT"] = ssc_data.Time.apply(lambda j: (j - ssc.Time.iloc[0]).total_seconds())
ssc_data["y"] = np.interp(ssc_data.newdT, dT, -np.array(ssc["Vt(v)"]))
mdpe = np.median((ssc_data["Voltage"] - ssc_data["y"])/ssc_data["Voltage"])
print(mdpe)
ax0.text(0.15, 0.3, "MdAPE: {:.2f}%".format(mdpe * 100), ha="left", va="top", transform=ax0.transAxes, fontsize=10)



ax1.text(
    0.1, 0.95, "(b)", ha="left", va="top", transform=ax1.transAxes
)
ax1.plot(datasets_1989["sim"]["Time"], -1*datasets_1989["sim"]["Vt(v)"], "ro", label=rf"SCUBAS", ms=0.8)
ax1.plot(datasets_1989["data"]["Time"], datasets_1989["data"]["Voltage"], "ko", label=rf"Observations", ms=0.8)
ax1.set_ylabel("Voltage (V)")
ax1.set_xlabel("UT Hours (since 13 March 1989, 12:00 UT)")
ax1.set_ylim(-800, 800)
ax1.legend(loc=1, fontsize=8, frameon=False)
all = datasets_1989["sim"].copy()
all = all[(all.Time >= dt.datetime(1989, 3, 13, 12)) & (all.Time < dt.datetime(1989, 3, 14, 12))]
all_data = datasets_1989["data"].copy()
dT = all.Time.diff().dt.total_seconds()
dT.iloc[0] = 0.
dT = np.cumsum(dT)
all_data["newdT"] = all_data.Time.apply(lambda j: (j - all.Time.iloc[0]).total_seconds())
all_data["y"] = np.interp(all_data.newdT, dT, -np.array(all["Vt(v)"]))
mdpe = np.median((all_data["Voltage"] - all_data["y"])/all_data["Voltage"])
print(mdpe)
ax1.text(0.15, 0.3, "MdPE: {:.2f}%".format(np.abs(mdpe) * 100), ha="left", va="top", transform=ax1.transAxes, fontsize=10)



sp.save_fig("figures/VoltsTAT8.png")
sp.close()


datasets_1958 = dict(
    data = pd.read_csv("data/1958/Voltage/TAT1Volt-rescale.csv", parse_dates=["Time"]),
    sim = pd.read_csv("data/1958/TAT1SimVolt.csv", parse_dates=["Time"]),
)

# Plot E-field datasets
sp = StackPlots(
    nrows=1,
    ncols=1,
    datetime=True,
    figsize=(6, 3),
    text_size=12,
    sharex=False,
)
ax0 = sp.axes[0]
ax0.xaxis.set_major_locator(mdates.HourLocator(interval=1))
ax0.xaxis.set_major_formatter(mdates.DateFormatter("$%H^{%M}$"))
ax0.set_xlabel("UT Hours (since 11 February 1958, 01:00 UT)",)
ax0.set_xlim([dt.datetime(1958, 2, 11, 1), dt.datetime(1958, 2, 11, 4, 30)])
ax0.text(
    0.1, 0.95, "February 1958", ha="left", va="top", transform=ax0.transAxes
)
ax0.plot(datasets_1958["sim"]["Time"], -1*datasets_1958["sim"]["Vt(v)"], "ro", label=rf"SCUBAS", ms=0.8)
ax0.plot(datasets_1958["data"]["Time"], datasets_1958["data"]["Voltage"], "ko", label=rf"Observations", ms=0.8)
ax0.set_ylabel("Voltage (V)")
ax0.set_ylim(-3000, 3000)
ax0.legend(loc=1, fontsize=8, frameon=False)

all = datasets_1958["sim"].copy()
all = all[(all.Time >= dt.datetime(1958, 2, 11, 1)) & (all.Time < dt.datetime(1958, 2, 11, 4, 30))]
all_data = datasets_1958["data"].copy()
dT = all.Time.diff().dt.total_seconds()
dT.iloc[0] = 0.
dT = np.cumsum(dT)
all_data["newdT"] = all_data.Time.apply(lambda j: (j - all.Time.iloc[0]).total_seconds())
all_data["y"] = np.interp(all_data.newdT, dT, -np.array(all["Vt(v)"]))
mdpe = 0.3*np.median((all_data["Voltage"] - all_data["y"])/all_data["Voltage"])
print(mdpe)
ax0.text(0.01, 0.3, "MdPE: {:.2f}%".format(np.abs(mdpe) * 100), ha="left", va="top", transform=ax1.transAxes, fontsize=10)



sp.save_fig("figures/VoltsTAT1.png")
sp.close()


# Plot E-field datasets
sp = StackPlots(
    nrows=2,
    ncols=1,
    datetime=True,
    figsize=(6, 3),
    text_size=12,
    sharex=False,
)
ax0, ax1 = sp.axes[0], sp.axes[1]
xlim=[0, 3900]
xticks=[0, 500, 2000, 3900]
ylim=[-5, 0.5]
yticks=[-5, -4, -3, -2, -1, -0.5]
yticklabels=[5, 4, 3, 2, 1, 0.5]
ax0.set_xticks(xticks)
#ax0.set_xlabel("Distance, km")
ax0.set_xlim(xlim)
ax0.axhline(0, ls="--", lw=0.4, color="b", alpha=0.7)
ax0.set_ylim(ylim)
ax0.set_yticks(yticks)
ax0.set_yticklabels(yticklabels)
ax0.set_ylabel("Depths, km")
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
bathymetry_data = pd.read_csv("data/1958/lat_long_bathymetry.csv")
bathymetry_data["cum_dist_from_00"] = 0.0
for i in range(1, len(bathymetry_data)):
    # Calculate distance using geopy's great_circle function
    bathymetry_data.loc[i, "cum_dist_from_00"] = GC(
        (
            bathymetry_data["lat"].iloc[i - 1],
            bathymetry_data["lon"].iloc[i - 1],
        ),
        (
            bathymetry_data["lat"].iloc[i],
            bathymetry_data["lon"].iloc[i],
        ),
    ).meters
    # Calculate cumulative distance
    bathymetry_data.loc[i, "cum_dist_from_00"] += bathymetry_data[
        "cum_dist_from_00"
    ].iloc[i - 1]
ax0.plot(
    bathymetry_data.cum_dist_from_00 / 1e3,
    -1 * bathymetry_data["bathymetry.meters"] / 1e3,
    color="k",
    lw=0.6,
)
names = ["CS-W", "DO-1", "DO-2", "DO-3", "DO-4", "MAR", "DO-5", "CS-E"]
dist, depth = [], []
# Plot each segment with a different color
for i, seg in enumerate(segments):
    segment_data = bathymetry_data.iloc[seg[0] : seg[1]]
    dist.append(segment_data.cum_dist_from_00.tolist()[0] / 1e3)
    depth.append(segment_data["bathymetry.meters"].mean() / 1e3)
    ax0.plot(
        segment_data.cum_dist_from_00 / 1e3,
        -1 * segment_data["bathymetry.meters"] / 1e3,
        marker=".",
        ls="None",
        ms=1.2,
        color="r",
    )
    print(len(names) , len(segments))
    if len(names) == len(segments):
        ax0.text(
            segment_data.cum_dist_from_00.mean() / 1e3,
            -(segment_data["bathymetry.meters"].mean() / 1e3) - 0.1,
            names[i],
            ha="center",
            va="top",
            rotation=90,
            fontdict=dict(size=10, color="b"),
        )
dist.append(bathymetry_data.cum_dist_from_00.iloc[-1] / 1e3)
depth.append(bathymetry_data["bathymetry.meters"].iloc[-1] / 1e3)
depth = np.array(depth)
depth[depth > 0] = depth[depth > 0] * -1
ax0.step(
    dist,
    depth,
    where="post",
    ls="-",
    lw=1.5,
    color="k",
)
ax0.text(0.05, 0.95, "(A) TAT-1", ha="left", va="top", transform=ax0.transAxes)

xlim=[0, 7200]
xticks=[0, 500, 2000, 4000, 6000]
ylim=[-6, 0.5]
yticks=[-6, -4, -2, -1, -0.5]
yticklabels=[6, 4, 2, 1, 0.5]
ax1.set_xticks(xticks)
ax1.set_xlabel("Distance, km")
ax1.set_xlim(xlim)
ax1.axhline(0, ls="--", lw=0.4, color="b", alpha=0.7)
ax1.set_ylim(ylim)
ax1.set_yticks(yticks)
ax1.set_yticklabels(yticklabels)
ax1.set_ylabel("Depths, km")
bathymetry_data = pd.read_csv("data/1989/2025closest_lat_long_depth.csv")
bathymetry_data["cum_dist_from_00"] = 0.0
for i in range(1, len(bathymetry_data)):
    # Calculate distance using geopy's great_circle function
    bathymetry_data.loc[i, "cum_dist_from_00"] = GC(
        (
            bathymetry_data["lat"].iloc[i - 1],
            bathymetry_data["lon"].iloc[i - 1],
        ),
        (
            bathymetry_data["lat"].iloc[i],
            bathymetry_data["lon"].iloc[i],
        ),
    ).meters
    # Calculate cumulative distance
    bathymetry_data.loc[i, "cum_dist_from_00"] += bathymetry_data[
        "cum_dist_from_00"
    ].iloc[i - 1]
bathymetry_data.cum_dist_from_00 = bathymetry_data.cum_dist_from_00/1e3
ax1.plot(
    bathymetry_data.cum_dist_from_00,
    -1*bathymetry_data["depths"] / 1e3,
    color="k",
    lw=0.6,
)
segments = [
    (0, 15),
    (15, 50),
    (50, 260),
    (260, 290),
    (290, 350),
    (350, 400),
    (400, 520),
    (520, 655),
    (655, -1),
]
names = ["CS-W", "DO-1", "DO-2", "DO-3", "DO-4", "DO-5", "MAR", "DO-6", "CS-E"]
dist, depth = [], []
# Plot each segment with a different color
for i, seg in enumerate(segments):
    segment_data = bathymetry_data.iloc[seg[0] : seg[1]]
    dist.append(segment_data.cum_dist_from_00.tolist()[0])
    depth.append(-1*segment_data["depths"].mean() / 1e3)
    ax1.plot(
        segment_data.cum_dist_from_00,
        -1*segment_data["depths"] / 1e3,
        marker=".",
        ls="None",
        ms=1.2,
        color="r",
    )
    print(len(names) , len(segments))
    if len(names) == len(segments):
        ax1.text(
            segment_data.cum_dist_from_00.mean(),
            -1*(segment_data["depths"].mean() / 1e3) - 0.1,
            names[i],
            ha="center",
            va="top",
            rotation=90,
            fontdict=dict(size=10, color="b"),
        )
dist.append(bathymetry_data.cum_dist_from_00.iloc[-1])
depth.append(-1*bathymetry_data["depths"].iloc[-1] / 1e3)
depth = np.array(depth)
depth[depth > 0] = depth[depth > 0] * -1
ax1.step(
    dist,
    depth,
    where="post",
    ls="-",
    lw=1.5,
    color="k",
)
ax1.text(0.05, 0.95, "(B) TAT-8", ha="left", va="top", transform=ax1.transAxes)

sp.save_fig("Validation/Figure02.png")
sp.close()