import sys
sys.path.append("py/")
from utils import StackPlots
import datetime as dt
import pandas as pd
import numpy as np
import matplotlib.dates as mdates


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
ax0.text(0.1, 0.3, "MdPE: {:.2f}%".format(np.abs(mdpe) * 100), ha="left", va="top", transform=ax1.transAxes, fontsize=10)



sp.save_fig("figures/VoltsTAT1.png")
sp.close()