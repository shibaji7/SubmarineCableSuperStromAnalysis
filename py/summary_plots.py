import sys
sys.path.append("py/")
from utils import StackPlots
import datetime as dt
import pandas as pd
import numpy as np

Dst1958 = pd.read_csv(
    "data/1958/Dst.csv",
    skiprows=17,                # Skip metadata/header lines
    dtype={"DATE": str, "TIME": str, "DOY": int, "DST": float},
    sep="\\s+",              # Use regex to split on whitespace
)
Dst1989 = pd.read_csv(
    "data/1989/Dst.csv",
    skiprows=17,                # Skip metadata/header lines
    dtype={"DATE": str, "TIME": str, "DOY": int, "DST": float},
    sep="\\s+",              # Use regex to split on whitespace
)

Ae1958 = pd.read_csv(
    "data/1958/AE.csv",
    skiprows=17,  # Skip metadata/header lines
    names=["DATE", "TIME", "DOY", "AE", "AU", "AL", "AO"],
    dtype={"DATE": str, "TIME": str, "DOY": int, "AE": float, "AU": float, "AL": float, "AO": float},
    sep="\\s+",              # Use regex to split on whitespace
)
Ae1989 = pd.read_csv(
    "data/1989/AE.csv",
    skiprows=17,  # Skip metadata/header lines
    # names=["DATE", "TIME", "DOY", "AE", "AU", "AL", "AO"],
    dtype={"DATE": str, "TIME": str, "DOY": int, "AE": float, "AU": float, "AL": float, "AO": float},
    sep="\\s+",              # Use regex to split on whitespace
)



# Optionally, combine DATE and TIME into a single datetime column
Dst1958["DATETIME"] = pd.to_datetime(Dst1958["DATE"] + " " + Dst1958["TIME"])
Dst1958.drop(columns=["DATE", "TIME", "|"], inplace=True)
Dst1989["DATETIME"] = pd.to_datetime(Dst1989["DATE"] + " " + Dst1989["TIME"])
Dst1989.drop(columns=["DATE", "TIME", "|"], inplace=True)
Ae1958["DATETIME"] = pd.to_datetime(Ae1958["DATE"] + " " + Ae1958["TIME"])
Ae1958.drop(columns=["DATE", "TIME", "DOY"], inplace=True)
Ae1989["DATETIME"] = pd.to_datetime(Ae1989["DATE"] + " " + Ae1989["TIME"])
Ae1989.drop(columns=["DATE", "TIME", "DOY", "|"], inplace=True)


# Example: print the first few rows
print(Ae1989.head())

sp = StackPlots(
    nrows=2,
    ncols=1,
    figsize=(8, 3),
    datetime=True,
    text_size=12,
    sharex=False,
)
_, ax = sp.plot_stack_plots(
    Dst1958["DATETIME"].tolist(),
    Dst1958["DST"],
    text="February 1958",
    ylabel="Dst (nT)",
    color="blue",
    xlim=[dt.datetime(1958, 2, 6), dt.datetime(1958, 2, 18)],
    ylim=[-700, 100],
    interval=24,
    dfx="%d",
    xlabel="Day of Month (Feb 1958)",
)
ax.axhline(0, color="black", linestyle="--", linewidth=0.5)
ax.hlines(Dst1958["DST"].min(), dt.datetime(1958, 2, 11, 4), dt.datetime(1958, 2, 11, 20), color="red", linestyle=":", linewidth=1)
ax.text(0.95, 0.1, f"(A) Dst$_m$={Dst1958['DST'].min()} nT", ha="right", va="bottom", transform=ax.transAxes)
ax.vlines(dt.datetime(1958, 2, 11, 1), -200, 100, color="m", linestyle=":", linewidth=1.5)
ax.text(dt.datetime(1958, 2, 11, 2), 60, "1 UT (2/11)", ha="left", va="top", color="m")
ax.tick_params(axis="y", labelcolor="b")
ax.set_ylabel("Dst (nT)", color="b")
tax = ax.twinx()
tax.plot(Ae1958["DATETIME"], Ae1958["AE"], color="orange", linewidth=1)
tax.set_ylabel("AE (nT)", color="orange")
tax.tick_params(axis="y", labelcolor="orange")
tax.set_ylim(0, 3000)


_, ax = sp.plot_stack_plots(
    Dst1989["DATETIME"].tolist(),
    Dst1989["DST"],
    text="March 1989",
    ylabel="Dst (nT)",
    color="blue",
    xlim=[dt.datetime(1989, 3, 8), dt.datetime(1989, 3, 20)],
    ylim=[-700, 100],
    interval=24,
    dfx="%d",
    xlabel="Day of Month (March 1989)",
)
ax.axhline(0, color="black", linestyle="--", linewidth=0.5)
ax.hlines(Dst1989["DST"].min(), dt.datetime(1989, 3, 13, 18), dt.datetime(1989, 3, 14, 6), color="red", linestyle=":", linewidth=1)
ax.text(0.95, 0.1, f"(B) Dst$_m$={Dst1989['DST'].min()} nT", ha="right", va="bottom", transform=ax.transAxes)
ax.vlines(dt.datetime(1989, 3, 13, 1), -200, 100, color="m", linestyle=":", linewidth=1.5)
ax.text(dt.datetime(1989, 3, 13, 2), 60, "1 UT (3/13)", ha="left", va="top", color="m")
ax.tick_params(axis="y", labelcolor="b")
ax.set_ylabel("Dst (nT)", color="b")
tax = ax.twinx()
tax.plot(Ae1989["DATETIME"], Ae1989["AE"], color="orange", linewidth=1)
tax.set_ylabel("AE (nT)", color="orange")
tax.tick_params(axis="y", labelcolor="orange")
tax.set_ylim(0, 3000)

sp.save_fig("figures/Dst_StackPlots.png")
sp.close()


datasets_1989 = {
    "FRD": pd.read_csv("data/1989/FRD.csv", parse_dates=["Date"]),
    "STJ": pd.read_csv("data/1989/STJ.csv", parse_dates=["Date"]),
    "HAD": pd.read_csv("data/1989/HAD.csv", parse_dates=["Date"]),
}
(
    datasets_1989["FRD"]["H"],
    datasets_1989["STJ"]["H"],
    datasets_1989["HAD"]["H"]
) = (
    np.sqrt(
        datasets_1989["FRD"]["X"]**2 + datasets_1989["FRD"]["Y"]**2
    ),
    np.sqrt(
        datasets_1989["STJ"]["X"]**2 + datasets_1989["STJ"]["Y"]**2
    ),
    np.sqrt(
        datasets_1989["HAD"]["X"]**2 + datasets_1989["HAD"]["Y"]**2
    )
)
(
    datasets_1989["FRD"]["H"],
    datasets_1989["STJ"]["H"],
    datasets_1989["HAD"]["H"]
) = (
    datasets_1989["FRD"]["H"] - np.median(datasets_1989["FRD"]["H"][:60]),
    datasets_1989["STJ"]["H"] - np.median(datasets_1989["STJ"]["H"][:60]),
    datasets_1989["HAD"]["H"] - np.median(datasets_1989["HAD"]["H"][:60]),
)

datasets_1958 = {
    "ESK": pd.read_csv("data/1958/compiled.csv", parse_dates=["Date"]),
}
(
    datasets_1958["ESK"]["H"]
) = (
    np.sqrt(
        datasets_1958["ESK"]["X"]**2 + datasets_1958["ESK"]["Y"]**2
    )
)
(
    datasets_1958["ESK"]["H"]
) = (
    datasets_1958["ESK"]["H"] - np.median(datasets_1958["ESK"]["H"][:60])
)

sp = StackPlots(
    nrows=2,
    ncols=1,
    figsize=(8, 3),
    datetime=True,
    text_size=12,
    sharex=False,
)

_, ax = sp.plot_stack_plots(
    datasets_1958["ESK"]["Date"].tolist(),
    datasets_1958["ESK"]["H"],
    text="(A) February 1958",
    ylabel=r"$B_H$ (nT)",
    color="blue",
    # xlim=[dt.datetime(1958, 2, 10, 12), dt.datetime(1958, 2, 11, 12)],
    ylim=[-1000, 1000],
    interval=6,
    dfx=r"%H",
    label="ESK",
    xlabel="UT Hours (since 10 February 1958, 12:00 UT)",
)
ax.legend(loc=2)

_, ax = sp.plot_stack_plots(
    datasets_1989["FRD"]["Date"].tolist(),
    datasets_1989["FRD"]["H"],
    text="(B) March 1989",
    ylabel=r"$B_H$ (nT)",
    color="blue",
    xlim=[dt.datetime(1989, 3, 12), dt.datetime(1989, 3, 15)],
    ylim=[-2000, 2000],
    interval=12,
    dfx=r"%H",
    label="FRD",
    xlabel="UT Hours (since 12 March 1989, 12:00 UT)",
)
sp.plot_stack_plots(
    datasets_1989["STJ"]["Date"].tolist(),
    datasets_1989["STJ"]["H"],
    color="red",
    xlim=[dt.datetime(1989, 3, 12), dt.datetime(1989, 3, 15)],
    ylim=[-2000, 2000],
    interval=12,
    ax=ax,
    label="STJ",
    dfx=r"%H",
)
sp.plot_stack_plots(
    datasets_1989["HAD"]["Date"].tolist(),
    datasets_1989["HAD"]["H"],
    color="k",
    xlim=[dt.datetime(1989, 3, 12), dt.datetime(1989, 3, 15)],
    ylim=[-2000, 2000],
    interval=12,
    ax=ax,
    label="HAD",
    dfx=r"%H",
)
ax.legend(loc=2)


sp.save_fig("figures/Mag_StackPlots.png")
sp.close()


datasets_1989 = pd.read_csv("data/1989/Voltage/TAT8Volt-rescale.csv", parse_dates=["Time"])

