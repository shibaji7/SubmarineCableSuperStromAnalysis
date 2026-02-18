import sys
sys.path.extend(["py/", "py/tat1/"])

import numpy as np
import pandas as pd
import glob
from utils import read_iaga, scale_to_dec
from bathymetry import BathymetryAnalysis
import datetime as dt

def percentile(n):
    def percentile_(x):
        return x.quantile(n)
    percentile_.__name__ = 'percentile_{:02.0f}'.format(n*100)
    return percentile_

def create_bathymetrystacks():
    file_path = "data/1958/lat_long_bathymetry-modified.csv"
    segments = [
        # (0, 10),
        # (10, 32),
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
    # names = ["BAY", "CS-W", "DO-1", "DO-2", "DO-3", "RDG-1", "DO-4", "MAR", "DO-5", "CS-E"]
    names = ["CS-W", "DO-1", "DO-2", "DO-3", "RDG-1", "DO-4", "MAR", "DO-5", "CS-E"]
    bathymetry.plot_bathymetry(
        "figures/bathymetry_TAT1.png", 
        names=names, 
        xlim=[0, 3900],
        # method=[percentile(0.25)]+[np.mean]*7+[percentile(0.4)],
        method=[np.min]*9,
        step_color="b",
    )
    segment_coordinates = np.array(bathymetry.get_segment_coordinates())
    print(f"dx Segments>>, {segment_coordinates}")
    return segment_coordinates

def convert_datasets(base_path: str = "data/1958/scaled_data/", scale=2.2):
    import pyIGRF
    from utils import StackPlots    
    from mpl_toolkits.axisartist.grid_finder import DictFormatter, FixedLocator
    import matplotlib.dates as mdates

    stn = "ESK"
    frame = pd.DataFrame()
    files = glob.glob(base_path + f"{stn}*.dat")
    files.sort()
    frame = pd.concat([read_iaga(f) for f in files])

    xlim=[dt.datetime(1958, 2, 11,), dt.datetime(1958, 2, 11, 5)]
    segment_coordinates = create_bathymetrystacks()

    sp = StackPlots(
        nrows=1, ncols=1, datetime=True, 
        figsize=(6, 4), text_size=12, 
    )
    ax = sp.axes[0]
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H"))
    ax.set_xlabel("Time, UT (11 Feb 1958)")
    ax.set_yticklabels([])
    ax.set_xlim(xlim)
    tax = ax.twinx()
    tax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
    tax.xaxis.set_major_formatter(mdates.DateFormatter("%H"))
    tax.set_yticklabels([])

    names = ["CS-W", "DO-1", "DO-2", "DO-3", "RDG-1", "DO-4", "MAR", "DO-5", "CS-E"]
    for j in range(len(segment_coordinates)-1):
        seg = segment_coordinates[j:j+2,:]
        seg_m = np.mean(seg, axis=0)
        D, I, H, X, Y, Z, F = pyIGRF.igrf_value(seg_m[0], seg_m[1], 0, 1958)
        # print(">>>", j, seg_m, D)
        dfn = scale_to_dec(frame.copy(), -10.53, D)
        dfn = dfn[(dfn.index >= xlim[0]) & (dfn.index < xlim[1])]
        dfn.index = dfn.index - dt.timedelta(minutes=2)
        dfn.Z, dfn.X, dfn.Y = dfn.Z*scale, dfn.X*scale, dfn.Y*scale
        dfn.to_csv(f"data/1958/{names[j]}_scaled.csv", header=True, index=True, float_format="%g")
        print(names[j], seg_m, dfn.X.min(), dfn.Y.min())

        ax.plot(
            dfn.index,
            j*3500+dfn.X-np.mean(dfn.X[:60]),
            color="r",
            ls="-",
            lw=0.6,
        )
        tax.plot(
            dfn.index,
            j*3000+dfn.Y-np.mean(dfn.Y[:60]),
            color="k",
            ls="-",
            lw=0.6,
        )
        # ax.text(
        #     xlim[1] + dt.timedelta(minutes=5),
        #     400*j + model_out[f"V(v).0{ix}"].iloc[-1],
        #     name,
        #     color="k",
        #     fontsize=6,
        #     rotation=90,
        #     va="center",
        #     ha="center",
        # )
        # ix -= 1

    # ax.set_ylim(-8500, 8500)
    ax.set_xlim(xlim)
    ax.set_ylabel("$B_x$ fields, nT", color="r")
    tax.set_ylabel("$B_y$ fields, nT")
    ax.set_xlabel("Time, UT (11 Feb 1958)")
    # ax.set_yticklabels([])
    sp.save_fig(f"figures/tat1/1958.BFields.png")
    return

if __name__ == "__main__":
    convert_datasets()