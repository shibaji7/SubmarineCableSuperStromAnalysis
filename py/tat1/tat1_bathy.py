import sys
sys.path.extend(["py/", "py/tat1/"])

import numpy as np
import pandas as pd
import glob
from utils import read_iaga, scale_to_dec, NatureStackPlots
from bathymetry import BathymetryAnalysis
import datetime as dt

def percentile(n):
    def percentile_(x):
        return x.quantile(n)
    percentile_.__name__ = 'percentile_{:02.0f}'.format(n*100)
    return percentile_

def create_bathymetrystacks():
    import matplotlib.pyplot as plt
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Helvetica', 'Arial']
    plt.rcParams['font.size'] = 12

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
    names = ["CS-W", "DO-1", "DO-2", "DO-3", "MAR", "DO-4", "RDG-1", "DO-5", "CS-E"]
    bathymetry.plot_bathymetry(
        "figures/bathymetry_TAT1.png", 
        names=names, 
        xlim=[0, 3900],
        method=[percentile(0.25)]+[np.mean]*7+[percentile(0.4)],
        # method=[np.min]*9,
        step_color="#0072B2",
    )
    segment_coordinates = np.array(bathymetry.get_segment_coordinates())
    print(f"dx Segments>>, {segment_coordinates}")
    return segment_coordinates

def convert_datasets(base_path: str = "data/1958/scaled_data/", scale=2.2):
    import pyIGRF
    from utils import StackPlots    
    from mpl_toolkits.axisartist.grid_finder import DictFormatter, FixedLocator
    import matplotlib.dates as mdates

    from geopy.distance import geodesic

    segment_coordinates = create_bathymetrystacks()
    distances = []
    print(">>> Segment coordinates:", segment_coordinates)
    for i in range(len(segment_coordinates)-1):
        start = (segment_coordinates[i, 0], segment_coordinates[i, 1])
        end = (segment_coordinates[i+1, 0], segment_coordinates[i+1, 1])
        distance_km = geodesic(start, end).kilometers
        distances.append(distance_km)
        print(f"Segment {i}: Start {start}, End {end}, Distance: {distance_km:.2f} km")

    total_distance = sum(distances)
    print(f"Total distance along the path: {total_distance:.2f} km")

    length_scales = [(1 + ((0.80*d)/total_distance)) for d in distances]
    stn = "ESK"
    frame = pd.DataFrame()
    files = glob.glob(base_path + f"{stn}*.dat")
    files.sort()
    frame = pd.concat([read_iaga(f) for f in files])

    xlim=[dt.datetime(1958, 2, 11,), dt.datetime(1958, 2, 11, 5)]
    segment_coordinates = create_bathymetrystacks()

    import matplotlib.pyplot as plt
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Helvetica', 'Arial']
    plt.rcParams['font.size'] = 7
    
    fig, ax = plt.subplots(1, 1, figsize=(3.5, 2.5), dpi=1000)
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H"))
    ax.set_xlabel("Time, UT (11 Feb 1958)")
    ax.set_yticklabels([])
    ax.set_xlim(xlim)

    names = ["CS-W", "DO-1", "DO-2", "DO-3", "RDG-1", "DO-4", "MAR", "DO-5", "CS-E"]
    for j in range(len(segment_coordinates)-1):
        l_scale = length_scales[j]
        scale = scale * l_scale
        seg = segment_coordinates[j:j+2,:]
        seg_m = np.mean(seg, axis=0)
        D, I, H, X, Y, Z, F = pyIGRF.igrf_value(seg_m[0], seg_m[1], 0, 1958)
        dfn = scale_to_dec(frame.copy(), -10.53, D)
        dfn = dfn[(dfn.index >= xlim[0]) & (dfn.index < xlim[1])]
        dfn.index = dfn.index - dt.timedelta(minutes=2)
        dfn.Z, dfn.X, dfn.Y = dfn.Z*scale, dfn.X*scale, dfn.Y*scale
        dfn.to_csv(f"data/1958/{names[j]}_scaled.csv", header=True, index=True, float_format="%g")
        print(names[j], seg_m, dfn.X.min(), dfn.Y.min())

        ax.plot(
            dfn.index,
            j*3500+dfn.X-np.mean(dfn.X[:60]),
            color="#D55E00",
            ls="-",
            lw=1.0,
        )

    ax.set_xlim(xlim)
    ax.set_ylabel("$B_x$ fields, nT", color="#D55E00")
    ax.set_xlabel("Time, UT (11 Feb 1958)")
    fig.savefig(f"figures/tat1/1958.BFields.png", bbox_inches='tight')
    fig.savefig(f"figures/tat1/1958.BFields.pdf", bbox_inches='tight')
    plt.close()
    return

if __name__ == "__main__":
    # convert_datasets()
    create_bathymetrystacks()