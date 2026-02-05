import sys
sys.path.extend(["py/", "py/tat1/"])

import numpy as np
from bathymetry import BathymetryAnalysis

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
        
    return

if __name__ == "__main__":
    create_bathymetrystacks()