import sys
sys.path.extend(["py/", "py/tat1/"])

import numpy as np
from bathymetry import BathymetryAnalysis

def create_bathymetrystacks():
    file_path = "data/1958/lat_long_bathymetry.csv"
    segments = [
        (0, 32),
        (32, 50),
        (50, 60),
        (60, 170),
        (170, 210),
        (210, 335),
        (335, 390),
        (390, 440),
        (440, -1),
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
    names = ["CS-W", "DO-1", "DO-2", "DO-3", "RDG-1", "DO-4", "MAR", "DO-5", "CS-E"]
    bathymetry.plot_bathymetry(
        "figures/bathymetry_TAT1.png", 
        names=names, 
        xlim=[0, 3900],
        method=[np.mean]*4 + [np.min, np.mean, np.min, np.mean, np.median],
    )
    segment_coordinates = np.array(bathymetry.get_segment_coordinates())
    print(f"dx Segments>>, {segment_coordinates}")
        
    return

if __name__ == "__main__":
    create_bathymetrystacks()