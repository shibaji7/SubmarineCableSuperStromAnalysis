import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, 'py/')
from bathymetry import BathymetryAnalysis
from math import radians, cos, sin, sqrt, atan2


def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    phi1, phi2 = radians(lat1), radians(lat2)
    dphi = radians(lat2 - lat1)
    dlambda = radians(lon2 - lon1)
    a = sin(dphi/2)**2 + cos(phi1)*cos(phi2)*sin(dlambda/2)**2
    return 2 * R * atan2(sqrt(a), sqrt(1-a))


def plot_bathymetry_tat1():
    names = ["CS-W", "DO-1", "DO-2", "DO-3", "RDG-1", "DO-4", "MAR", "DO-5", "CS-E"]
    colors = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#56B4E9", "#E69F00", "#F0E442", "#999999", "#0072B2"]
    
    bathy = pd.read_csv("data/1958/lat_long_bathymetry-modified.csv")
    
    distances = [0]
    for i in range(1, len(bathy)):
        d = haversine(bathy.iloc[i-1]['lat'], bathy.iloc[i-1]['lon'],
                       bathy.iloc[i]['lat'], bathy.iloc[i]['lon'])
        distances.append(distances[-1] + d)
    bathy['distance'] = distances
    
    bathymetry = BathymetryAnalysis("data/1958/lat_long_bathymetry-modified.csv", [], colors)
    bathymetry.load_data()
    bathymetry.bathymetry_data['distance'] = bathy['distance']
    
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Helvetica', 'Arial']
    plt.rcParams['font.size'] = 7
    
    fig, ax = plt.subplots(1, 1, figsize=(3.5, 2.5), dpi=1000)
    
    ax.plot(
        bathy['distance'],
        bathy["bathymetry.meters"] / 1e3,
        color="k",
        lw=0.8,
    )
    
    segment_indices = [
        [0, 50], [50, 100], [100, 150], [150, 200], [200, 250],
        [250, 300], [300, 350], [350, 400], [400, 486]
    ]
    method = [np.mean] * 9
    
    for i, (start, end) in enumerate(segment_indices):
        segment_data = bathy.iloc[start:end]
        
        ax.scatter(
            segment_data['distance'],
            segment_data["bathymetry.meters"] / 1e3,
            color=colors[i],
            s=3,
            alpha=0.7,
        )
        
        seg_depth = segment_data["bathymetry.meters"].mean() / 1e3
        seg_dist = segment_data['distance'].mean()
        
        ax.text(
            seg_dist,
            seg_depth + 0.1,
            names[i],
            ha="center",
            va="bottom",
            rotation=90,
            fontdict=dict(size=5, color="k"),
        )
    
    ax.set_xlim([0, bathy['distance'].max()])
    ax.set_ylim([0, 4.5])
    ax.set_ylabel("Depth, km")
    ax.set_xlabel("Distance along cable, km")
    ax.grid(True, lw=0.3, alpha=0.3)
    
    fig.savefig("figures/tat1/bathymetry_TAT-1.png", bbox_inches='tight')
    fig.savefig("figures/tat1/bathymetry_TAT-1.pdf", bbox_inches='tight')
    plt.close()
    print("Saved: bathymetry_TAT-1.png/pdf")


if __name__ == "__main__":
    plot_bathymetry_tat1()