import datetime as dt
import os
import sys
sys.path.extend(["py/", "py/tat1/"])

import numpy as np
import pandas as pd
import glob
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def read_esk_data(base_path: str = "data/1958/scaled_data/") -> pd.DataFrame:
    from utils import read_iaga
    
    stns = ["ESK"]
    frames = {}
    for stn in stns:
        files = glob.glob(base_path + f"{stn}*.dat")
        files.sort()
        frames[stn] = pd.concat([read_iaga(f) for f in files])
    return frames["ESK"]


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial']
    plt.rcParams['font.size'] = 7
    
    data = read_esk_data()
    data.drop_duplicates().sort_index(inplace=True)
    
    xlim = [dt.datetime(1958, 2, 11, 1), dt.datetime(1958, 2, 11, 4)]
    
    gs = plt.GridSpec(1, 1)
    fig = plt.figure(figsize=(3.5, 2.5), dpi=1000)
    ax = fig.add_subplot(gs[0])
    
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H"))
    
    data_z_shifted = data.Z.copy()
    data_z_shifted.index = data_z_shifted.index + dt.timedelta(minutes=10)
    
    ax.plot(data.index, data.X - np.median(data.X.iloc[:60]), 
          color="#0072B2", ls="-", lw=1.0, label="$B_x$")
    ax.plot(data.index, data.Y - np.median(data.Y.iloc[:60]), 
          color="#D55E00", ls="-", lw=1.0, label="$B_y$")
    ax.plot(data_z_shifted.index, data_z_shifted - np.median(data.Z.iloc[:60]), 
          color="#009E73", ls="-", lw=1.0, label="$B_z$")
    
    ax.legend(loc=2, fontsize=6)
    ax.set_ylim(-1000, 1000)
    ax.set_xlim(xlim)
    ax.set_ylabel("$B$, nT")
    ax.set_xlabel("Time, UT (11 Feb 1958)")
    
    fig.savefig("figures/tat1/1958.ESK.png", bbox_inches='tight')
    fig.savefig("figures/tat1/1958.ESK.pdf", bbox_inches='tight')
    plt.close()
    
    print("Saved: figures/tat1/1958.ESK.png and 1958.ESK.pdf")