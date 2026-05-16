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


def plot_b_e_fields():
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial']
    plt.rcParams['font.size'] = 7
    
    xlim = [dt.datetime(1958, 2, 11, 1), dt.datetime(1958, 2, 11, 4)]
    
    data = read_esk_data()
    data.drop_duplicates().sort_index(inplace=True)
    
    model_out = pd.read_csv("data/1958/TAT1SimVolt_1.0.csv", parse_dates=["Time"])
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(3.5, 4), dpi=1000, sharex=True)
    plt.subplots_adjust(hspace=0.35)
    
    ax1.xaxis.set_major_locator(mdates.HourLocator(interval=1))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%H"))
    
    data_z_shifted = data.Z.copy()
    data_z_shifted.index = data_z_shifted.index + dt.timedelta(minutes=10)
    
    ax1.plot(data.index, data.X - np.median(data.X.iloc[:60]), 
          color="#0072B2", ls="-", lw=1.0, label="$B_x$")
    ax1.plot(data.index, data.Y - np.median(data.Y.iloc[:60]), 
          color="#D55E00", ls="-", lw=1.0, label="$B_y$")
    ax1.plot(data_z_shifted.index, data_z_shifted - np.median(data.Z.iloc[:60]), 
          color="#009E73", ls="-", lw=1.0, label="$B_z$")
    
    ax1.legend(loc=2, fontsize=6)
    ax1.set_ylim(-1000, 1000)
    ax1.set_xlim(xlim)
    ax1.set_ylabel("$B$, nT")
    ax1.text(-0.1, 0.98, "(a)", transform=ax1.transAxes, fontdict=dict(size=10, weight='bold'), ha='left', va='top')
    ax1.grid(True, lw=0.3, alpha=0.3)
    
    ax2.xaxis.set_major_locator(mdates.HourLocator(interval=1))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%H"))
    ax2.set_ylabel("$E_{||}$, mV/km")
    ax2.set_xlabel("Time, UT (11 Feb 1958)")
    ax2.set_yticklabels([])
    ax2.set_xlim(xlim)
    
    names = ["CS-E", "DO-5", "RDG-1", "DO-4", "MAR", "DO-3", "DO-2", "DO-1", "CS-W"]
    names.reverse()
    y_positions = np.array([9000, 8000, 7000, 6000, 5000, 4000, 3000, 2000, 1000])
    label_x = xlim[1] + dt.timedelta(minutes=2)
    
    for j, name in enumerate(names):
        ex_col = f"E.X.0{j}"
        ey_col = f"E.Y.0{j}"
        
        if ex_col in model_out.columns and ey_col in model_out.columns:
            Epar = model_out[ex_col] * 0.5 + model_out[ey_col] * 0.5
            
            ax2.plot(
                model_out.Time,
                y_positions[j] + Epar - np.median(Epar),
                color="k",
                ls="-",
                lw=1.0,
            )
            ax2.text(
                label_x,
                y_positions[j],
                name,
                color="#0072B2",
                fontsize=4,
                va="center",
                ha="left",
            )
    
    ax2.set_ylim(-2000, 11000)
    ax2.axvline(dt.datetime(1958, 2, 11, 1, 30), ymin=6000/13000, ymax=7000/13000, color="#009E73", ls="-", lw=1.5)
    ax2.text(dt.datetime(1958, 2, 11, 1, 32), 4200, "1000 mV/km", color="#009E73", fontsize=6)
    ax2.text(-0.1, 0.98, "(b)", transform=ax2.transAxes, fontdict=dict(size=10, weight='bold'), ha='left', va='top')
    ax2.grid(True, lw=0.3, alpha=0.3)
    
    fig.subplots_adjust(wspace=0, hspace=0.05)
    fig.savefig("figures/tat1/1958.B_and_E.png", bbox_inches='tight')
    fig.savefig("figures/tat1/1958.B_and_E.pdf", bbox_inches='tight')
    plt.close()
    
    print("Saved: figures/tat1/1958.B_and_E.png/pdf")


if __name__ == "__main__":
    plot_b_e_fields()