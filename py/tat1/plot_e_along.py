import datetime as dt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def plot_e_field_along_cable():
    model_out = pd.read_csv("data/1958/TAT1SimVolt_1.0.csv", parse_dates=["Time"])
    
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial']
    plt.rcParams['font.size'] = 7
    
    xlim = [dt.datetime(1958, 2, 11, 1), dt.datetime(1958, 2, 11, 4)]
    names = ["CS-E", "DO-5", "MAR", "DO-4", "RDG-1", "DO-3", "DO-2", "DO-1", "CS-W"]
    
    fig, ax = plt.subplots(1, 1, figsize=(3.5, 2.5), dpi=1000)
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H"))
    ax.set_ylabel("$E_{||}$, mV/km")
    ax.set_xlabel("Time, UT (11 Feb 1958)")
    ax.set_yticklabels([])
    ax.set_xlim(xlim)
    
    y_positions = np.array([9000, 8000, 7000, 6000, 5000, 4000, 3000, 2000, 1000])
    label_x = xlim[1] + dt.timedelta(minutes=2)
    
    for j, name in enumerate(names):
        ex_col = f"E.X.0{j}"
        ey_col = f"E.Y.0{j}"
        
        if ex_col in model_out.columns and ey_col in model_out.columns:
            Epar = model_out[ex_col] * 0.5 + model_out[ey_col] * 0.5
            
            ax.plot(
                model_out.Time,
                y_positions[j] + Epar - np.median(Epar),
                color="k",
                ls="-",
                lw=1.0,
            )
            ax.text(
                label_x,
                y_positions[j],
                name,
                color="k",
                fontsize=5,
                # rotation=90,
                va="center",
                ha="left",
            )
    
    ax.set_ylim(-2000, 11000)
    ax.axvline(dt.datetime(1958, 2, 11, 1, 30), ymin=6000/13000, ymax=7000/13000, color="#009E73", ls="-", lw=1.5)
    ax.text(dt.datetime(1958, 2, 11, 1, 32), 4200, "1000 mV/km", color="#009E73", fontsize=6)
    
    fig.savefig("figures/tat1/1958.E_along_Cable.png", bbox_inches='tight')
    fig.savefig("figures/tat1/1958.E_along_Cable.pdf", bbox_inches='tight')
    plt.close()
    print("Saved: 1958.E_along_Cable.png/pdf")


if __name__ == "__main__":
    plot_e_field_along_cable()