import datetime as dt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def plot_scubas_compare():
    obs = pd.read_csv("data/1958/Voltage/TAT1Volt-rescale.csv", parse_dates=["Time"])
    sim = pd.read_csv("data/1958/TAT1SimVolt.csv", parse_dates=["Time"])
    sim_scaled = pd.read_csv("data/1958/TAT1SimVolt_1.5.csv", parse_dates=["Time"])
    
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Helvetica', 'Arial']
    plt.rcParams['font.size'] = 7
    
    date_lims = [dt.datetime(1958, 2, 11, 1), dt.datetime(1958, 2, 11, 4)]
    ylim = [-3, 3]
    
    fig, ax = plt.subplots(1, 1, figsize=(6, 2.5), dpi=1000)
    ax.xaxis.set_major_locator(mdates.MinuteLocator(byminute=range(0, 60, 30)))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("$%H^{%M}$"))
    
    ax.scatter(
        obs.Time,
        obs.Voltage / 1e3,
        color="#0072B2",
        marker="s",
        s=3,
        label="Observations",
        alpha=0.8,
    )
    
    sim_v = sim.copy()
    sim_v['Vt(v)'] = -sim_v['Vt(v)'] / 1e3
    
    ax.plot(
        sim_v.Time,
        sim_v['Vt(v)'],
        color="#D55E00",
        lw=1.2,
        label="SCUBAS",
    )
    
    sim_scaled_v = sim_scaled.copy()
    sim_scaled_v['Vt(v)'] = -sim_scaled_v['Vt(v)'] / 1e3
    
    ax.plot(
        sim_scaled_v.Time,
        sim_scaled_v['Vt(v)'],
        color="#009E73",
        lw=1.0,
        label="SCUBAS [50% scaled |B| at western edge]",
    )
    
    obs_times = np.array([(t - obs.Time.min()).total_seconds() for t in obs.Time])
    sim_times = np.array([(t - sim_v.Time.min()).total_seconds() for t in sim_v.Time])
    sim_scaled_times = np.array([(t - sim_scaled_v.Time.min()).total_seconds() for t in sim_scaled_v.Time])
    
    sim_interp = np.interp(obs_times, sim_times, -np.array(sim['Vt(v)']/1e3))
    sim_scaled_interp = np.interp(obs_times, sim_scaled_times, -np.array(sim_scaled['Vt(v)']/1e3))
    obs_vals = np.array(obs.Voltage/1e3)
    
    rmse = np.sqrt(np.mean((sim_interp - obs_vals)**2))
    mae = np.median(np.abs(sim_interp - obs_vals))
    rmse_scaled = np.sqrt(np.mean((sim_scaled_interp - obs_vals)**2))
    mae_scaled = np.median(np.abs(sim_scaled_interp - obs_vals))
    
    print(f"SCUBAS - RMSE: {rmse:.3f} kV, MAE: {mae:.3f} kV")
    print(f"SCUBAS scaled - RMSE: {rmse_scaled:.3f} kV, MAE: {mae_scaled:.3f} kV")
    
    # ax.text(0.98, 0.02, f"RMSE: {rmse:.2f} kV, MAE: {mae:.2f} kV", 
    #       transform=ax.transAxes, ha='right', va='bottom', fontsize=5, color="#D55E00")
    # ax.text(0.98, 0.08, f"RMSE: {rmse_scaled:.2f} kV, MAE: {mae_scaled:.2f} kV", 
    #       transform=ax.transAxes, ha='right', va='bottom', fontsize=5, color="#009E73")
    
    ax.legend(loc=2, fontsize=6)
    ax.set_ylim(ylim)
    ax.set_xlim(date_lims)
    ax.set_ylabel(r"Voltage, $\times 10^3$ V (kV)")
    ax.set_xlabel("Time, UT (11 Feb 1958)")
    
    fig.savefig("figures/tat1/1958.Scubas.Compare.png", bbox_inches='tight')
    fig.savefig("figures/tat1/1958.Scubas.Compare.pdf", bbox_inches='tight')
    plt.close()
    print("Saved: 1958.Scubas.Compare.png/pdf")


if __name__ == "__main__":
    plot_scubas_compare()