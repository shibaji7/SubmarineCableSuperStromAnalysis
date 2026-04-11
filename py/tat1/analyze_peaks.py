import datetime as dt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def analyze_peaks():
    obs = pd.read_csv("data/1958/Voltage/TAT1Volt-rescale.csv", parse_dates=["Time"])
    sim = pd.read_csv("data/1958/TAT1SimVolt.csv", parse_dates=["Time"])
    sim_scaled = pd.read_csv("data/1958/TAT1SimVolt_1.5.csv", parse_dates=["Time"])
    
    obs_times = np.array([(t - obs.Time.min()).total_seconds() for t in obs.Time])
    sim_times = np.array([(t - sim.Time.min()).total_seconds() for t in sim.Time])
    
    sim_interp = np.interp(obs_times, sim_times, -np.array(sim['Vt(v)']/1e3))
    obs_vals = np.array(obs.Voltage/1e3)
    errors = sim_interp - obs_vals
    
    peak_threshold = 1.0
    peak_mask = np.abs(obs_vals) > peak_threshold
    quiet_mask = np.abs(obs_vals) <= peak_threshold
    
    rmse_peaks = np.sqrt(np.mean(errors[peak_mask]**2)) if peak_mask.any() else np.nan
    rmse_quiet = np.sqrt(np.mean(errors[quiet_mask]**2)) if quiet_mask.any() else np.nan
    
    mae_peaks = np.median(np.abs(errors[peak_mask])) if peak_mask.any() else np.nan
    mae_quiet = np.median(np.abs(errors[quiet_mask])) if quiet_mask.any() else np.nan
    
    print("=" * 50)
    print("PEAK ANALYSIS (|V| > 1 kV)")
    print("=" * 50)
    print(f"Peak RMSE: {rmse_peaks:.3f} kV, Peak MAE: {mae_peaks:.3f} kV")
    print(f"Quiet RMSE: {rmse_quiet:.3f} kV, Quiet MAE: {mae_quiet:.3f} kV")
    print(f"Peak samples: {peak_mask.sum()}, Quiet samples: {quiet_mask.sum()}")
    
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Helvetica', 'Arial']
    plt.rcParams['font.size'] = 7
    
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5), dpi=1000)
    
    ax1 = axes[0, 0]
    ax1.scatter(obs_vals, errors, c="#0072B2", s=3, alpha=0.5)
    ax1.axhline(0, color="k", lw=0.5)
    ax1.axvline(0, color="k", lw=0.5)
    ax1.set_xlabel("Observed Voltage (kV)")
    ax1.set_ylabel("Error (kV)")
    ax1.set_title("Error vs Observed")
    
    ax2 = axes[0, 1]
    ax2.hist(errors, 30, color="#0072B2", alpha=0.7, edgecolor="k")
    ax2.axvline(0, color="r", lw=1)
    ax2.set_xlabel("Error (kV)")
    ax2.set_ylabel("Count")
    ax2.set_title("Error Distribution")
    
    ax3 = axes[1, 0]
    t = np.arange(len(errors))
    ax3.plot(obs.Time, obs_vals, color="#0072B2", lw=0.5, label="Obs")
    ax3.plot(sim.Time, -sim['Vt(v)']/1e3, color="#D55E00", lw=0.5, label="SCUBAS")
    ax3.axhline(peak_threshold, color="g", lw=0.5, ls="--", label=f"Peak threshold")
    ax3.set_xlabel("Time")
    ax3.set_ylabel("Voltage (kV)")
    ax3.set_title("Time Series")
    ax3.legend(loc=2, fontsize=6)
    
    ax4 = axes[1, 1]
    ax4.scatter(np.abs(obs_vals), np.abs(errors), c="#0072B2", s=3, alpha=0.5)
    ax4.set_xlabel("|Observed| (kV)")
    ax4.set_ylabel("|Error| (kV)")
    ax4.set_title("|Error| vs |Observed|")
    
    fig.savefig("figures/tat1/1958.ErrorAnalysis.png", bbox_inches='tight')
    fig.savefig("figures/tat1/1958.ErrorAnalysis.pdf", bbox_inches='tight')
    plt.close()
    
    print(f"\nSaved: 1958.ErrorAnalysis.png/pdf")
    print("\n" + "=" * 50)
    print("RECOMMENDATIONS FOR BETTER PEAK CAPTURE:")
    print("=" * 50)
    print("1. Try different scaling factors (e.g., 0.7, 0.8, 0.9, 1.1, 1.2)")
    print("2. Use time-dependent scaling during storms")
    print("3. Add conductance correction for peak conductivities")
    print("4. Include ionospheric induced currents model")


if __name__ == "__main__":
    analyze_peaks()