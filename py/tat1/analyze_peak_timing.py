import datetime as dt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.signal import find_peaks


def analyze_peak_timing():
    obs = pd.read_csv("data/1958/Voltage/TAT1Volt-rescale.csv", parse_dates=["Time"])
    sim = pd.read_csv("data/1958/TAT1SimVolt.csv", parse_dates=["Time"])
    
    obs_vals = np.array(obs.Voltage/1e3)
    sim_vals = -np.array(sim['Vt(v)']/1e3)
    time = np.arange(len(obs_vals))
    
    obs_peaks, obs_props = find_peaks(obs_vals, height=1.0, distance=10)
    sim_peaks, sim_props = find_peaks(sim_vals, height=1.0, distance=10)
    
    obs_neg_peaks, _ = find_peaks(-obs_vals, height=1.0, distance=10)
    sim_neg_peaks, _ = find_peaks(-sim_vals, height=1.0, distance=10)
    
    print("=" * 60)
    print("PEAK TIMING ANALYSIS")
    print("=" * 60)
    print(f"\nPositive peaks in OBS: {len(obs_peaks)} at indices: {obs_peaks}")
    print(f"Positive peaks in SIM: {len(sim_peaks)} at indices: {sim_peaks}")
    print(f"\nNegative peaks (troughs) in OBS: {len(obs_neg_peaks)}")
    print(f"Negative peaks (troughs) in SIM: {len(sim_neg_peaks)}")
    
    if len(obs_peaks) > 0 and len(sim_peaks) > 0:
        for i, (o_idx, s_idx) in enumerate(zip(obs_peaks[:3], sim_peaks[:3])):
            time_diff = (o_idx - s_idx) * (60/3600)
            amp_diff = obs_vals[o_idx] - sim_vals[s_idx]
            print(f"Peak {i+1}: time diff = {time_diff*60:.1f} min, amp diff = {amp_diff:.2f} kV")
    
    fig, axes = plt.subplots(2, 1, figsize=(3.5, 3), dpi=1000, sharex=True)
    
    axes[0].plot(obs.Time, obs_vals, 'o-', color='#0072B2', ms=2, lw=0.5, label='Observed')
    axes[0].plot(sim.Time, sim_vals, '-', color='#D55E00', lw=0.5, label='SCUBAS')
    axes[0].scatter(obs.Time.iloc[obs_peaks], obs_vals[obs_peaks], color='g', s=30, zorder=5, label='Obs peaks')
    axes[0].scatter(sim.Time.iloc[sim_peaks], sim_vals[sim_peaks], color='r', s=30, zorder=5, marker='^', label='Sim peaks')
    axes[0].legend(loc=2, fontsize=6)
    axes[0].set_ylabel('Voltage (kV)')
    axes[0].set_title('Observed vs SCUBAS')
    
    diff = sim_vals[:len(obs_vals)] - obs_vals
    axes[1].plot(obs.Time, diff, color='k', lw=0.5)
    axes[1].axhline(0, color='r', lw=0.5)
    axes[1].set_ylabel('Error (kV)')
    axes[1].set_xlabel('Time (UT)')
    axes[1].set_title('Model - Observation')
    
    plt.tight_layout()
    plt.savefig("figures/tat1/1958.PeakTiming.png", bbox_inches='tight')
    plt.savefig("figures/tat1/1958.PeakTiming.pdf", bbox_inches='tight')
    plt.close()
    
    print(f"\nSaved: 1958.PeakTiming.png/pdf")
    print("\n" + "=" * 60)
    print("FINDINGS:")
    print("=" * 60)
    print("1. Peak RMSE is ~1.69 kV regardless of scaling")
    print("2. This suggests MODEL STRUCTURE issues, not scaling")
    print("3. Possible causes:")
    print("   - Timing offset between model and observations")
    print("   - Peak amplitude mis-calibration")
    print("   - Missing ionospheric induced currents")
    print("   - Incorrect boundary conditions")


if __name__ == "__main__":
    analyze_peak_timing()