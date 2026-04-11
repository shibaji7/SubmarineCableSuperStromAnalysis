import datetime as dt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def find_optimal_scaling():
    obs = pd.read_csv("data/1958/Voltage/TAT1Volt-rescale.csv", parse_dates=["Time"])
    sim = pd.read_csv("data/1958/TAT1SimVolt.csv", parse_dates=["Time"])
    
    obs_times = np.array([(t - obs.Time.min()).total_seconds() for t in obs.Time])
    sim_times = np.array([(t - sim.Time.min()).total_seconds() for t in sim.Time])
    sim_vals = -np.array(sim['Vt(v)']/1e3)
    sim_interp = np.interp(obs_times, sim_times, sim_vals)
    obs_vals = np.array(obs.Voltage/1e3)
    
    scaling_factors = np.arange(0.5, 1.6, 0.1)
    results = []
    
    print("=" * 60)
    print("SCALING FACTOR OPTIMIZATION")
    print("=" * 60)
    print(f"{'Scale':<10} {'RMSE (kV)':<12} {'MAE (kV)':<12} {'Peak RMSE':<12} {'Peak MAE':<12}")
    print("-" * 60)
    
    peak_threshold = 1.0
    peak_mask = np.abs(obs_vals) > peak_threshold
    
    for scale in scaling_factors:
        scaled = sim_interp * scale
        errors = scaled - obs_vals
        
        rmse = np.sqrt(np.mean(errors**2))
        mae = np.median(np.abs(errors))
        
        if peak_mask.any():
            peak_rmse = np.sqrt(np.mean(errors[peak_mask]**2))
            peak_mae = np.median(np.abs(errors[peak_mask]))
        else:
            peak_rmse = np.nan
            peak_mae = np.nan
        
        results.append({
            'scale': scale,
            'rmse': rmse,
            'mae': mae,
            'peak_rmse': peak_rmse,
            'peak_mae': peak_mae
        })
        
        print(f"{scale:<10.1f} {rmse:<12.3f} {mae:<12.3f} {peak_rmse:<12.3f} {peak_mae:<12.3f}")
    
    results_df = pd.DataFrame(results)
    
    plt.figure(figsize=(3.5, 2.5), dpi=1000)
    plt.plot(results_df['scale'], results_df['rmse'], 'o-', color='#0072B2', label='RMSE', lw=1.5)
    plt.plot(results_df['scale'], results_df['mae'], 's-', color='#D55E00', label='MAE', lw=1.5)
    plt.plot(results_df['scale'], results_df['peak_rmse'], '^-', color='#009E73', label='Peak RMSE', lw=1.5)
    plt.xlabel("Scaling Factor")
    plt.ylabel("Error (kV)")
    plt.legend(loc=2, fontsize=6)
    plt.grid(True, lw=0.3)
    plt.tight_layout()
    plt.savefig("figures/tat1/1958.ScalingOptimization.png", bbox_inches='tight')
    plt.savefig("figures/tat1/1958.ScalingOptimization.pdf", bbox_inches='tight')
    plt.close()
    
    best_scale_rmse = results_df.loc[results_df['rmse'].idxmin(), 'scale']
    best_scale_peak = results_df.loc[results_df['peak_rmse'].idxmin(), 'scale']
    
    print("-" * 60)
    print(f"\nBest scale by overall RMSE: {best_scale_rmse:.1f}")
    print(f"Best scale by peak RMSE: {best_scale_peak:.1f}")
    print(f"\nSaved: 1958.ScalingOptimization.png/pdf")


if __name__ == "__main__":
    find_optimal_scaling()