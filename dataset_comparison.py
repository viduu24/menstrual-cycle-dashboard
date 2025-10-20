import pandas as pd
import numpy as np
from scipy.stats import pearsonr, ks_2samp
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Load your CSV files
real_ecg = pd.read_csv('real_ecg.csv')
simulated_ecg = pd.read_csv('simulated_ecg.csv')

# 1. Statistical Distribution Comparison (Kolmogorov-Smirnov test)
def compare_distributions(real_df, sim_df):
    results = {}
    for column in real_df.columns:
        if column in sim_df.columns and real_df[column].dtype.kind in 'fc' and sim_df[column].dtype.kind in 'fc':
            # KS test returns statistic and p-value
            statistic, p_value = ks_2samp(real_df[column].dropna(), sim_df[column].dropna())
            results[column] = {
                'KS_statistic': statistic,  # Lower is better (more similar)
                'p_value': p_value  # Higher is better (more similar)
            }
    return results

# 2. Compare histograms
def compare_histograms(real_df, sim_df):
    for column in real_df.columns:
        if column in sim_df.columns and real_df[column].dtype.kind in 'fc' and sim_df[column].dtype.kind in 'fc':
            plt.figure(figsize=(10, 6))
            # Get range covering both datasets
            min_val = min(real_df[column].min(), sim_df[column].min())
            max_val = max(real_df[column].max(), sim_df[column].max())
            bins = np.linspace(min_val, max_val, 30)
            
            plt.hist(real_df[column], bins=bins, alpha=0.5, label='Real ECG')
            plt.hist(sim_df[column], bins=bins, alpha=0.5, label='Simulated ECG')
            plt.title(f'Distribution Comparison of {column}')
            plt.legend()
            plt.grid(True)
            plt.show()

# 3. Basic statistics comparison
def compare_statistics(real_df, sim_df):
    stats = {}
    for column in real_df.columns:
        if column in sim_df.columns and real_df[column].dtype.kind in 'fc' and sim_df[column].dtype.kind in 'fc':
            real_stats = {
                'mean': real_df[column].mean(),
                'std': real_df[column].std(),
                'min': real_df[column].min(),
                'max': real_df[column].max(),
                'median': real_df[column].median(),
                'IQR': real_df[column].quantile(0.75) - real_df[column].quantile(0.25)
            }
            
            sim_stats = {
                'mean': sim_df[column].mean(),
                'std': sim_df[column].std(),
                'min': sim_df[column].min(),
                'max': sim_df[column].max(),
                'median': sim_df[column].median(),
                'IQR': sim_df[column].quantile(0.75) - sim_df[column].quantile(0.25)
            }
            
            stats[column] = {
                'real': real_stats,
                'simulated': sim_stats,
                'differences': {
                    key: abs(real_stats[key] - sim_stats[key]) for key in real_stats
                }
            }
    return stats

# Calculate all metrics
dist_comparison = compare_distributions(real_ecg, simulated_ecg)
stats_comparison = compare_statistics(real_ecg, simulated_ecg)

# Print results
print("\nDistribution comparison (Kolmogorov-Smirnov test):")
for column, values in dist_comparison.items():
    print(f"  {column}:")
    print(f"    KS statistic: {values['KS_statistic']:.4f} (closer to 0 = more similar)")
    print(f"    p-value: {values['p_value']:.4f} (higher = more similar)")

print("\nStatistical comparison:")
for column, stats in stats_comparison.items():
    print(f"\n  {column}:")
    print(f"    Real mean: {stats['real']['mean']:.2f}, Simulated mean: {stats['simulated']['mean']:.2f}")
    print(f"    Real std: {stats['real']['std']:.2f}, Simulated std: {stats['simulated']['std']:.2f}")
    print(f"    Real median: {stats['real']['median']:.2f}, Simulated median: {stats['simulated']['median']:.2f}")
    print(f"    Real range: {stats['real']['min']:.2f} to {stats['real']['max']:.2f}")
    print(f"    Simulated range: {stats['simulated']['min']:.2f} to {stats['simulated']['max']:.2f}")

# Visual comparison of distributions
compare_histograms(real_ecg, simulated_ecg)