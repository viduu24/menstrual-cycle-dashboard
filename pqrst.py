import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, savgol_filter
import os
import csv

# Get current file name without extension
file_name = "cycle_ecg_b3_b1_0.0087_b1_0.01000.0032.csv"  # You can also extract this from command line args if needed
file_name_no_ext = os.path.splitext(os.path.basename(file_name))[0]

# Load ECG CSV
df = pd.read_csv(file_name, header=None, names=['Potential', 'Time'])
df = df[1:]
df['Potential'] = pd.to_numeric(df['Potential'])
df['Time'] = pd.to_numeric(df['Time'])

# Convert to arrays
potential = df['Potential'].values
time = df['Time'].values

# Detect R-peaks (keeping your original approach)
r_peaks, _ = find_peaks(potential, prominence=0.5, distance=200)

# MATLAB-based fixed sample windows (from your original code)
q_window_before_r = 50
s_window_after_r = 100
t_window_after_s = 150

# Modified P wave detection approach
p_window_before_q = 200  # Slightly larger window
smooth_window = 15       # For Savitzky-Golay filter

# Apply Savitzky-Golay filter to smooth the signal for better P wave detection
smooth_potential = savgol_filter(potential, smooth_window, 3)

# Safe argmin/max helpers
def safe_argmin(arr, start, end):
    if end > start and end <= len(arr):
        return np.argmin(arr[start:end]) + start
    return start

def safe_argmax(arr, start, end):
    if end > start and end <= len(arr):
        return np.argmax(arr[start:end]) + start
    return start

# Improved P wave detection function
def find_p_wave(potential, smooth_potential, q_idx, window_size):
    p_search_start = max(0, q_idx - window_size)
    
    # Find local maxima in the smoothed signal within the P wave region
    p_candidates, _ = find_peaks(smooth_potential[p_search_start:q_idx], height=0, distance=20)
    
    if len(p_candidates) > 0:
        # Get the most prominent P wave - usually the last peak before Q
        p_idx = p_search_start + p_candidates[-1]
        return p_idx
    else:
        # Fallback to the original method if no peaks found
        return safe_argmax(potential, p_search_start, q_idx)

# Detected points
P_points, Q_points, R_points, S_points, T_points = [], [], [], [], []

for r_idx in r_peaks:
    # Q wave (min 50 samples before R) - from your original code
    q_search_start = max(0, r_idx - q_window_before_r)
    q_idx = safe_argmin(potential, q_search_start, r_idx)

    # Improved P wave detection
    p_idx = find_p_wave(potential, smooth_potential, q_idx, p_window_before_q)

    # S wave (min after R) - from your original code
    s_search_end = min(len(potential), r_idx + s_window_after_r)
    s_idx = safe_argmin(potential, r_idx, s_search_end)

    # T wave (max after S) - from your original code
    t_search_end = min(len(potential), s_idx + t_window_after_s)
    t_idx = safe_argmax(potential, s_idx, t_search_end)

    # Append points
    P_points.append(p_idx)
    Q_points.append(q_idx)
    R_points.append(r_idx)
    S_points.append(s_idx)
    T_points.append(t_idx)

# Plotting full signal with detected points
plt.figure(figsize=(14, 6))
plt.plot(time, potential, label='ECG Signal', linewidth=1)
plt.plot(time[P_points], potential[P_points], 'o', label='P', color='blue')
plt.plot(time[Q_points], potential[Q_points], 'o', label='Q', color='orange')
plt.plot(time[R_points], potential[R_points], 'o', label='R', color='red')
plt.plot(time[S_points], potential[S_points], 'o', label='S', color='green')
plt.plot(time[T_points], potential[T_points], 'o', label='T', color='purple')

# Annotate first cycle
first = 0
for label, idx in zip(['P', 'Q', 'R', 'S', 'T'], 
                      [P_points[first], Q_points[first], R_points[first], S_points[first], T_points[first]]):
    plt.text(time[idx], potential[idx] + 0.05, label, fontsize=9, ha='center')

plt.title("ECG Signal with Improved P Wave Detection")
plt.xlabel("Time (s)")
plt.ylabel("Potential (mV)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{file_name_no_ext}_analysis.png")
plt.show()

# Convert points to NumPy arrays for indexing
P_points = np.array(P_points)
Q_points = np.array(Q_points)
R_points = np.array(R_points)
S_points = np.array(S_points)
T_points = np.array(T_points)

# Compute intervals (in seconds)
RR_intervals = np.diff(time[R_points])
QRS_durations = time[S_points] - time[Q_points]
QT_intervals = time[T_points] - time[Q_points]
PQ_intervals = time[Q_points] - time[P_points]

# Convert to milliseconds
RR_intervals_ms = RR_intervals * 1000
QRS_durations_ms = QRS_durations * 1000
QT_intervals_ms = QT_intervals * 1000
PQ_intervals_ms = PQ_intervals * 1000

# Adjust lengths to match (cut last point for non-diff intervals)
min_len = min(len(RR_intervals_ms), len(QRS_durations_ms), len(QT_intervals_ms), len(PQ_intervals_ms))

# Bundle into a DataFrame for detailed intervals (in milliseconds)
interval_df = pd.DataFrame({
    'RR_interval_ms': RR_intervals_ms[:min_len],
    'QRS_duration_ms': QRS_durations_ms[:min_len],
    'QT_interval_ms': QT_intervals_ms[:min_len],
    'PQ_interval_ms': PQ_intervals_ms[:min_len]
})

# Save detailed intervals to CSV
interval_df.to_csv(f"{file_name_no_ext}_detailed_intervals.csv", index=False)

# Calculate the average RR, PQ, QT, QRS intervals (in milliseconds)
avg_qrs_duration_ms = np.mean(QRS_durations_ms)
avg_rr_interval_ms = np.mean(RR_intervals_ms)
avg_pq_interval_ms = np.mean(PQ_intervals_ms)
avg_qt_interval_ms = np.mean(QT_intervals_ms)

# Calculate heart rate (beats per minute) - keep this in the same units as before
heart_rate = 60 / np.mean(RR_intervals)  # Original calculation using seconds

# Print the results (in milliseconds)
print(f"File: {file_name}")
print(f"Average QRS Duration: {avg_qrs_duration_ms:.2f} ms")  # Changed to 2 decimal places for ms
print(f"Average RR Interval: {avg_rr_interval_ms:.2f} ms")
print(f"Average PQ Interval: {avg_pq_interval_ms:.2f} ms")
print(f"Average QT Interval: {avg_qt_interval_ms:.2f} ms")
print(f"Heart Rate: {heart_rate:.2f} BPM")

# Create summary CSV with the requested order: QRS duration, RR peak, PQ, QT, and file name (in milliseconds)
summary_file = "ecg_summary_results.csv"
summary_header = ["QRS_duration_ms", "RR_interval_ms", "PQ_interval_ms", "QT_interval_ms", "heart_rate", "file_name"]

# Check if summary file exists, create with header if not
file_exists = os.path.isfile(summary_file)
with open(summary_file, 'a', newline='') as f:
    writer = csv.writer(f)
    if not file_exists:
        writer.writerow(summary_header)
    
    # Write results in the requested order (in milliseconds)
    writer.writerow([
        f"{avg_qrs_duration_ms:.2f}",
        f"{avg_rr_interval_ms:.2f}",
        f"{avg_pq_interval_ms:.2f}",
        f"{avg_qt_interval_ms:.2f}",
        f"{heart_rate:.2f}",
        file_name
    ])

print(f"Summary data added to {summary_file}")