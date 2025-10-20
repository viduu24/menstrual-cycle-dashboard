import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import find_peaks, savgol_filter
import warnings
warnings.filterwarnings('ignore')

def read_ecg_data(filepath):
    """Read ECG data from CSV file with flexible handling of different formats."""
    try:
        # First try standard read
        try:
            df = pd.read_csv("C:/Users/vidus/Downloads/App (2)/App Installation/ecg_data/cycle_ecg_75_bpm0.0105.csv")
        except:
            # If that fails, try with no header
            df = pd.read_csv(filepath, header=None)
        
        # Handle different data formats
        if df.shape[1] > 1:
            # If multiple columns, assume first is ecg and second is time
            ecg = df.iloc[:, 0].values
            time = df.iloc[:, 1].values
        else:
            # If single column, create time vector based on sampling rate
            ecg = df.iloc[:, 0].values
            time = np.arange(len(ecg)) / 1000.0  # Assume 1000 Hz if not specified
            
        return time, ecg
    except Exception as e:
        print(f"Error reading file: {e}")
        return None, None

def preprocess_ecg(ecg, fs=357.4):
    """Enhanced preprocessing with better filtering and baseline correction."""
    # Remove baseline wander using high-pass filter
    b, a = signal.butter(3, 0.5/(fs/2), 'high')
    ecg_no_baseline = signal.filtfilt(b, a, ecg)
    
    # Apply bandpass filter to isolate ECG frequency components (0.5-40 Hz)
    b, a = signal.butter(3, [0.5/(fs/2), 40/(fs/2)], 'band')
    ecg_filtered = signal.filtfilt(b, a, ecg_no_baseline)
    
    # Smooth signal with Savitzky-Golay filter
    ecg_smoothed = savgol_filter(ecg_filtered, window_length=int(fs/20)+1 if int(fs/20) % 2 == 0 else int(fs/20), polyorder=3)
    
    # Normalize
    ecg_normalized = (ecg_smoothed - np.mean(ecg_smoothed)) / np.std(ecg_smoothed)
    
    return ecg_normalized

def detect_qrs_complexes(ecg, time, fs=357.4):
    """Enhanced QRS detection with adaptive thresholding."""
    # Apply specific bandpass filter for QRS enhancement (5-15 Hz)
    b, a = signal.butter(3, [5/(fs/2), 15/(fs/2)], 'band')
    ecg_qrs = signal.filtfilt(b, a, ecg)
    
    # Square the signal to enhance peaks
    ecg_qrs_squared = ecg_qrs**2
    
    # Apply moving average integration window
    window_size = int(0.08 * fs)  # 80 ms window
    ecg_integrated = np.convolve(ecg_qrs_squared, np.ones(window_size)/window_size, mode='same')
    
    # Apply adaptive threshold for R peak detection
    threshold = 0.6 * np.mean(ecg_integrated)
    min_distance = int(0.2 * fs)  # Minimum 200 ms between peaks
    
    r_peaks, _ = find_peaks(ecg_integrated, height=threshold, distance=min_distance)
    
    # Refine R peak positions using original signal
    refined_r_peaks = []
    for peak in r_peaks:
        # Look for maximum in original signal within +/- 25 ms window
        window_start = max(0, peak - int(0.025 * fs))
        window_end = min(len(ecg), peak + int(0.025 * fs))
        refined_peak = window_start + np.argmax(ecg[window_start:window_end])
        refined_r_peaks.append(refined_peak)
    
    r_peaks = np.array(refined_r_peaks)
    
    return r_peaks, time[r_peaks], ecg[r_peaks]

def find_pqrst_points(ecg, time, r_peaks, fs=357.4):
    """Advanced fiducial point detection for P, Q, R, S, T waves."""
    beat_data = []
    
    # Calculate first derivative of ECG
    ecg_derivative = np.gradient(ecg) * fs
    
    # Process each heartbeat
    for i, r_peak in enumerate(r_peaks):
        beat_info = {'r_idx': r_peak, 'r_time': time[r_peak], 'r_amp': ecg[r_peak]}
        
        # Define search windows
        # For Q point (before R)
        q_window_start = max(0, r_peak - int(0.1 * fs))  # Up to 100 ms before R
        q_search_window = ecg[q_window_start:r_peak]
        q_deriv_window = ecg_derivative[q_window_start:r_peak]
        
        # For S point (after R)
        s_window_end = min(len(ecg), r_peak + int(0.1 * fs))  # Up to 100 ms after R
        s_search_window = ecg[r_peak:s_window_end]
        s_deriv_window = ecg_derivative[r_peak:s_window_end]
        
        # Find Q point - find where derivative crosses zero from negative to positive before R
        q_idx = None
        for j in range(len(q_deriv_window)-1, 0, -1):  # Search backwards from R
            if q_deriv_window[j] >= 0 and q_deriv_window[j-1] < 0:
                q_idx = q_window_start + j
                break
        
        # If zero-crossing method fails, use minimum amplitude
        if q_idx is None and len(q_search_window) > 0:
            q_idx = q_window_start + np.argmin(q_search_window)
        
        # Find S point - find where derivative crosses zero from negative to positive after R
        s_idx = None
        for j in range(1, len(s_deriv_window)):
            if s_deriv_window[j] >= 0 and s_deriv_window[j-1] < 0:
                s_idx = r_peak + j
                break
                
        # If zero-crossing method fails, use minimum amplitude
        if s_idx is None and len(s_search_window) > 0:
            s_idx = r_peak + np.argmin(s_search_window)
        
        # Store Q and S indices if found
        if q_idx is not None:
            beat_info['q_idx'] = q_idx
            beat_info['q_time'] = time[q_idx]
            beat_info['q_amp'] = ecg[q_idx]
        
        if s_idx is not None:
            beat_info['s_idx'] = s_idx
            beat_info['s_time'] = time[s_idx]
            beat_info['s_amp'] = ecg[s_idx]
        
        # Find T wave segment - search after S point
        if s_idx is not None:
            # Define T wave search window (up to 400 ms after S point or to next R peak)
            next_r = r_peaks[i+1] if i < len(r_peaks)-1 else len(ecg)
            t_search_end = min(next_r - int(0.05 * fs), s_idx + int(0.4 * fs))
            
            if t_search_end > s_idx + 10:  # Make sure we have enough points to analyze
                t_search_window = ecg[s_idx:t_search_end]
                t_deriv_window = ecg_derivative[s_idx:t_search_end]
                
                # Find T peak (maximum in the T search window)
                t_peak_idx = s_idx + np.argmax(t_search_window)
                
                # Find T wave onset - inflection point after S and before T peak
                t_onset_window = ecg_derivative[s_idx:t_peak_idx]
                if len(t_onset_window) > 10:
                    # Look for maximum in derivative (steepest upslope)
                    t_onset_idx = s_idx + np.argmax(t_onset_window)
                    beat_info['t_onset_idx'] = t_onset_idx
                    beat_info['t_onset_time'] = time[t_onset_idx]
                    beat_info['t_onset_amp'] = ecg[t_onset_idx]
                
                # Find T wave offset - where curve flattens after T peak
                if t_peak_idx + 5 < len(ecg):
                    t_offset_search = ecg_derivative[t_peak_idx:t_search_end]
                    if len(t_offset_search) > 5:
                        # Look for where derivative approaches zero
                        t_offset_candidates = np.where(abs(t_offset_search) < 0.05)[0]
                        if len(t_offset_candidates) > 0:
                            t_offset_idx = t_peak_idx + t_offset_candidates[0]
                        else:
                            # Fallback - use the minimum of derivative after T peak
                            t_offset_idx = t_peak_idx + np.argmin(abs(t_offset_search))
                        
                        beat_info['t_offset_idx'] = t_offset_idx
                        beat_info['t_offset_time'] = time[t_offset_idx]
                        beat_info['t_offset_amp'] = ecg[t_offset_idx]
        
        # Add beat info if we found the essential points (Q, S, T onset, T offset)
        if all(k in beat_info for k in ['q_idx', 's_idx', 't_onset_idx', 't_offset_idx']):
            # Calculate intervals
            beat_info['qt_interval'] = beat_info['t_offset_time'] - beat_info['q_time']
            beat_info['st_duration'] = beat_info['t_onset_time'] - beat_info['s_time']
            beat_data.append(beat_info)
    
    return beat_data

def analyze_ecg(time, ecg, fs=357.4):
    """Main function to analyze ECG and extract intervals."""
    # Apply preprocessing
    ecg_processed = preprocess_ecg(ecg, fs)
    
    # Detect R peaks
    r_peaks, _, _ = detect_qrs_complexes(ecg_processed, time, fs)
    
    # Find all points and calculate intervals
    beat_data = find_pqrst_points(ecg_processed, time, r_peaks, fs)
    
    return beat_data, ecg_processed, r_peaks

def visualize_results(time, ecg_original, ecg_processed, beat_data, r_peaks):
    """Create detailed visualization of the results."""
    plt.figure(figsize=(15, 10))
    
    # Plot original and processed signals
    plt.subplot(2, 1, 1)
    plt.plot(time, ecg_original, 'b-', label='Original ECG')
    plt.title('Original ECG Signal')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    plt.plot(time, ecg_processed, 'g-', label='Processed ECG')
    
    # Mark all R peaks with black dots
    plt.plot(time[r_peaks], ecg_processed[r_peaks], 'ko', label='R Peaks')
    
    # Define colors for different points
    colors = {
        'q_idx': 'red',
        's_idx': 'purple', 
        't_onset_idx': 'orange',
        't_offset_idx': 'blue'
    }
    
    labels = {
        'q_idx': 'Q Point',
        's_idx': 'S Point',
        't_onset_idx': 'T Onset',
        't_offset_idx': 'T Offset'
    }
    
    # Keep track of which labels have been added
    added_labels = set()
    
    # Plot fiducial points for each beat
    for beat in beat_data:
        for point_type in ['q_idx', 's_idx', 't_onset_idx', 't_offset_idx']:
            if point_type in beat:
                point_idx = beat[point_type]
                label = labels[point_type] if point_type not in added_labels else None
                plt.plot(time[point_idx], ecg_processed[point_idx], 'o', 
                         color=colors[point_type], label=label)
                if label:
                    added_labels.add(point_type)
        
        # Draw QT interval
        if 'q_idx' in beat and 't_offset_idx' in beat:
            q_idx, t_offset_idx = beat['q_idx'], beat['t_offset_idx']
            plt.hlines(
                y=ecg_processed[q_idx] - 0.2,
                xmin=time[q_idx],
                xmax=time[t_offset_idx],
                color='red',
                linestyles='dashed',
                label='QT Interval' if 'QT Interval' not in added_labels else None
            )
            if 'QT Interval' not in added_labels:
                added_labels.add('QT Interval')
        
        # Draw ST duration
        if 's_idx' in beat and 't_onset_idx' in beat:
            s_idx, t_onset_idx = beat['s_idx'], beat['t_onset_idx']
            plt.hlines(
                y=ecg_processed[s_idx] - 0.4,
                xmin=time[s_idx],
                xmax=time[t_onset_idx],
                color='green',
                linestyles='dashed',
                label='ST Duration' if 'ST Duration' not in added_labels else None
            )
            if 'ST Duration' not in added_labels:
                added_labels.add('ST Duration')
    
    plt.title('Processed ECG with Detected Features')
    plt.xlabel('Time (s)')
    plt.ylabel('Normalized Amplitude')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Print detailed statistics
    if beat_data:
        qt_intervals = [beat['qt_interval'] for beat in beat_data]
        st_durations = [beat['st_duration'] for beat in beat_data]
        
        print("\nQT Interval Statistics:")
        print(f"Mean: {np.mean(qt_intervals):.4f} seconds")
        print(f"Std Dev: {np.std(qt_intervals):.4f} seconds")
        print(f"Min: {np.min(qt_intervals):.4f} seconds")
        print(f"Max: {np.max(qt_intervals):.4f} seconds")
        
        print("\nST Duration Statistics:")
        print(f"Mean: {np.mean(st_durations):.4f} seconds")
        print(f"Std Dev: {np.std(st_durations):.4f} seconds")
        print(f"Min: {np.min(st_durations):.4f} seconds")
        print(f"Max: {np.max(st_durations):.4f} seconds")

def main():
    # File path to your CSV file - replace with your actual file path
    filepath = "C:/Users/vidus/Downloads/App (2)/App Installation/ecg_data/cycle_ecg_75_bpm0.0105.csv"  # MODIFY THIS LINE with your CSV file path
    
    # Read data
    time, ecg = read_ecg_data(filepath)
    
    if time is None or ecg is None:
        print("Failed to read ECG data. Please check the file path and format.")
        return
    
    # Estimate sampling frequency
    fs = 357.4  # Default assumption (can be modified)
    if len(time) > 1:
        fs = 1 / (time[1] - time[0])
        print(f"Estimated sampling frequency: {fs:.2f} Hz")
    
    # Analyze ECG
    print("Analyzing ECG signal...")
    beat_data, ecg_processed, r_peaks = analyze_ecg(time, ecg, fs)
    
    # Display results
    if beat_data:
        print(f"Successfully analyzed {len(beat_data)} complete heartbeats")
        
        # Calculate average intervals
        qt_intervals = [beat['qt_interval'] for beat in beat_data]
        st_durations = [beat['st_duration'] for beat in beat_data]
        
        print(f"\nAverage QT Interval: {np.mean(qt_intervals):.4f} seconds")
        print(f"Average ST Duration: {np.mean(st_durations):.4f} seconds")
        
        # Visualize results
        visualize_results(time, ecg, ecg_processed, beat_data, r_peaks)
    else:
        print("No valid beats were fully detected for analysis. Try adjusting parameters.")

if __name__ == "__main__":
    main()