import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

def process_ecg(ecg_signal, sampling_rate=250):
    """
    Process ECG signal following specified steps (without FIR filtering):
    1. Detect R-peaks using Hamilton-like approach
    2. Calculate heart rate in BPM
    3. Find QRS complex
    4. Detect P wave
    5. Detect T wave
    
    Parameters:
    -----------
    ecg_signal : numpy array
        ECG signal
    sampling_rate : int
        Sampling rate in Hz
        
    Returns:
    --------
    dict
        Dictionary containing processed data
    """
    result = {}
    
    # Store the original signal
    result['filtered_signal'] = ecg_signal
    
    # Step 1: Detect R-peaks using Hamilton-like approach
    # Implementation of principles from the Hamilton segmenter
    
    # Differentiate
    diff_signal = np.diff(ecg_signal)
    diff_signal = np.append(diff_signal, diff_signal[-1])  # Maintain length
    
    # Square
    squared_signal = diff_signal ** 2
    
    # Moving window integration
    window_size = int(0.08 * sampling_rate)  # 80ms window
    integrated_signal = np.convolve(squared_signal, np.ones(window_size)/window_size, mode='same')
    
    # Find peaks with adaptive threshold
    threshold = 0.33 * np.max(integrated_signal)
    r_peaks, _ = find_peaks(integrated_signal, height=threshold, distance=int(0.25 * sampling_rate))
    
    # Refine R peak detection by finding the maximum in the original signal around each detected peak
    refined_r_peaks = []
    window_size = int(0.05 * sampling_rate)  # 50ms window to search for real R peak
    
    for peak in r_peaks:
        start = max(0, peak - window_size)
        end = min(len(ecg_signal), peak + window_size)
        real_peak = start + np.argmax(ecg_signal[start:end])
        refined_r_peaks.append(real_peak)
    
    r_peaks = np.array(refined_r_peaks, dtype=int)
    result['r_peaks'] = r_peaks
    
    # Step 2: Calculate heart rate in BPM
    # Number of R-peaks divided by recording time in minutes
    recording_time_in_minutes = len(ecg_signal) / sampling_rate / 60
    heart_rate = len(r_peaks) / recording_time_in_minutes if recording_time_in_minutes > 0 else 0
    
    # Alternative calculation based on RR intervals
    if len(r_peaks) > 1:
        rr_intervals = np.diff(r_peaks) / sampling_rate
        valid_intervals = rr_intervals[(rr_intervals >= 0.33) & (rr_intervals <= 1.5)]
        
        if len(valid_intervals) > 0:
            mean_rr_interval = np.mean(valid_intervals)
            heart_rate = 60 / mean_rr_interval  # Convert to BPM
    
    result['heart_rate_bpm'] = heart_rate
    
    # Initialize arrays for the waves
    q_indices = []
    s_indices = []
    p_indices = []
    t_indices = []
    
    # Step 3: Find QRS complex (Q and S points)
    for r_idx in r_peaks:
        # Get Q point (lowest point in 80ms before R)
        q_search_start = max(0, r_idx - int(0.08 * sampling_rate))
        q_search_window = ecg_signal[q_search_start:r_idx]
        if len(q_search_window) > 0:
            q_idx = q_search_start + np.argmin(q_search_window)
            q_indices.append(int(q_idx))
        
        # Get S point (lowest point in 80ms after R)
        s_search_end = min(len(ecg_signal) - 1, r_idx + int(0.08 * sampling_rate))
        s_search_window = ecg_signal[r_idx:s_search_end]
        if len(s_search_window) > 0:
            s_idx = r_idx + np.argmin(s_search_window)
            s_indices.append(int(s_idx))
    
    result['q_indices'] = np.array(q_indices, dtype=int)
    result['s_indices'] = np.array(s_indices, dtype=int)
    
    # Step 4: Find P wave (highest point in 200ms before Q)
    for q_idx in q_indices:
        p_search_start = max(0, q_idx - int(0.2 * sampling_rate))
        p_search_window = ecg_signal[p_search_start:q_idx]
        if len(p_search_window) > 0:
            p_idx = p_search_start + np.argmax(p_search_window)
            p_indices.append(int(p_idx))
    
    result['p_indices'] = np.array(p_indices, dtype=int)
    
    # Step 5: Find T wave (highest point in 400ms after S)
    for i, s_idx in enumerate(s_indices):
        # Limit T-wave search to before the next P-wave or end of signal
        t_search_end = len(ecg_signal) - 1
        if i < len(p_indices) - 1:
            t_search_end = min(t_search_end, p_indices[i+1])
        t_search_end = min(t_search_end, s_idx + int(0.4 * sampling_rate))
        
        t_search_window = ecg_signal[s_idx:t_search_end]
        if len(t_search_window) > 0:
            t_idx = s_idx + np.argmax(t_search_window)
            t_indices.append(int(t_idx))
    
    result['t_indices'] = np.array(t_indices, dtype=int)
    
    return result

def load_ecg_from_csv(file_path, column_name=None, header=0):
    """
    Load ECG data from a CSV file
    
    Parameters:
    -----------
    file_path : str
        Path to the CSV file
    column_name : str or int, optional
        Name or index of the column containing ECG data
    header : int, optional
        Row number to use as header (0-indexed), None means no header
        
    Returns:
    --------
    numpy.ndarray
        ECG signal
    """
    try:
        # Read the CSV file
        df = pd.read_csv('ecg_2600ms.csv', header=header)
        
        # If column_name is provided, use it to get the ECG data
        if column_name is not None:
            if column_name in df.columns:
                ecg_data = df[column_name].values
            elif isinstance(column_name, int) and column_name < len(df.columns):
                ecg_data = df.iloc[:, column_name].values
            else:
                print(f"Column '{column_name}' not found. Using first column.")
                ecg_data = df.iloc[:, 0].values
        else:
            # If no column_name provided, use the first column
            ecg_data = df.iloc[:, 0].values
        
        return ecg_data
    
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        return None

def plot_ecg_features(ecg_signal, result, sampling_rate=250):
    """
    Plot ECG signal with detected features
    """
    time = np.arange(len(ecg_signal)) / sampling_rate
    
    plt.figure(figsize=(15, 10))
    
    # Main plot for full signal
    plt.subplot(211)
    plt.plot(time, ecg_signal, label='ECG Signal')
    
    # Plot R peaks
    if len(result['r_peaks']) > 0:
        plt.scatter(result['r_peaks']/sampling_rate, 
                    ecg_signal[result['r_peaks']], 
                    color='red', s=50, marker='x', label='R peaks')
    
    # Plot Q points
    if len(result['q_indices']) > 0:
        plt.scatter(result['q_indices']/sampling_rate, 
                    ecg_signal[result['q_indices']], 
                    color='green', label='Q points')
    
    # Plot S points
    if len(result['s_indices']) > 0:
        plt.scatter(result['s_indices']/sampling_rate, 
                    ecg_signal[result['s_indices']], 
                    color='purple', label='S points')
    
    # Plot P waves
    if len(result['p_indices']) > 0:
        plt.scatter(result['p_indices']/sampling_rate, 
                    ecg_signal[result['p_indices']], 
                    color='orange', label='P waves')
    
    # Plot T waves
    if len(result['t_indices']) > 0:
        plt.scatter(result['t_indices']/sampling_rate, 
                    ecg_signal[result['t_indices']], 
                    color='brown', label='T waves')
    
    plt.title(f'ECG Signal Analysis (Heart Rate: {result["heart_rate_bpm"]:.1f} BPM)')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True)
    
    # Subplot for detailed view
    plt.subplot(212)
    
    # Show a 5-second segment or the first beat, whichever is shorter
    if len(result['r_peaks']) > 1:
        segment_end = min(result['r_peaks'][1] + int(0.4 * sampling_rate), len(ecg_signal))
        segment_start = max(0, result['r_peaks'][0] - int(0.2 * sampling_rate))
    else:
        segment_duration = min(5, len(ecg_signal)/sampling_rate)
        segment_start = 0
        segment_end = int(segment_duration * sampling_rate)
    
    plt.plot(time[segment_start:segment_end], ecg_signal[segment_start:segment_end])
    
    # Plot features in the segment
    for feature_name, color, label in [
        ('r_peaks', 'red', 'R peaks'),
        ('q_indices', 'green', 'Q points'),
        ('s_indices', 'purple', 'S points'),
        ('p_indices', 'orange', 'P waves'),
        ('t_indices', 'brown', 'T waves')
    ]:
        indices = result[feature_name]
        segment_indices = indices[(indices >= segment_start) & (indices < segment_end)]
        if len(segment_indices) > 0:
            plt.scatter(segment_indices/sampling_rate, 
                        ecg_signal[segment_indices], 
                        color=color, label=label)
    
    plt.title('ECG Detail View')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.show()

# Example usage
if __name__ == "__main__":
    # Set your parameters
    csv_file_path = "your_ecg_data.csv"  # Replace with your CSV file path
    sampling_rate = 100  # Hz - adjust according to your data
    
    # Specify which column contains the ECG data (by name or index)
    # If your CSV has headers, use column name; otherwise use column index (0-based)
    # Example: column_name="ecg_values" or column_name=0
    column_name = 0  # Replace with your column name or index
    
    # Load ECG data from CSV
    ecg_signal = load_ecg_from_csv(csv_file_path, column_name=column_name)
    
    if ecg_signal is not None:
        # Process the ECG signal
        result = process_ecg(ecg_signal, sampling_rate)
        
        # Print results
        print(f"Heart Rate: {result['heart_rate_bpm']:.1f} BPM")
        print(f"Detected {len(result['r_peaks'])} R-peaks")
        print(f"Detected {len(result['q_indices'])} Q-points")
        print(f"Detected {len(result['s_indices'])} S-points")
        print(f"Detected {len(result['p_indices'])} P-waves")
        print(f"Detected {len(result['t_indices'])} T-waves")
        
        # Plot the ECG with detected features
        plot_ecg_features(ecg_signal, sampling_rate=sampling_rate, result=result)
    else:
        print("Failed to load ECG data. Please check your file path and format.")