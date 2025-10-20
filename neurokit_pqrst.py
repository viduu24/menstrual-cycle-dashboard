import neurokit2 as nk
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.gridspec import GridSpec
import matplotlib.patches as patches

def create_detailed_ecg_plots(folder_path, sampling_rate=250):
    """
    Create detailed plots to verify ECG interval detection algorithm accuracy.
    
    Parameters:
    -----------
    folder_path : str
        Path to the folder containing ECG CSV files
    sampling_rate : int
        Sampling rate of the ECG signals in Hz (default: 250 Hz)
    
    Returns:
    --------
    dict
        Dictionary with processing results and validation metrics
    """
    # Get all CSV files in the folder
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    
    # Create directory for detailed plots
    detailed_plots_dir = os.path.join(folder_path, "ecg_detailed_plots")
    os.makedirs(detailed_plots_dir, exist_ok=True)
    
    # Dictionary to store validation results
    validation_results = {}
    
    # Process each file
    for file in csv_files:
        file_path = os.path.join(folder_path, file)
        print(f"Creating detailed plots for {file}...")
        
        try:
            # Read the CSV file
            data = pd.read_csv(file_path)
            
            # Determine which column contains ECG data
            if len(data.columns) == 1:
                ecg_signal = data.iloc[:, 0].values
            else:
                # Try to automatically detect ECG data column
                for col in data.columns:
                    if 'ecg' in col.lower() or 'signal' in col.lower():
                        ecg_signal = data[col].values
                        break
                else:
                    # If no obvious column name, use the second column (assuming first might be time)
                    ecg_signal = data.iloc[:, 1].values if len(data.columns) > 1 else data.iloc[:, 0].values
            
            signal_duration = len(ecg_signal) / sampling_rate
            print(f"  Signal length: {len(ecg_signal)} samples ({signal_duration:.2f} seconds)")
            
            # Clean the ECG signal
            ecg_cleaned = nk.ecg_clean(ecg_signal, sampling_rate=sampling_rate)
            
            # Find R-peaks
            rpeaks, info = nk.ecg_peaks(ecg_cleaned, sampling_rate=sampling_rate)
            r_peak_indices = np.where(rpeaks["ECG_R_Peaks"] == 1)[0]
            
            if len(r_peak_indices) < 2:
                print(f"  Not enough R-peaks detected in {file} (found {len(r_peak_indices)}). Trying alternative method...")
                # Try with a more sensitive method for challenging signals
                from scipy.signal import find_peaks
                normalized = (ecg_cleaned - np.mean(ecg_cleaned)) / np.std(ecg_cleaned)
                r_peak_indices, _ = find_peaks(normalized, height=1.0, distance=int(0.5 * sampling_rate))
                
                if len(r_peak_indices) < 2:
                    print(f"  Still not enough R-peaks found. Cannot create detailed plots.")
                    continue
            
            print(f"  Detected {len(r_peak_indices)} R-peaks")
            
            # Try both delineation methods and compare
            try:
                # Method 1: Peak delineation
                _, waves_peak = nk.ecg_delineate(ecg_cleaned, r_peak_indices, sampling_rate=sampling_rate, method="peaks")
                
                # Method 2: Wavelet delineation (usually more accurate)
                _, waves_dwt = nk.ecg_delineate(ecg_cleaned, r_peak_indices, sampling_rate=sampling_rate, method="dwt")
                
                # Choose the method that produced better results
                # (typically dwt is better but might fail on some signals)
                waves = waves_dwt  # Prefer wavelet method
                delineation_method = "wavelet"
                
                # If dwt failed to detect important points, fall back to peaks method
                key_waves = ['ECG_P_Peaks', 'ECG_Q_Peaks', 'ECG_R_Peaks', 'ECG_S_Peaks', 'ECG_T_Peaks']
                missing_waves = [w for w in key_waves if waves_dwt.get(w) is None or len(waves_dwt.get(w, [])) == 0]
                
                if missing_waves:
                    print(f"  Wavelet method missing: {missing_waves}. Trying peak method.")
                    # Check if peaks method detected these missing waves
                    missing_in_peaks = [w for w in missing_waves if waves_peak.get(w) is None or len(waves_peak.get(w, [])) == 0]
                    
                    if len(missing_in_peaks) < len(missing_waves):
                        print("  Peak method detected more waves. Using combined approach.")
                        # Copy the missing points from peaks method
                        for wave in missing_waves:
                            if wave in waves_peak and waves_peak[wave] is not None and len(waves_peak[wave]) > 0:
                                waves[wave] = waves_peak[wave]
                        delineation_method = "combined"
                    elif len(missing_in_peaks) == len(missing_waves):
                        print("  Both methods missing same waves. Using available points.")
                
            except Exception as e:
                print(f"  Error in standard delineation: {str(e)}. Using peak method only.")
                _, waves = nk.ecg_delineate(ecg_cleaned, r_peak_indices, sampling_rate=sampling_rate, method="peaks")
                delineation_method = "peaks"
            
            # Now create the detailed plots
            
            # 1. OVERVIEW PLOT - Full signal with all detected points
            plt.figure(figsize=(15, 10))
            plt.suptitle(f"ECG Analysis Overview - {file}", fontsize=16)
            
            # Time axis in seconds
            time = np.arange(len(ecg_cleaned)) / sampling_rate
            
            # Plot the full cleaned signal
            plt.subplot(2, 1, 1)
            plt.plot(time, ecg_cleaned, 'b-', label='Cleaned ECG')
            plt.scatter(r_peak_indices / sampling_rate, ecg_cleaned[r_peak_indices], color='red', s=50, label='R-peaks')
            
            # Highlight regions that will be examined in detail
            cycle_to_highlight = min(3, len(r_peak_indices)-1)  # Use 3rd cycle or last if fewer
            if cycle_to_highlight >= 1:
                start_sample = r_peak_indices[cycle_to_highlight-1]
                end_sample = r_peak_indices[cycle_to_highlight+1] if cycle_to_highlight+1 < len(r_peak_indices) else len(ecg_cleaned)-1
                plt.axvspan(start_sample/sampling_rate, end_sample/sampling_rate, color='yellow', alpha=0.3, label='Detailed View')
            
            plt.title("Full ECG Signal")
            plt.xlabel("Time (s)")
            plt.ylabel("Amplitude")
            plt.legend()
            
            # Plot heart rate over time
            plt.subplot(2, 1, 2)
            
            # Calculate instantaneous heart rate
            if len(r_peak_indices) > 1:
                rr_intervals = np.diff(r_peak_indices) / sampling_rate  # in seconds
                hr = 60 / rr_intervals  # convert to BPM
                hr_times = r_peak_indices[1:] / sampling_rate  # time points for heart rates
                
                plt.plot(hr_times, hr, 'r-o', label='Heart Rate')
                plt.axhline(y=np.mean(hr), color='k', linestyle='--', label=f'Mean HR: {np.mean(hr):.1f} BPM')
                plt.title("Heart Rate Variability")
                plt.xlabel("Time (s)")
                plt.ylabel("Heart Rate (BPM)")
                plt.legend()
            else:
                plt.text(0.5, 0.5, "Not enough R-peaks to calculate heart rate", 
                         horizontalalignment='center', verticalalignment='center')
            
            plt.tight_layout()
            plt.savefig(os.path.join(detailed_plots_dir, f"{Path(file).stem}_overview.png"))
            plt.close()
            
            # 2. DETAILED CYCLE ANALYSIS - Zoom in on specific cardiac cycles
            # For the most accurate interval detection verification
            
            # Choose a representative cycle to analyze in detail
            # Try to select a cycle where most waves were detected
            best_cycle = None
            for i in range(min(len(r_peak_indices)-1, 10)):  # Check first 10 cycles or fewer
                has_all_waves = True
                for wave_type in ['ECG_P_Peaks', 'ECG_Q_Peaks', 'ECG_S_Peaks', 'ECG_T_Peaks']:
                    if (waves.get(wave_type) is None or i >= len(waves.get(wave_type, [])) or 
                        waves[wave_type][i] is None):
                        has_all_waves = False
                        break
                
                if has_all_waves:
                    best_cycle = i
                    break
            
            if best_cycle is None:
                # If no complete cycle found, just use cycle_to_highlight
                best_cycle = cycle_to_highlight
            
            # Create a detailed plot of the selected cardiac cycle
            if best_cycle < len(r_peak_indices)-1:
                # Define window around the selected beat (from previous R-peak to next R-peak)
                if best_cycle > 0:
                    start_idx = r_peak_indices[best_cycle-1]
                else:
                    start_idx = max(0, r_peak_indices[best_cycle] - int(0.4 * sampling_rate))
                
                end_idx = r_peak_indices[best_cycle+1]
                
                # Allow a margin before and after for better visualization
                margin = int(0.1 * sampling_rate)  # 100ms margin
                plot_start = max(0, start_idx - margin)
                plot_end = min(len(ecg_cleaned), end_idx + margin)
                
                fig = plt.figure(figsize=(15, 12))
                gs = GridSpec(3, 1, height_ratios=[3, 1, 1])
                
                # Main ECG plot with annotations
                ax1 = fig.add_subplot(gs[0])
                ax1.plot(time[plot_start:plot_end], ecg_cleaned[plot_start:plot_end], 'b-', linewidth=2)
                
                # Mark R-peaks
                for idx in r_peak_indices:
                    if plot_start <= idx <= plot_end:
                        ax1.axvline(x=idx/sampling_rate, color='red', linestyle='--', alpha=0.5)
                        ax1.plot(idx/sampling_rate, ecg_cleaned[idx], 'ro', markersize=8, label='_')
                
                # Dictionary to store detected points for this beat
                detected_points = {}
                
                # Color and marker for each wave type
                wave_properties = {
                    'ECG_P_Peaks': ('green', 'o', 'P'),
                    'ECG_P_Onsets': ('lightgreen', '^', 'P onset'),
                    'ECG_P_Offsets': ('lightgreen', 'v', 'P offset'),
                    'ECG_Q_Peaks': ('purple', 's', 'Q'),
                    'ECG_R_Peaks': ('red', 'o', 'R'),
                    'ECG_S_Peaks': ('orange', 's', 'S'),
                    'ECG_T_Peaks': ('blue', 'o', 'T'),
                    'ECG_T_Onsets': ('lightblue', '^', 'T onset'),
                    'ECG_T_Offsets': ('lightblue', 'v', 'T offset')
                }
                
                # Add all detected points to plot
                legend_elements = []
                for wave_type, (color, marker, label) in wave_properties.items():
                    if waves.get(wave_type) is not None and best_cycle < len(waves[wave_type]):
                        point = waves[wave_type][best_cycle]
                        if point is not None and plot_start <= point <= plot_end:
                            ax1.plot(point/sampling_rate, ecg_cleaned[point], marker, color=color, 
                                    markersize=8, label=label)
                            detected_points[wave_type] = point
                            # Only add to legend once
                            if label not in [l.get_label() for l in legend_elements]:
                                legend_elements.append(ax1.plot([], [], marker, color=color, label=label)[0])
                
                # Calculate and mark intervals if possible
                intervals = {}
                
                # QRS duration (Q to S)
                if 'ECG_Q_Peaks' in detected_points and 'ECG_S_Peaks' in detected_points:
                    q_point = detected_points['ECG_Q_Peaks']
                    s_point = detected_points['ECG_S_Peaks']
                    qrs_duration = (s_point - q_point) / sampling_rate * 1000  # in ms
                    intervals['QRS'] = qrs_duration
                    
                    # Draw QRS interval
                    y_level = np.min(ecg_cleaned[plot_start:plot_end]) - 0.1
                    ax1.annotate('', xy=(s_point/sampling_rate, y_level), xytext=(q_point/sampling_rate, y_level),
                                arrowprops=dict(arrowstyle='<->', color='purple', lw=2))
                    ax1.text((q_point + s_point)/(2*sampling_rate), y_level-0.05, 
                            f'QRS: {qrs_duration:.1f} ms', ha='center', fontsize=10, color='purple')
                
                # PQ/PR interval (P onset to Q)
                if 'ECG_P_Onsets' in detected_points and 'ECG_Q_Peaks' in detected_points:
                    p_onset = detected_points['ECG_P_Onsets']
                    q_point = detected_points['ECG_Q_Peaks']
                    pq_interval = (q_point - p_onset) / sampling_rate * 1000  # in ms
                    intervals['PQ'] = pq_interval
                    
                    # Draw PQ interval
                    y_level = np.min(ecg_cleaned[plot_start:plot_end]) - 0.2
                    ax1.annotate('', xy=(q_point/sampling_rate, y_level), xytext=(p_onset/sampling_rate, y_level),
                                arrowprops=dict(arrowstyle='<->', color='green', lw=2))
                    ax1.text((p_onset + q_point)/(2*sampling_rate), y_level-0.05, 
                            f'PQ: {pq_interval:.1f} ms', ha='center', fontsize=10, color='green')
                
                # QT interval (Q to T offset)
                if 'ECG_Q_Peaks' in detected_points and 'ECG_T_Offsets' in detected_points:
                    q_point = detected_points['ECG_Q_Peaks']
                    t_offset = detected_points['ECG_T_Offsets']
                    qt_interval = (t_offset - q_point) / sampling_rate * 1000  # in ms
                    intervals['QT'] = qt_interval
                    
                    # Draw QT interval
                    y_level = np.min(ecg_cleaned[plot_start:plot_end]) - 0.3
                    ax1.annotate('', xy=(t_offset/sampling_rate, y_level), xytext=(q_point/sampling_rate, y_level),
                                arrowprops=dict(arrowstyle='<->', color='blue', lw=2))
                    ax1.text((q_point + t_offset)/(2*sampling_rate), y_level-0.05, 
                            f'QT: {qt_interval:.1f} ms', ha='center', fontsize=10, color='blue')
                
                # RR interval
                if best_cycle < len(r_peak_indices)-1:
                    current_r = r_peak_indices[best_cycle]
                    next_r = r_peak_indices[best_cycle+1]
                    rr_interval = (next_r - current_r) / sampling_rate * 1000  # in ms
                    intervals['RR'] = rr_interval
                    hr = 60000 / rr_interval  # heart rate in BPM
                    
                    # Draw RR interval
                    y_level = np.min(ecg_cleaned[plot_start:plot_end]) - 0.4
                    ax1.annotate('', xy=(next_r/sampling_rate, y_level), xytext=(current_r/sampling_rate, y_level),
                                arrowprops=dict(arrowstyle='<->', color='red', lw=2))
                    ax1.text((current_r + next_r)/(2*sampling_rate), y_level-0.05, 
                            f'RR: {rr_interval:.1f} ms (HR: {hr:.1f} BPM)', ha='center', fontsize=10, color='red')
                
                ax1.set_title(f"Detailed Cardiac Cycle Analysis - Beat {best_cycle+1}", fontsize=14)
                ax1.set_xlabel("Time (s)")
                ax1.set_ylabel("Amplitude")
                ax1.legend(handles=legend_elements, loc='upper right')
                ax1.grid(True, alpha=0.3)
                
                # Plot the first derivative to help identify wave onsets/offsets
                ax2 = fig.add_subplot(gs[1], sharex=ax1)
                # Calculate first derivative
                derivative = np.gradient(ecg_cleaned[plot_start:plot_end])
                ax2.plot(time[plot_start:plot_end], derivative, 'g-')
                ax2.set_title("First Derivative (dV/dt)")
                ax2.set_ylabel("dV/dt")
                ax2.grid(True, alpha=0.3)
                
                # Plot detected intervals as bars for comparison
                ax3 = fig.add_subplot(gs[2])
                interval_colors = {'QRS': 'purple', 'PQ': 'green', 'QT': 'blue', 'RR': 'red'}
                
                if intervals:
                    interval_names = list(intervals.keys())
                    interval_values = [intervals[key] for key in interval_names]
                    
                    bars = ax3.bar(interval_names, interval_values, color=[interval_colors[key] for key in interval_names])
                    
                    # Add reference lines for normal values
                    normal_values = {'QRS': 100, 'PQ': 160, 'QT': 400, 'RR': 800}
                    for i, key in enumerate(interval_names):
                        if key in normal_values:
                            ax3.axhline(y=normal_values[key], color='k', linestyle='--', alpha=0.5, 
                                      xmin=i/len(interval_names), xmax=(i+1)/len(interval_names))
                    
                    # Add value labels on top of bars
                    for bar in bars:
                        height = bar.get_height()
                        ax3.text(bar.get_x() + bar.get_width()/2., height + 5,
                                f'{height:.1f}', ha='center', va='bottom')
                    
                    ax3.set_title("Measured Intervals Compared to Normal Values")
                    ax3.set_ylabel("Duration (ms)")
                else:
                    ax3.text(0.5, 0.5, "No intervals could be measured", 
                             horizontalalignment='center', verticalalignment='center')
                
                plt.tight_layout()
                plt.savefig(os.path.join(detailed_plots_dir, f"{Path(file).stem}_detailed_beat.png"))
                plt.close()
                
                # 3. CREATE VERIFICATION GRID - Multiple beats for comparison
                # This helps verify consistency across different beats
                
                num_beats_to_show = min(4, len(r_peak_indices)-1)
                if num_beats_to_show >= 2:
                    fig, axes = plt.subplots(num_beats_to_show, 1, figsize=(12, 3*num_beats_to_show))
                    fig.suptitle(f"Multiple Beat Verification - {file}", fontsize=16)
                    
                    all_beat_intervals = []
                    
                    for i in range(num_beats_to_show):
                        if i >= len(r_peak_indices)-1:
                            break
                            
                        # Define window around this beat
                        current_r = r_peak_indices[i]
                        next_r = r_peak_indices[i+1]
                        
                        # Window with margins
                        margin = int(0.2 * sampling_rate)  # 200ms margin
                        beat_start = max(0, current_r - margin)
                        beat_end = min(len(ecg_cleaned), next_r + margin)
                        
                        ax = axes[i] if num_beats_to_show > 1 else axes
                        ax.plot(time[beat_start:beat_end], ecg_cleaned[beat_start:beat_end], 'b-')
                        
                        # Mark key points for this beat
                        beat_points = {}
                        for wave_type, (color, marker, _) in wave_properties.items():
                            if waves.get(wave_type) is not None and i < len(waves[wave_type]) and waves[wave_type][i] is not None:
                                point = waves[wave_type][i]
                                if beat_start <= point <= beat_end:
                                    ax.plot(point/sampling_rate, ecg_cleaned[point], marker, color=color, markersize=6)
                                    beat_points[wave_type] = point
                        
                        # Calculate intervals for this beat
                        beat_intervals = {}
                        
                        # QRS duration
                        if 'ECG_Q_Peaks' in beat_points and 'ECG_S_Peaks' in beat_points:
                            qrs = (beat_points['ECG_S_Peaks'] - beat_points['ECG_Q_Peaks']) / sampling_rate * 1000
                            beat_intervals['QRS'] = qrs
                        
                        # PQ interval
                        if 'ECG_P_Onsets' in beat_points and 'ECG_Q_Peaks' in beat_points:
                            pq = (beat_points['ECG_Q_Peaks'] - beat_points['ECG_P_Onsets']) / sampling_rate * 1000
                            beat_intervals['PQ'] = pq
                        
                        # QT interval
                        if 'ECG_Q_Peaks' in beat_points and 'ECG_T_Offsets' in beat_points:
                            qt = (beat_points['ECG_T_Offsets'] - beat_points['ECG_Q_Peaks']) / sampling_rate * 1000
                            beat_intervals['QT'] = qt
                        
                        # RR interval
                        rr = (next_r - current_r) / sampling_rate * 1000
                        beat_intervals['RR'] = rr
                        
                        all_beat_intervals.append(beat_intervals)
                        
                        # Add interval text
                        interval_text = []
                        for int_name, int_val in beat_intervals.items():
                            interval_text.append(f"{int_name}: {int_val:.1f} ms")
                        
                        ax.set_title(f"Beat {i+1}: " + " | ".join(interval_text))
                        ax.set_xlabel("Time (s)")
                        ax.set_ylabel("Amplitude")
                        ax.grid(True, alpha=0.3)
                    
                    plt.tight_layout()
                    plt.savefig(os.path.join(detailed_plots_dir, f"{Path(file).stem}_multibeat_verification.png"))
                    plt.close()
                    
                    # Calculate consistency metrics across beats
                    interval_consistency = {}
                    for int_type in ['QRS', 'PQ', 'QT', 'RR']:
                        values = [beat.get(int_type, np.nan) for beat in all_beat_intervals]
                        valid_values = [v for v in values if not np.isnan(v)]
                        
                        if valid_values:
                            mean_val = np.mean(valid_values)
                            std_val = np.std(valid_values)
                            cv = std_val / mean_val * 100 if mean_val > 0 else np.nan  # coefficient of variation
                            
                            interval_consistency[int_type] = {
                                'mean': mean_val,
                                'std': std_val,
                                'cv_percent': cv,  # lower is better, <10% typically good
                                'num_beats_detected': len(valid_values)
                            }
                
                    # Save consistency info
                    validation_results[file] = {
                        'delineation_method': delineation_method,
                        'num_beats_analyzed': num_beats_to_show,
                        'signal_duration': signal_duration,
                        'consistency': interval_consistency
                    }
            
            print(f"  Detailed plots created successfully for {file}")
            
        except Exception as e:
            print(f"Error creating detailed plots for {file}: {str(e)}")
    
    # Create summary visualization of consistency across files
    if validation_results:
        try:
            # Extract consistency metrics
            files = []
            qrs_cv = []
            pq_cv = []
            qt_cv = []
            
            for file, results in validation_results.items():
                cons = results.get('consistency', {})
                files.append(Path(file).stem)
                
                qrs_cv.append(cons.get('QRS', {}).get('cv_percent', np.nan))
                pq_cv.append(cons.get('PQ', {}).get('cv_percent', np.nan))
                qt_cv.append(cons.get('QT', {}).get('cv_percent', np.nan))
            
            # Create consistency plot
            plt.figure(figsize=(12, 8))
            
            width = 0.25
            x = np.arange(len(files))
            
            plt.bar(x - width, qrs_cv, width, label='QRS CV%', color='purple')
            plt.bar(x, pq_cv, width, label='PQ CV%', color='green')
            plt.bar(x + width, qt_cv, width, label='QT CV%', color='blue')
            
            # Add reference line for good consistency threshold (10%)
            plt.axhline(y=10, color='r', linestyle='--', label='Good Consistency Threshold (10%)')
            
            plt.xlabel('ECG Files')
            plt.ylabel('Coefficient of Variation (%)')
            plt.title('Consistency of Interval Measurements Across Beats')
            plt.xticks(x, files, rotation=45)
            plt.legend()
            plt.tight_layout()
            
            plt.savefig(os.path.join(folder_path, "ecg_algorithm_consistency.png"))
            plt.close()
            
            # Save validation metrics to CSV
            validation_df = []
            for file, metrics in validation_results.items():
                cons = metrics.get('consistency', {})
                for int_type in ['QRS', 'PQ', 'QT', 'RR']:
                    if int_type in cons:
                        row = {
                            'file': file,
                            'interval_type': int_type,
                            'mean_ms': cons[int_type]['mean'],
                            'std_ms': cons[int_type]['std'],
                            'cv_percent': cons[int_type]['cv_percent'],
                            'num_beats_detected': cons[int_type]['num_beats_detected'],
                            'delineation_method': metrics['delineation_method']
                        }
                        validation_df.append(row)
            
            if validation_df:
                validation_metrics = pd.DataFrame(validation_df)
                validation_metrics.to_csv(os.path.join(folder_path, "ecg_algorithm_validation.csv"), index=False)
                print(f"\nValidation metrics saved to {os.path.join(folder_path, 'ecg_algorithm_validation.csv')}")
                
                # Print summary statistics
                print("\nAlgorithm Validation Summary:")
                summary = validation_metrics.groupby('interval_type').agg({
                    'cv_percent': ['mean', 'min', 'max'],
                    'num_beats_detected': 'mean'
                })
                print(summary)
        
        except Exception as e:
            print(f"Error creating summary visualization: {str(e)}")
    
    return validation_results

def main():
    # Replace with the path to your folder containing ECG CSV files
    folder_path = "C:/Users/vidus/OneDrive/Desktop/8th sem project/normal ecg digitised"
    
    # Set the sampling rate to match your data
    sampling_rate = 189.9999962  # Hz - adjust this to match your recording's sample rate
    
    print(f"Creating detailed ECG plots for files in {folder_path}...")
    validation_results = create_detailed_ecg_plots(folder_path, sampling_rate)
    
    print("\nDetailed plots generated successfully!")
    print(f"The plots can be found in {os.path.join(folder_path, 'ecg_detailed_plots')}")
    
    if validation_results:
        # Count the number of files where each interval was detected successfully
        interval_detection = {'QRS': 0, 'PQ': 0, 'QT': 0, 'RR': 0}
        
        for file, results in validation_results.items():
            cons = results.get('consistency', {})
            for int_type in interval_detection.keys():
                if int_type in cons and cons[int_type].get('num_beats_detected', 0) > 0:
                    interval_detection[int_type] += 1
        
        print("\nInterval Detection Success Rates:")
        total_files = len(validation_results)
        for int_type, count in interval_detection.items():
            success_rate = count / total_files * 100 if total_files > 0 else 0
            print(f"  {int_type}: {count}/{total_files} files ({success_rate:.1f}%)")

if __name__ == "__main__":
    main()