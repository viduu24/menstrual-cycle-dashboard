function [pq_segment, st_segment] = extract_ecg_segments(results, t_window)
    % Extract PQ and ST segments from the model output
    %
    % Parameters:
    % results - The output from the model's simulate method
    % t_window - Optional time window [t_min, t_max] to analyze
    %
    % Returns:
    % pq_segment - Structure with time, values, and mean of the PQ segment
    % st_segment - Structure with time, values, and mean of the ST segment
    
    % Filter for time window if provided
    if nargin >= 2 && ~isempty(t_window)
        mask = (results.t >= t_window(1)) & (results.t <= t_window(2));
        t = results.t(mask);
        ecg = results.ecg(mask);
        at_potential = results.at_potential(mask);
        vn_potential = results.vn_potential(mask);
    else
        % Use the full signal but skip initial transient (first 10 seconds)
        mask = results.t >= 10;
        t = results.t(mask);
        ecg = results.ecg(mask);
        at_potential = results.at_potential(mask);
        vn_potential = results.vn_potential(mask);
    end
    
    % Initialize output structures
    pq_segment = struct('time', [], 'values', [], 'mean', NaN);
    st_segment = struct('time', [], 'values', [], 'mean', NaN);
    
    % Find cardiac cycles 
    % (Use x1 peaks as markers for the start of each cycle)
    x1 = results.x1(mask);
    [~, cycle_starts] = findpeaks(x1, 'MinPeakHeight', 0.5, 'MinPeakDistance', 100);
    
    if length(cycle_starts) < 2
        warning('Not enough cardiac cycles found in the data');
        return;
    end
    
    % Process each cardiac cycle (skip first and last for stability)
    for i = 2:(length(cycle_starts)-1)
        cycle_start = cycle_starts(i);
        cycle_end = cycle_starts(i+1);
        
        % Get data for this cardiac cycle
        cycle_t = t(cycle_start:cycle_end);
        cycle_ecg = ecg(cycle_start:cycle_end);
        cycle_at = at_potential(cycle_start:cycle_end);
        cycle_vn = vn_potential(cycle_start:cycle_end);
        
        % Find P wave peak
        [~, p_peak_idx] = findpeaks(cycle_at, 'NPeaks', 1, 'SortStr', 'descend');
        if isempty(p_peak_idx)
            continue;
        end
        
        % Find end of P wave (estimate as when atrial potential drops to 20% of peak)
        p_end_idx = find(cycle_at(p_peak_idx:end) < 0.2 * cycle_at(p_peak_idx), 1, 'first') + p_peak_idx - 1;
        if isempty(p_end_idx)
            continue;
        end
        
        % Find QRS onset (when ventricular potential starts rising rapidly)
        qrs_start_t = cycle_t(p_end_idx) + 0.04; % Typical PR interval is at least 120-200ms
        [~, qrs_start_idx] = min(abs(cycle_t - qrs_start_t));
        for j = qrs_start_idx:length(cycle_vn)
            if abs(diff(cycle_vn(j-1:j))) > 0.01 % Detect rapid change
                qrs_start_idx = j;
                break;
            end
        end
        
        % PQ segment is between P end and QRS onset
        pq_indices = p_end_idx:qrs_start_idx;
        if length(pq_indices) < 3 % Ensure enough points
            continue;
        end
        
        % Find QRS end (R+S waves) - look for peak then rapid decline
        [~, r_peak_idx] = max(cycle_vn(qrs_start_idx:qrs_start_idx+50));
        r_peak_idx = r_peak_idx + qrs_start_idx - 1;
        
        % Find end of S wave (when ventricular signal stops declining rapidly)
        s_end_idx = r_peak_idx;
        for j = (r_peak_idx+1):min(r_peak_idx+50, length(cycle_vn))
            if diff(cycle_vn(j-1:j)) > -0.001 % Detect when decline slows
                s_end_idx = j;
                break;
            end
        end
        
        % Find T wave onset (typically 80-120ms after QRS end)
        t_start_idx = s_end_idx + round(0.08 / (cycle_t(2) - cycle_t(1)));
        
        % ST segment is between S end and T onset
        st_indices = s_end_idx:min(t_start_idx, length(cycle_ecg));
        if length(st_indices) < 3 % Ensure enough points
            continue;
        end
        
        % Store PQ segment data
        pq_segment.time = [pq_segment.time; cycle_t(pq_indices)'];
        pq_segment.values = [pq_segment.values; cycle_ecg(pq_indices)'];
        
        % Store ST segment data
        st_segment.time = [st_segment.time; cycle_t(st_indices)'];
        st_segment.values = [st_segment.values; cycle_ecg(st_indices)'];
    end
    
    % Calculate average values
    pq_segment.mean = mean(pq_segment.values);
    st_segment.mean = mean(st_segment.values);
end