function [pq_segment, st_segment] = extract_ecg_segments(results, t_window)
    % Check if results parameter exists
    if nargin < 1 || isempty(results)
        error('Results parameter is required');
    end
    
    % Verify results structure has required fields
    required_fields = {'t', 'ecg', 'x1', 'at_potential', 'vn_potential'};
    for i = 1:length(required_fields)
        if ~isfield(results, required_fields{i})
            error(['Results structure missing required field: ', required_fields{i}]);
        end
    end
    
    % Filter for time window if provided
    if nargin >= 2 && ~isempty(t_window)
        mask = (results.t >= t_window(1)) & (results.t <= t_window(2));
        t = results.t(mask);
        ecg = results.ecg(mask);
        at_potential = results.at_potential(mask);
        vn_potential = results.vn_potential(mask);
        x1 = results.x1(mask);
    else
        % Use the full signal but skip initial transient (first 10 seconds)
        % Changed from 10 to 0 to handle shorter simulations
        mask = results.t >= 0;
        t = results.t(mask);
        ecg = results.ecg(mask);
        at_potential = results.at_potential(mask);
        vn_potential = results.vn_potential(mask);
        x1 = results.x1(mask);
    end
    
    % Initialize output structures
    pq_segment = struct('time', [], 'values', [], 'mean', NaN);
    st_segment = struct('time', [], 'values', [], 'mean', NaN);
    
    % Find cardiac cycles 
    % Use lower MinPeakDistance for short recordings
    [~, cycle_starts] = findpeaks(x1, 'MinPeakHeight', 0.5, 'MinPeakDistance', 50);
    
    if length(cycle_starts) < 2
        warning('Not enough cardiac cycles found in the data. Trying with lower peak criteria.');
        [~, cycle_starts] = findpeaks(x1, 'MinPeakHeight', 0.3, 'MinPeakDistance', 40);
        
        if length(cycle_starts) < 2
            warning('Still not enough cardiac cycles found. Using entire signal.');
            pq_segment.time = t;
            pq_segment.values = ecg;
            pq_segment.mean = mean(ecg);
            
            st_segment.time = t;
            st_segment.values = ecg;
            st_segment.mean = mean(ecg);
            return;
        end
    end
    
    % Rest of the function remains the same...
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
            if j > 1 && abs(diff(cycle_vn(j-1:j))) > 0.01 % Detect rapid change
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
        qrs_window = min(50, length(cycle_vn) - qrs_start_idx);
        r_peak_range = qrs_start_idx:(qrs_start_idx+qrs_window);
        if isempty(r_peak_range) || length(r_peak_range) < 2 || max(r_peak_range) > length(cycle_vn)
            continue;
        end
        [~, r_peak_idx] = max(cycle_vn(r_peak_range));
        r_peak_idx = r_peak_idx + qrs_start_idx - 1;
        
        % Find end of S wave (when ventricular signal stops declining rapidly)
        s_end_idx = r_peak_idx;
        s_window = min(50, length(cycle_vn) - r_peak_idx);
        for j = (r_peak_idx+1):min(r_peak_idx+s_window, length(cycle_vn))
            if j > 1 && diff(cycle_vn(j-1:j)) > -0.001 % Detect when decline slows
                s_end_idx = j;
                break;
            end
        end
        
        % Find T wave onset (typically 80-120ms after QRS end)
        dt = mean(diff(cycle_t));  % Average time step
        t_step_count = round(0.08 / dt);  % Number of time steps for 80ms
        t_start_idx = min(s_end_idx + t_step_count, length(cycle_ecg));
        
        % ST segment is between S end and T onset
        st_indices = s_end_idx:t_start_idx;
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