function cardiac_model_main()
    % Create model
    model = ecg_100();
    
    % Define simulation parameters
    t_span = [0, 20];
    initial_conditions = [0.01, 0.01, 0.01, 0.0, 0.05, 0.0, ...
                         0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    
    % Run simulation
    disp('Running simulation...');
    results = model.simulate(t_span, initial_conditions);
    disp('Simulation complete.');
    
    % Export full ECG data
    full_df = model.export_ecg_data(results, 'full_ecg_simulation.csv');
    
    % Export specific time window (2600ms segment)
    cycle_df = model.export_ecg_data(results, 'ecg_2600ms.csv', 15, 20);
    
    % Call extract_ecg_segments with results and time window
    [pq_segment, st_segment] = extract_ecg_segments(results, [15, 20]);
    
    % Display segment information
    disp('PQ Segment Mean: ');
    disp(pq_segment.mean);
    disp('ST Segment Mean: ');
    disp(st_segment.mean);
    
    % Plotting
    disp('Generating plots...');
    figure(1);
    model.plot_results_standard(results, 15, 20);
    title('Cardiac Signal');
    
    figure(2);
    model.plot_results_diagram(results, 15, 20);
    title('Detailed Segment Analysis');
    
    disp('Done.');
end