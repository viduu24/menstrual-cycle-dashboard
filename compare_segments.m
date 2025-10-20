function [relation_equation, st_elevation] = compare_segments(model_pq, model_st, paper_pq, paper_st)
    % Compare PQ and ST segments between model and paper ECG
    %
    % Parameters:
    % model_pq - PQ segment data from model
    % model_st - ST segment data from model
    % paper_pq - PQ segment data from paper ECG
    % paper_st - ST segment data from paper ECG
    %
    % Returns:
    % relation_equation - Equation relating model to paper values
    % st_elevation - ST elevation in both model and paper
    
    % Calculate ST elevation
    model_st_elevation = model_st.mean - model_pq.mean;
    paper_st_elevation = paper_st.mean - paper_pq.mean;
    
    % Find scaling factor between model and paper
    scaling_factor = paper_st_elevation / model_st_elevation;
    
    % Create relation equation
    offset = paper_pq.mean - model_pq.mean * scaling_factor;
    
    % Form of equation: paper_value = scaling_factor * model_value + offset
    relation_equation = struct('scaling_factor', scaling_factor, 'offset', offset, ...
        'equation', sprintf('paper_value = %.4f * model_value + %.4f', scaling_factor, offset));
    
    % Return ST elevation information
    st_elevation = struct('model', model_st_elevation, 'paper', paper_st_elevation, ...
        'ratio', scaling_factor);
    
    % Display the results
    fprintf('ST Elevation (model): %.4f\n', model_st_elevation);
    fprintf('ST Elevation (paper): %.4f\n', paper_st_elevation);
    fprintf('Relation Equation: %s\n', relation_equation.equation);
end