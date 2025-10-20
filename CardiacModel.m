classdef CardiacModel < handle
    properties
        % Parameters for SA, AV, and HP oscillators (i=1,2,3)
        alpha = 77.5
        model.b = [0.0130, 0.0120, 0.0110];
% model.mu1 = [0.230, 0.230, 0.130];
        % Oscillator parameters
        k = [5, 5, 4.5]  % k_i
        b = [0.0113, 0.0105, 0.0095]  % b_i
        a = [0.18, 0.1, 0.3]  % a_i
        
        epsilon = [0.004, 0.004, 0.003]  % ε_i
        mu1 = [0.2, 0.2, 0.1]  % μ1_i
        mu2 = [0.3, 0.3, 0.34]  % μ2_i
        
        % Coupling coefficients
        C_AV_SA = 0.5
        C_SA_AV = 2.5
        C_HP_AV = 0.4
        C_AV_HP = 2.5
        
        % Time delays
        tau_AV_SA = 0.02
        tau_SA_AV = 0.05
        tau_HP_AV = 0.02
        tau_AV_HP = 0.07
        %tau_AV_HP=0.1
        % Parameters for ECG wave equations
        % P wave
        k1 = 4e3
        c1 = 0.26
        %c1=0.30
        b1 = 0.0
        d1 = 0.4
        h1 = 0.004
        g1 = 1.0
        w11 = 0.13
        w12 = 1.0
        
        % Ta wave
        k2 = 4e2
        c2 = 0.26
        b2 = 0.0
        d2 = 0.4
        h2 = 0.004
        %h2=0.006
        g2 = 1.0
        w21 = 0.19
        w22 = 1.0
        
        % QRS complex
        k3 = 1e4
        c3 = 0.12
        b3 = 0.015
        d3 = 0.09
        h3 = 0.008
        g3 = 1.0
        w31 = 0.12
        w32 = 1.1
        
        % T wave
        k4 = 2e3
        c4 = 0.1   % original
        %c4=0.12
        b4 = 0.0
        d4 = 0.1
        h4 = 0.008   % Faster stabilization
        g4 = 1.0
        w41 = 0.22  % Slightly delayed peak
        %w41=0.25
        w42 = 0.8   % Smoother decay
        %w42=0.7
        
        % Coupling coefficients for muscle excitation
        K_ATDe = 4e-5
        K_ATRe = 4e-5
        K_VNDe = 9e-5
        K_VNRe = 6e-5
        
        % All time delays in the system
        delays
    end
    
    methods
        function obj = CardiacModel()
            % Apply specific modifications from the original code
            %obj.mu1(3) = 0.12;   % Faster y3 recovery (was 0.10)
            %obj.mu2(3) = 0.30;   % Sooner stabilization (was 0.34)
            %obj.b(3) = 0.008;    % Smoother x3 decay (was 0.0095)
            
            % Define all delays for dde23
            obj.delays = [obj.tau_AV_SA, obj.tau_SA_AV, obj.tau_HP_AV, obj.tau_AV_HP];
        end
        
        function dydt = system(obj, t, y, Z)
            % Extract state variables
            x1 = y(1); y1 = y(2);
            x2 = y(3); y2 = y(4);
            x3 = y(5); y3 = y(6);
            z1 = y(7); v1 = y(8);
            z2 = y(9); v2 = y(10);
            z3 = y(11); v3 = y(12);
            z4 = y(13); v4 = y(14);
            
            % Extract delayed values - Z contains the delayed values in the same order as delays
            Z1 = Z(:,1);  % Values at t - tau_AV_SA
            Z2 = Z(:,2);  % Values at t - tau_SA_AV
            Z3 = Z(:,3);  % Values at t - tau_HP_AV
            Z4 = Z(:,4);  % Values at t - tau_AV_HP
            
            % Get delayed variables
            x2_tau_AV_SA = Z1(3);  % x2 at t - tau_AV_SA
            x1_tau_SA_AV = Z2(1);  % x1 at t - tau_SA_AV
            x3_tau_HP_AV = Z3(5);  % x3 at t - tau_HP_AV
            x2_tau_AV_HP = Z4(3);  % x2 at t - tau_AV_HP
            
            % Calculate coupling currents
            I1_coupl = obj.C_AV_SA * (x2_tau_AV_SA - x1);
            I2_coupl = obj.C_SA_AV * (x1_tau_SA_AV - x2) + obj.C_HP_AV * (x3_tau_HP_AV - x2);
            I3_coupl = obj.C_AV_HP * (x2_tau_AV_HP - x3);
            
            % First calculate the derivatives for oscillators without muscle feedback
            % This gives us the basic oscillator behavior
            dx1_dt_basic = obj.alpha * (obj.k(1) * x1 * (x1 + obj.b(1)) * (1 - x1) - (y1 * x1)) + I1_coupl;
            dy1_dt = obj.alpha * (obj.epsilon(1) + ((y1 * obj.mu1(1)) / (x1 + obj.mu2(1)))) * (-y1 - obj.k(1) * x1 * (x1 - obj.a(1) - 1));
            
            dx2_dt = obj.alpha * (obj.k(2) * x2 * (x2 + obj.b(2)) * (1 - x2) - y2 * x2) + I2_coupl;
            dy2_dt = obj.alpha * (obj.epsilon(2) + ((y2 * obj.mu1(2)) / (x2 + obj.mu2(2)))) * (-y2 - obj.k(2) * x2 * (x2 - obj.a(2) - 1));
            
            dx3_dt_basic = obj.alpha * (obj.k(3) * x3 * (x3 + obj.b(3)) * (1 - x3) - y3 * x3) + I3_coupl;
            dy3_dt = obj.alpha * (obj.epsilon(3) + ((y3 * obj.mu1(3)) / (x3 + obj.mu2(3)))) * (-y3 - obj.k(3) * x3 * (x3 - obj.a(3) - 1));
            
            % Now use these derivatives to calculate muscle excitation currents
            % Calculate atrial excitation currents
            if dx1_dt_basic <= 0
                I_ATDe = 0;
                I_ATRe = -obj.K_ATRe * dx1_dt_basic;
            else
                I_ATDe = obj.K_ATDe * dx1_dt_basic;
                I_ATRe = 0;
            end
            
            % Calculate ventricular excitation currents
            if dx3_dt_basic <= 0
                I_VNDe = 0;
                I_VNRe = -obj.K_VNRe * dx3_dt_basic;
            else
                I_VNDe = obj.K_VNDe * dx3_dt_basic;
                I_VNRe = 0;
            end
            
            % The final dx1_dt and dx3_dt are the same as the basic ones
            dx1_dt = dx1_dt_basic;
            dx3_dt = dx3_dt_basic;
            
            % Calculate derivatives for P wave
            dz1_dt = obj.k1 * (-obj.c1 * z1 * (z1 - obj.w11) * (z1 - obj.w12) - obj.b1 * v1 - obj.d1 * v1 * z1 + I_ATDe);
            dv1_dt = obj.k1 * obj.h1 * (z1 - obj.g1 * v1);
            
            % Calculate derivatives for Ta wave
            dz2_dt = obj.k2 * (-obj.c2 * z2 * (z2 - obj.w21) * (z2 - obj.w22) - obj.b2 * v2 - obj.d2 * v2 * z2 + I_ATRe);
            dv2_dt = obj.k2 * obj.h2 * (z2 - obj.g2 * v2);
            
            % Calculate derivatives for QRS complex
            dz3_dt = obj.k3 * (-obj.c3 * z3 * (z3 - obj.w31) * (z3 - obj.w32) - obj.b3 * v3 - obj.d3 * v3 * z3 + I_VNDe);
            dv3_dt = obj.k3 * obj.h3 * (z3 - obj.g3 * v3);
            
            % Calculate derivatives for T wave
            dz4_dt = obj.k4 * (-obj.c4 * z4 * (z4 - obj.w41) * (z4 - obj.w42) - obj.b4 * v4 - obj.d4 * v4 * z4 + I_VNRe);
            dv4_dt = obj.k4 * obj.h4 * (z4 - obj.g4 * v4);
            
            % Return the derivatives
            dydt = [dx1_dt; dy1_dt; dx2_dt; dy2_dt; dx3_dt; dy3_dt; ...
                   dz1_dt; dv1_dt; dz2_dt; dv2_dt; dz3_dt; dv3_dt; dz4_dt; dv4_dt];
        end
        
        function ecg = calculate_ecg(~, z1, z2, z3, z4)
            ecg = z1 - z2 + z3 + z4;
        end
        
        function results = simulate(obj, t_span, initial_conditions)
            if nargin < 3 || isempty(initial_conditions)
                initial_conditions = [0.01, 0.0, 0.01, 0.0, 0.01, 0.0, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01];
            end
            
            % Create history function for the delays
            history = @(t) history_function(t, initial_conditions);
            
            % Higher resolution for better visualization
            t_eval = linspace(t_span(1), t_span(2), round((t_span(2) - t_span(1)) / 0.005));
            
            % Set DDE solver options
            options = ddeset('RelTol', 1e-4, 'AbsTol', 1e-6, 'MaxStep', 0.01);
            
            % Solve the system of equations using dde23
            sol = dde23(@(t, y, Z) obj.system(t, y, Z), obj.delays, history, t_eval, options);
            
            % Extract solution data
            t = sol.x';
            y = sol.y';
            
            x1 = y(:, 1);
            y1 = y(:, 2);
            x2 = y(:, 3);
            y2 = y(:, 4);
            x3 = y(:, 5);
            y3 = y(:, 6);
            z1 = y(:, 7);
            v1 = y(:, 8);
            z2 = y(:, 9);
            v2 = y(:, 10);
            z3 = y(:, 11);
            v3 = y(:, 12);
            z4 = y(:, 13);
            v4 = y(:, 14);
            
            % Calculate ECG and potentials
            ecg = zeros(size(t));
            at_potential = zeros(size(t));
            vn_potential = zeros(size(t));
            
            for i = 1:length(t)
                ecg(i) = obj.calculate_ecg(z1(i), z2(i), z3(i), z4(i));
                at_potential(i) = z1(i) - z2(i);  % Atrial potential
                vn_potential(i) = z3(i) + z4(i);  % Ventricular potential
            end
            
            % Store results in a struct
            results = struct('t', t, ...
                            'x1', x1, 'y1', y1, ...  % SA node
                            'x2', x2, 'y2', y2, ...  % AV node
                            'x3', x3, 'y3', y3, ...  % HP system
                            'z1', z1, 'v1', v1, ...  % P wave
                            'z2', z2, 'v2', v2, ...  % Ta wave
                            'z3', z3, 'v3', v3, ...  % QRS complex
                            'z4', z4, 'v4', v4, ...  % T wave
                            'at_potential', at_potential, ...  % Atrial potential
                            'vn_potential', vn_potential, ...  % Ventricular potential
                            'ecg', ecg);  % ECG signal
        end
        
        function plot_results_standard(~, results, t_min, t_max)
            if nargin < 3
                t_min = 15;
            end
            if nargin < 4
                t_max = 20;
            end
            
            % Get time slice
            mask = (results.t >= t_min) & (results.t <= t_max);
            t = results.t(mask);
            
            % Set up plot
            figure('Position', [100, 100, 800, 960]);
            
            % Plot SA oscillator (x1)
            subplot(5, 1, 1);
            plot(t, results.x1(mask), 'r-');
            ylabel('x_1');
            ylim([0, 1]);
            text(t_min + 0.2, 0.8, 'SA', 'FontSize', 12);
            grid on;
            
            % Plot AV oscillator (x2)
            subplot(5, 1, 2);
            plot(t, results.x2(mask), 'k-');
            ylabel('x_2');
            ylim([0, 1]);
            text(t_min + 0.2, 0.8, 'AV', 'FontSize', 12);
            grid on;
            
            % Plot HP oscillator (x3)
            subplot(5, 1, 3);
            plot(t, results.x3(mask), 'b-');
            ylabel('x_3');
            ylim([0, 1]);
            text(t_min + 0.2, 0.8, 'HP', 'FontSize', 12);
            grid on;
            
            % Plot potentials
            subplot(5, 1, 4);
            plot(t, results.at_potential(mask), 'r-');
            hold on;
            plot(t, results.vn_potential(mask), 'b-');
            hold off;
            ylabel('Potential');
            grid on;
            legend('AT', 'VN', 'Location', 'northeast');
            
            % Plot ECG
            subplot(5, 1, 5);
            plot(t, results.ecg(mask), 'k-');
            ylabel('ECG');
            xlabel('Time (s)');
            grid on;
        end
        
        function plot_results_diagram(~, results, t_min, t_max)
            if nargin < 3
                t_min = 17.2;
            end
            if nargin < 4
                t_max = 18.0;
            end
            
            % Get time slice
            mask = (results.t >= t_min) & (results.t <= t_max);
            t = results.t(mask);
            
            % Calculate absolute values of derivatives
            y1_abs = abs(results.y1(mask));
            y3_abs = abs(results.y3(mask));
            
            % Set up the figure
            figure('Position', [100, 100, 800, 640]);
            
            % TOP PANEL - SA node and atrial activation
            subplot(4, 1, 1);
            plot(t, results.x1(mask), 'k-', 'LineWidth', 1.5);
            hold on;
            plot(t, y1_abs, 'k--', 'LineWidth', 1.5);
            hold off;
            ylabel('x, z, y', 'FontSize', 12);
            ylim([0, 1]);
            legend('x_1', '|y_1|', 'Location', 'northeast');
            
            % Atrial potential
            subplot(4, 1, 2);
            at_potential = results.at_potential(mask);
            plot(t, at_potential, 'k-', 'LineWidth', 1.5);
            ylabel('z_1-z_2', 'FontSize', 12);
            
            % Add labels for P and Ta waves
            % Find P wave peak
            [~, p_idx] = max(at_potential(t < t_min + 0.4));
            p_wave_time = t(p_idx);
            p_wave_val = at_potential(p_idx);
            text(p_wave_time - 0.05, p_wave_val + 0.02, 'P', 'FontSize', 12);
            
            % Find Ta wave location (negative deflection after P)
            ta_mask = (t > p_wave_time + 0.2) & (t < t_max);
            if any(ta_mask)
                [~, ta_idx] = min(at_potential(ta_mask));
                % Convert to index in the original array
                ta_idx = find(ta_mask, 1) + ta_idx - 1;
                ta_time = t(ta_idx);
                ta_val = at_potential(ta_idx);
                text(ta_time, ta_val - 0.05, 'Ta', 'FontSize', 12);
            end
            
            % BOTTOM PANEL - His-Purkinje system and ventricular activation
            subplot(4, 1, 3);
            plot(t, results.x3(mask), 'k-', 'LineWidth', 1.5);
            hold on;
            plot(t, y3_abs, 'k--', 'LineWidth', 1.5);
            hold off;
            ylabel('x, z, y', 'FontSize', 12);
            ylim([0, 1]);
            legend('x_3', '|y_3|', 'Location', 'northeast');
            
            % Ventricular potential
            subplot(4, 1, 4);
            vn_potential = results.vn_potential(mask);
            plot(t, vn_potential, 'k-', 'LineWidth', 1.5);
            ylabel('z_3+z_4', 'FontSize', 12);
            xlabel('Time (s)', 'FontSize', 12);
            
            % Add labels for QRS complex and T wave
            % Find R wave (highest peak)
            [~, r_idx] = max(vn_potential);
            r_wave_time = t(r_idx);
            r_wave_val = vn_potential(r_idx);
            text(r_wave_time - 0.02, r_wave_val * 0.8, 'R', 'FontSize', 12);
            
            % Find Q wave (negative deflection before R)
            q_mask = (t < r_wave_time) & (vn_potential < 0.05);
            if any(q_mask)
                q_idx = find(q_mask, 1, 'last');
                q_time = t(q_idx);
                q_val = vn_potential(q_idx);
                text(q_time - 0.05, q_val, 'Q', 'FontSize', 12);
            end
            
            % Find S wave (negative deflection after R)
            s_mask = (t > r_wave_time) & (t < r_wave_time + 0.15);
            if any(s_mask)
                [~, s_idx] = min(vn_potential(s_mask));
                % Convert to index in the original array
                s_idx = find(s_mask, 1) + s_idx - 1;
                s_time = t(s_idx);
                s_val = vn_potential(s_idx);
                text(s_time - 0.02, s_val - 0.05, 'S', 'FontSize', 12);
            end
            
            % Find T wave (second peak after QRS)
            t_wave_mask = (t > t(find(s_mask, 1, 'last')) + 0.1) & (t < t_max);
            if any(t_wave_mask)
                [~, t_idx] = max(vn_potential(t_wave_mask));
                % Convert to index in the original array
                t_idx = find(t_wave_mask, 1) + t_idx - 1;
                t_wave_time = t(t_idx);
                t_wave_val = vn_potential(t_idx);
                text(t_wave_time, t_wave_val + 0.02, 'T', 'FontSize', 12);
            end
            
            % Remove top and right spines for all subplots
            for i = 1:4
                subplot(4, 1, i);
                box off;
                set(gca, 'XGrid', 'on', 'YGrid', 'on', 'GridLineStyle', ':');
            end
            
            % Adjust spacing
            set(gcf, 'Position', [100, 100, 800, 640]);
        end
        
        function df = export_ecg_data(~, results, filename, t_min, t_max)
            % Create data for export
            t = results.t;
            ecg = results.ecg;
            
            % Filter data if time bounds are specified
            if nargin >= 4 && ~isempty(t_min) && nargin >= 5 && ~isempty(t_max)
                mask = (t >= t_min) & (t <= t_max);
                t = t(mask);
                ecg = ecg(mask);
            end
            
            % Create table for export
            df = table(t, ecg, 'VariableNames', {'time', 'ecg'});
            
            % Save to CSV if filename is provided
            if nargin >= 3 && ~isempty(filename)
                writetable(df, filename);
                fprintf('ECG data exported to %s\n', filename);
            end
        end
    end
end

% History function for initialization of the DDE solver
function y = history_function(t, initial_conditions)
    % This function provides the history for the delay differential equation
    % For times t < 0, it returns the initial conditions
    y = initial_conditions;
end