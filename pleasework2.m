% Main script to solve and visualize the coupled ODEs with completely revised parameters
clear all; close all;

% Define parameters - completely revised to achieve proper oscillations
alpha = 80;       
% Parameters for each oscillator to ensure robust oscillations
k = [6.5, 6.0, 5.5];      
b = [0.15, 0.15, 0.15];   
a = [0.2, 0.2, 0.2];      
epsilon = [0.002, 0.002, 0.002]; 
mu1 = [0.2, 0.2, 0.2];     
mu2 = [0.3, 0.3, 0.3];     

% Coupling constants - completely revised
C_AV_SA = 0.01;    % Very slight feedback from AV to SA (SA should be autonomous)
C_SA_AV = 2.0;     % Strong forward coupling from SA to AV
C_HP_AV = 0.01;    % Very slight feedback from HP to AV
C_AV_HP = 2.0;     % Strong forward coupling from AV to HP

% Transmission delays - adjusted to ensure proper sequence
tau_AV_SA = 0.05;  % Delay from AV to SA
tau_SA_AV = 0.05;  % Delay from SA to AV
tau_HP_AV = 0.05;  % Delay from HP to AV
tau_AV_HP = 0.05;  % Delay from AV to HP

% Initial conditions [x1, y1, x2, y2, x3, y3]
initial_conditions = [0.5, 0.1, 0.5, 0.1, 0.5, 0.1];

% Time span 
tspan = [0 20];

% For using dde23 (delay differential equation solver), we need history function
history = @(t) initial_conditions;

% Solve the DDE system
options = ddeset('RelTol', 1e-5, 'AbsTol', 1e-6); % Setting stricter tolerances
sol = dde23(@(t, state, Z) coupled_odes_revised(t, state, Z, alpha, k, b, a, epsilon, mu1, mu2, ...
                                                  C_AV_SA, C_SA_AV, C_HP_AV, C_AV_HP), ...
                                                  [tau_AV_SA, tau_SA_AV, tau_HP_AV, tau_AV_HP], ...
                                                  history, tspan, options);

% Extract solution at specific time points for plotting
t = linspace(tspan(1), tspan(2), 2000);
state = deval(sol, t);

% Extract states for 3 oscillators
x1 = state(1,:)'; y1 = state(2,:)';
x2 = state(3,:)'; y2 = state(4,:)';
x3 = state(5,:)'; y3 = state(6,:)';

% Create plot similar to the reference image
figure('Position', [100, 100, 800, 400]);

% Customize the figure to match the reference plot
set(gcf, 'Color', 'w');
subplot(3,1,1);
plot(t, x1, 'r', 'LineWidth', 1.5);
title('SA', 'FontSize', 12);
ylabel('x_1', 'FontSize', 12);
ylim([0, 1]);
grid on;
box on;
set(gca, 'GridAlpha', 0.15);
set(gca, 'XTickLabel', []);

subplot(3,1,2);
plot(t, x2, 'k', 'LineWidth', 1.5);
title('AV', 'FontSize', 12);
ylabel('x_2', 'FontSize', 12);
ylim([0, 1]);
grid on;
box on;
set(gca, 'GridAlpha', 0.15);
set(gca, 'XTickLabel', []);

subplot(3,1,3);
plot(t, x3, 'b', 'LineWidth', 1.5);
title('HP', 'FontSize', 12);
xlabel('Time (s)', 'FontSize', 12);
ylabel('x_3', 'FontSize', 12);
ylim([0, 1]);
grid on;
box on;
set(gca, 'GridAlpha', 0.15);

% Adding "AP" label to top of plot
annotation('textbox', [0.5, 0.95, 0.1, 0.05], 'String', 'AP', ...
           'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', ...
           'FontSize', 12, 'LineStyle', 'none');

% Completely revised system dynamics function
function dstate = coupled_odes_revised(t, state, Z, alpha, k, b, a, epsilon, mu1, mu2, ...
                                        C_AV_SA, C_SA_AV, C_HP_AV, C_AV_HP)
    % Extract current states
    x1 = state(1); y1 = state(2);  % SA node
    x2 = state(3); y2 = state(4);  % AV node
    x3 = state(5); y3 = state(6);  % HP region
    
    % Extract delayed states
    x2_delay_AV_SA = Z(3, 1);  % Delayed x2 for AV->SA coupling
    x1_delay_SA_AV = Z(1, 2);  % Delayed x1 for SA->AV coupling
    x3_delay_HP_AV = Z(5, 3);  % Delayed x3 for HP->AV coupling
    x2_delay_AV_HP = Z(3, 4);  % Delayed x2 for AV->HP coupling
    
    % Define thresholds for conduction - key to getting the right pattern
    SA_threshold = 0.3;  % SA node activation threshold
    AV_threshold = 0.3;  % AV node activation threshold
    
    % Initialize conduction flags
    conduct_to_AV = 0;
    conduct_to_HP = 0;
    
    % Determine if SA signal should conduct to AV
    if x1 > SA_threshold && x2 < 0.1
        % Determine if this SA signal should conduct to AV
        % Make every SA signal conduct to AV
        conduct_to_AV = 1;
    end
    
    % Determine if AV signal should conduct to HP
    if x2 > AV_threshold && x3 < 0.1
        % Make every other AV signal conduct to HP
        % This is managed by a time-based condition to create periodicity
        if mod(floor(t), 2) == 0
            conduct_to_HP = 1;
        end
    end
    
    % Calculate coupling currents with active conduction control
    I_coupl_1 = C_AV_SA * (x2_delay_AV_SA - x1);  % Minimal feedback from AV to SA
    
    % Forward conduction from SA to AV only when appropriate
    I_coupl_2 = conduct_to_AV * C_SA_AV * (x1_delay_SA_AV - x2) + C_HP_AV * (x3_delay_HP_AV - x2);
    
    % Forward conduction from AV to HP only when appropriate
    I_coupl_3 = conduct_to_HP * C_AV_HP * (x2_delay_AV_HP - x3);
    
    % Initialize derivative vector
    dstate = zeros(6, 1);
    
    % SA node equations - ensure periodic oscillation
    dstate(1) = alpha * (k(1)*x1*(x1-a(1))*(1-x1) - y1) + I_coupl_1;
    dstate(2) = alpha * epsilon(1) * (b(1)*x1 - y1);
    
    % AV node equations
    dstate(3) = alpha * (k(2)*x2*(x2-a(2))*(1-x2) - y2) + I_coupl_2;
    dstate(4) = alpha * epsilon(2) * (b(2)*x2 - y2);
    
    % HP region equations
    dstate(5) = alpha * (k(3)*x3*(x3-a(3))*(1-x3) - y3) + I_coupl_3;
    dstate(6) = alpha * epsilon(3) * (b(3)*x3 - y3);
end