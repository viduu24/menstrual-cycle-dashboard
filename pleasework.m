% Main script to solve and visualize the coupled ODEs with modified parameters
clear all; close all;

% Define parameters - modified to achieve the desired behavior
alpha = 77.5;       
% Adjusting parameters to show different frequencies
k = [5, 5, 4.5];      
b = [0.113, 0.0105, 0.0095];   
a = [0.18, 0.1, 0.3];      
epsilon = [0.004, 0.004, 0.003]; 
mu1 = [0.2, 0.2, 0.10];     
mu2 = [0.3, 0.3, 0.34];     
% For muscle part
k1=2e3; k2=4e2; k3=1e4; k4=2e3;
c1=0.26;c2=0.26; c3 = 0.12; c4 = 0.1;
b1 =0; b2 =0; b4 = 0; b3 = 0.015;
d1 =0.4; d2 = 0.4; d3 = 0.09; d4 = 0.1;
h1 =0.004; h2 = 0.004; h3 =0.008; h4 = 0.008;
g1 =1; g2 =1; g3 =1; g4 = 1;
w11=0.13;w12=1;w22=1;w21=0.19;w31=0.12;w32=1.1;w41=0.22;w42=0.8;
% Coupling coefficients for the muscle models
K_ATDe = 4e-5; K_ATRe = 4e-5;
K_VNDe = 9e-5; K_VNRe = 6e-5;
% Modified coupling constants to create the desired pattern
C_AV_SA = 0.5;    % Reduced feedback from AV to SA
C_SA_AV = 2.5;     % Adjusted forward coupling from SA to AV
C_HP_AV = 0.4;    % Reduced feedback from HP to AV
C_AV_HP = 2.5;     % Strong forward coupling, but intermittent conduction

% Transmission delays - adjusted to achieve desired behavior
tau_AV_SA = 0.02;  % Delay from SA to AV
tau_SA_AV = 0.05;  % Delay from AV to SA
tau_HP_AV = 0.02;  % Delay from AV to HP
tau_AV_HP = 0.07;  % Delay from HP to AV

% Initial conditions [x1, y1, x2, y2, x3, y3]
initial_conditions = [0.5, 0.1, 0.5, 0.1, 0.5, 0.1,0, 0, 0, 0, 0, 0, 0, 0];

% Time span - increased to see multiple cycles
tspan = [0 20];

% For using dde23 (delay differential equation solver), we need history function
history = @(t) initial_conditions;

% Solve the DDE system
sol = dde23(@(t, state, Z) coupled_odes_with_muscle(t, state, Z, alpha, k, b, a, epsilon, mu1, mu2, ...
                                                C_AV_SA, C_SA_AV, C_HP_AV, C_AV_HP, ...
                                                k1, k2, k3, k4, c1, c2, c3, c4, ...
                                                b1, b2, b3, b4, d1, d2, d3, d4, ...
                                                h1, h2, h3, h4, g1, g2, g3, g4, ...
                                                w11, w12, w21, w22, w31, w32, w41, w42, ...
                                                K_ATDe, K_ATRe, K_VNDe, K_VNRe), ...
                                              [tau_AV_SA, tau_SA_AV, tau_HP_AV, tau_AV_HP], ...
                                              history, tspan);

% Extract solution at specific time points for plotting
t = linspace(tspan(1), tspan(2), 2000);
state = deval(sol, t);

% Extract states for 3 cardiac oscillators
x1 = state(1,:)'; y1 = state(2,:)';  % SA node
x2 = state(3,:)'; y2 = state(4,:)';  % AV node
x3 = state(5,:)'; y3 = state(6,:)';  % HP system

% Extract states for 4 muscle models
z1 = state(7,:)'; v1 = state(8,:)';  % P wave (AT muscle)
z2 = state(9,:)'; v2 = state(10,:)'; % Ta wave (AT muscle)
z3 = state(11,:)'; v3 = state(12,:)'; % QRS (VN muscle)
z4 = state(13,:)'; v4 = state(14,:)'; % T wave (VN muscle)

% Create plots for cardiac oscillators
figure('Position', [100, 100, 800, 600]);

% Customize the figure to match the reference plot
set(gcf, 'Color', 'w');

% Cardiac nodes plot
subplot(3,2,1);
plot(t, x1, 'r', 'LineWidth', 1.5);
title('SA Node', 'FontSize', 12);
ylabel('x_1', 'FontSize', 12);
ylim([0, 1]);
grid on;
box on;
set(gca, 'GridAlpha', 0.15);
set(gca, 'XTickLabel', []);

subplot(3,2,3);
plot(t, x2, 'k', 'LineWidth', 1.5);
title('AV Node', 'FontSize', 12);
ylabel('x_2', 'FontSize', 12);
ylim([0, 1]);
grid on;
box on;
set(gca, 'GridAlpha', 0.15);
set(gca, 'XTickLabel', []);

subplot(3,2,5);
plot(t, x3, 'b', 'LineWidth', 1.5);
title('HP System', 'FontSize', 12);
xlabel('Time (s)', 'FontSize', 12);
ylabel('x_3', 'FontSize', 12);
ylim([0, 1]);
grid on;
box on;
set(gca, 'GridAlpha', 0.15);

% Adding "AP" label to top of plot
annotation('textbox', [0.2, 0.95, 0.1, 0.05], 'String', 'AP', ...
           'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', ...
           'FontSize', 12, 'LineStyle', 'none');

% Muscle models plot
subplot(4,2,2);
plot(t, z1, 'm', 'LineWidth', 1.5);
title('P wave (AT muscle)', 'FontSize', 12);
ylabel('z_1', 'FontSize', 12);
grid on;
box on;
set(gca, 'GridAlpha', 0.15);
set(gca, 'XTickLabel', []);

subplot(4,2,4);
plot(t, z2, 'c', 'LineWidth', 1.5);
title('Ta wave (AT muscle)', 'FontSize', 12);
ylabel('z_2', 'FontSize', 12);
grid on;
box on;
set(gca, 'GridAlpha', 0.15);
set(gca, 'XTickLabel', []);

subplot(4,2,6);
plot(t, z3, 'g', 'LineWidth', 1.5);
title('QRS (VN muscle)', 'FontSize', 12);
ylabel('z_3', 'FontSize', 12);
grid on;
box on;
set(gca, 'GridAlpha', 0.15);
set(gca, 'XTickLabel', []);

subplot(4,2,8);
plot(t, z4, 'color', [0.8 0.4 0], 'LineWidth', 1.5);
title('T wave (VN muscle)', 'FontSize', 12);
xlabel('Time (s)', 'FontSize', 12);
ylabel('z_4', 'FontSize', 12);
grid on;
box on;
set(gca, 'GridAlpha', 0.15);

% Adding "ECG" label to top of muscle plot
annotation('textbox', [0.65, 0.95, 0.1, 0.05], 'String', 'ECG Components', ...
           'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', ...
           'FontSize', 12, 'LineStyle', 'none');

% Plot combined ECG-like signal
figure('Position', [100, 700, 800, 300]);
set(gcf, 'Color', 'w');

% Create a simple approximation of an ECG by combining muscle components
ecg_signal = z1 + z2 + 2*z3 + z4;  % Scale QRS to be more prominent
plot(t, ecg_signal, 'k', 'LineWidth', 1.5);
title('Simulated ECG Signal', 'FontSize', 14);
xlabel('Time (s)', 'FontSize', 12);
ylabel('Amplitude', 'FontSize', 12);
grid on;
box on;
set(gca, 'GridAlpha', 0.15);

% Function that defines the system of ODEs with delay coupling and muscle models
function dstate = coupled_odes_with_muscle(t, state, Z, alpha, k, b, a, epsilon, mu1, mu2, ...
                                         C_AV_SA, C_SA_AV, C_HP_AV, C_AV_HP, ...
                                         k1, k2, k3, k4, c1, c2, c3, c4, ...
                                         b1, b2, b3, b4, d1, d2, d3, d4, ...
                                         h1, h2, h3, h4, g1, g2, g3, g4, ...
                                         w11, w12, w21, w22, w31, w32, w41, w42, ...
                                         K_ATDe, K_ATRe, K_VNDe, K_VNRe)
    % Extract current states
    % Cardiac nodes
    x1 = state(1); y1 = state(2);  % SA node
    x2 = state(3); y2 = state(4);  % AV node
    x3 = state(5); y3 = state(6);  % HP region
    
    % Muscle models
    z1 = state(7); v1 = state(8);    % P wave (AT muscle)
    z2 = state(9); v2 = state(10);   % Ta wave (AT muscle)
    z3 = state(11); v3 = state(12);  % QRS (VN muscle)
    z4 = state(13); v4 = state(14);  % T wave (VN muscle)
    
    % Extract delayed states for cardiac coupling
    x2_delay_AV_SA = Z(3, 1);  % Delayed x2 for AV->SA coupling
    x1_delay_SA_AV = Z(1, 2);  % Delayed x1 for SA->AV coupling
    x3_delay_HP_AV = Z(5, 3);  % Delayed x3 for HP->AV coupling
    x2_delay_AV_HP = Z(3, 4);  % Delayed x2 for AV->HP coupling
    
    % Calculate coupling currents between cardiac nodes
    I_coupl_1 = C_AV_SA * (x2_delay_AV_SA - x1);
    I_coupl_2 = C_SA_AV * (x1_delay_SA_AV - x2) + C_HP_AV * (x3_delay_HP_AV - x2);
    I_coupl_3 = C_AV_HP * (x2_delay_AV_HP - x3);
    
    % Calculate muscle coupling currents (from Image 2)
    % AT muscle coupling
    I_ATDe = (y1 > 0) * K_ATDe * y1;
    I_ATRe = (y1 <= 0) * (-K_ATRe * y1);
    
    % VN muscle coupling
    I_VNDe = (y3 > 0) * K_VNDe * y3;
    I_VNRe = (y3 <= 0) * (-K_VNRe * y3);
    
    % Initialize derivative vector
    dstate = zeros(14, 1);
    
    % SA node equations (index 1)
    dstate(1) = alpha * (k(1)*x1*(x1 + b(1))*(1 - x1) - y1*x1) + I_coupl_1;
    dstate(2) = alpha * (epsilon(1) + ((y1*mu1(1))/(x1 + mu2(1)))) * (-y1 - k(1)*x1*(x1 - a(1) - 1));
    
    % AV node equations (index 2)
    dstate(3) = alpha * (k(2)*x2*(x2 + b(2))*(1 - x2) - y2*x2) + I_coupl_2;
    dstate(4) = alpha * (epsilon(2) + ((y2*mu1(2))/(x2 + mu2(2)))) * (-y2 - k(2)*x2*(x2 - a(2) - 1));
    
    % HP region equations (index 3)
    dstate(5) = alpha * (k(3)*x3*(x3 + b(3))*(1 - x3) - y3*x3) + I_coupl_3;
    dstate(6) = alpha * (epsilon(3) + ((y3*mu1(3))/(x3 + mu2(3)))) * (-y3 - k(3)*x3*(x3 - a(3) - 1));
    
    % Muscle model equations (from Image 1)
    % P wave (AT muscle)
    dstate(7) = k1 * (-c1*z1*(z1 - w11)*(z1 - w12) - b1*v1 - d1*v1*z1 + I_ATDe);
    dstate(8) = k1 * h1 * (z1 - g1*v1);
    % Ta wave (AT muscle)
    dstate(9) = k2 * (-c2*z2*(z2 - w21)*(z2 - w22) - b2*v2 - d2*v2*z2 + I_ATRe);
    dstate(10) = k2 * h2 * (z2 - g2*v2);
    
    % QRS (VN muscle)
    dstate(11) = k3 * (-c3*z3*(z3 - w31)*(z3 - w32) - b3*v3 - d3*v3*z3 + I_VNDe);
    dstate(12) = k3 * h3 * (z3 - g3*v3);
    
    % T wave (VN muscle)
    dstate(13) = k4 * (-c4*z4*(z4 - w41)*(z4 - w42) - b4*v4 - d4*v4*z4 + I_VNRe);
    dstate(14) = k4 * h4 * (z4 - g4*v4);
end