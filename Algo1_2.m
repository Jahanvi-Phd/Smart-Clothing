
% Part 1: Algorithm 1 and Algorithm 2 with Case Studies

clear all; close all; clc;


%% ALGORITHM 1


fprintf('=== ALGORITHM 1: SMART FABRIC SELECTION ===\n\n');

% Universal set of fabrics (from case study)
fabrics = {'f1', 'f2', 'f3', 'f4', 'f5', 'f6'};
n_fabrics = length(fabrics);

% Parameters (from case study)
parameters = {'e1_Durability', 'e2_Comfort', 'e3_MoistureWicking', ...
              'e4_ThermalRegulation', 'e5_Flexibility', 'e6_UVProtection', ...
              'e7_AntimicrobialProperties', 'e8_UserConsentLevel', ...
              'e9_DataAnonymizationCapability'};
n_params = length(parameters);

% Weights (from case study)
weights = [0.05, 0.15, 0.20, 0.20, 0.05, 0.05, 0.10, 0.10, 0.10];

% Soft set construction F(e_j) - fabrics satisfying each parameter
soft_set = {
    [1, 2, 4, 6];        % e1: Durability
    [2, 3, 4, 5];        % e2: Comfort  
    [1, 3, 5, 6];        % e3: Moisture Wicking
    [2, 3, 5];           % e4: Thermal Regulation
    [1, 4, 6];           % e5: Flexibility
    [2, 5, 6];           % e6: UV Protection
    [1, 2, 3, 4];        % e7: Antimicrobial Properties
    [3, 4, 5, 6];        % e8: User Consent Level
    [1, 2, 6]            % e9: Data Anonymization Capability
};

% Algorithm 1 Implementation
fabric_scores = zeros(1, n_fabrics);

% Calculate weighted scores for each fabric
for i = 1:n_fabrics
    for j = 1:n_params
        if ismember(i, soft_set{j})
            fabric_scores(i) = fabric_scores(i) + weights(j);
        end
    end
end

% Display results
fprintf('Fabric Scores:\n');
for i = 1:n_fabrics
    fprintf('%s: %.3f\n', fabrics{i}, fabric_scores(i));
end

[max_score, best_fabric_idx] = max(fabric_scores);
fprintf('\nOptimal Fabric: %s with score %.3f\n\n', fabrics{best_fabric_idx}, max_score);

% Ranking fabrics
[sorted_scores, ranking_idx] = sort(fabric_scores, 'descend');
fprintf('Ranking Order:\n');
for i = 1:n_fabrics
    fprintf('%d. %s (%.3f)\n', i, fabrics{ranking_idx(i)}, sorted_scores(i));
end

% Visualization
figure('Position', [100, 100, 1200, 400]);

% Plot 1: Fabric scores
subplot(1, 3, 1);
bar(fabric_scores, 'FaceColor', [0.2, 0.6, 0.8]);
title('Algorithm 1: Fabric Selection Scores');
xlabel('Fabric');
ylabel('Weighted Score');
set(gca, 'XTickLabel', fabrics);
grid on;

% Plot 2: Parameter weights
subplot(1, 3, 2);
bar(weights, 'FaceColor', [0.8, 0.2, 0.2]);
title('Parameter Weights');
xlabel('Parameter');
ylabel('Weight');
set(gca, 'XTickLabel', parameters, 'XTickLabelRotation', 45);
grid on;

% Plot 3: Soft set membership matrix
subplot(1, 3, 3);
membership_matrix = zeros(n_params, n_fabrics);
for i = 1:n_params
    for j = 1:n_fabrics
        if ismember(j, soft_set{i})
            membership_matrix(i, j) = 1;
        end
    end
end
imagesc(membership_matrix);
colormap([1 1 1; 0.2 0.6 0.8]); % White for 0, Blue for 1
title('Soft Set Membership Matrix');
xlabel('Fabric');
ylabel('Parameter');
set(gca, 'XTick', 1:n_fabrics, 'XTickLabel', fabrics);
set(gca, 'YTick', 1:n_params, 'YTickLabel', parameters, 'FontSize', 8);
colorbar;

sgtitle('Algorithm 1: Smart Fabric Selection Analysis');


%% ALGORITHM 2: Multi-input Adaptive Fuzzy Logic for Ventilation Control


fprintf('\n=== ALGORITHM 2: ADAPTIVE FUZZY LOGIC CONTROL ===\n\n');

% Case study data (Tropical training scenario)
T = 27;          % Temperature (°C)
H = 70;          % Humidity (%)
B = 38;          % Body heat (°C)

% User Profile (from case study)
t_min = 20; t_max = 28;
h_min = 40; h_max = 75;
b_min = 36.5; b_max = 38.5;
w_T = 0.4; w_H = 0.3; w_B = 0.3;

% Climate Zone adjustments
c_T = 1; c_H = 0; c_B = 0;

% Max ventilation setting
max_ventilation = 10;

% Algorithm 2 Implementation
fprintf('Original sensor values:\n');
fprintf('Temperature: %.1f°C, Humidity: %.1f%%, Body Heat: %.1f°C\n', T, H, B);

% Step 2: Adjust inputs
T_adj = T + c_T;
H_adj = H + c_H;
B_adj = B + c_B;

fprintf('Adjusted values:\n');
fprintf('T_adj: %.1f°C, H_adj: %.1f%%, B_adj: %.1f°C\n', T_adj, H_adj, B_adj);

% Step 3: Normalize inputs
T_n = (T_adj - t_min) / (t_max - t_min);
H_n = (H_adj - h_min) / (h_max - h_min);
B_n = (B_adj - b_min) / (b_max - b_min);

fprintf('Normalized values:\n');
fprintf('T_n: %.3f, H_n: %.3f, B_n: %.3f\n', T_n, H_n, B_n);

% Step 4: Calculate fuzzy membership values (triangular function)
mu_T = fuzzy_membership(T_n);
mu_H = fuzzy_membership(H_n);
mu_B = fuzzy_membership(B_n);

fprintf('Membership values:\n');
fprintf('μ_T: %.3f, μ_H: %.3f, μ_B: %.3f\n', mu_T, mu_H, mu_B);

% Step 5: Weighted aggregation
S = w_T * mu_T + w_H * mu_H + w_B * mu_B;
fprintf('Aggregated score S: %.3f\n', S);

% Step 6: Final ventilation level
V = S * max_ventilation;
fprintf('Final ventilation level: %.2f units\n\n', V);

% Visualization of Algorithm 2
figure('Position', [100, 500, 1200, 600]);

% Plot 1: Input normalization
subplot(2, 3, 1);
inputs = [T, H, B];
inputs_adj = [T_adj, H_adj, B_adj];
inputs_norm = [T_n, H_n, B_n];
x = categorical({'Temperature', 'Humidity', 'Body Heat'});
bar(x, [inputs; inputs_adj; inputs_norm]');
title('Input Processing Steps');
ylabel('Value');
legend('Original', 'Adjusted', 'Normalized', 'Location', 'best');

% Plot 2: Membership functions visualization
subplot(2, 3, 2);
x_range = 0:0.01:1;
mu_range = arrayfun(@fuzzy_membership, x_range);
plot(x_range, mu_range, 'LineWidth', 2);
hold on;
plot([T_n, H_n, B_n], [mu_T, mu_H, mu_B], 'ro', 'MarkerSize', 8, 'LineWidth', 2);
title('Triangular Membership Function');
xlabel('Normalized Input');
ylabel('Membership Value');
legend('μ(x)', 'Case Study Points', 'Location', 'best');
grid on;

% Plot 3: Weight distribution
subplot(2, 3, 3);
weights_fuzzy = [w_T, w_H, w_B];
pie(weights_fuzzy, {'Temperature', 'Humidity', 'Body Heat'});
title('Parameter Weights');

% Plot 4: Membership comparison
subplot(2, 3, 4);
memberships = [mu_T, mu_H, mu_B];
bar(categorical({'Temperature', 'Humidity', 'Body Heat'}), memberships, ...
    'FaceColor', [0.8, 0.4, 0.2]);
title('Fuzzy Membership Values');
ylabel('Membership Value');
ylim([0, 1]);

% Plot 5: Ventilation response surface
subplot(2, 3, 5);
temp_range = 20:0.5:35;
humid_range = 40:1:80;
[T_grid, H_grid] = meshgrid(temp_range, humid_range);
V_grid = zeros(size(T_grid));

for i = 1:size(T_grid, 1)
    for j = 1:size(T_grid, 2)
        T_temp = T_grid(i, j) + c_T;
        H_temp = H_grid(i, j) + c_H;
        B_temp = B + c_B; % Keep body heat constant
        
        T_norm = max(0, min(1, (T_temp - t_min) / (t_max - t_min)));
        H_norm = max(0, min(1, (H_temp - h_min) / (h_max - h_min)));
        B_norm = max(0, min(1, (B_temp - b_min) / (b_max - b_min)));
        
        mu_T_temp = fuzzy_membership(T_norm);
        mu_H_temp = fuzzy_membership(H_norm);
        mu_B_temp = fuzzy_membership(B_norm);
        
        S_temp = w_T * mu_T_temp + w_H * mu_H_temp + w_B * mu_B_temp;
        V_grid(i, j) = S_temp * max_ventilation;
    end
end

contourf(T_grid, H_grid, V_grid, 20);
colorbar;
hold on;
plot(T, H, 'r*', 'MarkerSize', 15, 'LineWidth', 3);
title('Ventilation Response Surface');
xlabel('Temperature (°C)');
ylabel('Humidity (%)');

% Plot 6: Final output
subplot(2, 3, 6);
bar(categorical({'Ventilation Level'}), V, 'FaceColor', [0.2, 0.8, 0.2]);
title('Algorithm 2 Output');
ylabel('Ventilation Units');
ylim([0, max_ventilation]);
text(1, V/2, sprintf('%.2f units', V), 'HorizontalAlignment', 'center', ...
     'FontSize', 12, 'FontWeight', 'bold');

sgtitle('Algorithm 2: Adaptive Fuzzy Logic Control Analysis');

% Function definition for fuzzy membership
function mu = fuzzy_membership(x_n)
    % Triangular membership function as defined in the manuscript
    if x_n <= 0.3
        mu = 0;
    elseif x_n < 0.7
        mu = (x_n - 0.3) / 0.4;
    else
        mu = 1;
    end
end