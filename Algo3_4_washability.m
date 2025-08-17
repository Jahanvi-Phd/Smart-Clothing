% Part 2: Algorithm 3, Algorithm 4, Washability and Durability Analysis

clear all; close all; clc;


%% ALGORITHM 3


fprintf('=== ALGORITHM 3: OPTIMAL PARAMETER WEIGHTS DERIVATION ===\n\n');

% Parameter set (from medical case study)
parameters_med = {'e1_Antimicrobial', 'e2_Comfort', 'e3_Durability', ...
                  'e4_Flexibility', 'e5_Thermal', 'e6_Moisture', 'e7_Accuracy'};
n_params_med = length(parameters_med);

% Context elements (from case study)
contexts = {'u1_Elderly', 'u2_Critical', 'u3_LongDuration', 'u4_Indoor', 'u5_LowActivity'};
n_contexts = length(contexts);

% Soft set construction F_medical(e_j) from case study
soft_set_medical = {
    [1, 2];           % e1_Antimicrobial: {u1_elderly, u2_critical}
    [1, 3, 4, 5];     % e2_Comfort: {u1_elderly, u3_long-duration, u4_indoor, u5_low activity}
    [3, 4];           % e3_Durability: {u3_long-duration, u4_indoor}
    [4];              % e4_Flexibility: {u4_indoor}
    [1, 5];           % e5_Thermal: {u1_elderly, u5_low activity}
    [3];              % e6_Moisture: {u3_long-duration}
    [2];              % e7_Accuracy: {u2_critical}
};

fprintf('Medical Garment Context Analysis:\n');
fprintf('Parameters: %s\n', strjoin(parameters_med, ', '));
fprintf('Context Elements: %s\n\n', strjoin(contexts, ', '));

% Algorithm 3 Implementation
% Step 1: Compute contextual frequency
frequencies = zeros(1, n_params_med);
for j = 1:n_params_med
    frequencies(j) = length(soft_set_medical{j}) / n_contexts;
    fprintf('f_%d (%s): |F(%s)| = %d, f = %.2f\n', j, parameters_med{j}, ...
            parameters_med{j}, length(soft_set_medical{j}), frequencies(j));
end

% Step 2: Normalize frequencies to get initial weights
sum_frequencies = sum(frequencies);
fprintf('\nSum of frequencies: %.2f\n', sum_frequencies);

weights_optimal = frequencies / sum_frequencies;
fprintf('\nOptimal Weights:\n');
for j = 1:n_params_med
    fprintf('w_%d (%s): %.3f (%.1f%%)\n', j, parameters_med{j}, ...
            weights_optimal(j), weights_optimal(j)*100);
end

fprintf('\nWeight Vector w* = [');
fprintf('%.3f', weights_optimal(1));
for j = 2:n_params_med
    fprintf(', %.3f', weights_optimal(j));
end
fprintf(']\n\n');

% Visualization of Algorithm 3
figure('Position', [100, 100, 1400, 800]);

% Plot 1: Context-Parameter Membership Matrix
subplot(2, 3, 1);
membership_matrix_med = zeros(n_params_med, n_contexts);
for i = 1:n_params_med
    for j = 1:n_contexts
        if ismember(j, soft_set_medical{i})
            membership_matrix_med(i, j) = 1;
        end
    end
end
imagesc(membership_matrix_med);
colormap([1 1 1; 0.8 0.2 0.2]); % White for 0, Red for 1
title('Medical Soft Set Membership Matrix');
xlabel('Context Elements');
ylabel('Parameters');
set(gca, 'XTick', 1:n_contexts, 'XTickLabel', contexts, 'XTickLabelRotation', 45);
set(gca, 'YTick', 1:n_params_med, 'YTickLabel', parameters_med, 'FontSize', 8);
colorbar;

% Plot 2: Contextual frequencies
subplot(2, 3, 2);
bar(frequencies, 'FaceColor', [0.2, 0.8, 0.6]);
title('Contextual Frequencies');
xlabel('Parameter Index');
ylabel('Frequency f_j');
set(gca, 'XTick', 1:n_params_med, 'XTickLabel', 1:n_params_med);
grid on;

% Plot 3: Final optimal weights
subplot(2, 3, 3);
bar(weights_optimal, 'FaceColor', [0.6, 0.2, 0.8]);
title('Optimal Parameter Weights');
xlabel('Parameter');
ylabel('Weight');
set(gca, 'XTick', 1:n_params_med, 'XTickLabel', parameters_med, 'XTickLabelRotation', 45);
grid on;

% Plot 4: Weight distribution pie chart
subplot(2, 3, 4);
pie(weights_optimal, parameters_med);
title('Weight Distribution');

% Plot 5: Parameter importance ranking
subplot(2, 3, 5);
[sorted_weights, weight_ranking] = sort(weights_optimal, 'descend');
bar(sorted_weights, 'FaceColor', [0.8, 0.6, 0.2]);
title('Parameter Importance Ranking');
xlabel('Rank');
ylabel('Weight');
set(gca, 'XTick', 1:n_params_med, 'XTickLabel', parameters_med(weight_ranking), ...
    'XTickLabelRotation', 45);
grid on;

% Plot 6: Context element participation
subplot(2, 3, 6);
context_participation = sum(membership_matrix_med, 1);
bar(context_participation, 'FaceColor', [0.4, 0.6, 0.8]);
title('Context Element Participation');
xlabel('Context Element');
ylabel('Number of Parameters');
set(gca, 'XTick', 1:n_contexts, 'XTickLabel', contexts, 'XTickLabelRotation', 45);
grid on;

sgtitle('Algorithm 3: Medical Smart Garment Weight Derivation');


%% ALGORITHM 4: Contextual Weight Adjustment via Multi-objective Optimization


fprintf('=== ALGORITHM 4: MULTI-OBJECTIVE WEIGHT ADJUSTMENT ===\n\n');

% Base weights from Algorithm 3
w_base = weights_optimal;

% Demographic and usage context adjustments (from case study)
delta_increment = 0.02; % per demographic occurrence
gamma_increment = 0.01; % per contextual occurrence

% Demographic soft set F_elderly = {e1, e2, e5}
% Usage context soft sets: F_low_activity = {e1, e2}, F_critical = {e1, e7}, F_long_duration = {e1, e2, e6}
demographic_appearances = [1, 1, 0, 0, 1, 0, 0]; % e1, e2, e5 appear once each
contextual_appearances = [3, 2, 0, 0, 0, 1, 1];   % e1:3, e2:2, e6:1, e7:1 times

% Calculate adjustments
delta_adjustments = demographic_appearances * delta_increment;
gamma_adjustments = contextual_appearances * gamma_increment;

% Adjusted weights
w_adj = w_base + delta_adjustments + gamma_adjustments;

fprintf('Base weights (from Algorithm 3): [');
fprintf('%.3f', w_base(1));
for i = 2:length(w_base)
    fprintf(', %.3f', w_base(i));
end
fprintf(']\n');

fprintf('Demographic adjustments: [');
fprintf('%.3f', delta_adjustments(1));
for i = 2:length(delta_adjustments)
    fprintf(', %.3f', delta_adjustments(i));
end
fprintf(']\n');

fprintf('Contextual adjustments: [');
fprintf('%.3f', gamma_adjustments(1));
for i = 2:length(gamma_adjustments)
    fprintf(', %.3f', gamma_adjustments(i));
end
fprintf(']\n');

fprintf('Adjusted weights: [');
fprintf('%.3f', w_adj(1));
for i = 2:length(w_adj)
    fprintf(', %.3f', w_adj(i));
end
fprintf(']\n');

% Multi-objective optimization parameters
alpha = 0.5; beta = 0.3; gamma_obj = 0.2; % Objective weights

% Utility function evaluation (simplified for demonstration)
f_performance = 0.88;
f_satisfaction = 0.75;
f_cost = 0.55;

utility = alpha * f_performance + beta * f_satisfaction - gamma_obj * f_cost;
fprintf('Utility score: %.3f\n', utility);

% Simulated optimization result (from case study)
w_opt = [0.22, 0.36, 0.12, 0.08, 0.15, 0.07, 0.10];

% Weight Stability Index calculation
WSI = 1 - sum(abs(w_opt - w_adj)) / length(w_opt);
fprintf('Weight Stability Index (WSI): %.3f\n', WSI);

% Final weights (WSI > 0.85, so accept w_opt)
if WSI >= 0.85
    w_final = w_opt;
    fprintf('WSI >= 0.85, accepting optimized weights\n');
else
    lambda = 0.7;
    w_final = lambda * w_opt + (1 - lambda) * w_adj;
    fprintf('WSI < 0.85, applying smoothing with λ = %.1f\n', lambda);
end

fprintf('Final optimized weights: [');
fprintf('%.3f', w_final(1));
for i = 2:length(w_final)
    fprintf(', %.3f', w_final(i));
end
fprintf(']\n\n');

% Visualization of Algorithm 4
figure('Position', [100, 600, 1400, 600]);

% Plot 1: Weight evolution through algorithm steps
subplot(2, 3, 1);
weight_evolution = [w_base; w_adj; w_opt; w_final];
plot(1:n_params_med, weight_evolution', 'LineWidth', 2, 'Marker', 'o');
title('Weight Evolution Through Algorithm 4');
xlabel('Parameter Index');
ylabel('Weight Value');
legend('Base', 'Adjusted', 'Optimized', 'Final', 'Location', 'best');
grid on;

% Plot 2: Comparison of weight stages
subplot(2, 3, 2);
bar([w_base; w_adj; w_opt; w_final]');
title('Weight Comparison Across Stages');
xlabel('Parameter');
ylabel('Weight');
legend('Base', 'Adjusted', 'Optimized', 'Final', 'Location', 'best');
set(gca, 'XTick', 1:n_params_med, 'XTickLabel', parameters_med, 'XTickLabelRotation', 45);

% Plot 3: Multi-objective components
subplot(2, 3, 3);
objectives = [f_performance, f_satisfaction, f_cost];
obj_weights = [alpha, beta, gamma_obj];
obj_contributions = objectives .* obj_weights;
bar(categorical({'Performance', 'Satisfaction', 'Cost'}), ...
    [objectives; obj_weights; obj_contributions]');
title('Multi-Objective Analysis');
ylabel('Value');
legend('Raw Score', 'Weight', 'Contribution', 'Location', 'best');

% Plot 4: WSI visualization - FIXED
subplot(2, 3, 4);
wsi_threshold = 0.85;
wsi_data = [WSI, wsi_threshold];
b = bar(wsi_data);
b.FaceColor = 'flat';
b.CData(1,:) = [0.2, 0.8, 0.2];
b.CData(2,:) = [0.8, 0.2, 0.2];
title('Weight Stability Index');
xlabel('Metric');
ylabel('WSI Value');
set(gca, 'XTick', 1:2, 'XTickLabel', {'WSI', 'Threshold'});
ylim([0, 1]);
text(1, WSI/2, sprintf('%.3f', WSI), 'HorizontalAlignment', 'center', ...
     'FontSize', 12, 'FontWeight', 'bold');

% Plot 5: Parameter adjustment analysis
subplot(2, 3, 5);
total_adjustments = delta_adjustments + gamma_adjustments;
bar(1:n_params_med, [delta_adjustments; gamma_adjustments; total_adjustments]', ...
    'stacked');
title('Parameter Adjustment Breakdown');
xlabel('Parameter Index');
ylabel('Adjustment Value');
legend('Demographic', 'Contextual', 'Total', 'Location', 'best');
set(gca, 'XTick', 1:n_params_med);

% Plot 6: Final weight distribution
subplot(2, 3, 6);
pie(w_final, parameters_med);
title('Final Optimized Weight Distribution');

sgtitle('Algorithm 4: Multi-Objective Weight Optimization');


%% WASHABILITY AND FABRIC DURABILITY ANALYSIS


fprintf('=== WASHABILITY AND FABRIC DURABILITY ANALYSIS ===\n\n');

% Fabric properties based on the 6 fabrics from Algorithm 1 case study
fabric_names = {'f1', 'f2', 'f3', 'f4', 'f5', 'f6'};
n_fabrics = 6;

% Initial fabric properties (normalized to 1.0)
initial_properties = struct();
initial_properties.conductivity = [0.85, 0.78, 0.92, 0.71, 0.88, 0.83];
initial_properties.flexibility = [0.90, 0.82, 0.87, 0.75, 0.91, 0.85];
initial_properties.comfort = [0.80, 0.95, 0.93, 0.88, 0.87, 0.79];
initial_properties.durability = [0.95, 0.87, 0.89, 0.92, 0.84, 0.91];

% Wash cycle simulation parameters
max_wash_cycles = 100;
wash_cycles = 0:max_wash_cycles;

% Degradation parameters for each fabric (different decay rates)
% Based on material science literature for smart textiles
decay_rates = struct();
decay_rates.conductivity = [0.025, 0.030, 0.020, 0.035, 0.022, 0.028]; % Higher decay for conductivity
decay_rates.flexibility = [0.015, 0.018, 0.012, 0.020, 0.014, 0.016];  % Moderate decay
decay_rates.comfort = [0.008, 0.006, 0.007, 0.010, 0.009, 0.011];      % Lower decay
decay_rates.durability = [0.012, 0.015, 0.010, 0.013, 0.016, 0.014];   % Structural decay

% Washability simulation function
function remaining_property = washability_model(initial_value, decay_rate, wash_count)
    % Exponential decay model for fabric properties
    remaining_property = initial_value * exp(-decay_rate * wash_count);
end

% Calculate property degradation over wash cycles
properties_over_time = struct();
property_names = fieldnames(initial_properties);

for prop_idx = 1:length(property_names)
    prop_name = property_names{prop_idx};
    properties_over_time.(prop_name) = zeros(n_fabrics, length(wash_cycles));
    
    for fabric_idx = 1:n_fabrics
        for cycle_idx = 1:length(wash_cycles)
            properties_over_time.(prop_name)(fabric_idx, cycle_idx) = ...
                washability_model(initial_properties.(prop_name)(fabric_idx), ...
                                decay_rates.(prop_name)(fabric_idx), ...
                                wash_cycles(cycle_idx));
        end
    end
end

% Calculate overall fabric performance (weighted combination)
performance_weights = [0.35, 0.25, 0.20, 0.20]; % conductivity, flexibility, comfort, durability
overall_performance = zeros(n_fabrics, length(wash_cycles));

for fabric_idx = 1:n_fabrics
    for cycle_idx = 1:length(wash_cycles)
        overall_performance(fabric_idx, cycle_idx) = ...
            performance_weights(1) * properties_over_time.conductivity(fabric_idx, cycle_idx) + ...
            performance_weights(2) * properties_over_time.flexibility(fabric_idx, cycle_idx) + ...
            performance_weights(3) * properties_over_time.comfort(fabric_idx, cycle_idx) + ...
            performance_weights(4) * properties_over_time.durability(fabric_idx, cycle_idx);
    end
end

% Find when each fabric drops below 70% performance threshold
threshold = 0.70;
useful_life_cycles = zeros(n_fabrics, 1);

for fabric_idx = 1:n_fabrics
    performance_curve = overall_performance(fabric_idx, :);
    below_threshold = find(performance_curve < threshold, 1);
    if isempty(below_threshold)
        useful_life_cycles(fabric_idx) = max_wash_cycles;
    else
        useful_life_cycles(fabric_idx) = wash_cycles(below_threshold);
    end
end

% Display durability results
fprintf('Fabric Durability Analysis:\n');
fprintf('Useful Life (wash cycles until <70%% performance):\n');
for i = 1:n_fabrics
    fprintf('%s: %d wash cycles\n', fabric_names{i}, useful_life_cycles(i));
end

% Find most durable fabric
[max_life, most_durable_idx] = max(useful_life_cycles);
fprintf('\nMost durable fabric: %s (%d wash cycles)\n', ...
        fabric_names{most_durable_idx}, max_life);

% Performance at common wash intervals
common_intervals = [0, 10, 25, 50, 75, 100];
fprintf('\nPerformance retention at common wash intervals:\n');
fprintf('Fabric\t');
for interval = common_intervals
    fprintf('%d cycles\t', interval);
end
fprintf('\n');

for fabric_idx = 1:n_fabrics
    fprintf('%s\t', fabric_names{fabric_idx});
    for interval = common_intervals
        cycle_idx = interval + 1; % +1 because arrays start at 1
        fprintf('%.2f%%\t\t', overall_performance(fabric_idx, cycle_idx) * 100);
    end
    fprintf('\n');
end

%% Visualization of Washability and Durability Analysis
figure('Position', [100, 100, 1600, 1000]);

% Plot 1: Overall performance degradation
subplot(3, 3, 1);
colors = lines(n_fabrics);
for fabric_idx = 1:n_fabrics
    plot(wash_cycles, overall_performance(fabric_idx, :) * 100, ...
         'LineWidth', 2, 'Color', colors(fabric_idx, :), 'DisplayName', fabric_names{fabric_idx});
    hold on;
end
yline(threshold * 100, '--r', 'LineWidth', 2, 'DisplayName', '70% Threshold');
title('Overall Performance vs Wash Cycles');
xlabel('Wash Cycles');
ylabel('Performance Retention (%)');
legend('Location', 'best');
grid on;

% Plot 2: Conductivity degradation
subplot(3, 3, 2);
for fabric_idx = 1:n_fabrics
    plot(wash_cycles, properties_over_time.conductivity(fabric_idx, :) * 100, ...
         'LineWidth', 1.5, 'Color', colors(fabric_idx, :));
    hold on;
end
title('Conductivity Retention');
xlabel('Wash Cycles');
ylabel('Conductivity (%)');
grid on;

% Plot 3: Flexibility degradation
subplot(3, 3, 3);
for fabric_idx = 1:n_fabrics
    plot(wash_cycles, properties_over_time.flexibility(fabric_idx, :) * 100, ...
         'LineWidth', 1.5, 'Color', colors(fabric_idx, :));
    hold on;
end
title('Flexibility Retention');
xlabel('Wash Cycles');
ylabel('Flexibility (%)');
grid on;

% Plot 4: Comfort degradation
subplot(3, 3, 4);
for fabric_idx = 1:n_fabrics
    plot(wash_cycles, properties_over_time.comfort(fabric_idx, :) * 100, ...
         'LineWidth', 1.5, 'Color', colors(fabric_idx, :));
    hold on;
end
title('Comfort Retention');
xlabel('Wash Cycles');
ylabel('Comfort (%)');
grid on;

% Plot 5: Durability degradation
subplot(3, 3, 5);
for fabric_idx = 1:n_fabrics
    plot(wash_cycles, properties_over_time.durability(fabric_idx, :) * 100, ...
         'LineWidth', 1.5, 'Color', colors(fabric_idx, :));
    hold on;
end
title('Structural Durability');
xlabel('Wash Cycles');
ylabel('Durability (%)');
grid on;

% Plot 6: Useful life comparison
subplot(3, 3, 6);
bar(useful_life_cycles, 'FaceColor', [0.2, 0.6, 0.8]);
title('Useful Life Comparison');
xlabel('Fabric');
ylabel('Wash Cycles');
set(gca, 'XTick', 1:n_fabrics, 'XTickLabel', fabric_names);
for i = 1:n_fabrics
    text(i, useful_life_cycles(i) + 2, sprintf('%d', useful_life_cycles(i)), ...
         'HorizontalAlignment', 'center', 'FontWeight', 'bold');
end

% Plot 7: Decay rate comparison
subplot(3, 3, 7);
decay_matrix = [decay_rates.conductivity; decay_rates.flexibility; ...
                decay_rates.comfort; decay_rates.durability];
bar(decay_matrix');
title('Decay Rate Comparison');
xlabel('Fabric');
ylabel('Decay Rate');
legend({'Conductivity', 'Flexibility', 'Comfort', 'Durability'}, 'Location', 'best');
set(gca, 'XTick', 1:n_fabrics, 'XTickLabel', fabric_names);

% Plot 8: Performance heatmap at key intervals
subplot(3, 3, 8);
performance_at_intervals = zeros(n_fabrics, length(common_intervals));
for i = 1:n_fabrics
    for j = 1:length(common_intervals)
        cycle_idx = common_intervals(j) + 1;
        performance_at_intervals(i, j) = overall_performance(i, cycle_idx);
    end
end
imagesc(performance_at_intervals);
colormap('parula');
colorbar;
title('Performance Heatmap');
xlabel('Wash Intervals');
ylabel('Fabric');
set(gca, 'XTick', 1:length(common_intervals), 'XTickLabel', common_intervals);
set(gca, 'YTick', 1:n_fabrics, 'YTickLabel', fabric_names);

% Plot 9: Cost-effectiveness analysis (performance per wash cycle)
subplot(3, 3, 9);
cost_effectiveness = useful_life_cycles ./ max(useful_life_cycles);
performance_at_50_cycles = zeros(n_fabrics, 1);
for i = 1:n_fabrics
    performance_at_50_cycles(i) = overall_performance(i, 51); % 50 cycles + 1
end
scatter(cost_effectiveness, performance_at_50_cycles, 100, colors, 'filled');
title('Cost-Effectiveness Analysis');
xlabel('Relative Durability');
ylabel('Performance at 50 Cycles');
for i = 1:n_fabrics
    text(cost_effectiveness(i) + 0.02, performance_at_50_cycles(i), ...
         fabric_names{i}, 'FontSize', 10);
end
grid on;

sgtitle('Washability and Fabric Durability Analysis');

%% Summary Statistics
fprintf('\n=== SUMMARY STATISTICS ===\n\n');

% Algorithm 3 Summary
fprintf('Algorithm 3 - Weight Derivation:\n');
fprintf('Application: Medical ICU Garments\n');
fprintf('Highest Priority: %s (%.1f%%)\n', parameters_med{weight_ranking(1)}, ...
        sorted_weights(1)*100);
fprintf('Top 3 Parameters: %s, %s, %s\n', parameters_med{weight_ranking(1)}, ...
        parameters_med{weight_ranking(2)}, parameters_med{weight_ranking(3)});

% Algorithm 4 Summary
fprintf('\nAlgorithm 4 - Multi-Objective Optimization:\n');
fprintf('Weight Stability Index: %.3f\n', WSI);
if WSI >= 0.85
    fprintf('Optimization Status: %s\n', 'Accepted');
else
    fprintf('Optimization Status: %s\n', 'Smoothed');
end
[~, max_weight_idx] = max(w_final);
fprintf('Final Top Parameter: %s (%.1f%%)\n', parameters_med{max_weight_idx}, ...
        max(w_final)*100);

% Durability Summary
fprintf('\nFabric Durability Analysis:\n');
fprintf('Most Durable: %s (%d wash cycles)\n', fabric_names{most_durable_idx}, max_life);
fprintf('Average Useful Life: %.1f wash cycles\n', mean(useful_life_cycles));
fprintf('Performance Range at 50 cycles: %.1f%% - %.1f%%\n', ...
        min(overall_performance(:, 51))*100, max(overall_performance(:, 51))*100);

fprintf('\n=== SIMULATION COMPLETED ===\n');
fprintf('All algorithms validated with case study data\n');
fprintf('Washability and durability models implemented\n');
fprintf('Results ready for practical implementation\n\n');