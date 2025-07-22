% Load dataset
data = readtable('electric_vehicles_spec_2025cop.csv');

% Select two numeric features and the target variable
X = data(:, {'battery_capacity_kWh', 'efficiency_wh_per_km'});
y = data.range_km;

% Remove rows with missing values
validRows = ~any(ismissing([X, table(y)]), 2);
X = X(validRows, :);
y = y(validRows);

% Combine predictors and target into one table
dataClean = [X, table(y)];
dataClean.Properties.VariableNames{end} = 'range_km';  % Rename 'y' to 'range_km'


% Clean variable names to be valid MATLAB identifiers
dataClean.Properties.VariableNames = matlab.lang.makeValidName(dataClean.Properties.VariableNames);

% Split data into training and testing sets (70/30 split)
cv = cvpartition(height(dataClean), 'HoldOut', 0.3);
trainData = dataClean(training(cv), :);
testData = dataClean(test(cv), :);

% Train linear regression model
mdl = fitlm(trainData, 'range_km ~ battery_capacity_kWh + efficiency_wh_per_km');

% Predict on test data
y_pred = predict(mdl, testData);

% Calculate RMSE
rmse = sqrt(mean((testData.range_km - y_pred).^2));
fprintf('Test RMSE: %.2f km\n', rmse);

% Display model summary
disp(mdl)

% Plot actual vs predicted
figure;
scatter(testData.range_km, y_pred, 'filled');
xlabel('Actual Range (km)');
ylabel('Predicted Range (km)');
title('Actual vs Predicted Range');
grid on;
