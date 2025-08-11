# Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest

from matplotlib.lines import Line2D

# Load the dataset
df = pd.read_csv("electric_vehicles_spec_2025(in).csv")

# Data Preprocessing
df = df.dropna(subset=['battery_capacity_kWh', 'efficiency_wh_per_km', 'torque_nm', 'range_km'])  # Drop rows with NaN in these columns

# Feature Selecting for Linear Regression 
features_lr = ['battery_capacity_kWh', 'efficiency_wh_per_km', 'torque_nm']
X_lr = df[features_lr]
y_lr = df['range_km']

# Train Linear Regression Model
model_lr = LinearRegression()
model_lr.fit(X_lr, y_lr)

# Creating a prediction feature for Linear Regression
df['predicted_range_km']= model_lr.predict(X_lr)

# Data Preprocessing for Random Forest Regressor
features_rf = ['battery_capacity_kWh', 'efficiency_wh_per_km', 'torque_nm', 'predicted_range_km']
X_rf = df[features_rf]
y_rf = df['range_km']  # Target variable for Random Forest model

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_rf, y_rf, test_size=0.20, random_state=42)

# Train Random Forest Classifier Model
reg = RandomForestRegressor(n_estimators=100, random_state=42)
reg.fit(X_train, y_train)

# Prediction and Evaluation for Random Forest Classifier
y_pred = reg.predict(X_test)

# Isolation Forest for anomaly detection
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_rf)

iso_forest = IsolationForest(contamination=0.05, random_state=42)
df['anomaly_scores'] = iso_forest.fit_predict(X_scaled)
df['anomaly'] = df['anomaly_scores'] == -1

# Map anomaly status to numeric values for coloring
color_map = df['anomaly'].map({False: 0, True: 1})

# Plotting augemented data with anomalies
plt.figure(figsize=(10, 6))
plt.scatter(df.index, y_rf, c=color_map, cmap='coolwarm', label='Anomaly')

# Custom legend
legend_elements = [
    Line2D([0], [0], marker='o', color='w', label='Normal', markerfacecolor='blue', markersize=8),
    Line2D([0], [0], marker='o', color='w', label='Anomaly', markerfacecolor='red', markersize=8)
]

plt.xlabel("Vehicle Index (For Visualization)")
plt.ylabel("Range (km)")
plt.title("Anomaly Detection in EV Range Predictions")
plt.legend(handles=legend_elements)
plt.grid(True)
plt.tight_layout()
plt.savefig("mod_anomaly_detection_plot.png")

# Implementing Cross Validation for Random Forest
cv_scores = cross_val_score(reg, X_rf, y_rf, cv=5, scoring='neg_mean_squared_error')
mse_cv_scores = -cv_scores
mean_mse_cv = mse_cv_scores.mean()

# Output MSE and number of anomalies
mse = mean_squared_error(y_test, y_pred)
num_anomalies = df['anomaly'].sum()

print(f"Random Forest MSE (Test Set): {mse:.2f}")
print(f"Root Mean Squared Error for RF model (RMSE): {np.sqrt(mse):.2f}")
print(f"Number of anomalies detected: {num_anomalies}")
print(f"Cross-validated MSE (5-fold): {mean_mse_cv:.2f}")
print(f"Root Mean Square Error (CV): {np.sqrt(mean_mse_cv):.2f}")


# # Feature Importance
# importances = clf.feature_importances_
# feature_importance_df = pd.DataFrame({
#     'Feature': features_rf,
#     'Importance': importances
# }).sort_values(by='Importance', ascending=False)

# # Plot feature importance
# plt.figure(figsize=(10, 6))
# sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
# plt.title('Feature Importance in Random Forest Model')
# plt.tight_layout()
# plt.savefig("feature_importance.png")

# # Output results
# print(f"Mean Squared Error of the Random Forest model with augmented features: {mse:.4f}")
# print("\nFeature Importances:")
# print(feature_importance_df)
