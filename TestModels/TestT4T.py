#imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error, r2_score



data = pd.read_csv("electric_vehicles_spec_2025(in).csv") # Load the dataset
data.info() # Display basic information about the dataset
print(data.isnull().sum()) # Print the missing values (if any) in the dataset, prints sum of missing values


# Feature Selecting

X = data[['battery_capacity_kWh', 'efficiency_wh_per_km']]
y = data['range_km']

# Check for missing values in features and target
mask = X.join(y).notna().all(axis=1)
X = X[mask]
y = y[mask]

# Establish a train/test split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

# Model Training

model = LinearRegression()
model.fit(X_train, y_train)

# Model Prediction
y_pred = model.predict(X_test) 

# Model Evaluation
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
r2   = r2_score(y_test, y_pred)

print(f"Test RMSE: {rmse:.2f} km")
print(f"R² (test): {r2:.4f}\n")

print("Coefficients:")
for name, coef in zip(X.columns, model.coef_):
    print(f"  {name}: {coef:.4f}")
print(f"Intercept: {model.intercept_:.4f}")

# Plot: Actual vs. Predicted
plt.figure(figsize=(6, 6))
sns.scatterplot(x=y_test, y=y_pred, s=60)
max_val = max(y_test.max(), y_pred.max())
plt.plot([0, max_val], [0, max_val], 'k--', linewidth=1.2)  # 45° reference
plt.xlabel("Actual Range (km)")
plt.ylabel("Predicted Range (km)")
plt.title("Actual vs. Predicted Range")
plt.grid(True)
plt.tight_layout()
plt.show()

# Residual plot
plt.figure(figsize=(6, 4))
residuals = y_test - y_pred
sns.scatterplot(x=y_pred, y=residuals, s=60)
plt.axhline(0, color='k', linestyle='--', linewidth=1.2)
plt.xlabel("Predicted Range (km)")
plt.ylabel("Residuals (km)")
plt.title("Residuals vs. Predicted")
plt.grid(True)
plt.tight_layout()
plt.show()