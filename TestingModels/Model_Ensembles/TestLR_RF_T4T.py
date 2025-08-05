#imports

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("electric_vehicles_spec_2025(in).csv")  # Load the dataset

# Data Preprocessing
df = df.dropna(subset=['battery_capacity_kWh', 'efficiency_wh_per_km', 'torque_nm', 'range_km'])  # Drop rows with NaN in these columns

# Feature Selecting for Linear Regression 
features_lr = ['battery_capacity_kWh', 'efficiency_wh_per_km', 'torque_nm', ]
X_lr = df[features_lr]
y_lr = df['range_km']

# Train Linear Regression Model
model_lr = LinearRegression()
model_lr.fit(X_lr, y_lr)

# Creating a prediction feature for Linear Regression
df['predicted_range_km']= model_lr.predict(X_lr)

# Data Preprocessing for Random Forest Classifier
features_rf = ['battery_capacity_kWh', 'efficiency_wh_per_km', 'torque_nm', 'predicted_range_km']
X_rf = df[features_rf]
y_rf = df['range_km']  # Target variable for Random Forest model

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_rf, y_rf, test_size=0.20, random_state=42)

# Train Random Forest Classifier Model
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Prediction and Evaluation for Random Forest Classifier
y_pred = clf.predict(X_test)
mse = mean_squared_error(y_test, y_pred)

# Feature Importance
importances = clf.feature_importances_
feature_importance_df = pd.DataFrame({
    'Feature': features_rf,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

# Plot feature importance
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
plt.title('Feature Importance in Random Forest Model')
plt.tight_layout()
plt.savefig("feature_importance.png")

# Output results
print(f"Mean Squared Error of the Random Forest model with augmented features: {mse:.4f}")
print("\nFeature Importances:")
print(feature_importance_df)

