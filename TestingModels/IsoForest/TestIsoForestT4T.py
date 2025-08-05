# Imports

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample

# Load the dataset
df = pd.read_csv("electric_vehicles_spec_2025(in).csv")

# Feature Selection and Data preprocessing (For generating heatmap with multi-demensional data)

# features = [
#     'top_speed_kmh',
#     'battery_capacity_kWh',
#     'torque_nm',
#     'efficiency_wh_per_km',
#     'range_km',
#     'acceleration_0_100_s'
# ]

# df_clean = df[features].dropna() # Drop null rows

##################
# For 2D contour plot visualization, we need to ensure that the data is 2D for the Isolation Forest model.


features_2d = ['range_km', 'efficiency_wh_per_km']  # Select the two features for 2D visualization
df_2d = df[features_2d].dropna()
##################

# Scaling features
scaler = StandardScaler()  
#X_scaled = scaler.fit_transform(df_clean)
X_scaled_2d = scaler.fit_transform(df_2d) #For 2D visualization case


# Train Isolation Forest Model
# iso_forest = IsolationForest(contamination=0.05, random_state=42)
# df_clean['anomaly'] = iso_forest.fit_predict(X_scaled)
# df_clean['anomaly'] = df_clean['anomaly'].map({1: 'normal', -1: 'anomaly'})  # Map results to labels

########################################################################
# Fit Isolation Forest on the 2D data

iso_forest_2d = IsolationForest(contamination=0.05, random_state=42)
df_2d['anomaly'] = iso_forest_2d.fit_predict(X_scaled_2d)
df_2d['anomaly'] = df_2d['anomaly'].map({1: 'normal', -1: 'anomaly'})
########################################################################

##########################
# Commented out following lines as they were not needed for generating the heatmap, rather they were used for trying to create a contour plot

# scores = iso_forest.decision_function(X_scaled)
# preds = iso_forest.predict(X_scaled)
##########################

# # Introduce anomalies scores
# df_clean['anomaly_score'] = iso_forest.decision_function(X_scaled)

# # Summary
# print("Anomaly Detection Summary:")
# print(df_clean['anomaly'].value_counts())

# # Sample anomalies
# print("\nSample Anomalies:")
# print(df_clean[df_clean['anomaly'] == 'anomaly'].head())

# # Create a pivot table for the heatmap
# heatmap_data = df_clean.pivot_table(
#     index='range_km',
#     columns='efficiency_wh_per_km',
#     values='anomaly_score',
#     aggfunc='mean'
# )

# # Plot the heatmap
# plt.figure(figsize=(12, 8))
# sns.heatmap(heatmap_data, cmap='coolwarm', cbar_kws={'label': 'Anomaly Score'})
# plt.title('Heatmap of Anomaly Scores by Range and Efficiency')
# plt.xlabel('Efficiency (Wh/km)')
# plt.ylabel('Range (km)')
# plt.tight_layout()
# plt.show()


#####################################################################################################
# Create a meshgrid for contour plot

xx, yy = np.meshgrid(
    np.linspace(X_scaled_2d[:, 0].min() - 0.5, X_scaled_2d[:, 0].max() + 0.5, 500),
    np.linspace(X_scaled_2d[:, 1].min() - 0.5, X_scaled_2d[:, 1].max() + 0.5, 500)
)
grid = np.c_[xx.ravel(), yy.ravel()]
Z = iso_forest_2d.decision_function(grid)
Z = Z.reshape(xx.shape)

# Plot the decision function as a contour plot
plt.figure(figsize=(12, 8))
plt.contourf(xx, yy, Z, levels=np.linspace(Z.min(), Z.max(), 50), cmap=plt.cm.coolwarm, alpha=0.7)

# Overlay the data points
colors = {'normal': 'gold', 'anomaly': 'purple'}
for label in ['normal', 'anomaly']:
    subset = df_2d[df_2d['anomaly'] == label]
    plt.scatter(
        scaler.transform(subset[features_2d])[:, 0],
        scaler.transform(subset[features_2d])[:, 1],
        c=colors[label],
        label=label,
        edgecolor='k',
        s=40
    )

# Add labels and title
plt.title('Isolation Forest Anomaly Detection Contour Plot')
plt.xlabel('Range (km) [Standardized]')
plt.ylabel('Efficiency (Wh/km) [Standardized]')
plt.legend()
plt.colorbar(label='Anomaly Score')
plt.tight_layout()

# Show the plot
plt.show()
#####################################################################################################

