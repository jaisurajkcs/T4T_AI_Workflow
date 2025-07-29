# Imports

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("electric_vehicles_spec_2025(in).csv")  # Load the dataset

df.dropna(subset=['segment'], inplace=True)  # Drop rows where 'segment' is NaN
df['cargo_volume_l'] = pd.to_numeric(df['cargo_volume_l'], errors='coerce')  # Convert 'cargo_volume_l' to numeric
df.dropna()

# Define Features
X = df.drop(columns=['brand', 'model', 'segment', 'source_url'])
y = df['segment']  # Target variable

# Create Categorical features and encode them
cata_cols = X.select_dtypes(include=['object']).columns  # Categorical columns
X_encoded = pd.get_dummies(X, columns=cata_cols)  # One-hot encoding

le = LabelEncoder()
y_encoded = le.fit_transform(y)  # Encode target variable
all_labels = list(range(len(le.classes_)))
class_names = le.classes_

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y_encoded, test_size=0.3, random_state=42)

# Initialize and train the Random Forest Classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

#Prediction time and Evaluation
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)   
report = classification_report(y_test, y_pred, labels=all_labels, target_names=le.classes_, zero_division=0)

# Print the results
print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:\n", report)

# Plot heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted Segment')
plt.ylabel('Actual Segment')
plt.title('Confusion Matrix Heatmap for Vehicle Segment Classification')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()


