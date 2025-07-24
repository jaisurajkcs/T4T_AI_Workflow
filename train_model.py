import pandas as pd
from sklearn.linear_model import LinearRegression
from joblib import dump
from config import MODEL_PATH

# Load merged + cleaned + feature-engineered data manually
df = pd.read_csv('output/merged_features.csv')  # you can use output from your pipeline
X = df[['need_index']]
y = df['coverage']

model = LinearRegression()
model.fit(X, y)

# Save model
dump(model, MODEL_PATH)
print(f"Model saved to {MODEL_PATH}")
