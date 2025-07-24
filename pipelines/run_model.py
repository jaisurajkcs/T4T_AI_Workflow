import pandas as pd
from joblib import load
from config import MODEL_PATH

def run_model(df):
    model = load(MODEL_PATH)
    X = df[['need_index']]
    df['predicted_coverage'] = model.predict(X)
    return df
