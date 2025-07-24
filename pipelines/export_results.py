from config import PREDICTION_PATH

def export(df):
    df.to_csv(PREDICTION_PATH, index=False)
    print(f"Predictions saved to {PREDICTION_PATH}")
