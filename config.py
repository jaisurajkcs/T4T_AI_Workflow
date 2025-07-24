from datetime import datetime

TODAY = datetime.now().strftime("%Y_%m_%d")
PREDICTION_PATH = f"output/predictions_{TODAY}.csv"
MODEL_PATH = "models/coverage_predictor.joblib"
