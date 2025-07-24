import pandas as pd

def fetch_external_data():
    # Placeholder: load from World Bank or UNICEF
    # return pd.read_csv("data/external/socioeconomic.csv")
    return pd.read_csv("data/external/mhm.csv")

def fetch_internal_data():
    return pd.read_csv("data/internal/distribution_simulated.csv")

def merge_data(ext_df, int_df):
    return pd.merge(ext_df, int_df, on="country", how="inner")
