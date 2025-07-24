def clean_data(df):
    df = df.dropna(subset=['poverty_rate', 'female_teens', 'coverage'])
    df['poverty_rate'] = df['poverty_rate'].clip(0, 100)
    return df

# def clean_data(df):
#     # Example: check for missing estimated_need or gni_group
#     required_columns = ['estimated_need', 'gni_group']
#     missing_cols = [col for col in required_columns if col not in df.columns]
#     if missing_cols:
#         raise KeyError(f"Missing required columns: {missing_cols}")

#     return df.dropna(subset=required_columns)
