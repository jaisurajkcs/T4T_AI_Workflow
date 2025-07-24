def add_need_index(df):
    df['need_index'] = (df['poverty_rate'] * 0.6 +
                        df['female_teens'] * 0.4)
    return df[['region', 'coverage', 'need_index']]

# def add_need_index(df):
#     # Map GNI group to a priority score (higher = more need)
#     gni_score = {
#         "Low-income": 100,
#         "Lower-middle-income": 75,
#         "Upper-middle-income": 50,
#         "High-income": 25
#     }

#     df["gni_priority"] = df["gni_group"].map(gni_score)

#     # Need index = combination of GNI priority + unmet need (estimated - distributed)
#     df["unmet_need"] = df["estimated_need"] - df["kits_distributed"]
#     df["unmet_need"] = df["unmet_need"].clip(lower=0)

#     # Normalize unmet need (0 to 100 scale)
#     max_unmet = df["unmet_need"].max()
#     df["unmet_scaled"] = (df["unmet_need"] / max_unmet) * 100 if max_unmet > 0 else 0

#     # Final need index is a weighted combo
#     df["need_index"] = (0.6 * df["gni_priority"] + 0.4 * df["unmet_scaled"]).round(1)

#     return df
