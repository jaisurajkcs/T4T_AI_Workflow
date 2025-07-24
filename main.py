# from pipelines.fetch_data import fetch_external_data, fetch_internal_data, merge_data
# from pipelines.clean_data import clean_data
# from pipelines.engineer_features import add_need_index
# from pipelines.run_model import run_model
# from pipelines.export_results import export

# def full_pipeline():
#     ext = fetch_external_data()
#     internal = fetch_internal_data()
#     merged = merge_data(ext, internal)
#     cleaned = clean_data(merged)
#     features = add_need_index(cleaned)
#     predictions = run_model(features)
#     export(predictions)

# if __name__ == "__main__":
#     full_pipeline()

from pipelines.fetch_data import fetch_external_data, fetch_internal_data, merge_data
from pipelines.clean_data import clean_data
from pipelines.engineer_features import add_need_index

def export_features():
    ext = fetch_external_data()
    internal = fetch_internal_data()
    merged = merge_data(ext, internal)
    cleaned = clean_data(merged)
    features = add_need_index(cleaned)

    # NEW: Export to CSV so you can train your model
    features.to_csv("output/merged_features.csv", index=False)
    print("Feature dataset exported to output/merged_features.csv")

if __name__ == "__main__":
    export_features()
