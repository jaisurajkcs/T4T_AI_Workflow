import pandas as pd
import numpy as np

def generate_internal_distribution_from_external(
    input_csv="data/external/mhm.csv",
    output_csv="data/internal/distribution_simulated.csv",
    seed=42
):
    np.random.seed(seed)

    # Load CSV
    df = pd.read_csv(input_csv)

    # Normalize and clean string columns
    df["indicator"] = df["indicator"].astype(str).str.strip().str.lower()
    df["method"] = df["method"].astype(str).str.strip().str.lower()
    df["scenario"] = df["scenario"].astype(str).str.strip().str.lower()

    # Debug: print unique values
    print("Unique values in indicator:", df["indicator"].unique())
    print("Unique values in method:", df["method"].unique())
    print("Unique values in scenario:", df["scenario"].unique())

    # Flexible filtering
    filtered = df[
        df["indicator"].str.contains("menstruator", na=False) &
        df["method"].str.contains("single-use", na=False) &
        df["scenario"].str.contains("s1", na=False)
    ].copy()

    print("Filtered rows found:", len(filtered))
    if filtered.empty:
        print("No matching rows found. Exiting.")
        return

    # Rename columns
    filtered = filtered.rename(columns={
        "geo": "region",
        "country": "country_name",
        "value": "estimated_need"
    })

    # Define coverage rates based on income
    coverage_lookup = {
        "High-income": 0.85,
        "Upper-middle-income": 0.65,
        "Lower-middle-income": 0.45,
        "Low-income": 0.25
    }

    def simulate_coverage(row):
        base = coverage_lookup.get(row["gni_group"], 0.5)
        noise = np.random.normal(0, 0.05)
        return np.clip(base + noise, 0.1, 0.95)

    # Generate fields
    filtered["coverage_pct"] = filtered.apply(simulate_coverage, axis=1)
    filtered["kits_distributed"] = (filtered["estimated_need"] * filtered["coverage_pct"]).round().astype(int)
    filtered["distribution_events"] = (
        (filtered["kits_distributed"] / 50000 + np.random.randint(1, 4, len(filtered)))
    ).round().astype(int)

    # Prepare final output
    output = filtered[[
        "country_name", "region", "year", "estimated_need",
        "kits_distributed", "coverage_pct", "distribution_events"
    ]].sort_values(by="country_name")

    # Save
    output.to_csv(output_csv, index=False)
    print(f"Internal distribution simulation saved to {output_csv}")

if __name__ == "__main__":
    generate_internal_distribution_from_external()