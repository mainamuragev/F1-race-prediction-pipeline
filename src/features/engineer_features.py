import pandas as pd

def engineer_features(input_csv="data/processed/multiseason_results.csv",
                      output_csv="data/processed/features_multiseason.csv"):
    df = pd.read_csv(input_csv)

    # Ensure chronological order per driver
    df = df.sort_values(by=["DriverId", "Year", "Race"])

    # Rolling driver form (last 3 races)
    df["driver_form_last3"] = (
        df.groupby("DriverId")["Points"]
          .transform(lambda x: x.rolling(3, min_periods=1).mean())
    )

    # Rolling constructor form (last 3 races) using TeamName
    if "TeamName" in df.columns:
        df["constructor_form_last3"] = (
            df.groupby("TeamName")["Points"]
              .transform(lambda x: x.rolling(3, min_periods=1).mean())
        )

    # Grid advantage (normalize grid position if available)
    if "GridPosition" in df.columns:
        max_grid = df["GridPosition"].max()
        df["grid_advantage"] = (max_grid - df["GridPosition"]) / max_grid

    df.to_csv(output_csv, index=False)
    print(f"📂 Features engineered and saved to {output_csv}")

if __name__ == "__main__":
    engineer_features()
