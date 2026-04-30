import pandas as pd
import os

def load_local_dataset():
    # Load the renamed dataset file
    df = pd.read_csv("data/raw/f1_drivers_dataset_1950_2026.csv")
    return df

if __name__ == "__main__":
    df = load_local_dataset()
    os.makedirs("data/raw", exist_ok=True)
    df.to_csv("data/raw/race_results.csv", index=False)
    print(f"✅ Raw race data saved to data/raw/race_results.csv with {len(df)} rows")
