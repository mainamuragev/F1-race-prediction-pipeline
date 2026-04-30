import pandas as pd
import os

if __name__ == "__main__":
    df = pd.read_csv("data/raw/race_results.csv")

    # Example cleaning
    df['driver'] = df['driver'].str.strip().str.lower()
    df['constructor'] = df['constructor'].str.strip().str.lower()

    os.makedirs("data/processed", exist_ok=True)
    df.to_csv("data/processed/clean_results.csv", index=False)
    print(" Cleaned data saved to data/processed/clean_results.csv")
