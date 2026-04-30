import fastf1
import pandas as pd
from fastf1.core import DataNotLoadedError

YEARS = range(2018, 2024)  # backfill 2018–2023
RACES = [
    "Australian Grand Prix", "Bahrain Grand Prix", "Chinese Grand Prix",
    "Monaco Grand Prix", "Italian Grand Prix", "United States Grand Prix",
    "Abu Dhabi Grand Prix"
]

def ingest_season(year):
    records = []
    for gp in RACES:
        try:
            session = fastf1.get_session(year, gp, 'R')
            session.load()
            print(f"✅ Loaded {gp} {year}")
            try:
                df = session.results
                df['Year'] = year
                df['Race'] = gp
                records.append(df)
            except DataNotLoadedError:
                print(f"⚠️ Results not available for {gp} {year}")
        except Exception as e:
            print(f"❌ Failed {gp} {year}: {e}")
    return pd.concat(records) if records else pd.DataFrame()

if __name__ == "__main__":
    all_data = []
    for year in YEARS:
        season_df = ingest_season(year)
        if not season_df.empty:
            all_data.append(season_df)

    if all_data:
        df = pd.concat(all_data)
        df.to_csv("data/processed/multiseason_results.csv", index=False)
        print("📂 Saved multi-season results to data/processed/multiseason_results.csv")
    else:
        print("⚠️ No data ingested.")
