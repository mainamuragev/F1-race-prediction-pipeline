import pandas as pd
import numpy as np

# Load historical features
hist = pd.read_csv('data/processed/features.csv')
# Load latest race (raw) to get current drivers
latest_race = pd.read_csv('data/raw/f1_drivers_dataset_1950_2026.csv')
# Get most recent stats per driver from historical data
driver_col = 'driver' if 'driver' in hist.columns else 'driver_name'
latest_stats = hist.sort_values(['year','round']).groupby(driver_col).last().reset_index()
# Merge with current drivers (use driver name)
current = latest_race[['driver_name', 'team']].drop_duplicates('driver_name')
merged = current.merge(latest_stats, left_on='driver_name', right_on=driver_col, how='left')
# Fill missing features with median from historical
features = ['driver_form_last5','grid','constructor_form_last5','pit_form_last5','cum_season_points','lap_consistency_last5','circuit_avg_points','grid_form_last5']
for f in features:
    if f not in merged.columns:
        merged[f] = hist[f].median()
    else:
        merged[f] = merged[f].fillna(hist[f].median())
merged[['driver_name','team'] + features].to_csv('data/processed/upcoming_features.csv', index=False)
print(f"✅ Upcoming features for {len(merged)} drivers")
