import pandas as pd
import numpy as np
import os

FEATURE_COLS = [
    'driver_form_last5', 'grid', 'constructor_form_last5', 'pit_form_last5',
    'cum_season_points', 'lap_consistency_last5', 'circuit_avg_points', 'grid_form_last5'
]

def build_upcoming_features():
    hist = pd.read_csv('data/processed/features.csv')
    # Get most recent stats per driver
    last_race = hist.sort_values('year', ascending=False).drop_duplicates('driver_name')
    # Load next race grid if available
    grid_file = 'data/processed/next_race_grid.csv'
    if os.path.exists(grid_file):
        grid = pd.read_csv(grid_file)
        grid = grid.rename(columns={'driver_name': 'driver'})
        upcoming = grid.merge(last_race, on='driver', how='left')
    else:
        # fallback: use last known stats for drivers appearing in latest race
        latest_race = pd.read_csv('data/raw/f1_drivers_dataset_1950_2026.csv')
        drivers = latest_race['driver_name'].unique()
        upcoming = last_race[last_race['driver_name'].isin(drivers)].copy()
        upcoming['grid'] = None  # will be filled with last race's grid or median
    # Ensure all feature columns exist
    for col in FEATURE_COLS:
        if col not in upcoming.columns:
            upcoming[col] = 0
        else:
            upcoming[col] = upcoming[col].fillna(0)
    upcoming = upcoming[['driver_name', 'team'] + FEATURE_COLS].rename(columns={'driver_name': 'driver'})
    upcoming.to_csv('data/processed/upcoming_features.csv', index=False)
    print(f"✅ Upcoming features for {len(upcoming)} drivers")

if __name__ == "__main__":
    build_upcoming_features()
