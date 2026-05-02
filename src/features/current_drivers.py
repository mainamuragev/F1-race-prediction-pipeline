import pandas as pd
import numpy as np
import os

# Load historical features
hist = pd.read_csv('data/processed/features.csv')

# Load latest race raw data
latest_race = pd.read_csv('data/raw/f1_drivers_dataset_1950_2026.csv')

# Determine columns for driver name and team
if 'driver_name' in latest_race.columns:
    driver_col = 'driver_name'
elif 'driver' in latest_race.columns:
    driver_col = 'driver'
else:
    raise KeyError("No driver name column found in raw file")

if 'team' in latest_race.columns:
    team_col = 'team'
else:
    team_col = 'constructor' # fallback

# Get unique current drivers and their teams
current = latest_race[[driver_col, team_col]].drop_duplicates(driver_col)
current.columns = ['driver_name', 'team']

# Map driver names to driverRef using drivers.csv if exists
driver_map = {}
drivers_file = 'data/raw/drivers.csv'
if os.path.exists(drivers_file):
    drivers_df = pd.read_csv(drivers_file)
    if 'driverRef' in drivers_df.columns and 'forename' in drivers_df.columns and 'surname' in drivers_df.columns:
        drivers_df['full_name'] = drivers_df['forename'] + ' ' + drivers_df['surname']
        driver_map = dict(zip(drivers_df['full_name'].str.lower(), drivers_df['driverRef']))
    elif 'driverRef' in drivers_df.columns and 'name' in drivers_df.columns:
        driver_map = dict(zip(drivers_df['name'].str.lower(), drivers_df['driverRef']))

current['driverRef'] = current['driver_name'].str.lower().map(driver_map)
# If mapping fails for some, we'll leave driverRef as None; those will get median values later.

# Get latest stats per driverRef from hist
latest_stats = hist.sort_values(['year','round']).groupby('driverRef').last().reset_index()

# Merge
merged = current.merge(latest_stats, on='driverRef', how='left')

# Feature list
features = ['driver_form_last5','grid','constructor_form_last5','pit_form_last5',
            'cum_season_points','lap_consistency_last5','circuit_avg_points','grid_form_last5']

# Fill missing with median from hist
for f in features:
    median_val = hist[f].median() if f in hist.columns else 0
    if f not in merged.columns:
        merged[f] = median_val
    else:
        merged[f] = merged[f].fillna(median_val)

# Ensure grid column (if missing, use median)
if 'grid' not in merged.columns:
    merged['grid'] = hist['grid'].median()

# Save only needed columns
merged[['driver_name','team'] + features].to_csv('data/processed/upcoming_features.csv', index=False)
print(f"✅ Upcoming features saved for {len(merged)} current drivers")
