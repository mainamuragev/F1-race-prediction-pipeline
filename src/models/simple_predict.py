import joblib
import pandas as pd
import numpy as np

model = joblib.load('artifacts/race_predictor.pkl')
upcoming = pd.read_csv('data/processed/upcoming_features.csv')
features = ['driver_form_last5','grid','constructor_form_last5','pit_form_last5','cum_season_points','lap_consistency_last5','circuit_avg_points','grid_form_last5']
X = upcoming[features].fillna(0)
preds = model.predict(X)
upcoming['predicted_position'] = preds
sorted_df = upcoming.sort_values('predicted_position')
print("\n🏁 PREDICTED FINISHING ORDER FOR THE NEXT RACE 🏁\n")
print(f"{'Pos':<4} {'Driver':<30} {'Team':<20} {'Pred Finish':<12}")
for i, (_, row) in enumerate(sorted_df.iterrows(), 1):
    pred = max(1, min(20, round(row['predicted_position'], 1)))
    print(f"{i:<4} {row['driver_name']:<30} {row['team']:<20} {pred:<12}")
sorted_df.to_csv('data/processed/predictions.csv', index=False)
