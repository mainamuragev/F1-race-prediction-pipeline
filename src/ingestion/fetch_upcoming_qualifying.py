import fastf1
from fastf1 import get_event_schedule
from datetime import datetime
import pandas as pd
import os

def fetch_qualifying_for_next_race():
    now = datetime.now()
    for year in [now.year, now.year+1]:
        try:
            schedule = get_event_schedule(year)
            events = schedule[schedule['RoundNumber'] > 0].copy()
            events['Date'] = pd.to_datetime(events['EventDate'])
            future = events[events['Date'] > now]
            if not future.empty:
                next_race = future.iloc[0]
                year = next_race['EventDate'].year
                round_num = next_race['RoundNumber']
                # Check if qualifying has happened (date < now? but qualifying is usually day before race)
                # For simplicity, we try to load; if fails, skip
                try:
                    session = fastf1.get_session(year, round_num, 'Q')
                    session.load(laps=False, telemetry=False)
                    results = session.results
                    if results is not None and not results.empty:
                        grid_data = results[['DriverNumber', 'FullName', 'TeamName', 'Position']]
                        grid_data.columns = ['driver_number', 'driver_name', 'team', 'grid']
                        os.makedirs('data/processed', exist_ok=True)
                        grid_data.to_csv('data/processed/next_race_grid.csv', index=False)
                        print(f"✅ Grid for {next_race['EventName']} saved.")
                    else:
                        print(f"Qualifying for {next_race['EventName']} not yet available.")
                except Exception as e:
                    print(f"Could not fetch qualifying: {e}")
                return
        except Exception:
            continue
    print("No upcoming race found.")

if __name__ == "__main__":
    fetch_qualifying_for_next_race()
