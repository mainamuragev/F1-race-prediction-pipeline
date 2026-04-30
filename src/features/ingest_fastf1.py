import fastf1
import pandas as pd
import os

def fetch_race(year, gp_name):
    session = fastf1.get_session(year, gp_name, 'R')
    session.load()
    laps = session.laps
    pitstops = session.pit_stops

    rows = []
    for drv in session.drivers:
        drv_laps = laps.pick_driver(drv)
        avg_lap = drv_laps['LapTime'].mean().total_seconds() if not drv_laps.empty else None
        consistency = drv_laps['LapTime'].std().total_seconds() if not drv_laps.empty else None
        pit_time = pitstops.loc[pitstops['Driver'] == drv, 'Duration'].mean()

        rows.append({
            "driverId": drv,
            "points": session.results.loc[drv, 'Points'],
            "constructorId": session.results.loc[drv, 'Team'],
            "grid": session.results.loc[drv, 'GridPosition'],
            "pit_stop_time": pit_time if not pd.isna(pit_time) else 2.5,
            "lap_consistency": consistency if consistency is not None else 0.95
        })

    return pd.DataFrame(rows)

if __name__ == "__main__":
    # Dynamically fetch the 2026 calendar
    schedule = fastf1.get_event_schedule(2026)
    completed = schedule.loc[schedule['EventDate'] < pd.Timestamp.today()]

    dfs = []
    for _, event in completed.iterrows():
        gp_name = event['EventName']
        try:
            df = fetch_race(2026, gp_name)
            dfs.append(df)
            print(f"✅ Fetched {gp_name}")
        except Exception as e:
            print(f"⚠️ Failed to fetch {gp_name}: {e}")

    if dfs:
        all_results = pd.concat(dfs)
        os.makedirs("data/processed", exist_ok=True)
        all_results.to_csv("data/processed/race_results.csv", index=False)
        print("✅ Saved race_results.csv with FastF1 data")
    else:
        print("⚠️ No race data fetched")
