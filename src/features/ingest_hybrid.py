import pandas as pd
import os
import fastf1
from datetime import datetime
import json

# --- Kaggle Results Loader ---
def load_kaggle_results():
    results = pd.read_csv("data/raw/results.csv")
    results["constructor_points"] = results.groupby(
        ["raceId","constructorId"]
    )["points"].transform("sum")

    # placeholders until FastF1 fills telemetry
    results["pit_stop_time"] = 2.5
    results["lap_consistency"] = 0.95

    return results[[
        "raceId","driverId","constructorId","points","constructor_points","grid",
        "pit_stop_time","lap_consistency"
    ]]

# --- FastF1 Telemetry Loader ---
def load_fastf1_race(year, gp_name):
    try:
        session = fastf1.get_session(year, gp_name, 'R')
        session.load()
        laps = session.laps
        pitstops = session.pit_stops

        rows = []
        for drv in session.drivers:
            drv_laps = laps.pick_driver(drv)
            consistency = drv_laps['LapTime'].std().total_seconds() if not drv_laps.empty else None
            pit_time = pitstops.loc[pitstops['Driver'] == drv, 'Duration'].mean()

            rows.append({
                "raceId": session.event['EventName'],
                "driverId": drv,
                "constructorId": session.results.loc[drv, 'Team'],
                "points": session.results.loc[drv, 'Points'],
                "constructor_points": None,  # merged later
                "grid": session.results.loc[drv, 'GridPosition'],
                "pit_stop_time": pit_time if not pd.isna(pit_time) else 2.5,
                "lap_consistency": consistency if consistency is not None else 0.95
            })
        print(f"✅ Using FastF1 telemetry for {gp_name}")
        return pd.DataFrame(rows)
    except Exception as e:
        print(f"⚠️ FastF1 failed for {gp_name}: {e}")
        return pd.DataFrame()

# --- Hybrid Merge ---
def build_hybrid_dataset():
    kaggle_df = load_kaggle_results()
    schedule = fastf1.get_event_schedule(2026)
    completed = schedule.loc[schedule['EventDate'] < pd.Timestamp.today()]

    dfs = []
    telemetry_count = 0
    fallback_count = 0

    for _, event in completed.iterrows():
        gp_name = event['EventName']
        f1_df = load_fastf1_race(2026, gp_name)
        if not f1_df.empty:
            dfs.append(f1_df)
            telemetry_count += 1
        else:
            print(f"⚠️ Falling back to Kaggle data for {gp_name}")
            fallback_count += 1

    telemetry_df = pd.concat(dfs) if dfs else pd.DataFrame()

    if not telemetry_df.empty:
        hybrid = pd.merge(
            kaggle_df, telemetry_df,
            on=["raceId","driverId","constructorId"],
            how="outer", suffixes=("_kaggle","_fastf1")
        )

        # Priority rules: prefer FastF1 values if present
        hybrid["points"] = hybrid["points_fastf1"].fillna(hybrid["points_kaggle"])
        hybrid["constructor_points"] = hybrid["constructor_points_fastf1"].fillna(hybrid["constructor_points_kaggle"])
        hybrid["grid"] = hybrid["grid_fastf1"].fillna(hybrid["grid_kaggle"])
        hybrid["pit_stop_time"] = hybrid["pit_stop_time_fastf1"].fillna(hybrid["pit_stop_time_kaggle"])
        hybrid["lap_consistency"] = hybrid["lap_consistency_fastf1"].fillna(hybrid["lap_consistency_kaggle"])

        # Keep only clean columns
        hybrid = hybrid[["raceId","driverId","constructorId","points","constructor_points","grid","pit_stop_time","lap_consistency"]]
    else:
        hybrid = kaggle_df

    # Add timestamp column
    run_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    hybrid["last_ingested"] = run_time

    os.makedirs("data/processed", exist_ok=True)
    hybrid.to_csv("data/processed/race_results.csv", index=False)

    # Save metadata summary
    summary = {
        "run_timestamp": run_time,
        "fastf1_telemetry_races": telemetry_count,
        "kaggle_fallback_races": fallback_count,
        "total_races": telemetry_count + fallback_count
    }
    with open("data/processed/ingestion_summary.json", "w") as f:
        json.dump(summary, f, indent=4)

    # Append to rolling log file
    log_line = f"{run_time} | FastF1: {telemetry_count} | Kaggle: {fallback_count} | Total: {telemetry_count + fallback_count}\n"
    with open("data/processed/ingestion_log.txt", "a") as log_file:
        log_file.write(log_line)

    print("✅ Saved hybrid race_results.csv")
    print("✅ Saved ingestion_summary.json")
    print("✅ Appended ingestion_log.txt")
    print(f"Summary: {telemetry_count} races used FastF1 telemetry, {fallback_count} races used Kaggle fallback.")

if __name__ == "__main__":
    build_hybrid_dataset()
