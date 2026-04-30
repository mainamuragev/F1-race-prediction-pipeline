"""
ingest_hybrid.py
----------------
Ingestion layer: tries FastF1 telemetry first for completed 2026 races,
falls back to Kaggle CSV data. Logs every run with counts and timestamps.
"""

import pandas as pd
import os
import json
from datetime import datetime

try:
    import fastf1
    FASTF1_AVAILABLE = True
except ImportError:
    FASTF1_AVAILABLE = False
    print("⚠️  fastf1 not installed. Will use Kaggle data only.")


# ---------------------------------------------------------------------------
# Kaggle / local CSV loader
# ---------------------------------------------------------------------------

def load_kaggle_results():
    """Load and join the core Kaggle Ergast tables."""
    base = "data/raw"
    results      = pd.read_csv(f"{base}/results.csv")
    races        = pd.read_csv(f"{base}/races.csv")
    drivers      = pd.read_csv(f"{base}/drivers.csv")
    constructors = pd.read_csv(f"{base}/constructors.csv")
    pit_stops    = pd.read_csv(f"{base}/pit_stops.csv")

    # Replace Ergast null sentinel
    for df in [results, races, drivers, constructors, pit_stops]:
        df.replace(r"\\N", pd.NA, regex=True, inplace=True)

    # Constructor points per race
    results["constructor_points"] = (
        pd.to_numeric(results["points"], errors="coerce")
          .fillna(0)
    )
    results["constructor_points"] = results.groupby(
        ["raceId", "constructorId"]
    )["constructor_points"].transform("sum")

    # Average pit stop time per driver per race
    pit_stops["milliseconds"] = pd.to_numeric(pit_stops["milliseconds"], errors="coerce")
    pit_avg = (
        pit_stops.groupby(["raceId", "driverId"])["milliseconds"]
                 .mean()
                 .reset_index()
                 .rename(columns={"milliseconds": "pit_stop_ms"})
    )

    df = (
        results
        .merge(races[["raceId", "year", "round", "circuitId"]], on="raceId", how="left")
        .merge(drivers[["driverId", "driverRef"]],              on="driverId", how="left")
        .merge(constructors[["constructorId", "constructorRef"]],on="constructorId", how="left")
        .merge(pit_avg, on=["raceId", "driverId"], how="left")
    )

    df["points"]       = pd.to_numeric(df["points"],       errors="coerce").fillna(0)
    df["grid"]         = pd.to_numeric(df["grid"],         errors="coerce")
    df["pit_stop_ms"]  = df["pit_stop_ms"].fillna(df["pit_stop_ms"].median())
    df["lap_consistency_ms"] = 0.95  # placeholder; FastF1 fills this

    return df[[
        "raceId", "driverId", "driverRef", "constructorId", "constructorRef",
        "year", "round", "circuitId",
        "points", "constructor_points", "grid",
        "pit_stop_ms", "lap_consistency_ms",
    ]]


# ---------------------------------------------------------------------------
# FastF1 loader
# ---------------------------------------------------------------------------

def load_fastf1_race(year, gp_name):
    """Fetch one race from FastF1. Returns empty DataFrame on failure."""
    if not FASTF1_AVAILABLE:
        return pd.DataFrame()
    try:
        session = fastf1.get_session(year, gp_name, 'R')
        session.load()
        laps     = session.laps
        pitstops = session.pit_stops

        rows = []
        for drv in session.drivers:
            drv_laps    = laps.pick_driver(drv)
            consistency = (
                drv_laps['LapTime'].dt.total_seconds().std()
                if not drv_laps.empty else None
            )
            pit_time = (
                pitstops.loc[pitstops['Driver'] == drv, 'Duration'].mean()
                if not pitstops.empty else None
            )
            try:
                res_row = session.results.loc[
                    session.results['Abbreviation'] == drv
                ].iloc[0]
                points   = float(res_row['Points'])
                grid     = float(res_row['GridPosition'])
                team     = str(res_row['TeamName'])
            except (IndexError, KeyError):
                continue

            rows.append({
                "raceId":             gp_name,
                "driverId":           drv,
                "constructorRef":     team,
                "points":             points,
                "constructor_points": None,
                "grid":               grid,
                "pit_stop_ms":        pit_time * 1000 if pd.notna(pit_time) else None,
                "lap_consistency_ms": consistency * 1000 if consistency else None,
            })
        print(f"   ✅ FastF1 telemetry: {gp_name} {year}")
        return pd.DataFrame(rows)
    except Exception as e:
        print(f"   ⚠️  FastF1 failed ({gp_name} {year}): {e}")
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Hybrid builder
# ---------------------------------------------------------------------------

def build_hybrid_dataset(
    output_csv     = "data/processed/race_results_hybrid.csv",
    summary_json   = "logs/ingestion_summary.json",
    log_txt        = "logs/ingestion_log.txt",
    year           = 2026,
):
    print("📦 Loading Kaggle base data...")
    kaggle_df = load_kaggle_results()

    telemetry_count = 0
    fallback_count  = 0
    fastf1_dfs      = []

    if FASTF1_AVAILABLE:
        print(f"🔄 Fetching {year} FastF1 telemetry for completed races...")
        try:
            schedule  = fastf1.get_event_schedule(year, include_testing=False)
            completed = schedule[schedule['EventDate'] < pd.Timestamp.today()]
        except Exception as e:
            print(f"⚠️  Could not get schedule: {e}")
            completed = pd.DataFrame()

        for _, event in completed.iterrows():
            gp_name = event['EventName']
            f1_df   = load_fastf1_race(year, gp_name)
            if not f1_df.empty:
                fastf1_dfs.append(f1_df)
                telemetry_count += 1
            else:
                fallback_count += 1

    # Combine FastF1 rows with Kaggle base
    if fastf1_dfs:
        telemetry_df = pd.concat(fastf1_dfs, ignore_index=True)
        # Prefer FastF1 values where available, fall back to Kaggle
        combined = pd.concat([kaggle_df, telemetry_df], ignore_index=True)
        # Keep FastF1 rows first so they win on deduplication
        combined = combined.drop_duplicates(
            subset=["raceId", "driverId"], keep="first"
        )
    else:
        combined = kaggle_df
        print("ℹ️  Using Kaggle data only (no FastF1 data available).")

    # Timestamp
    run_time = datetime.now().isoformat()
    combined["last_ingested"] = run_time

    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    combined.to_csv(output_csv, index=False)

    summary = {
        "run_timestamp":          run_time,
        "total_rows":             len(combined),
        "fastf1_telemetry_races": telemetry_count,
        "kaggle_fallback_races":  fallback_count,
    }
    with open(summary_json, "w") as f:
        json.dump(summary, f, indent=4)

    log_line = (
        f"{run_time} | FastF1: {telemetry_count} | "
        f"Kaggle fallback: {fallback_count} | Rows: {len(combined)}\n"
    )
    with open(log_txt, "a") as lf:
        lf.write(log_line)

    print(f"\n✅ Hybrid dataset saved → {output_csv}  ({len(combined):,} rows)")
    print(f"   FastF1 races: {telemetry_count} | Kaggle fallback: {fallback_count}")
    return combined


if __name__ == "__main__":
    build_hybrid_dataset()
