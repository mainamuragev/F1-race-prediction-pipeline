"""
clean_data.py
-------------
Cleans raw Ergast CSV tables before feature engineering.
Handles: Ergast \\N nulls, type coercion, lap time string → ms,
         DNF position filling, whitespace normalization.
"""

import pandas as pd
import os


def parse_lap_time(val):
    """Convert '1:23.456' string to milliseconds float."""
    if pd.isna(val):
        return None
    val = str(val).strip()
    try:
        if ":" in val:
            parts   = val.split(":")
            minutes = int(parts[0])
            seconds = float(parts[1])
            return (minutes * 60 + seconds) * 1000
        return float(val)
    except Exception:
        return None


def clean_results(df):
    df = df.copy()
    df.replace(r"\\N", pd.NA, regex=True, inplace=True)
    df["points"]        = pd.to_numeric(df["points"],        errors="coerce").fillna(0)
    df["grid"]          = pd.to_numeric(df["grid"],          errors="coerce")
    df["positionOrder"] = pd.to_numeric(df["positionOrder"], errors="coerce")
    df["laps"]          = pd.to_numeric(df["laps"],          errors="coerce")
    # Fill DNF positions with a large number (not a win)
    df["positionOrder"] = df["positionOrder"].fillna(20)
    return df


def clean_qualifying(df):
    df = df.copy()
    df.replace(r"\\N", pd.NA, regex=True, inplace=True)
    for col in ["q1", "q2", "q3"]:
        if col in df.columns:
            df[f"{col}_ms"] = df[col].apply(parse_lap_time)
    df["position"] = pd.to_numeric(df["position"], errors="coerce")
    return df


def clean_lap_times(df):
    df = df.copy()
    df.replace(r"\\N", pd.NA, regex=True, inplace=True)
    df["milliseconds"] = pd.to_numeric(df["milliseconds"], errors="coerce")
    if "time" in df.columns:
        df["time_ms"] = df["time"].apply(parse_lap_time)
        # Use parsed value if milliseconds column is missing
        df["milliseconds"] = df["milliseconds"].fillna(df["time_ms"])
    return df


def clean_pit_stops(df):
    df = df.copy()
    df.replace(r"\\N", pd.NA, regex=True, inplace=True)
    df["milliseconds"] = pd.to_numeric(df["milliseconds"], errors="coerce")
    df["duration"]     = pd.to_numeric(df["duration"],     errors="coerce")
    # Clamp unrealistic pit stop times (< 1 s or > 120 s are bad data)
    if "duration" in df.columns:
        df.loc[df["duration"] < 1,   "duration"] = pd.NA
        df.loc[df["duration"] > 120, "duration"] = pd.NA
    return df


def clean_drivers(df):
    df = df.copy()
    df.replace(r"\\N", pd.NA, regex=True, inplace=True)
    if "driverRef" in df.columns:
        df["driverRef"] = df["driverRef"].str.strip().str.lower()
    return df


def clean_constructors(df):
    df = df.copy()
    df.replace(r"\\N", pd.NA, regex=True, inplace=True)
    if "constructorRef" in df.columns:
        df["constructorRef"] = df["constructorRef"].str.strip().str.lower()
    return df


def run_all(raw_dir="data/raw", processed_dir="data/processed"):
    os.makedirs(processed_dir, exist_ok=True)

    tasks = [
        ("results.csv",              clean_results,       "results_clean.csv"),
        ("qualifying.csv",           clean_qualifying,    "qualifying_clean.csv"),
        ("lap_times.csv",            clean_lap_times,     "lap_times_clean.csv"),
        ("pit_stops.csv",            clean_pit_stops,     "pit_stops_clean.csv"),
        ("drivers.csv",              clean_drivers,       "drivers_clean.csv"),
        ("constructors.csv",         clean_constructors,  "constructors_clean.csv"),
    ]

    for fname, cleaner, out_name in tasks:
        path = os.path.join(raw_dir, fname)
        if not os.path.exists(path):
            print(f"⚠️  Skipping {fname} (not found)")
            continue
        df      = pd.read_csv(path)
        cleaned = cleaner(df)
        out     = os.path.join(processed_dir, out_name)
        cleaned.to_csv(out, index=False)
        print(f"✅ {fname:<35} → {out_name}  ({len(cleaned):,} rows)")


if __name__ == "__main__":
    print("🧹 Cleaning raw data tables...\n")
    run_all()
    print("\nDone.")
