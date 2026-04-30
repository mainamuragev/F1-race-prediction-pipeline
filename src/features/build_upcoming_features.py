"""
build_upcoming_features.py
--------------------------
Builds per-driver feature rows for the NEXT upcoming race.
Uses the last 5 completed races from features.csv as the rolling window.
Fetches next race info from the OpenF1 API (Ergast fallback).
"""

import pandas as pd
import requests
import os
import json
from datetime import datetime


FEATURE_COLS = [
    "driver_form_last5",
    "constructor_form_last5",
    "grid_form_last5",
    "pit_form_last5",
    "lap_consistency_last5",
    "circuit_avg_points",
    "cum_season_points",
    "grid",
]


def fetch_next_race_info():
    """Try Ergast for the next race. Returns dict or None."""
    # Ergast is being deprecated — try it but don't fail hard
    try:
        url = "http://ergast.com/api/f1/current/next.json"
        resp = requests.get(url, timeout=8)
        if resp.status_code == 200:
            race = resp.json()["MRData"]["RaceTable"]["Races"]
            if race:
                r = race[0]
                return {
                    "race_name":    r["raceName"],
                    "circuit_name": r["Circuit"]["circuitName"],
                    "circuit_id":   r["Circuit"]["circuitId"],
                    "date":         r["date"],
                    "round":        int(r["round"]),
                    "season":       int(r["season"]),
                }
    except Exception as e:
        print(f"⚠️  Ergast unavailable: {e}")
    return None


def build_upcoming_features(
    features_csv   = "data/processed/features.csv",
    output_csv     = "data/processed/upcoming_features.csv",
    race_info_json = "data/processed/next_race_info.json",
):
    print("📂 Loading historical features...")
    df = pd.read_csv(features_csv)
    df["year"]  = df["year"].astype(int)
    df["round"] = df["round"].astype(int)

    # Get next race info
    next_race = fetch_next_race_info()
    if next_race:
        print(f"🏁 Next race: {next_race['race_name']} — {next_race['date']}")
        with open(race_info_json, "w") as f:
            json.dump(next_race, f, indent=4)
        next_circuit_id = next_race.get("circuit_id")
        next_season     = next_race["season"]
        next_round      = next_race["round"]
    else:
        print("⚠️  Could not fetch next race. Using latest available data.")
        next_circuit_id = None
        next_season     = df["year"].max()
        next_round      = df["round"].max() + 1

    # For each driver, take their most recent 5 races as the feature window
    df_sorted = df.sort_values(["driverId", "year", "round"])

    upcoming_rows = []
    for driver_id, grp in df_sorted.groupby("driverId"):
        last5 = grp.tail(5)

        # Rolling averages over last 5 for form features
        row = {
            "driverId":               driver_id,
            "driverRef":              grp["driverRef"].iloc[-1] if "driverRef" in grp.columns else driver_id,
            "constructorId":          grp["constructorId"].iloc[-1],
            "constructorRef":         grp["constructorRef"].iloc[-1] if "constructorRef" in grp.columns else "",
            "driver_form_last5":      last5["points"].mean(),
            "constructor_form_last5": last5["constructor_form_last5"].mean(),
            "grid_form_last5":        last5["grid_form_last5"].mean(),
            "pit_form_last5":         last5["pit_form_last5"].mean(),
            "lap_consistency_last5":  last5["lap_consistency_last5"].mean(),
            "cum_season_points":      grp[grp["year"] == grp["year"].max()]["points"].sum(),
            "grid":                   last5["grid"].mean(),  # placeholder; update after qualifying
        }

        # Circuit history for the upcoming circuit
        if next_circuit_id and "circuitId" in grp.columns:
            circuit_hist = grp[grp["circuitId"] == next_circuit_id]["points"]
            row["circuit_avg_points"] = circuit_hist.mean() if not circuit_hist.empty else 0.0
        else:
            row["circuit_avg_points"] = grp["circuit_avg_points"].iloc[-1] if "circuit_avg_points" in grp.columns else 0.0

        upcoming_rows.append(row)

    upcoming = pd.DataFrame(upcoming_rows)

    # Fill any remaining NaNs
    for col in FEATURE_COLS:
        if col in upcoming.columns:
            upcoming[col] = upcoming[col].fillna(0)

    os.makedirs("data/processed", exist_ok=True)
    upcoming.to_csv(output_csv, index=False)
    print(f"✅ Upcoming features saved → {output_csv}  ({len(upcoming)} drivers)")
    return upcoming


if __name__ == "__main__":
    build_upcoming_features()
