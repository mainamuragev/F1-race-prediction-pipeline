"""
build_features.py
-----------------
Single source of truth for all F1 feature engineering.
Reads from data/raw/ Kaggle CSVs and produces data/processed/features.csv
with 5 rolling form features + circuit + season context.
"""

import pandas as pd
import os


def load_raw_tables():
    base = "data/raw"
    results            = pd.read_csv(f"{base}/results.csv")
    races              = pd.read_csv(f"{base}/races.csv")
    drivers            = pd.read_csv(f"{base}/drivers.csv")
    constructors       = pd.read_csv(f"{base}/constructors.csv")
    constructor_stand  = pd.read_csv(f"{base}/constructor_standings.csv")
    qualifying         = pd.read_csv(f"{base}/qualifying.csv")
    pit_stops          = pd.read_csv(f"{base}/pit_stops.csv")
    lap_times          = pd.read_csv(f"{base}/lap_times.csv")
    circuits           = pd.read_csv(f"{base}/circuits.csv")

    # Replace Ergast null sentinel with NaN
    for df in [results, races, drivers, constructors, constructor_stand,
               qualifying, pit_stops, lap_times, circuits]:
        df.replace(r"\\N", pd.NA, regex=True, inplace=True)

    return results, races, drivers, constructors, constructor_stand, qualifying, pit_stops, lap_times, circuits


def build_base(results, races, drivers, constructors, circuits):
    """Join all core tables into one wide DataFrame."""
    df = (
        results
        .merge(races[["raceId", "year", "round", "circuitId", "name"]], on="raceId", how="left")
        .merge(drivers[["driverId", "driverRef"]], on="driverId", how="left")
        .merge(constructors[["constructorId", "constructorRef"]], on="constructorId", how="left")
        .merge(circuits[["circuitId", "circuitRef", "country"]], on="circuitId", how="left")
    )

    df["points"]       = pd.to_numeric(df["points"],       errors="coerce").fillna(0)
    df["grid"]         = pd.to_numeric(df["grid"],         errors="coerce")
    df["positionOrder"]= pd.to_numeric(df["positionOrder"],errors="coerce")
    df["year"]         = df["year"].astype(int)
    df["round"]        = df["round"].astype(int)

    # Cumulative season points at the time of each race (no leakage)
    df = df.sort_values(["driverId", "year", "round"])
    df["cum_season_points"] = (
        df.groupby(["driverId", "year"])["points"]
          .cumsum()
          .shift(1)
          .fillna(0)
    )

    return df


def add_driver_form(df):
    df = df.sort_values(["driverId", "year", "round"])
    df["driver_form_last5"] = (
        df.groupby("driverId")["points"]
          .transform(lambda x: x.rolling(5, min_periods=1).mean().shift(1))
          .fillna(0)
    )
    return df


def add_constructor_form(df, constructor_stand, races):
    cs = (
        constructor_stand
        .merge(races[["raceId", "year", "round"]], on="raceId", how="left")
    )
    cs["points"] = pd.to_numeric(cs["points"], errors="coerce").fillna(0)
    cs = cs.sort_values(["constructorId", "year", "round"])
    cs["constructor_form_last5"] = (
        cs.groupby("constructorId")["points"]
          .transform(lambda x: x.rolling(5, min_periods=1).mean().shift(1))
          .fillna(0)
    )
    df = df.merge(
        cs[["raceId", "constructorId", "constructor_form_last5"]],
        on=["raceId", "constructorId"], how="left"
    )
    df["constructor_form_last5"] = df["constructor_form_last5"].fillna(0)
    return df


def add_grid_form(df, qualifying, races):
    q = qualifying.merge(races[["raceId", "year", "round"]], on="raceId", how="left")
    q["position"] = pd.to_numeric(q["position"], errors="coerce")
    q = q.sort_values(["driverId", "year", "round"])
    q["grid_form_last5"] = (
        q.groupby("driverId")["position"]
          .transform(lambda x: x.rolling(5, min_periods=1).mean().shift(1))
          .fillna(0)
    )
    df = df.merge(
        q[["raceId", "driverId", "grid_form_last5"]],
        on=["raceId", "driverId"], how="left"
    )
    df["grid_form_last5"] = df["grid_form_last5"].fillna(df["grid"].fillna(10))
    return df


def add_pit_form(df, pit_stops, races):
    ps = pit_stops.merge(races[["raceId", "year", "round"]], on="raceId", how="left")
    ps["milliseconds"] = pd.to_numeric(ps["milliseconds"], errors="coerce")
    pit_summary = (
        ps.groupby(["driverId", "raceId", "year", "round"])["milliseconds"]
          .mean()
          .reset_index()
    )
    pit_summary = pit_summary.sort_values(["driverId", "year", "round"])
    pit_summary["pit_form_last5"] = (
        pit_summary.groupby("driverId")["milliseconds"]
                   .transform(lambda x: x.rolling(5, min_periods=1).mean().shift(1))
                   .fillna(pit_summary["milliseconds"].median())
    )
    df = df.merge(
        pit_summary[["raceId", "driverId", "pit_form_last5"]],
        on=["raceId", "driverId"], how="left"
    )
    df["pit_form_last5"] = df["pit_form_last5"].fillna(df["pit_form_last5"].median())
    return df


def add_lap_consistency(df, lap_times, races):
    lt = lap_times.merge(races[["raceId", "year", "round"]], on="raceId", how="left")
    lt["milliseconds"] = pd.to_numeric(lt["milliseconds"], errors="coerce")
    lap_summary = (
        lt.groupby(["driverId", "raceId", "year", "round"])["milliseconds"]
          .std()
          .reset_index()
          .rename(columns={"milliseconds": "lap_std"})
    )
    lap_summary = lap_summary.sort_values(["driverId", "year", "round"])
    lap_summary["lap_consistency_last5"] = (
        lap_summary.groupby("driverId")["lap_std"]
                   .transform(lambda x: x.rolling(5, min_periods=1).mean().shift(1))
                   .fillna(lap_summary["lap_std"].median())
    )
    df = df.merge(
        lap_summary[["raceId", "driverId", "lap_consistency_last5"]],
        on=["raceId", "driverId"], how="left"
    )
    df["lap_consistency_last5"] = df["lap_consistency_last5"].fillna(
        df["lap_consistency_last5"].median()
    )
    return df


def add_circuit_history(df):
    """Driver's average points at this specific circuit (prior races only)."""
    df = df.sort_values(["driverId", "year", "round"])
    df["circuit_avg_points"] = (
        df.groupby(["driverId", "circuitId"])["points"]
          .transform(lambda x: x.expanding().mean().shift(1))
          .fillna(0)
    )
    return df


def build_features(output_csv="data/processed/features.csv"):
    print("📦 Loading raw tables...")
    results, races, drivers, constructors, constructor_stand, qualifying, pit_stops, lap_times, circuits = load_raw_tables()

    print("🔗 Building base dataframe...")
    df = build_base(results, races, drivers, constructors, circuits)

    print("⚙️  Engineering features...")
    df = add_driver_form(df)
    df = add_constructor_form(df, constructor_stand, races)
    df = add_grid_form(df, qualifying, races)
    df = add_pit_form(df, pit_stops, races)
    df = add_lap_consistency(df, lap_times, races)
    df = add_circuit_history(df)

    # Final feature columns to keep
    keep_cols = [
        "raceId", "driverId", "driverRef", "constructorId", "constructorRef",
        "year", "round", "circuitId", "circuitRef", "country",
        "grid", "positionOrder", "points",
        "cum_season_points",
        "driver_form_last5",
        "constructor_form_last5",
        "grid_form_last5",
        "pit_form_last5",
        "lap_consistency_last5",
        "circuit_avg_points",
    ]
    df = df[[c for c in keep_cols if c in df.columns]]
    df = df.dropna(subset=["points"])

    os.makedirs("data/processed", exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"✅ Features saved → {output_csv}  ({len(df):,} rows, {df['year'].nunique()} seasons)")
    return df


if __name__ == "__main__":
    build_features()
