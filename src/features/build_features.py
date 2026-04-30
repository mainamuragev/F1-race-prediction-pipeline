import pandas as pd

def add_driver_form(df):
    df = df.sort_values(by=["Driver", "Seasons"])
    df["driver_form_last5"] = (
        df.groupby("Driver")["Points"]
          .rolling(window=5, min_periods=1)
          .mean()
          .reset_index(level=0, drop=True)
    )
    return df

def add_constructor_form(df, constructor_standings, constructors, races):
    standings = constructor_standings.merge(constructors, on="constructorId", how="left")
    standings = standings.merge(races[["raceId", "year"]], on="raceId", how="left")

    def extract_year(val):
        if isinstance(val, str) and val.startswith("[") and val.endswith("]"):
            return int(val.strip("[]").split(",")[0])
        try:
            return int(val)
        except:
            return None

    df["Seasons"] = df["Seasons"].apply(extract_year)
    standings["year"] = standings["year"].astype(int)

    standings = standings.sort_values(by=["name", "year"])
    standings["constructor_form_last5"] = (
        standings.groupby("name")["points"]
                 .rolling(window=5, min_periods=1)
                 .mean()
                 .reset_index(level=0, drop=True)
    )

    merged = df.merge(
        standings[["year", "name", "constructor_form_last5"]],
        left_on="Seasons", right_on="year", how="left"
    )
    return merged

def add_grid_form(df, qualifying, races):
    qualifying = qualifying.merge(races[["raceId", "year"]], on="raceId", how="left")
    qualifying = qualifying.sort_values(by=["driverId", "year"])

    qualifying["grid_form_last5"] = (
        qualifying.groupby("driverId")["position"]
                  .rolling(window=5, min_periods=1)
                  .mean()
                  .reset_index(level=0, drop=True)
    )

    merged = df.merge(
        qualifying[["year", "driverId", "grid_form_last5"]],
        left_on="Seasons", right_on="year", how="left"
    )
    return merged

def add_pit_form(df, pit_stops, races):
    pit_stops = pit_stops.merge(races[["raceId", "year"]], on="raceId", how="left")
    pit_summary = pit_stops.groupby(["driverId", "year"])["milliseconds"].mean().reset_index()

    pit_summary = pit_summary.sort_values(by=["driverId", "year"])
    pit_summary["pit_form_last5"] = (
        pit_summary.groupby("driverId")["milliseconds"]
                   .rolling(window=5, min_periods=1)
                   .mean()
                   .reset_index(level=0, drop=True)
    )

    merged = df.merge(
        pit_summary[["driverId", "year", "pit_form_last5"]],
        left_on=["driverId", "Seasons"],
        right_on=["driverId", "year"],
        how="left"
    ).drop(columns=["year"])

    merged = merged.drop_duplicates(subset=["Driver", "Seasons", "driverId"])
    return merged

def add_lap_consistency(df, lap_times, races):
    lap_times = lap_times.merge(races[["raceId", "year"]], on="raceId", how="left")
    lap_summary = lap_times.groupby(["driverId", "year"])["milliseconds"].std().reset_index()

    lap_summary = lap_summary.sort_values(by=["driverId", "year"])
    lap_summary["lap_consistency_last5"] = (
        lap_summary.groupby("driverId")["milliseconds"]
                   .rolling(window=5, min_periods=1)
                   .mean()
                   .reset_index(level=0, drop=True)
    )

    merged = df.merge(
        lap_summary[["driverId", "year", "lap_consistency_last5"]],
        left_on=["driverId", "Seasons"],
        right_on=["driverId", "year"],
        how="left"
    ).drop(columns=["year"])

    merged = merged.drop_duplicates(subset=["Driver", "Seasons", "driverId"])
    return merged

if __name__ == "__main__":
    df = pd.read_csv("data/raw/race_results.csv")
    constructor_standings = pd.read_csv("data/raw/constructor_standings.csv")
    constructors = pd.read_csv("data/raw/constructors.csv")
    races = pd.read_csv("data/raw/races.csv")
    qualifying = pd.read_csv("data/raw/qualifying.csv")
    pit_stops = pd.read_csv("data/raw/pit_stops.csv")
    lap_times = pd.read_csv("data/raw/lap_times.csv")

    df = add_driver_form(df)
    df = add_constructor_form(df, constructor_standings, constructors, races)
    df = add_grid_form(df, qualifying, races)
    df = add_pit_form(df, pit_stops, races)
    df = add_lap_consistency(df, lap_times, races)

    df.to_csv("data/processed/features.csv", index=False)
    print(f"✅ Features saved to data/processed/features.csv with {len(df)} rows")
