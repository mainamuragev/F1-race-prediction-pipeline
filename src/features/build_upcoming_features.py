import requests
import pandas as pd

def fetch_next_race():
    url = "http://ergast.com/api/f1/current.json"
    resp = requests.get(url)
    if resp.status_code != 200:
        raise RuntimeError(f"Failed to fetch race data: {resp.status_code}")
    try:
        data = resp.json()
    except ValueError:
        raise RuntimeError("Invalid JSON response from Ergast API")
    next_race = data["MRData"]["RaceTable"]["Races"][0]
    return next_race

def build_features():
    # Example: load your historical race results
    results = pd.read_csv("data/processed/race_results.csv")

    # Compute rolling averages for last 5 races per driver
    features = results.groupby("driverId").tail(5).groupby("driverId").agg({
        "points":"mean",
        "constructor_points":"mean",
        "grid":"mean",
        "pit_stop_time":"mean",
        "lap_consistency":"mean"
    }).reset_index()

    features.to_csv("data/processed/upcoming_features.csv", index=False)

if __name__ == "__main__":
    try:
        race = fetch_next_race()
        print(f"Next race: {race['raceName']} at {race['Circuit']['circuitName']}")
    except Exception as e:
        print(f"Error fetching next race: {e}")
    build_features()
