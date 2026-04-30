import requests
import pandas as pd
import os

def fetch_race_results(round_number):
    url = f"http://ergast.com/api/f1/current/{round_number}/results.json"
    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code != 200:
            print(f"Round {round_number} failed: {resp.status_code}")
            return pd.DataFrame()
        data = resp.json()
        races = data["MRData"]["RaceTable"]["Races"]
        if not races:
            return pd.DataFrame()
        race = races[0]
        results = race["Results"]
        rows = []
        for r in results:
            rows.append({
                "round": round_number,
                "driverId": r["Driver"]["driverId"],
                "points": float(r["points"]),
                "constructorId": r["Constructor"]["constructorId"],
                "grid": int(r["grid"]),
                "position": int(r["position"]),
            })
        return pd.DataFrame(rows)
    except Exception as e:
        print(f"Error fetching round {round_number}: {e}")
        return pd.DataFrame()

if __name__ == "__main__":
    dfs = []
    for rnd in range(1, 23):  # loop through all possible rounds
        df = fetch_race_results(rnd)
        if not df.empty:
            dfs.append(df)
    if dfs:
        all_results = pd.concat(dfs)
        os.makedirs("data/processed", exist_ok=True)
        all_results.to_csv("data/processed/race_results.csv", index=False)
        print("✅ Saved race_results.csv with available Ergast data")
    else:
        print("⚠️ Ergast API unavailable. Please use a local fallback dataset.")
