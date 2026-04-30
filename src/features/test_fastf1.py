import fastf1

def test_fastf1_session(year, gp_name):
    try:
        # Load a race session
        session = fastf1.get_session(year, gp_name, 'R')
        session.load()

        print(f"✅ Loaded {gp_name} {year} Race")

        # Print basic info
        print("Drivers:", session.drivers)
        print("Results head:\n", session.results.head())

        # Print a few lap times for the first driver
        laps = session.laps.pick_driver(session.drivers[0])
        print("Sample lap times:\n", laps[['LapNumber','LapTime']].head())

        # Print pit stop info
        pitstops = session.pit_stops
        print("Pit stops:\n", pitstops.head())

    except Exception as e:
        print(f"⚠️ Failed to load {gp_name} {year}: {e}")

if __name__ == "__main__":
    # Try a known race (replace with any GP name)
      test_fastf1_session(2022, "Monaco Grand Prix")
      test_fastf1_session(2023, "Italian Grand Prix")
