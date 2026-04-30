import sys
import fastf1
from fastf1.core import DataNotLoadedError

def test_fastf1_session(year, gp_name):
    session = fastf1.get_session(year, gp_name, 'R')
    try:
        session.load()
        print(f"✅ Loaded {gp_name} {year} Race")
        print("Drivers:", [drv for drv in session.drivers])
        try:
            print("Results head:")
            print(session.results.head())
        except DataNotLoadedError:
            print("⚠️ Results not available for this race.")
    except DataNotLoadedError:
        print(f"⚠️ Failed to load {gp_name} {year}: Data not available.")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python test_fastf1.py <year> \"<Grand Prix name>\"")
        sys.exit(1)

    year = int(sys.argv[1])
    gp_name = sys.argv[2]
    test_fastf1_session(year, gp_name)
