import sys
import fastf1

def test_fastf1_session(year, gp_name):
    session = fastf1.get_session(year, gp_name, 'R')
    session.load()
    print(f"✅ Loaded {gp_name} {year} Race")
    print("Drivers:", [drv for drv in session.drivers])
    print("Results head:")
    print(session.results.head())

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python test_fastf1.py <year> \"<Grand Prix name>\"")
        sys.exit(1)

    year = int(sys.argv[1])
    gp_name = sys.argv[2]
    test_fastf1_session(year, gp_name)
