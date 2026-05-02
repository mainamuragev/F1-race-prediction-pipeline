import pandas as pd
import fastf1
from fastf1 import get_event_schedule
from datetime import datetime
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_latest_finished_race():
    for year in [datetime.now().year, datetime.now().year-1, datetime.now().year-2]:
        try:
            schedule = get_event_schedule(year)
            events = schedule[schedule['RoundNumber'] > 0].copy()
            events['Date'] = pd.to_datetime(events['EventDate'])
            past = events[events['Date'] <= datetime.now()]
            if not past.empty:
                latest = past.iloc[-1]
                return year, int(latest['RoundNumber'])
        except:
            continue
    return 2024, 24

def fetch_race_advanced(year, round_num):
    session = fastf1.get_session(year, round_num, 'R')
    session.load(laps=False, telemetry=False, weather=True, messages=False)
    results = session.results
    rows = []
    for _, row in results.iterrows():
        rows.append({
            'driver_number': row['DriverNumber'],
            'driver_name': row['FullName'],
            'team': row['TeamName'],
            'position': row['Position'],
            'year': year,
            'round': round_num,
            'circuit': session.event['EventName']
        })
    df = pd.DataFrame(rows)
    # Get qualifying grid
    try:
        quali = session.get_qualifying()
        if quali is not None:
            quali_res = quali.results
            grid_map = dict(zip(quali_res['DriverNumber'], quali_res['Position']))
            df['grid'] = df['driver_number'].map(grid_map)
        else:
            df['grid'] = None
    except:
        df['grid'] = None
    # Get practice averages (FP1, FP2, FP3)
    for fp in [1,2,3]:
        try:
            fp_sess = fastf1.get_session(year, round_num, f'FP{fp}')
            fp_sess.load(laps=True, telemetry=False)
            avg_laps = fp_sess.laps.groupby('Driver')['LapTime'].mean().apply(lambda x: x.total_seconds())
            avg_map = dict(zip(avg_laps.index, avg_laps.values))
            df[f'fp{fp}_avg_lap'] = df['driver_number'].map(avg_map)
        except:
            df[f'fp{fp}_avg_lap'] = None
    # Get weather at race start (average temp, rain)
    if session.weather_data is not None and not session.weather_data.empty:
        start_weather = session.weather_data.iloc[0]
        df['race_temperature'] = start_weather.get('AirTemp', None)
        df['rainfall'] = 1 if start_weather.get('Rainfall', False) else 0
    else:
        df['race_temperature'] = None
        df['rainfall'] = 0
    os.makedirs('data/raw', exist_ok=True)
    df.to_csv('data/raw/f1_drivers_dataset_1950_2026.csv', index=False)
    logger.info(f"Saved advanced race data for {year} R{round_num}")
    return df

def fetch_recent_race(year=None, round_num=None):
    if year is None or round_num is None:
        year, round_num = get_latest_finished_race()
    return fetch_race_advanced(year, round_num)

if __name__ == "__main__":
    fetch_recent_race()
