import fastf1
import pandas as pd
import json


# --- USER PARAMETERS ---
YEAR = 2025
EVENT_NAME = 'Bahrain Grand Prix'


def parse_laps_to_sector(laps_df_copy):
    # Build per-sector DataFrame
    sector_list = []
    for sec_num in [1, 2, 3]:
        sec_df = laps_df_copy.copy()
        sec_df['SectorNumber'] = sec_num
        sec_df['SectorTime'] = sec_df[f'Sector{sec_num}Time']
        sec_df['SectorSessionTime_Start'] = (
            sec_df['LapStartTime'] if sec_num == 1 else sec_df[f'Sector{sec_num-1}SessionTime']
        )
        sec_df['SectorSessionTime_End'] = sec_df[f'Sector{sec_num}SessionTime']
        sector_list.append(sec_df)
    per_sector = pd.concat(sector_list, ignore_index=True)

    # Build sector interval table
    cols_to_copy = [
        'Year', 'EventName', 'Team', 'Driver',
        'Stint', 'LapNumber', 'SectorNumber',
        'Compound', 'TyreLife', 'SectorTime', 
        'SectorSessionTime_Start', 'SectorSessionTime_End'
        ]
    sector_intervals = per_sector[cols_to_copy].copy()
    return sector_intervals

def main():
    fastf1.Cache.enable_cache('./cache')

    year = YEAR
    event_name = EVENT_NAME

    session = fastf1.get_session(year, event_name, 'R')
    session.load(telemetry=True, weather=True)

    laps_df = session.laps.copy()
    laps_df['Year'] = year
    laps_df['EventName'] = event_name

    weather_df = session.weather_data.copy()
    weather_df['Year'] = year
    weather_df['EventName'] = event_name

    telemetry_dfs = []
    for idx, lap in session.laps.iterlaps():
        tel = lap.get_car_data().add_track_status()
        tel['Year'] = year
        tel['EventName'] = event_name
        tel['Driver'] = lap.Driver
        tel['LapNumber'] = lap.LapNumber
        telemetry_dfs.append(tel)
    if telemetry_dfs:
        telemetry_df = pd.concat(telemetry_dfs, ignore_index=True)
    else:
        telemetry_df = pd.DataFrame()  # fallback in case of no data

    laps_df = laps_df[[
        'Year', 'EventName', 'Team', 'Driver', 
        'Stint', 'LapNumber', 'LapStartTime',
        'Sector1Time', 'Sector2Time', 'Sector3Time',
        'Sector1SessionTime', 'Sector2SessionTime', 'Sector3SessionTime',
        'Compound', 'TyreLife', 'FreshTyre'
    ]]

    telemetry_df = telemetry_df[[
        'Year', 'EventName', 'Driver', 'LapNumber', 'SessionTime', 
        'RPM', 'Speed', 'nGear', 'Throttle', 'Brake', 'DRS', 'TrackStatus'
    ]]

    weather_df = weather_df[[
        'Year', 'EventName', 'SessionTime',
        'AirTemp', 'Humidity', 'Pressure', 'Rainfall',
        'TrackTemp', 'WindDirection', 'WindSpeed'
    ]]

    # Save to JSON
    telemetry_df.to_json("telemetry.json", orient="records", indent=2, date_format="iso")
    weather_df.to_json("weather.json", orient="records", indent=2, date_format="iso")
    sector_df = parse_laps_to_sector(laps_df)
    sector_df.to_json("sector.json", orient="records", indent=2, date_format="iso")

if __name__ == "__main__":
    main()