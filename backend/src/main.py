from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import os
from .config import DATA_FOLDER

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

sector = pd.read_json(os.path.join(DATA_FOLDER, "sector.json"))
telemetry = pd.read_json(os.path.join(DATA_FOLDER, "telemetry.json"))
weather = pd.read_json(os.path.join(DATA_FOLDER, "weather.json"))

sector['SectorSessionTime_Start'] = pd.to_timedelta(sector['SectorSessionTime_Start'])
sector['SectorSessionTime_End']   = pd.to_timedelta(sector['SectorSessionTime_End'])
telemetry['SessionTime'] = pd.to_timedelta(telemetry['SessionTime'])
weather['SessionTime'] = pd.to_timedelta(weather['SessionTime'])

@app.get("/data")
def get_data(since_time: float = Query(...), until_time: float = Query(...)):
    since = pd.Timedelta(seconds=since_time)
    until = pd.Timedelta(seconds=until_time)
    sector_out = sector[
        (sector['SectorSessionTime_Start'] > since) &
        (sector['SectorSessionTime_Start'] <= until)
    ]
    telemetry_out = telemetry[
        (telemetry['SessionTime'] > since) &
        (telemetry['SessionTime'] <= until)
    ]
    weather_out = weather[
        (weather['SessionTime'] > since) &
        (weather['SessionTime'] <= until)
    ]

    # Calculate max session time (race end)
    max_session_time = float(sector['SectorSessionTime_End'].max().total_seconds())
    race_ended = False
    if until_time >= max_session_time:
        race_ended = True

    return {
        "sector": sector_out.to_dict(orient="records"),
        "telemetry": telemetry_out.to_dict(orient="records"),
        "weather": weather_out.to_dict(orient="records"),
        "race_ended": race_ended,
    }

@app.get("/range")
def get_range():
    lap1_start = sector[sector['LapNumber'] == 1]['SectorSessionTime_Start'].min()
    tmax = float(sector['SectorSessionTime_End'].max().total_seconds())
    return {
        "min": float(lap1_start.total_seconds()),
        "max": tmax
    }