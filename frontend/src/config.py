# config.py
import os

BACKEND_URL = "http://localhost:8000"
# BACKEND_URL = "http://backend:8000"

event_name = "Bahrain Grand Prix"

# Feature columns
numeric_cols = [
    'LapNumber', 'SectorTime', 'TyreLife', 'Speed_P10',
    'Throttle_Median', 'Throttle_ZeroPct',
    'Gear_Range', 'DRS_ActivePct', 'TrackStatus_Mean',
    'AirTemp', 'Humidity', 'Pressure', 'TrackTemp',
    'WindDirection', 'WindSpeed'
]
categorical_cols = ['EventName', 'Team', 'Compound', 'Stint', 'TrackStatus_Mode']

feature_cols = numeric_cols + categorical_cols

window_size=15

# Model/encoder/scaler paths (relative to the 'models' folder)
MODEL_PATH = os.path.join(os.path.dirname(__file__), "../model/model_telemetry.pkl")
ENCODER_PATH = os.path.join(os.path.dirname(__file__), "../model/encoder_telemetry.pkl")
SCALER_PATH = os.path.join(os.path.dirname(__file__), "../model/scaler_dict_telemetry.pkl")