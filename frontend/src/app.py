import dash
from dash import dcc, html, Output, Input, State
import requests
import pandas as pd
import numpy as np
import pickle
import time
import os
import psutil
from .etl import (
    incremental_aggregate, encode, scale,
    is_consecutive_sector_window, prepare_window_for_prediction, predict
)
from .config import (
    numeric_cols, categorical_cols, feature_cols,
    BACKEND_URL, MODEL_PATH, ENCODER_PATH, SCALER_PATH,
    window_size, event_name
)

# Load encoder, scaler, and model once globally
with open(ENCODER_PATH, "rb") as f:
    encoder = pickle.load(f)
with open(SCALER_PATH, "rb") as f:
    scaler_dict = pickle.load(f)
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

app = dash.Dash(__name__)
app.layout = html.Div([
    dcc.Store(id='race-data', data={
        "sector": {},       # {driver: [sector dicts]}
        "telemetry": {},    # {driver: [telemetry dicts]}
        "weather": [],      # global, not per driver
        "agg_data": {},     # {driver: [agg dicts]}
        "buffers": {},      # {driver: [dicts]}
        "pit_preds": {},    # {driver: [dicts]}
        "last_time": 0.0,
        "ready": False,
        "sector_agg_id": {},     # {driver: set(row_ids)}
        "telemetry_agg_id": {},  # {driver: set(row_ids)}
    }),
    dcc.Interval(id='interval', interval=1000, n_intervals=0, disabled=False),
    html.Pre(id='data-info')
])

@app.callback(
    Output('race-data', 'data'),
    Output('data-info', 'children'),
    Output('interval', 'disabled'),
    Input('interval', 'n_intervals'),
    State('race-data', 'data')
)
def update(n_intervals, stored):
    # --- RESOURCE MONITOR ---
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / (1024 * 1024)  # in MB
    print(f"[Resource Monitor] Tick={n_intervals} | Memory usage: {mem:.1f} MB")
    if not isinstance(stored, dict) or not stored:
        msg = f"[ T-{n_intervals} ] Initializing stored data dictionary."
        stored = {
            "sector": {}, "telemetry": {}, "weather": [],
            "agg_data": {}, "buffers": {}, "pit_preds": {},
            "last_time": 0.0, "ready": False,
            "sector_agg_id": {},
            "telemetry_agg_id": {},
        }
        return stored, msg, False

    # --- INITIALIZATION ---
    if not stored.get("ready", False):
        try:
            r = requests.get(f"{BACKEND_URL}/range")
            if r.status_code != 200:
                msg = f"[ T-{n_intervals} ] Error: Could not fetch range."
                return stored, msg, False
            rng = r.json()
            stored['last_time'] = rng['min']
            stored['ready'] = True
            stored["sector"] = {}
            stored["telemetry"] = {}
            stored["weather"] = []
            stored["agg_data"] = {}
            stored["buffers"] = {}
            stored["pit_preds"] = {}
            stored["sector_agg_id"] = {}
            stored["telemetry_agg_id"] = {}
            msg = f"[ T-{n_intervals} ] Initialized at session min: {rng['min']}."
            return stored, msg, False
        except Exception as e:
            print(f"Error fetching range at tick {n_intervals}: {e}")
            msg = f"[ T-{n_intervals} ] Error fetching range."
            return stored, msg, False

    since_time = stored.get('last_time', 0.0)
    tick_size = 1.0
    new_time = since_time + tick_size

    # --- FETCH NEW DATA ---
    try:
        r = requests.get(f"{BACKEND_URL}/data", params={
            "since_time": since_time,
            "until_time": new_time
        })
        if r.status_code != 200:
            msg = f"[ T-{n_intervals} ] Error: Could not fetch new data."
            return stored, msg, False

        new = r.json()
    except Exception as e:
        print(f"Error fetching new data at tick {n_intervals}: {e}")
        msg = f"[ T-{n_intervals} ] Error fetching new data."
        return stored, msg, False

    race_ended = new.get("race_ended", False)

    # --- Append NEW DATA (per driver) ---
    for row in new["sector"]:
        driver = row["Driver"]
        row['row_id'] = f"sector-{row['Driver']}-{row['LapNumber']}-{row['SectorNumber']}-{n_intervals}"
        stored["sector"].setdefault(driver, []).append(row)
    for row in new["telemetry"]:
        driver = row["Driver"]
        row['row_id'] = f"telemetry-{row['Driver']}-{row['LapNumber']}-{row['SessionTime']}-{n_intervals}"
        stored["telemetry"].setdefault(driver, []).append(row)
    for row in new["weather"]:
        stored["weather"].append(row)
    stored['last_time'] = new_time

    # --- AGGREGATION: For each driver, process completed sectors ---
    agg_count = 0
    t0 = time.time()
    try:
        for driver in list(stored["sector"].keys()):
            driver_sectors = stored["sector"][driver]
            driver_telemetry = stored["telemetry"].get(driver, [])
            driver_weather = pd.DataFrame(stored["weather"])

            if not driver_sectors or not driver_telemetry:
                continue

            # Convert to DataFrame
            sector_df = pd.DataFrame(driver_sectors)
            telemetry_df = pd.DataFrame(driver_telemetry)
            weather_df = driver_weather  # Already a DataFrame

            # Type conversions for new data
            if not sector_df.empty:
                sector_df['SectorSessionTime_Start'] = parse_time_as_float(sector_df['SectorSessionTime_Start'])
                sector_df['SectorSessionTime_End'] = parse_time_as_float(sector_df['SectorSessionTime_End'])
                sector_df['SectorTime'] = parse_time_as_float(sector_df['SectorTime'])
                sector_df['LapNumber'] = sector_df['LapNumber'].astype(int, errors='ignore')
                sector_df = sector_df.dropna(subset=['SectorTime', 'SectorSessionTime_Start', 'SectorSessionTime_End'])

            if not telemetry_df.empty:
                telemetry_df['SessionTime'] = parse_time_as_float(telemetry_df['SessionTime'])
                telemetry_df['LapNumber'] = telemetry_df['LapNumber'].astype(int, errors='ignore')
                telemetry_df['TrackStatus'] = telemetry_df['TrackStatus'].astype(int, errors='ignore')

            if not weather_df.empty:
                weather_df['SessionTime'] = parse_time_as_float(weather_df['SessionTime'])

            # Skip if sector data is empty after conversion/filtering
            if sector_df.empty:
                continue

            # Init id sets if not present
            agg_id_set = set(stored["sector_agg_id"].get(driver, []))
            tel_id_set = set(stored["telemetry_agg_id"].get(driver, []))

            agg_df, sector_id_set_new, tel_id_set_new = incremental_aggregate(
                sector_df, telemetry_df, weather_df, agg_id_set, tel_id_set, new_time
            )
            if not agg_df.empty:
                # Encode/scale, and add to agg_data
                agg_df = encode(agg_df, categorical_cols, encoder)
                agg_df = scale(agg_df, numeric_cols, scaler_dict, event_name)
                agg_records = agg_df.to_dict(orient='records')
                stored["agg_data"].setdefault(driver, []).extend(agg_records)
                # Keep only the most recent 15
                stored["agg_data"][driver] = stored["agg_data"][driver][-15:]
                agg_count += len(agg_records)
                # Update id sets
                stored["sector_agg_id"][driver] = list(sector_id_set_new)
                stored["telemetry_agg_id"][driver] = list(tel_id_set_new)
    except Exception as e:
        print(f"Error during aggregation at tick {n_intervals}: {e}")
    
    # --- PREDICTION: For each driver, build/update buffer, run prediction ---
    t1 = time.time()
    agg_msg = f"\nAggregating took {t1-t0:.4f} seconds at tick {n_intervals}, time {new_time}"
    try:
        for driver, agg_list in stored["agg_data"].items():
            if not agg_list:
                continue
            # Get the last stint value
            last_stint = agg_list[-1].get('StintRaw')
            # Filter agg_list to the most recent window_size rows with the same stint
            same_stint = []
            for row in reversed(agg_list):
                if row.get('StintRaw') == last_stint:
                    same_stint.append(row)
                    if len(same_stint) == window_size:
                        break
                else:
                    break
            window = list(reversed(same_stint))
            stored["buffers"][driver] = window

            # Always append a prediction for the last sector in the buffer,
            # but only do model inference if buffer is full and consecutive and same stint.
            prob_val, pred_val = 0, 0
            if len(window) == window_size and is_consecutive_sector_window(window):
                X_cat, X_num = prepare_window_for_prediction(
                    window, feature_cols, categorical_cols, numeric_cols
                )
                prob, pred = predict(model, X_num, X_cat)
                if not np.isnan(prob[0]):
                    prob_val = float(prob[0])
                    pred_val = int(pred[0])
                else:
                    prob_val, pred_val = 0, 0

            # Always append (use 0 if not full/consecutive)
            if window:
                stored["pit_preds"].setdefault(driver, []).append({
                    "LapNumberRaw": window[-1].get("LapNumberRaw"),
                    "SectorNumberRaw": window[-1].get("SectorNumberRaw"),
                    "StintRaw": window[-1].get("StintRaw"),
                    "prob": prob_val,
                    "pred": pred_val,
                })
    except Exception as e:
        print(f"Error during prediction at tick {n_intervals}: {e}")

    # --- OUTPUT MESSAGE ---
    msg = f"[ T-{n_intervals} | {pd.to_timedelta(new_time, unit='s')} ]\n"
    msg += f"\nFetched: sector: {len(new['sector'])}, telemetry: {len(new['telemetry'])}, weather: {len(new['weather'])}"
    n_sector = sum(len(x) for x in stored["sector"].values())
    n_tel = sum(len(x) for x in stored["telemetry"].values())
    n_agg = sum(len(x) for x in stored["agg_data"].values())
    msg += f"\nTotals: sector: {n_sector}, telemetry: {n_tel}, weather: {len(stored['weather'])}"
    msg += f"\nNew agg rows: {agg_count} | Total agg: {n_agg}\n"
    msg += agg_msg + "\n"

    # Buffer lengths, show stint/lap/sector for last buffer entry if present
    buf_msg = ""
    for d, b in stored["buffers"].items():
        if b:
            last = b[-1]
            stint = last.get('StintRaw', '-')
            lap = last.get('LapNumberRaw', '-')
            sector = last.get('SectorNumberRaw', '-')
            buf_msg += f"{d}: {len(b)} (Stint={stint}, Lap={lap}, Sector={sector})\n"
        else:
            buf_msg += f"{d}: 0\n"
    msg += "\n[ Per-Driver Buffer Lengths ]\n" + buf_msg

    # Current leader
    all_agg = []
    for d, agg_list in stored["agg_data"].items():
        for rec in agg_list:
            all_agg.append((d, rec))
    if all_agg:
        leader = max(all_agg, key=lambda x: (x[1].get("LapNumberRaw", 0), x[1].get("SectorNumberRaw", 0)))
        msg += f"\n[ Current Leader ]\nDriver: {leader[0]} | Lap: {leader[1].get('LapNumberRaw', '-')} | Sector: {leader[1].get('SectorNumberRaw', '-')}\n"

    # Last pit predictions
    msg += "\n[ Current Pit Predictions ]"
    for d, preds in stored["pit_preds"].items():
        if preds:
            last_pred = preds[-1]
            prob_val = last_pred.get('prob', None)
            try:
                if prob_val is None or (isinstance(prob_val, float) and pd.isna(prob_val)):
                    prob_str = "-"
                else:
                    prob_str = f"{prob_val:.3f}"
            except Exception:
                prob_str = "-"
            msg += (
                f"\n[Driver={d}] Pit prob: {prob_str} | "
                f"Stint: {last_pred.get('StintRaw', '-')}, "
                f"Lap: {last_pred.get('LapNumberRaw', '-')}, "
                f"Sector: {last_pred.get('SectorNumberRaw', '-')}, "
                f"Pred: {last_pred.get('pred', '-')}"
            )
    msg += "\n"

    # --- ID-BASED PRUNING LOGIC ---
    for driver in stored["sector"]:
        # Get row_ids that have NOT been aggregated
        keep_sector_ids = set(row['row_id'] for row in stored["sector"][driver]) - set(stored["sector_agg_id"].get(driver, []))
        stored["sector"][driver] = [row for row in stored["sector"][driver] if row['row_id'] in keep_sector_ids]
        keep_telemetry_ids = set(row['row_id'] for row in stored["telemetry"][driver]) - set(stored["telemetry_agg_id"].get(driver, []))
        stored["telemetry"][driver] = [row for row in stored["telemetry"][driver] if row['row_id'] in keep_telemetry_ids]

        # Add hard limits on stored raw data
        if len(stored["sector"][driver]) > 2:
            stored["sector"][driver] = stored["sector"][driver][-2:]
        if len(stored["telemetry"][driver]) > 500:
            stored["telemetry"][driver] = stored["telemetry"][driver][-500:]

    if race_ended:
        msg += "\n\nRace has finished. No more data will be returned."
        return stored, msg, True  # disables the interval
    else:
        return stored, msg, False


def parse_time_as_float(col):
    # Handles either float seconds or "0 days ..." string
    if col.dtype == object:
        try:
            return pd.to_timedelta(col).dt.total_seconds()
        except Exception:
            return col.astype(float)
    return col.astype(float)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8050)