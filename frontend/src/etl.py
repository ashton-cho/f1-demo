import pandas as pd
import numpy as np


def incremental_aggregate(
    sector_df, telemetry_df, weather_df,
    sector_agg_id_set, telemetry_agg_id_set, current_time
):
    """
    Efficiently aggregate only new sector rows (not already in `sector_agg_id_set`)
    whose SectorSessionTime_End <= current_time, and telemetry rows not in `telemetry_agg_id_set`.
    Returns: (agg_df, updated_sector_agg_id_set, updated_telemetry_agg_id_set)
    """
    # Only consider sectors that have finished and not been aggregated
    finished_mask = sector_df['SectorSessionTime_End'] <= current_time
    not_agg_mask = ~sector_df['row_id'].isin(sector_agg_id_set)
    ready_sectors = sector_df[finished_mask & not_agg_mask]

    if ready_sectors.empty:
        return pd.DataFrame(), set(sector_agg_id_set), set(telemetry_agg_id_set)

    aggs = []
    new_sector_agg_id_set = set(sector_agg_id_set)
    new_telemetry_agg_id_set = set(telemetry_agg_id_set)

    # Columns to copy from sector
    cols_to_copy = [
        'EventName', 'Team', 'Driver', 'Stint', 'LapNumber', 'SectorNumber',
        'Compound', 'TyreLife', 'SectorTime', 'SectorSessionTime_Start', 'SectorSessionTime_End'
    ]
    duplicate_cols = ['Stint', 'LapNumber', 'SectorNumber']

    for idx, sec in ready_sectors.iterrows():
        # After masking for sector interval
        tel_time_mask = (
            (telemetry_df['Driver'] == sec['Driver']) &
            (telemetry_df['LapNumber'] == sec['LapNumber']) &
            (telemetry_df['SessionTime'] >= sec['SectorSessionTime_Start']) &
            (telemetry_df['SessionTime'] < sec['SectorSessionTime_End'])
        )
        tdf = telemetry_df[tel_time_mask]
        try:
            # Copy sector info, handling missing columns
            agg_row = {col: sec[col] if col in sec else None for col in cols_to_copy}
            # Duplicate columns as 'XRaw'
            for col in duplicate_cols:
                agg_row[f"{col}Raw"] = sec[col] if col in sec else None

            # Find telemetry in this sector interval for this driver/lap and not yet aggregated
            tmask = (
                (telemetry_df['Driver'] == sec.get('Driver', None)) &
                (telemetry_df['LapNumber'] == sec.get('LapNumber', None)) &
                (telemetry_df['SessionTime'] >= sec.get('SectorSessionTime_Start', pd.NaT)) &
                (telemetry_df['SessionTime'] < sec.get('SectorSessionTime_End', pd.NaT))
            )
            tdf = telemetry_df[tmask]
            # Use only telemetry rows not already aggregated for this driver
            tdf = tdf.loc[~tdf['row_id'].isin(telemetry_agg_id_set)]
            telemetry_ids = tdf['row_id'].tolist()

            if tdf.empty:
                new_sector_agg_id_set.add(sec['row_id'])
                continue  # Don't process if no telemetry

            # Aggregation logic (edit as needed for your features)
            agg_row['Speed_P10'] = np.percentile(tdf['Speed'], 10) if 'Speed' in tdf else np.nan
            agg_row['Throttle_Median'] = tdf['Throttle'].median() if 'Throttle' in tdf else np.nan
            agg_row['Throttle_ZeroPct'] = (tdf['Throttle'] == 0).mean() if 'Throttle' in tdf else np.nan
            agg_row['Gear_Range'] = tdf['nGear'].max() - tdf['nGear'].min() if 'nGear' in tdf else np.nan
            agg_row['DRS_ActivePct'] = tdf['DRS'].isin([10, 12, 14]).mean() if 'DRS' in tdf else np.nan
            agg_row['TrackStatus_Mean'] = tdf['TrackStatus'].mean() if 'TrackStatus' in tdf else np.nan
            agg_row['TrackStatus_Mode'] = tdf['TrackStatus'].mode().iloc[0] if 'TrackStatus' in tdf and not tdf['TrackStatus'].mode().empty else np.nan

            # Weather: merge_asof on sector start (backward)
            wx = weather_df[weather_df['SessionTime'] <= sec.get('SectorSessionTime_Start', pd.NaT)]
            if not wx.empty:
                wx_row = wx.iloc[-1].to_dict()
                for col, val in wx_row.items():
                    agg_row[col] = val

            aggs.append((sec['row_id'], agg_row))
            new_sector_agg_id_set.add(sec['row_id'])
            new_telemetry_agg_id_set.update(telemetry_ids)
        except Exception as e:
            print(f"Error during aggregation at index {idx}, driver {sec.get('Driver', None)}: {e}")
            raise

    if not aggs:
        return pd.DataFrame(), new_sector_agg_id_set, new_telemetry_agg_id_set
    
    print(f"[Aggregation] Aggregated {len(aggs)} sectors. "
      f"New sector agg ids: {len(new_sector_agg_id_set - set(sector_agg_id_set))}, "
      f"New telemetry agg ids: {len(new_telemetry_agg_id_set - set(telemetry_agg_id_set))}")

    agg_df = pd.DataFrame([row for row_id, row in aggs])

    return agg_df, new_sector_agg_id_set, new_telemetry_agg_id_set

def encode(X_df, categorical_cols, encoder):
    """
    Encode categorical columns using provided label encoders.
    Unknown categories get -1.
    """
    # X_df = X_df.copy()
    for col in categorical_cols:
        if col in encoder:
            classes = encoder[col].classes_
            class_to_int = {k: i for i, k in enumerate(classes)}
            X_df[col] = X_df[col].astype(str).map(class_to_int).fillna(-1).astype(int)
    return X_df

def scale(X_df, numeric_cols, scaler_dict, event_name):
    """
    Standardize numeric features per event using provided scaler dict.
    Unseen events use the global scaler if available.
    """
    scaler = scaler_dict.get(event_name, scaler_dict.get('__global__'))
    # X_scaled = X_df.copy()
    X_df[numeric_cols] = scaler.transform(X_df[numeric_cols])
    return X_df


#####


def is_consecutive_sector_window(buffer):
    if len(buffer) < 15:
        return False
    # Extract ordered (lap, sector) pairs
    pairs = [(row['LapNumberRaw'], row['SectorNumberRaw']) for row in buffer]
    # Define the valid transitions: (lap, sector) -> (next_lap, next_sector)
    def next_pair(lap, sector):
        if sector == 3:
            return (lap + 1, 1)
        else:
            return (lap, sector + 1)
    for i in range(len(pairs) - 1):
        if pairs[i + 1] != next_pair(*pairs[i]):
            return False
    return True

def prepare_window_for_prediction(buffer, feature_cols, categorical_cols, numeric_cols):
    """
    Given a 15-sector buffer (list of pre-encoded/scaled dicts), returns X_num, X_cat for prediction.
    No encoding or scaling is performed here.
    """
    window_df = pd.DataFrame(buffer)
    X = window_df[feature_cols].values
    X = X[np.newaxis, ...]  # Add batch dimension

    # Figure out categorical and numeric indices
    cat_idx = [feature_cols.index(col) for col in categorical_cols]
    num_idx = [feature_cols.index(col) for col in numeric_cols]

    X_cat = [X[..., idx].astype(np.int32) for idx in cat_idx]
    X_num = X[..., num_idx].astype(np.float32)
    return X_cat, X_num


#####


def predict(model, X_num, X_cat):
    """
    Predict probabilities for a dataset.
    Returns prediction probabilities and binarized predictions.
    """
    y_pred_prob = model.predict([X_num] + X_cat, verbose=0)
    y_pred_bin = (y_pred_prob > 0.5).astype(int).flatten()
    return y_pred_prob.flatten(), y_pred_bin