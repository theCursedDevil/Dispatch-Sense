"""
stage2_feature_engineering.py
==============================
Builds the model-ready features on top of the cleaned dataset: distance,
time-of-day/weekend/peak flags, kitchen-prep time (with midnight rollover
handled), and label-encoded categoricals.

Input:  data/train_cleaned.csv  (produced by stage1_load_clean.py)
Output: data/train_processed.csv

Run standalone:  python3 stage2_feature_engineering.py
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from common import CLEANED_PATH, PROCESSED_PATH, DISTANCE_JUMP_CUTOFF_KM, haversine_km


def run():
    df = pd.read_csv(CLEANED_PATH)
    print(f"[Stage 2] Loaded cleaned dataset from {CLEANED_PATH}  shape={df.shape}")

    # dist_km: rebuilt fresh from raw coordinates (kept distinct from the cleaning-step distance_km),
    # with the same two GPS-corruption fixes re-applied (recomputed here since this stage runs as its
    # own process and doesn't have stage 1's in-memory mask available).
    df["dist_km"] = haversine_km(
        df["Restaurant_latitude"], df["Restaurant_longitude"],
        df["Delivery_location_latitude"], df["Delivery_location_longitude"],
    )
    near_zero_gps = (
        (df["Restaurant_latitude"].abs() < 0.1) & (df["Restaurant_longitude"].abs() < 0.1)
    ) | (
        (df["Delivery_location_latitude"].abs() < 0.1) & (df["Delivery_location_longitude"].abs() < 0.1)
    )
    bad_distance = (df["dist_km"] > DISTANCE_JUMP_CUTOFF_KM) | near_zero_gps
    df.loc[bad_distance, "dist_km"] = np.nan
    df["dist_km"] = df["dist_km"].fillna(df["dist_km"].median())
    print(f"[Stage 2] dist_km built. Mean: {df['dist_km'].mean():.2f} km "
          f"(corrupted rows re-imputed: {int(bad_distance.sum())})")

    # hour_of_order / day_of_week: Time_Orderd hides real missingness as a literal "NaN" string,
    # same quirk as the category columns in Stage 1 -- convert before parsing as a timedelta.
    order_dt = pd.to_datetime(df["Order_Date"], format="%d-%m-%Y")
    time_ordered_raw = df["Time_Orderd"].astype(str).str.strip().replace("NaN", np.nan)
    time_ordered = pd.to_timedelta(time_ordered_raw)
    hour_of_order = time_ordered.dt.components["hours"]
    n_missing_hour = int(hour_of_order.isna().sum())
    df["hour_of_order"] = hour_of_order.fillna(hour_of_order.mode(dropna=True)[0]).astype(int)
    df["day_of_week"] = order_dt.dt.day_name()
    print(f"[Stage 2] hour_of_order / day_of_week extracted "
          f"({n_missing_hour} rows had a missing order time, filled with the mode hour)")

    # is_peak / is_weekend: simple derived flags from the time features above.
    df["is_peak"] = df["hour_of_order"].isin({12, 13, 14, 19, 20, 21}).astype(int)
    df["is_weekend"] = df["day_of_week"].isin(["Saturday", "Sunday"]).astype(int)
    print(f"[Stage 2] is_peak rate: {df['is_peak'].mean()*100:.1f}%   is_weekend rate: {df['is_weekend'].mean()*100:.1f}%")

    # prep_time_min: minutes between order placed and picked up, with midnight-rollover handled.
    # Computed from the ORIGINAL (un-imputed) time_ordered so a missing order time produces a missing
    # prep_time_min instead of a nonsense gap from a mismatched imputed time; the resulting MINUTES
    # (not the raw time) are then median-imputed.
    time_picked = pd.to_timedelta(df["Time_Order_picked"])
    prep_time_min = (time_picked - time_ordered).dt.total_seconds() / 60
    n_rollover = int((prep_time_min < 0).sum())
    prep_time_min = prep_time_min.where(prep_time_min.isna() | (prep_time_min >= 0), prep_time_min + 24 * 60)
    df["prep_time_min"] = prep_time_min.fillna(prep_time_min.median())
    print(f"[Stage 2] prep_time_min built (midnight-rollover fixed for {n_rollover} rows). "
          f"Range: {df['prep_time_min'].min():.1f}-{df['prep_time_min'].max():.1f} min")

    # Label-encode multi-class categoricals (tree models split on thresholds, so encoding order
    # doesn't imply false ordinal meaning the way it would for a linear model); map the binary
    # Festival flag directly to 0/1.
    for col in ["Weatherconditions", "Road_traffic_density", "Type_of_vehicle", "Type_of_order", "City"]:
        df[col + "_enc"] = LabelEncoder().fit_transform(df[col])
    df["Festival_enc"] = (df["Festival"] == "Yes").astype(int)
    print("[Stage 2] Label-encoded: Weatherconditions, Road_traffic_density, Type_of_vehicle, "
          "Type_of_order, City (+ Festival mapped to 0/1)")

    df.to_csv(PROCESSED_PATH, index=False)
    print(f"[Stage 2] Processed dataset saved to {PROCESSED_PATH}  shape={df.shape}")
    return df


if __name__ == "__main__":
    run()
