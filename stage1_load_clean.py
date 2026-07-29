"""
stage1_load_clean.py
=====================
Loads the raw Kaggle CSV and cleans it: fixes the text-encoded target,
exposes missing values hidden as the literal string "NaN", coerces
mis-typed numeric columns, and repairs corrupted GPS coordinates.

Input:  data/train.csv
Output: data/train_cleaned.csv

Run standalone:  python3 stage1_load_clean.py
"""

import numpy as np
import pandas as pd

from common import RAW_PATH, CLEANED_PATH, LATE_THRESHOLD_MIN, DISTANCE_JUMP_CUTOFF_KM, haversine_km


def run():
    df = pd.read_csv(RAW_PATH)
    n_start = len(df)
    print(f"[Stage 1] Loaded {n_start} rows, {df.shape[1]} columns from {RAW_PATH}")

    # Target column arrives as text like "(min) 24" -- extract the integer.
    df["Time_taken_min"] = df["Time_taken(min)"].astype(str).str.extract(r"(\d+)")[0].astype(int)

    # "Weatherconditions" arrives with a "conditions " prefix glued onto every value.
    df["Weatherconditions"] = (
        df["Weatherconditions"].astype(str).str.strip().str.replace("conditions ", "", regex=False)
    )

    # Several category columns hide real missingness as the LITERAL STRING "NaN" (not a true NaN),
    # which pandas' .isna() cannot see until it's replaced -- fix that first, then strip whitespace
    # everywhere else (IDs, vehicle/order type, etc. all carry stray leading/trailing spaces).
    nan_string_counts = {}
    for col in ["Road_traffic_density", "Weatherconditions", "Festival", "City", "multiple_deliveries"]:
        stripped = df[col].astype(str).str.strip()
        nan_string_counts[col] = int((stripped == "NaN").sum())
        df[col] = stripped.replace("NaN", np.nan)
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].astype(str).str.strip()

    print("[Stage 1] Literal-string 'NaN' values exposed as real missing values:")
    for col, n in nan_string_counts.items():
        print(f"           {col:<22}: {n}")

    # Age, Ratings, and multiple_deliveries are meant to be numeric but load as text because of the
    # embedded "NaN" strings above -- coerce explicitly rather than trusting pandas' inferred dtype.
    df["Delivery_person_Age"] = pd.to_numeric(df["Delivery_person_Age"], errors="coerce")
    df["Delivery_person_Ratings"] = pd.to_numeric(df["Delivery_person_Ratings"], errors="coerce")
    df["multiple_deliveries"] = pd.to_numeric(df["multiple_deliveries"], errors="coerce")

    # Impute remaining missing values: median for numeric columns (robust to outliers), mode for
    # categorical columns. We impute rather than drop rows, since missingness here is a data-quality
    # artifact (~1-4% per column), not a meaningful signal worth losing ~9% of the dataset over.
    missing_cols = df.columns[df.isna().any()]
    print(f"[Stage 1] Imputing {len(missing_cols)} columns with remaining missing values:")
    for col in missing_cols:
        n_missing = int(df[col].isna().sum())
        if pd.api.types.is_numeric_dtype(df[col]):
            fill_val = df[col].median()
            df[col] = df[col].fillna(fill_val)
            print(f"           {col:<26}: {n_missing:>5} rows -> median {fill_val:.2f}")
        else:
            fill_val = df[col].mode(dropna=True)[0]
            df[col] = df[col].fillna(fill_val)
            print(f"           {col:<26}: {n_missing:>5} rows -> mode {fill_val!r}")

    # Flag and fix impossible values:
    #  - GPS corruption shows up two ways: (a) a huge haversine distance from sign-flipped coordinates,
    #    and (b) an exact (0,0) placeholder pair that can coincidentally produce a small, plausible-
    #    looking distance and so isn't caught by the distance check alone.
    #  - Ratings outside the valid 1-5 range are data-entry errors.
    # All are treated as missing, then median-imputed (preserves every row for this tree model).
    df["distance_km"] = haversine_km(
        df["Restaurant_latitude"], df["Restaurant_longitude"],
        df["Delivery_location_latitude"], df["Delivery_location_longitude"],
    )
    near_zero_gps = (
        (df["Restaurant_latitude"].abs() < 0.1) & (df["Restaurant_longitude"].abs() < 0.1)
    ) | (
        (df["Delivery_location_latitude"].abs() < 0.1) & (df["Delivery_location_longitude"].abs() < 0.1)
    )
    bad_distance = (df["distance_km"] > DISTANCE_JUMP_CUTOFF_KM) | near_zero_gps
    df.loc[bad_distance, "distance_km"] = np.nan
    df["distance_km"] = df["distance_km"].fillna(df["distance_km"].median())

    bad_rating = (df["Delivery_person_Ratings"] < 1) | (df["Delivery_person_Ratings"] > 5)
    df.loc[bad_rating, "Delivery_person_Ratings"] = np.nan
    df["Delivery_person_Ratings"] = df["Delivery_person_Ratings"].fillna(df["Delivery_person_Ratings"].median())

    print(f"[Stage 1] Corrupted GPS rows fixed: {int(bad_distance.sum())}, "
          f"out-of-range ratings fixed: {int(bad_rating.sum())}. Row count unchanged: {n_start} -> {len(df)}")

    # Define the target: an order is "late" if it took more than LATE_THRESHOLD_MIN minutes.
    df["is_late"] = (df["Time_taken_min"] > LATE_THRESHOLD_MIN).astype(int)
    print(f"[Stage 1] Class split (threshold = {LATE_THRESHOLD_MIN} min): "
          f"{(1 - df['is_late'].mean()) * 100:.1f}% on-time, {df['is_late'].mean() * 100:.1f}% late")

    df.to_csv(CLEANED_PATH, index=False)
    print(f"[Stage 1] Cleaned dataset saved to {CLEANED_PATH}  shape={df.shape}")
    return df


if __name__ == "__main__":
    run()