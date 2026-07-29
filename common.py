"""
common.py
=========
Shared constants, file paths, and helper functions used by every stage
script in this pipeline. Each stageN_*.py file imports from here so the
cleaning thresholds, feature list, and random seed stay identical across
the whole pipeline (rather than risking drift if each stage redefined them).
"""

import os
import numpy as np

RANDOM_STATE = 42
LATE_THRESHOLD_MIN = 30        # an order counts as "late" past this many minutes
DISTANCE_JUMP_CUTOFF_KM = 100  # above this, a GPS pair is treated as corrupted, not a real trip

# ---------------------------------------------------------------------------
# Paths -- every stage reads/writes through these so the on-disk handoff
# between stages (CSV in, CSV/joblib out) stays consistent.
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)

RAW_PATH = os.path.join(DATA_DIR, "train.csv")
CLEANED_PATH = os.path.join(DATA_DIR, "train_cleaned.csv")
PROCESSED_PATH = os.path.join(DATA_DIR, "train_processed.csv")
SPLIT_PATH = os.path.join(OUTPUTS_DIR, "train_test_split.joblib")

# Feature set: numeric + encoded categoricals only. Excluded on purpose: Time_taken(min) /
# Time_taken_min / distance_km (these ARE or duplicate the target -> leakage), ID columns (not
# predictive), and raw date/time/text columns (already converted into the features below).
FEATURE_COLS = [
    "Delivery_person_Age", "Delivery_person_Ratings", "Vehicle_condition",
    "multiple_deliveries", "dist_km", "hour_of_order", "is_peak", "is_weekend",
    "prep_time_min", "Festival_enc", "Weatherconditions_enc", "Road_traffic_density_enc",
    "Type_of_vehicle_enc", "Type_of_order_enc", "City_enc",
]


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km between two (lat, lon) points."""
    R = 6371
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))
