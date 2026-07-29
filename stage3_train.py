"""
stage3_train.py
================
Splits the processed dataset (80/20, stratified) and trains both models:
a scaled Logistic Regression baseline and the main XGBoost classifier.
Saves both models, the scaler, and the exact train/test split (so later
stages evaluate on identically the same held-out rows).

Input:  data/train_processed.csv  (produced by stage2_feature_engineering.py)
Output: outputs/logistic_regression_model.joblib
        outputs/scaler.joblib
        outputs/xgboost_model.joblib
        outputs/train_test_split.joblib

Run standalone:  python3 stage3_train.py
"""

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from common import PROCESSED_PATH, OUTPUTS_DIR, SPLIT_PATH, FEATURE_COLS, RANDOM_STATE
import os


def run():
    df = pd.read_csv(PROCESSED_PATH)
    print(f"[Stage 3] Loaded processed dataset from {PROCESSED_PATH}  shape={df.shape}")

    X = df[FEATURE_COLS]
    y = df["is_late"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    print(f"[Stage 3] Train/test split: {len(X_train)} train rows ({y_train.mean()*100:.1f}% late), "
          f"{len(X_test)} test rows ({y_test.mean()*100:.1f}% late)")

    # Persist the exact split so stage 4 and stage 5 evaluate on identically these rows,
    # rather than relying on train_test_split() being re-called identically in another process.
    joblib.dump(
        {"X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test},
        SPLIT_PATH,
    )
    print(f"[Stage 3] Train/test split saved to {SPLIT_PATH}")

    # Model 1: Logistic Regression (baseline). Features are scaled since its coefficients are
    # sensitive to feature scale -- unscaled features with larger ranges would dominate the fit.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    log_reg = LogisticRegression(random_state=RANDOM_STATE, max_iter=1000)
    log_reg.fit(X_train_scaled, y_train)
    print("[Stage 3] Logistic Regression trained.")

    # Model 2: XGBoost (main model).
    xgb_model = XGBClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8, random_state=RANDOM_STATE, eval_metric="logloss",
    )
    xgb_model.fit(X_train, y_train)
    print("[Stage 3] XGBoost trained.")

    joblib.dump(log_reg, os.path.join(OUTPUTS_DIR, "logistic_regression_model.joblib"))
    joblib.dump(scaler, os.path.join(OUTPUTS_DIR, "scaler.joblib"))
    joblib.dump(xgb_model, os.path.join(OUTPUTS_DIR, "xgboost_model.joblib"))
    print(f"[Stage 3] Models saved to {OUTPUTS_DIR}")
    return log_reg, scaler, xgb_model


if __name__ == "__main__":
    run()
