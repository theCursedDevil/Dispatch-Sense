"""
stage4_evaluate.py
===================
Loads the trained models and the saved train/test split, then runs the
full evaluation: accuracy/precision/recall/F1/AUC comparison, confusion
matrix, 5-fold cross-validation, ROC + Precision-Recall curves, and a
calibration check (with Platt scaling attempted and kept only if it
measurably improves the Brier score).

Input:  data/train_processed.csv, outputs/train_test_split.joblib,
        outputs/logistic_regression_model.joblib, outputs/scaler.joblib,
        outputs/xgboost_model.joblib
Output: outputs/xgboost_full_evaluation.png
        outputs/calibration_curve.png
        outputs/xgboost_final.joblib (or xgboost_final_calibrated.joblib)

Run standalone:  python3 stage4_evaluate.py
"""

import os
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, roc_curve, precision_recall_curve, brier_score_loss,
)
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from xgboost import XGBClassifier

from common import PROCESSED_PATH, OUTPUTS_DIR, SPLIT_PATH, FEATURE_COLS, RANDOM_STATE


def evaluate_metrics(y_true, y_pred, y_proba):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1": f1_score(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_proba),
    }


def run():
    df = pd.read_csv(PROCESSED_PATH)
    split = joblib.load(SPLIT_PATH)
    X_train, X_test, y_train, y_test = split["X_train"], split["X_test"], split["y_train"], split["y_test"]
    X, y = df[FEATURE_COLS], df["is_late"]

    log_reg = joblib.load(os.path.join(OUTPUTS_DIR, "logistic_regression_model.joblib"))
    scaler = joblib.load(os.path.join(OUTPUTS_DIR, "scaler.joblib"))
    xgb_model = joblib.load(os.path.join(OUTPUTS_DIR, "xgboost_model.joblib"))
    print(f"[Stage 4] Loaded processed data {df.shape}, the saved train/test split, and both models.")

    X_test_scaled = scaler.transform(X_test)
    log_proba = log_reg.predict_proba(X_test_scaled)[:, 1]
    log_pred = log_reg.predict(X_test_scaled)
    xgb_proba = xgb_model.predict_proba(X_test)[:, 1]
    xgb_pred = xgb_model.predict(X_test)

    results = pd.DataFrame({
        "Logistic Regression": evaluate_metrics(y_test, log_pred, log_proba),
        "XGBoost": evaluate_metrics(y_test, xgb_pred, xgb_proba),
    }).T
    print("\n[Stage 4] MODEL COMPARISON\n" + results.round(4).to_string())

    cm_xgb = confusion_matrix(y_test, xgb_pred)
    print("\n[Stage 4] XGBoost confusion matrix (rows=actual, cols=predicted, [on-time, late]):")
    print(cm_xgb)

    # 5-fold stratified cross-validation, to confirm the single train/test split wasn't a lucky draw.
    cv_model = XGBClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8, random_state=RANDOM_STATE, eval_metric="logloss",
    )
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_val_score(cv_model, X, y, cv=skf, scoring="f1")
    print(f"\n[Stage 4] 5-fold CV F1: mean={cv_scores.mean():.4f}, std={cv_scores.std():.4f}")

    # ROC + Precision-Recall curves, plus the confusion matrix, in one figure.
    fpr, tpr, _ = roc_curve(y_test, xgb_proba)
    roc_auc = roc_auc_score(y_test, xgb_proba)
    precision_vals, recall_vals, _ = precision_recall_curve(y_test, xgb_proba)

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    im = axes[0].imshow(cm_xgb, cmap="Blues")
    axes[0].set_title("XGBoost Confusion Matrix")
    axes[0].set_xticks([0, 1]); axes[0].set_yticks([0, 1])
    axes[0].set_xticklabels(["Pred: On-time", "Pred: Late"])
    axes[0].set_yticklabels(["Actual: On-time", "Actual: Late"])
    for i in range(2):
        for j in range(2):
            axes[0].text(j, i, cm_xgb[i, j], ha="center", va="center",
                         color="white" if cm_xgb[i, j] > cm_xgb.max() / 2 else "black", fontsize=14)
    plt.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)

    axes[1].plot(fpr, tpr, color="#4C72B0", linewidth=2, label=f"XGBoost (AUC = {roc_auc:.4f})")
    axes[1].plot([0, 1], [0, 1], color="gray", linestyle="--", label="Random guess")
    axes[1].set_xlabel("False Positive Rate"); axes[1].set_ylabel("True Positive Rate")
    axes[1].set_title("ROC Curve"); axes[1].legend(loc="lower right")

    axes[2].plot(recall_vals, precision_vals, color="#C44E52", linewidth=2)
    axes[2].axhline(y.mean(), color="gray", linestyle="--", label=f"Baseline (= {y.mean():.3f} late rate)")
    axes[2].set_xlabel("Recall"); axes[2].set_ylabel("Precision")
    axes[2].set_title("Precision-Recall Curve"); axes[2].legend(loc="lower left")
    plt.tight_layout()
    eval_fig_path = os.path.join(OUTPUTS_DIR, "xgboost_full_evaluation.png")
    plt.savefig(eval_fig_path, dpi=150)
    print(f"[Stage 4] Confusion matrix / ROC / PR figure saved to {eval_fig_path}")

    # Calibration: are XGBoost's predicted probabilities trustworthy at face value?
    log_frac_pos, log_mean_pred = calibration_curve(y_test, log_proba, n_bins=10)
    xgb_frac_pos, xgb_mean_pred = calibration_curve(y_test, xgb_proba, n_bins=10)
    xgb_brier_before = brier_score_loss(y_test, xgb_proba)
    xgb_calibration_gap = np.mean(np.abs(xgb_mean_pred - xgb_frac_pos))

    plt.figure(figsize=(7, 7))
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")
    plt.plot(log_mean_pred, log_frac_pos, marker="o", color="#4C72B0", label="Logistic Regression")
    plt.plot(xgb_mean_pred, xgb_frac_pos, marker="o", color="#C44E52", label="XGBoost")
    plt.xlabel("Mean predicted probability"); plt.ylabel("Fraction of actual positives")
    plt.title("Calibration Curve"); plt.legend(loc="upper left")
    plt.tight_layout()
    calib_fig_path = os.path.join(OUTPUTS_DIR, "calibration_curve.png")
    plt.savefig(calib_fig_path, dpi=150)
    print(f"[Stage 4] Calibration curve saved to {calib_fig_path}")

    # Apply Platt scaling only if needed, and only keep it if it actually measures better --
    # never ship a "fix" that performs worse than the original just because it was attempted.
    final_model, final_model_name, final_brier = xgb_model, "xgboost_final.joblib", xgb_brier_before
    if xgb_calibration_gap > 0.02:
        calibrated_xgb = CalibratedClassifierCV(xgb_model, method="sigmoid", cv=5)
        calibrated_xgb.fit(X_train, y_train)
        xgb_brier_after = brier_score_loss(y_test, calibrated_xgb.predict_proba(X_test)[:, 1])
        if xgb_brier_after < xgb_brier_before:
            final_model, final_model_name, final_brier = calibrated_xgb, "xgboost_final_calibrated.joblib", xgb_brier_after
        print(f"[Stage 4] Platt scaling attempted: Brier {xgb_brier_before:.4f} -> {xgb_brier_after:.4f} "
              f"({'kept' if xgb_brier_after < xgb_brier_before else 'discarded, original was better'})")

    joblib.dump(final_model, os.path.join(OUTPUTS_DIR, final_model_name))
    print(f"[Stage 4] Brier score (Logistic Regression): {brier_score_loss(y_test, log_proba):.4f}")
    print(f"[Stage 4] Brier score (XGBoost, final, saved as {final_model_name}): {final_brier:.4f}")
    return results, cm_xgb, cv_scores


if __name__ == "__main__":
    run()
