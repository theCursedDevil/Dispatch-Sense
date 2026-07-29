"""
stage5_interpret.py
====================
Computes feature importance three ways for the XGBoost model: built-in
gain importance, permutation importance on the test set, and a SHAP
beeswarm summary (installed on the fly if missing, skipped cleanly if
installation fails).

Input:  data/train_processed.csv, outputs/train_test_split.joblib,
        outputs/xgboost_model.joblib
Output: outputs/feature_importance.png

Run standalone:  python3 stage5_interpret.py
"""

import os
import subprocess
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.inspection import permutation_importance

from common import PROCESSED_PATH, OUTPUTS_DIR, SPLIT_PATH, FEATURE_COLS, RANDOM_STATE


def run():
    df = pd.read_csv(PROCESSED_PATH)
    split = joblib.load(SPLIT_PATH)
    X_test, y_test = split["X_test"], split["y_test"]
    xgb_model = joblib.load(os.path.join(OUTPUTS_DIR, "xgboost_model.joblib"))
    print(f"[Stage 5] Loaded processed data {df.shape}, the saved test split, and the XGBoost model.")

    TOP_N = 12

    # --- Built-in gain importance ---
    gain_importance = xgb_model.get_booster().get_score(importance_type="gain")
    gain_series = pd.Series(gain_importance).reindex(FEATURE_COLS).fillna(0).sort_values(ascending=False)
    gain_top = gain_series.head(TOP_N)
    print(f"\n[Stage 5] XGBoost built-in gain importance (top {TOP_N}):")
    print(gain_top.round(2).to_string())

    # --- Permutation importance on the test set (model-agnostic, test-set grounded) ---
    perm_result = permutation_importance(
        xgb_model, X_test, y_test, n_repeats=10, random_state=RANDOM_STATE, scoring="f1", n_jobs=-1
    )
    perm_series = pd.Series(perm_result.importances_mean, index=FEATURE_COLS)
    perm_std_series = pd.Series(perm_result.importances_std, index=FEATURE_COLS)
    perm_order = perm_series.sort_values(ascending=False).head(TOP_N).index
    print(f"\n[Stage 5] Top 5 features by permutation importance (most trustworthy ranking):")
    print(perm_series.sort_values(ascending=False).head(5).round(4).to_string())

    # --- SHAP beeswarm (only if available; install on the fly, skip cleanly if it fails) ---
    shap_available = False
    try:
        import shap
        shap_available = True
    except ImportError:
        print("\n[Stage 5] shap not found -- attempting `pip install shap` ...")
        install = subprocess.run(
            ["pip", "install", "shap", "--break-system-packages", "--quiet"], capture_output=True, text=True
        )
        if install.returncode == 0:
            try:
                import shap
                shap_available = True
                print("[Stage 5] shap installed successfully.")
            except ImportError:
                print("[Stage 5] shap installed but import still failed -- skipping SHAP panel.")
        else:
            print("[Stage 5] shap could not be installed -- skipping SHAP panel.")

    if shap_available:
        shap_values = shap.TreeExplainer(xgb_model).shap_values(X_test)

    n_panels = 3 if shap_available else 2
    fig, axes = plt.subplots(1, n_panels, figsize=(8 * n_panels, 7))

    gain_plot = gain_top.sort_values()
    axes[0].barh(gain_plot.index, gain_plot.values, color="#4C72B0")
    axes[0].set_xlabel("Gain importance"); axes[0].set_title(f"XGBoost Gain Importance (top {TOP_N})")

    perm_plot = perm_series[perm_order].sort_values()
    axes[1].barh(perm_plot.index, perm_plot.values, xerr=perm_std_series[perm_plot.index].values,
                 color="#C44E52", ecolor="black", capsize=3)
    axes[1].set_xlabel("Mean F1 decrease when shuffled"); axes[1].set_title(f"Permutation Importance (top {TOP_N})")

    if shap_available:
        plt.sca(axes[2])
        shap.summary_plot(shap_values, X_test, show=False, plot_size=None)
        axes[2].set_title("SHAP Summary (Beeswarm)")

    plt.tight_layout()
    fig_path = os.path.join(OUTPUTS_DIR, "feature_importance.png")
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"\n[Stage 5] Feature importance figure saved to {fig_path}")
    return gain_series, perm_series


if __name__ == "__main__":
    run()
