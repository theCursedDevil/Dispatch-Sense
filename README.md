# Dispatch Sense

Predicting whether a food delivery order will arrive more than 30 minutes
after it was placed, using the Kaggle `gauravmalik26/food-delivery-dataset`.

This is the **multi-file pipeline** version of the project: each stage is
its own script, reading the previous stage's saved output from disk and
writing its own, instead of one long script.

## Problem
Late deliveries hurt customer trust and cost support time to handle after
the fact. This project predicts lateness risk *before* an order is out for
delivery, so a dispatch team could intervene early instead of reacting
after a complaint.

## Dataset
45,593 historical delivery records, covering rider details (age, rating),
restaurant and delivery GPS coordinates, order/pickup timestamps, weather,
road traffic, festival flag, vehicle type and condition, and the actual
delivery time in minutes. "Late" is defined as any order taking more than
30 minutes. Place `train.csv` in `data/` before running anything.

## Pipeline Stages

| File | Reads | Writes |
|---|---|---|
| `common.py` | — | shared constants, paths, the haversine distance helper |
| `stage1_load_clean.py` | `data/train.csv` | `data/train_cleaned.csv` |
| `stage2_feature_engineering.py` | `train_cleaned.csv` | `data/train_processed.csv` |
| `stage3_train.py` | `train_processed.csv` | both models, the scaler, and the exact train/test split |
| `stage4_evaluate.py` | `train_processed.csv` + saved split + models | evaluation figures, final (possibly calibrated) model |
| `stage5_interpret.py` | `train_processed.csv` + saved split + XGBoost model | feature importance figure |
| `run_all.py` | — | runs all five stages above in order |

Each `stageN_*.py` file can be run on its own as long as the stage before
it has already run and produced its output file. `stage3_train.py` saves
the exact train/test split to disk specifically so stages 4 and 5 evaluate
on identically the same held-out rows, rather than depending on every
stage re-deriving the same split independently.

## Key Results
- **93.5% test accuracy**, with XGBoost clearly outperforming the logistic
  regression baseline (84.9%).
- **82.8% recall** on the late class — catches roughly 5 out of 6 truly
  late orders (468 missed out of 2,721 actually-late test orders).
- **Brier score of 0.046** — the model's predicted probabilities are
  well-calibrated; a stated "75% chance of being late" is trustworthy.

## Limitations
- Trained on a handful of Indian cities and a narrow set of vehicle types —
  unlikely to generalize to new regions, rural delivery, or extreme weather
  the data never saw.
- No feature captures one-off events (a wrong turn, an unusually slow
  kitchen night) — these will always be a blind spot.
- ~9% of rows had corrupted GPS coordinates that were median-imputed rather
  than recovered, which slightly understates true distance variation.

## How to Run

Install dependencies:
```bash
pip3 install -r requirements.txt
```

Place `train.csv` in `data/`, then either run everything at once:
```bash
python3 run_all.py
```
or run each stage one at a time, in order:
```bash
python3 stage1_load_clean.py
python3 stage2_feature_engineering.py
python3 stage3_train.py
python3 stage4_evaluate.py
python3 stage5_interpret.py
```

### Troubleshooting (macOS)
If you see `ModuleNotFoundError: No module named 'xgboost'`, the package
isn't installed in the Python environment you're running from — install
everything from `requirements.txt` as shown above (use `pip3 install --user
-r requirements.txt` if you hit a permissions error).

If you see an error mentioning `libomp.dylib could not be loaded`, XGBoost
needs Apple's OpenMP runtime, which pip can't install on its own:
```bash
brew install libomp
```
then re-run your script. If you don't have Homebrew yet, install it from
https://brew.sh first.
