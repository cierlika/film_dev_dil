"""
Train ExtraTrees Baseline
=========================

Trains an ExtraTreesRegressor using the full feature pipeline:
    1. Load data + frozen splits
    2. Per fold: fit slopes, MF latent vectors, aggregate stats on train
    3. Transform train/val/test
    4. Train ExtraTrees, evaluate on val + test

Usage:
    python train_extratrees.py
"""

import time
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor

from data_split import DataSplitter, prepare_data
from film_slopes import (
    FilmSlopeEstimator,
    DevDilSlopeEstimator,
    DevDilutionSlopeEstimator,
)
from latent_features import LatentFeatureBuilder
from aggregate_features import AggregateFeatureBuilder


# ══════════════════════════════════════════════════════════════════════════════
# Feature pipeline
# ══════════════════════════════════════════════════════════════════════════════

# Columns to drop before training (metadata, text, target)
DROP_COLS = [
    "Film", "Developer", "Dilution", "ASA/ISO",
    "120", "Sheet", "Temp", "Notes",
    "dev_dil", "lv_available",
]
TARGET = "35mm"


def build_features(df_train, df_eval):
    """
    Fit all feature builders on df_train, transform both df_train and df_eval.

    Returns (X_train, y_train, X_eval, y_eval, feature_names)
    """
    # 1. Slopes
    film_est = FilmSlopeEstimator().fit(df_train)
    devdil_est = DevDilSlopeEstimator(box_isos=film_est.box_isos).fit(df_train)
    dil_est = DevDilutionSlopeEstimator().fit(df_train)

    def _apply_slopes(d):
        return dil_est.transform(devdil_est.transform(film_est.transform(d)))

    df_train_s = _apply_slopes(df_train)
    df_eval_s = _apply_slopes(df_eval)

    # 2. Latent features
    lv = LatentFeatureBuilder(n_factors=30, reg=0.005)
    lv_train = lv.fit_transform(df_train)
    lv_eval = lv.transform(df_eval)

    # 3. Aggregate features
    agg = AggregateFeatureBuilder()
    agg_train = agg.fit_transform(df_train)
    agg_eval = agg.transform(df_eval)

    # 4. Combine
    full_train = pd.concat([df_train_s, lv_train, agg_train], axis=1)
    full_eval = pd.concat([df_eval_s, lv_eval, agg_eval], axis=1)

    # 5. Select numeric features only, drop metadata
    drop = [c for c in DROP_COLS if c in full_train.columns] + [TARGET]
    feature_cols = [
        c for c in full_train.columns
        if c not in drop and full_train[c].dtype in [np.float64, np.float32, np.int64, np.int32, np.bool_]
    ]

    X_train = full_train[feature_cols].values.astype(np.float32)
    y_train = full_train[TARGET].values.astype(np.float32)
    X_eval = full_eval[feature_cols].values.astype(np.float32)
    y_eval = full_eval[TARGET].values.astype(np.float32)

    # Replace inf with NaN (tree models handle NaN natively)
    X_train[~np.isfinite(X_train)] = np.nan
    X_eval[~np.isfinite(X_eval)] = np.nan

    return X_train, y_train, X_eval, y_eval, feature_cols


def smape(actual, predicted):
    return np.mean(2 * np.abs(actual - predicted) / (np.abs(actual) + np.abs(predicted))) * 100


def evaluate(y_true, y_pred, label=""):
    s = smape(y_true, y_pred)
    mae = np.mean(np.abs(y_true - y_pred))
    mape = np.mean(np.abs(y_true - y_pred) / y_true) * 100
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - ss_res / ss_tot
    print(f"  {label:12s}  SMAPE={s:6.2f}%  MAE={mae:5.2f} min  MAPE={mape:6.2f}%  R²={r2:.4f}")
    return {"smape": s, "mae": mae, "mape": mape, "r2": r2}


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    df = prepare_data()
    splitter = DataSplitter.load("splits.json")

    print(f"\nDataset: {len(df)} rows")
    print(f"Test set: {len(splitter.test_indices)} rows")
    print(f"Train+val: {len(splitter.get_trainval_indices())} rows, {splitter.n_folds} folds")

    # ── Cross-validation ──
    print(f"\n{'='*80}")
    print("5-Fold Cross-Validation — ExtraTreesRegressor")
    print(f"{'='*80}")

    fold_results = []

    for fold in range(splitter.n_folds):
        train_idx, val_idx = splitter.get_fold(fold)
        df_train = df.loc[train_idx]
        df_val = df.loc[val_idx]

        t0 = time.time()
        X_train, y_train, X_val, y_val, feature_names = build_features(df_train, df_val)
        t_feat = time.time() - t0

        # Train ExtraTrees
        t0 = time.time()
        et = ExtraTreesRegressor(
            n_estimators=500,
            max_features=0.5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        )
        et.fit(X_train, y_train)
        t_train = time.time() - t0

        y_pred_train = et.predict(X_train)
        y_pred_val = et.predict(X_val)

        print(f"\nFold {fold}  (features: {t_feat:.1f}s, train: {t_train:.1f}s, "
              f"{X_train.shape[1]} features):")
        train_res = evaluate(y_train, y_pred_train, "train")
        val_res = evaluate(y_val, y_pred_val, "val")
        fold_results.append(val_res)

    # ── CV Summary ──
    print(f"\n{'='*80}")
    print("CV Summary")
    print(f"{'='*80}")
    for metric in ["smape", "mae", "mape", "r2"]:
        vals = [r[metric] for r in fold_results]
        print(f"  {metric:8s}  mean={np.mean(vals):.4f}  std={np.std(vals):.4f}  "
              f"min={np.min(vals):.4f}  max={np.max(vals):.4f}")

    # ── Final: train on all trainval, evaluate on test ──
    print(f"\n{'='*80}")
    print("Final Model — Train on all trainval, evaluate on test")
    print(f"{'='*80}")

    trainval_idx = splitter.get_trainval_indices()
    test_idx = splitter.test_indices

    df_trainval = df.loc[trainval_idx]
    df_test = df.loc[test_idx]

    X_trainval, y_trainval, X_test, y_test, feature_names = build_features(
        df_trainval, df_test
    )

    et_final = ExtraTreesRegressor(
        n_estimators=500,
        max_features=0.5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )
    et_final.fit(X_trainval, y_trainval)

    y_pred_trainval = et_final.predict(X_trainval)
    y_pred_test = et_final.predict(X_test)

    print()
    evaluate(y_trainval, y_pred_trainval, "trainval")
    evaluate(y_test, y_pred_test, "TEST")

    # ── Feature importance ──
    print(f"\n{'='*80}")
    print("Top 30 Feature Importances")
    print(f"{'='*80}")
    imp = pd.Series(et_final.feature_importances_, index=feature_names)
    imp = imp.sort_values(ascending=False)
    for i, (feat, val) in enumerate(imp.head(30).items()):
        print(f"  {i+1:2d}. {feat:45s}  {val:.4f}  ({val*100:.1f}%)")
