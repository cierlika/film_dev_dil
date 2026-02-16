"""
train_extratrees.py
===================
Flexible training pipeline for film-development-time regression.

Supports multiple regression algorithms, optional imputation and feature
scaling, and Optuna-based hyperparameter optimisation.

Algorithm choices (set ALGORITHM below):
    "extratrees"       – ExtraTreesRegressor           (default)
    "random_forest"    – RandomForestRegressor
    "lightgbm"         – LightGBM LGBMRegressor
    "gradient_boosting"– sklearn GradientBoostingRegressor
    "ridge"            – Ridge (linear; uses scaling + imputation)
    "lasso"            – Lasso (linear; uses scaling + imputation)
    "svr"              – SVR   (linear; uses scaling + imputation)

Optuna:
    Set N_OPTUNA_TRIALS > 0 to run hyperparameter search over OPTUNA_FOLDS
    cross-validation folds before the final evaluation run.

Feature pipeline order:
    1.  FilmSlopeEstimator    → film_slope, dev_dil_slope, box_iso, pred_log_ratio
    2.  LatentValueBuilder    → lv_pred_time, lv_film_bias, lv_dev_bias, lv_global_mean
    3.  AggregateFeatureBuilder → film/developer aggregate stats
    4.  FilmMetadataBuilder   → film_manufacturer_code, film_family_code, box_iso_category_code
    5.  DeveloperMetadataBuilder → developer_type_code
    6.  InteractionFeatureBuilder → 17 arithmetic interaction features
    7.  [Optional] SimpleImputer → fill remaining NaN
    8.  [Optional] StandardScaler → z-score normalisation
"""

from __future__ import annotations

import json
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ── Project-local builders ────────────────────────────────────────────────────
from data_split import prepare_data, DEFAULT_CSV, SPLITS_PATH
from film_metadata import FilmMetadataBuilder
from developer_metadata import DeveloperMetadataBuilder
from interaction_features import InteractionFeatureBuilder

# These three builders live in your existing modules:
from film_slope import FilmSlopeEstimator          # → film_slope, dev_dil_slope, box_iso
from latent_value import LatentValueBuilder         # → lv_pred_time, lv_*
from aggregate_features import AggregateFeatureBuilder  # → film_agg_*, dd_agg_*

# ══════════════════════════════════════════════════════════════════════════════
# ── CONFIGURATION ─────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

# Algorithm selection
ALGORITHM: str = "extratrees"   # see module docstring for options

# Target transformation – log target aligns the variance-reduction objective
# with SMAPE (which penalises on a percentage basis).  Highly recommended.
USE_LOG_TARGET: bool = True

# Optuna – set to 0 to skip hyperparameter search and use FIXED_PARAMS
N_OPTUNA_TRIALS: int = 60
OPTUNA_FOLDS: int    = 2     # number of CV folds used during search (speed vs quality)
OPTUNA_TIMEOUT: int  = 300   # max seconds for the search (None = no limit)

# Fixed parameters used when N_OPTUNA_TRIALS == 0.
# Keys must match the algorithm's sklearn/LightGBM constructor.
FIXED_PARAMS: dict[str, Any] = {
    "n_estimators": 500,
    "max_features": 0.6,
    "min_samples_leaf": 3,
    "n_jobs": -1,
    "random_state": 42,
}

# Random seed
RANDOM_STATE: int = 42

# Columns to drop BEFORE building model features.
# 'Temp' is excluded because prepare_data() parses it into temp_celsius.
# 'Time' is the regression target.
DROP_COLS: list[str] = [
    "Film", "Developer", "Dilution", "Temp",
    "ISO", "EI",
    "Time",           # target – always excluded from X
]

# Data paths
CSV_PATH:    Path = DEFAULT_CSV
SPLITS_FILE: Path = SPLITS_PATH


# ══════════════════════════════════════════════════════════════════════════════
# ── Algorithm registry ────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

# Which algorithms need NaN imputation and feature scaling before fitting
_NEEDS_IMPUTATION: set[str] = {"ridge", "lasso", "svr"}
_NEEDS_SCALING:    set[str] = {"ridge", "lasso", "svr"}

# LightGBM is imported lazily so the project works without it installed
_LIGHTGBM_AVAILABLE = False
try:
    from lightgbm import LGBMRegressor
    _LIGHTGBM_AVAILABLE = True
except ImportError:
    pass


def _make_base_model(algorithm: str, params: dict[str, Any]):
    """Instantiate the raw (unwrapped) regressor for the given algorithm."""
    alg = algorithm.lower()
    p = params.copy()

    if alg == "extratrees":
        p.setdefault("n_jobs", -1)
        p.setdefault("random_state", RANDOM_STATE)
        return ExtraTreesRegressor(**p)

    if alg == "random_forest":
        p.setdefault("n_jobs", -1)
        p.setdefault("random_state", RANDOM_STATE)
        return RandomForestRegressor(**p)

    if alg == "lightgbm":
        if not _LIGHTGBM_AVAILABLE:
            raise ImportError("LightGBM is not installed. Run: pip install lightgbm")
        p.setdefault("n_jobs", -1)
        p.setdefault("random_state", RANDOM_STATE)
        p.setdefault("verbose", -1)
        return LGBMRegressor(**p)

    if alg == "gradient_boosting":
        p.setdefault("random_state", RANDOM_STATE)
        return GradientBoostingRegressor(**p)

    if alg == "ridge":
        return Ridge(**{k: v for k, v in p.items() if k in ("alpha", "fit_intercept")})

    if alg == "lasso":
        p.setdefault("max_iter", 5000)
        return Lasso(**{k: v for k, v in p.items() if k in ("alpha", "fit_intercept", "max_iter")})

    if alg == "svr":
        return SVR(**{k: v for k, v in p.items() if k in ("C", "epsilon", "kernel", "gamma")})

    raise ValueError(f"Unknown algorithm: {algorithm!r}. "
                     "Choose from: extratrees, random_forest, lightgbm, "
                     "gradient_boosting, ridge, lasso, svr")


def _make_pipeline(algorithm: str, params: dict[str, Any]) -> Pipeline:
    """
    Build a sklearn Pipeline that optionally adds imputation and scaling
    before the base estimator.
    """
    steps: list[tuple[str, Any]] = []
    alg = algorithm.lower()

    if alg in _NEEDS_IMPUTATION:
        steps.append(("imputer", SimpleImputer(strategy="median")))

    if alg in _NEEDS_SCALING:
        steps.append(("scaler", StandardScaler()))

    steps.append(("model", _make_base_model(algorithm, params)))
    return Pipeline(steps)


# ══════════════════════════════════════════════════════════════════════════════
# ── Metrics ───────────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Symmetric Mean Absolute Percentage Error (%)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom  = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    with np.errstate(divide="ignore", invalid="ignore"):
        ratios = np.where(denom == 0, 0.0, np.abs(y_true - y_pred) / denom)
    return float(np.mean(ratios) * 100)


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


# ══════════════════════════════════════════════════════════════════════════════
# ── Feature pipeline ──────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def build_features(
    df_train: pd.DataFrame,
    df_test:  pd.DataFrame,
    y_train:  pd.Series,
    algorithm: str = ALGORITHM,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run all feature builders in sequence and return (X_train, X_test).

    All builders that require training-data statistics (slopes, latent values,
    aggregates) are fitted on df_train only and then applied to both frames,
    ensuring no data leakage.

    Metadata and interaction builders are purely deterministic / arithmetic.

    The returned DataFrames contain only numeric columns; DROP_COLS are excluded.
    Missing values are filled downstream inside the sklearn Pipeline when the
    algorithm requires it; tree-based algorithms receive them as-is.
    """
    # ── 1. Film slope estimator ───────────────────────────────────────────────
    slope_est = FilmSlopeEstimator(random_state=RANDOM_STATE)
    slope_est.fit(df_train, y_train)
    df_train_s = slope_est.transform(df_train)
    df_test_s  = slope_est.transform(df_test)

    # ── 2. Latent value builder (ALS-style matrix factorisation) ──────────────
    lv_builder = LatentValueBuilder(random_state=RANDOM_STATE)
    lv_builder.fit(df_train_s, y_train)
    lv_train = lv_builder.transform(df_train_s)
    lv_test  = lv_builder.transform(df_test_s)

    # ── 3. Aggregate feature builder ──────────────────────────────────────────
    agg_builder = AggregateFeatureBuilder()
    agg_builder.fit(df_train_s, y_train)
    agg_train = agg_builder.transform(df_train_s)
    agg_test  = agg_builder.transform(df_test_s)

    # ── 4 & 5. Static metadata builders (no fit needed) ───────────────────────
    film_meta_builder = FilmMetadataBuilder()
    dev_meta_builder  = DeveloperMetadataBuilder()

    film_meta_train = film_meta_builder.transform(df_train_s)
    film_meta_test  = film_meta_builder.transform(df_test_s)

    dev_meta_train  = dev_meta_builder.transform(df_train_s)
    dev_meta_test   = dev_meta_builder.transform(df_test_s)

    # ── First concatenation: all non-interaction features ─────────────────────
    def _base_cols(df: pd.DataFrame) -> pd.DataFrame:
        """Drop raw/string/target columns and return numeric-only frame."""
        drop = [c for c in DROP_COLS if c in df.columns]
        numeric_df = df.drop(columns=drop, errors="ignore")
        return numeric_df.select_dtypes(include=[np.number])

    full_train = pd.concat(
        [_base_cols(df_train_s), lv_train, agg_train, film_meta_train, dev_meta_train],
        axis=1,
    )
    full_test = pd.concat(
        [_base_cols(df_test_s), lv_test, agg_test, film_meta_test, dev_meta_test],
        axis=1,
    )

    # ── 6. Interaction features (operate on the full concatenated frame) ───────
    ixn_builder = InteractionFeatureBuilder()
    ixn_train = ixn_builder.transform(full_train)
    ixn_test  = ixn_builder.transform(full_test)

    full_train = pd.concat([full_train, ixn_train], axis=1)
    full_test  = pd.concat([full_test,  ixn_test],  axis=1)

    # ── Remove duplicate columns (safety; shouldn't normally occur) ────────────
    full_train = full_train.loc[:, ~full_train.columns.duplicated()]
    full_test  = full_test.loc[:,  ~full_test.columns.duplicated()]

    return full_train, full_test


# ══════════════════════════════════════════════════════════════════════════════
# ── Single fold evaluation ────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_fold(
    df:        pd.DataFrame,
    fold:      dict[str, list[int]],
    params:    dict[str, Any],
    algorithm: str = ALGORITHM,
    use_log_target: bool = USE_LOG_TARGET,
) -> dict[str, float]:
    """
    Train on fold["train"], evaluate on fold["val"] and fold["test"].
    Returns a dict with SMAPE and MAE for val and test splits.
    """
    train_idx = fold["train"]
    val_idx   = fold["val"]
    test_idx  = fold["test"]

    df_train = df.iloc[train_idx].copy()
    df_val   = df.iloc[val_idx].copy()
    df_test  = df.iloc[test_idx].copy()

    y_train_raw = df_train["Time"].values.astype(float)
    y_val_raw   = df_val["Time"].values.astype(float)
    y_test_raw  = df_test["Time"].values.astype(float)

    y_train = np.log(y_train_raw) if use_log_target else y_train_raw

    X_train, X_val_full = build_features(df_train, df_val,   pd.Series(y_train_raw, name="Time"), algorithm)
    _,       X_test_full = build_features(df_train, df_test, pd.Series(y_train_raw, name="Time"), algorithm)

    # Align column sets (test might differ if very few combos)
    X_val  = X_val_full.reindex(columns=X_train.columns, fill_value=np.nan)
    X_test = X_test_full.reindex(columns=X_train.columns, fill_value=np.nan)

    pipe = _make_pipeline(algorithm, params)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pipe.fit(X_train.values, y_train)

    def predict_minutes(pipe, X: pd.DataFrame) -> np.ndarray:
        preds = pipe.predict(X.values)
        if use_log_target:
            preds = np.exp(preds)
        return np.clip(preds, 0.1, None)   # development time can't be < 0

    val_preds  = predict_minutes(pipe, X_val)
    test_preds = predict_minutes(pipe, X_test)

    return {
        "val_smape":  smape(y_val_raw,  val_preds),
        "val_mae":    mae(y_val_raw,    val_preds),
        "test_smape": smape(y_test_raw, test_preds),
        "test_mae":   mae(y_test_raw,   test_preds),
    }


# ══════════════════════════════════════════════════════════════════════════════
# ── Optuna hyperparameter search ──────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def _suggest_params(trial: optuna.Trial, algorithm: str) -> dict[str, Any]:
    """Map algorithm name → Optuna parameter suggestions."""
    alg = algorithm.lower()

    if alg in ("extratrees", "random_forest"):
        return {
            "n_estimators":    trial.suggest_int("n_estimators",    100, 1000, step=100),
            "max_features":    trial.suggest_float("max_features",  0.2, 1.0),
            "min_samples_leaf":trial.suggest_int("min_samples_leaf",1, 20),
            "max_depth":       trial.suggest_categorical("max_depth", [None, 10, 20, 30, 50]),
            "n_jobs": -1,
            "random_state": RANDOM_STATE,
        }

    if alg == "lightgbm":
        return {
            "n_estimators":      trial.suggest_int("n_estimators",     100, 2000, step=100),
            "num_leaves":        trial.suggest_int("num_leaves",        20,  300),
            "learning_rate":     trial.suggest_float("learning_rate",   0.01, 0.3, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 5,   100),
            "subsample":         trial.suggest_float("subsample",        0.5, 1.0),
            "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "reg_alpha":         trial.suggest_float("reg_alpha",        1e-8, 10.0, log=True),
            "reg_lambda":        trial.suggest_float("reg_lambda",       1e-8, 10.0, log=True),
            "n_jobs": -1,
            "random_state": RANDOM_STATE,
            "verbose": -1,
        }

    if alg == "gradient_boosting":
        return {
            "n_estimators":    trial.suggest_int("n_estimators",    100, 800, step=50),
            "learning_rate":   trial.suggest_float("learning_rate",  0.01, 0.3, log=True),
            "max_depth":       trial.suggest_int("max_depth",         2,   8),
            "min_samples_leaf":trial.suggest_int("min_samples_leaf",  3,  30),
            "subsample":       trial.suggest_float("subsample",        0.5, 1.0),
            "max_features":    trial.suggest_float("max_features",     0.3, 1.0),
            "random_state": RANDOM_STATE,
        }

    if alg == "ridge":
        return {"alpha": trial.suggest_float("alpha", 1e-4, 1e4, log=True)}

    if alg == "lasso":
        return {
            "alpha":    trial.suggest_float("alpha", 1e-4, 10.0, log=True),
            "max_iter": 10000,
        }

    if alg == "svr":
        return {
            "C":       trial.suggest_float("C",       1e-2, 1e4, log=True),
            "epsilon": trial.suggest_float("epsilon", 1e-3, 1.0, log=True),
            "kernel":  trial.suggest_categorical("kernel", ["rbf", "linear"]),
        }

    raise ValueError(f"No Optuna suggestions defined for algorithm: {algorithm!r}")


def run_optuna(
    df:         pd.DataFrame,
    splits:     list[dict[str, list[int]]],
    algorithm:  str = ALGORITHM,
    n_trials:   int = N_OPTUNA_TRIALS,
    n_folds:    int = OPTUNA_FOLDS,
    timeout:    int | None = OPTUNA_TIMEOUT,
    use_log_target: bool = USE_LOG_TARGET,
) -> dict[str, Any]:
    """
    Run an Optuna study using the first *n_folds* cross-validation folds as
    the evaluation surface.  Returns the best hyperparameters found.
    """
    search_folds = splits[:n_folds]

    def objective(trial: optuna.Trial) -> float:
        params = _suggest_params(trial, algorithm)
        scores = []
        for fold in search_folds:
            try:
                result = evaluate_fold(df, fold, params, algorithm, use_log_target)
                scores.append(result["val_smape"])
            except Exception as e:
                # Prune bad parameter combinations rather than crashing
                raise optuna.exceptions.TrialPruned() from e
        return float(np.mean(scores))

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
    )

    t0 = time.time()
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=True)
    elapsed = time.time() - t0

    best = study.best_params
    print(f"\n[Optuna] {len(study.trials)} trials in {elapsed:.0f}s – "
          f"best val SMAPE: {study.best_value:.2f}%")
    print(f"[Optuna] Best params: {best}")
    return best


# ══════════════════════════════════════════════════════════════════════════════
# ── Feature importance ────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def _extract_feature_importance(
    pipe: Pipeline,
    feature_names: list[str],
    algorithm: str,
    top_n: int = 20,
) -> pd.DataFrame | None:
    """Return a sorted feature importance DataFrame, or None if not available."""
    model = pipe.named_steps["model"]
    alg   = algorithm.lower()

    if hasattr(model, "feature_importances_"):
        imp = model.feature_importances_
    elif alg in ("ridge", "lasso") and hasattr(model, "coef_"):
        imp = np.abs(model.coef_)
    else:
        return None

    fi = (
        pd.DataFrame({"feature": feature_names, "importance": imp})
        .sort_values("importance", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )
    return fi


# ══════════════════════════════════════════════════════════════════════════════
# ── Full evaluation run ────────────────────────────────════════════════════════
# ══════════════════════════════════════════════════════════════════════════════

def full_cv_run(
    df:      pd.DataFrame,
    splits:  list[dict[str, list[int]]],
    params:  dict[str, Any],
    algorithm: str = ALGORITHM,
    use_log_target: bool = USE_LOG_TARGET,
) -> dict[str, Any]:
    """
    Train and evaluate across all folds.  Prints per-fold metrics and returns
    a summary dict with mean ± std for val and test SMAPE / MAE.
    Also trains one final model on the last fold and reports feature importances.
    """
    fold_results = []
    last_pipe  = None
    last_feat_names = None

    print(f"\n{'─'*60}")
    print(f"Algorithm : {algorithm}   Log-target : {use_log_target}")
    print(f"{'─'*60}")
    header = f"{'Fold':>4}  {'Val SMAPE':>10}  {'Val MAE':>8}  {'Test SMAPE':>10}  {'Test MAE':>8}"
    print(header)
    print("─" * len(header))

    for fold in splits:
        fold_idx = fold["fold"]
        t0 = time.time()

        train_idx = fold["train"]
        val_idx   = fold["val"]
        test_idx  = fold["test"]

        df_train = df.iloc[train_idx].copy()
        df_val   = df.iloc[val_idx].copy()
        df_test  = df.iloc[test_idx].copy()

        y_train_raw = df_train["Time"].values.astype(float)
        y_val_raw   = df_val["Time"].values.astype(float)
        y_test_raw  = df_test["Time"].values.astype(float)

        y_train = np.log(y_train_raw) if use_log_target else y_train_raw

        X_train, X_val_full = build_features(
            df_train, df_val, pd.Series(y_train_raw, name="Time"), algorithm
        )
        _, X_test_full = build_features(
            df_train, df_test, pd.Series(y_train_raw, name="Time"), algorithm
        )

        X_val  = X_val_full.reindex(columns=X_train.columns,  fill_value=np.nan)
        X_test = X_test_full.reindex(columns=X_train.columns, fill_value=np.nan)

        pipe = _make_pipeline(algorithm, params)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pipe.fit(X_train.values, y_train)

        def predict_minutes(X: pd.DataFrame) -> np.ndarray:
            preds = pipe.predict(X.values)
            if use_log_target:
                preds = np.exp(preds)
            return np.clip(preds, 0.1, None)

        val_smape  = smape(y_val_raw,  predict_minutes(X_val))
        val_mae    = mae(y_val_raw,    predict_minutes(X_val))
        test_smape = smape(y_test_raw, predict_minutes(X_test))
        test_mae   = mae(y_test_raw,   predict_minutes(X_test))
        elapsed    = time.time() - t0

        print(
            f"  {fold_idx:2d}  "
            f"{val_smape:10.2f}%  "
            f"{val_mae:8.3f}  "
            f"{test_smape:10.2f}%  "
            f"{test_mae:8.3f}  "
            f"({elapsed:.1f}s)"
        )

        fold_results.append({
            "fold": fold_idx,
            "val_smape": val_smape,
            "val_mae": val_mae,
            "test_smape": test_smape,
            "test_mae": test_mae,
        })

        last_pipe        = pipe
        last_feat_names  = list(X_train.columns)

    print("─" * len(header))
    test_smapes = [r["test_smape"] for r in fold_results]
    test_maes   = [r["test_mae"]   for r in fold_results]
    val_smapes  = [r["val_smape"]  for r in fold_results]

    summary = {
        "algorithm":       algorithm,
        "use_log_target":  use_log_target,
        "params":          params,
        "n_features":      len(last_feat_names) if last_feat_names else 0,
        "val_smape_mean":  float(np.mean(val_smapes)),
        "val_smape_std":   float(np.std(val_smapes)),
        "test_smape_mean": float(np.mean(test_smapes)),
        "test_smape_std":  float(np.std(test_smapes)),
        "test_mae_mean":   float(np.mean(test_maes)),
        "test_mae_std":    float(np.std(test_maes)),
        "fold_results":    fold_results,
    }

    print(
        f"\n  Val  SMAPE : {summary['val_smape_mean']:.2f}% ± {summary['val_smape_std']:.2f}%\n"
        f"  Test SMAPE : {summary['test_smape_mean']:.2f}% ± {summary['test_smape_std']:.2f}%\n"
        f"  Test MAE   : {summary['test_mae_mean']:.3f} min ± {summary['test_mae_std']:.3f} min\n"
        f"  # features : {summary['n_features']}"
    )

    # ── Feature importance (last fold model) ───────────────────────────────────
    if last_pipe and last_feat_names:
        fi = _extract_feature_importance(last_pipe, last_feat_names, algorithm)
        if fi is not None:
            print(f"\n  Top features (fold {splits[-1]['fold']}):")
            for _, row in fi.head(15).iterrows():
                bar = "█" * int(row["importance"] / fi["importance"].max() * 30)
                print(f"    {row['feature']:<35} {bar}")

    return summary


# ══════════════════════════════════════════════════════════════════════════════
# ── Entry point ───────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Train film development time regressor")
    parser.add_argument("--csv",       default=str(CSV_PATH),    help="Path to raw CSV")
    parser.add_argument("--splits",    default=str(SPLITS_FILE), help="Path to splits.json")
    parser.add_argument("--algorithm", default=ALGORITHM,
                        choices=["extratrees", "random_forest", "lightgbm",
                                 "gradient_boosting", "ridge", "lasso", "svr"],
                        help="Regression algorithm to use")
    parser.add_argument("--n-trials",  type=int, default=N_OPTUNA_TRIALS,
                        help="Number of Optuna trials (0 = use FIXED_PARAMS)")
    parser.add_argument("--optuna-folds", type=int, default=OPTUNA_FOLDS,
                        help="Number of CV folds used for Optuna search")
    parser.add_argument("--no-log-target", action="store_true",
                        help="Train on raw target instead of log(target)")
    parser.add_argument("--results-out", default="results.json",
                        help="Path to write results JSON")
    args = parser.parse_args()

    algorithm      = args.algorithm
    n_trials       = args.n_trials
    use_log_target = not args.no_log_target

    # ── Load data ─────────────────────────────────────────────────────────────
    df = prepare_data(csv_path=args.csv, temp_filter=True)

    # ── Load splits ────────────────────────────────────────────────────────────
    splits_path = Path(args.splits)
    if not splits_path.exists():
        raise FileNotFoundError(
            f"{splits_path} not found. Run `python data_split.py` first."
        )
    with open(splits_path) as f:
        splits = json.load(f)

    print(f"\n[train] algorithm={algorithm!r}  n_folds={len(splits)}  "
          f"n_rows={len(df)}  log_target={use_log_target}")

    # ── Hyperparameter optimisation ────────────────────────────────────────────
    if n_trials > 0:
        print(f"\n[Optuna] Starting search: {n_trials} trials, "
              f"{args.optuna_folds} folds for evaluation\n")
        best_params = run_optuna(
            df         = df,
            splits     = splits,
            algorithm  = algorithm,
            n_trials   = n_trials,
            n_folds    = args.optuna_folds,
            timeout    = OPTUNA_TIMEOUT,
            use_log_target = use_log_target,
        )
    else:
        print("\n[train] Skipping Optuna search – using FIXED_PARAMS")
        best_params = FIXED_PARAMS.copy()

    # ── Full cross-validated evaluation with best params ───────────────────────
    summary = full_cv_run(
        df            = df,
        splits        = splits,
        params        = best_params,
        algorithm     = algorithm,
        use_log_target= use_log_target,
    )

    # ── Save results ───────────────────────────────────────────────────────────
    out_path = Path(args.results_out)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[train] Results written to {out_path}")


if __name__ == "__main__":
    main()
