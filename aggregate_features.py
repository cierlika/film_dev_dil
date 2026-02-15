"""
Aggregate Feature Builder
=========================

Computes per-entity aggregate statistics from development time data.
All features are computed on training data only and then mapped to
any dataset (train/val/test) to avoid leakage.

Entity levels:
    - Per Film         (e.g., Ilford HP5+)
    - Per Film @ ISO   (e.g., Ilford HP5+ @ 800)
    - Per dev_dil      (e.g., HC-110 B)
    - Per Developer    (e.g., HC-110, across all dilutions)

Features per entity:
    - count (popularity)
    - median, mean, std
    - q25, q75, iqr
    - skewness, kurtosis
    - min, max, range

Additional row-level features:
    - stops (push/pull from box ISO)
    - log_dil_factor (numeric dilution)
    - is_box_iso (binary: shooting at box speed)

Usage:
    from aggregate_features import AggregateFeatureBuilder

    agg = AggregateFeatureBuilder()
    df_train = agg.fit_transform(df_train)
    df_val   = agg.transform(df_val)
    df_test  = agg.transform(df_test)
"""

from __future__ import annotations

import re
import warnings
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as sp_stats


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

_HC110_CODES: dict[str, str] = {
    "A": "1+15", "B": "1+31", "C": "1+19", "D": "1+39",
    "E": "1+47", "F": "1+79", "G": "1+119", "H": "1+63", "J": "1+150",
}


def _parse_dilution_factor(dilution: str) -> float | None:
    """Parse dilution string → numeric factor. stock=1, 1+50=51, etc."""
    d = str(dilution).strip()
    if d.lower() in ("stock", "nan", "", "none"):
        return 1.0
    if d.upper() in _HC110_CODES:
        d = _HC110_CODES[d.upper()]
    m = re.match(r"^1\+(\d+(?:\.\d+)?)$", d)
    if m:
        return 1.0 + float(m.group(1))
    m = re.match(r"^1\+(\d+)\+(\d+)$", d)
    if m:
        return 1.0 + float(m.group(1)) + float(m.group(2))
    return None


def _compute_group_stats(times: pd.Series) -> dict[str, float]:
    """Compute aggregate statistics for a group of development times."""
    vals = times.dropna().values
    n = len(vals)
    if n == 0:
        return {}

    result = {
        "count": float(n),
        "median": float(np.median(vals)),
        "mean": float(np.mean(vals)),
    }

    if n >= 2:
        result["std"] = float(np.std(vals, ddof=1))
        result["q25"] = float(np.percentile(vals, 25))
        result["q75"] = float(np.percentile(vals, 75))
        result["iqr"] = result["q75"] - result["q25"]
        result["min"] = float(np.min(vals))
        result["max"] = float(np.max(vals))
        result["range"] = result["max"] - result["min"]
    else:
        result["std"] = 0.0
        result["q25"] = result["median"]
        result["q75"] = result["median"]
        result["iqr"] = 0.0
        result["min"] = result["median"]
        result["max"] = result["median"]
        result["range"] = 0.0

    if n >= 3:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            result["skew"] = float(sp_stats.skew(vals, bias=False))
    else:
        result["skew"] = 0.0

    if n >= 4:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            result["kurtosis"] = float(sp_stats.kurtosis(vals, bias=False))
    else:
        result["kurtosis"] = 0.0

    return result


# ══════════════════════════════════════════════════════════════════════════════
# Entity statistics builder
# ══════════════════════════════════════════════════════════════════════════════

class _EntityStatsBuilder:
    """
    Compute and store aggregate statistics for a single entity level.

    Parameters
    ----------
    entity_name : str
        Prefix for generated column names (e.g., "film", "dev_dil").
    """

    def __init__(self, entity_name: str):
        self.entity_name = entity_name
        self.stats: dict[str, dict[str, float]] = {}

    def fit(self, keys: pd.Series, times: pd.Series) -> "_EntityStatsBuilder":
        """Compute stats per unique key from training data."""
        self.stats = {}
        df_tmp = pd.DataFrame({"key": keys.values, "time": times.values})
        for key, grp in df_tmp.groupby("key"):
            s = _compute_group_stats(grp["time"])
            if s:
                self.stats[key] = s
        return self

    def transform(self, keys: pd.Series) -> pd.DataFrame:
        """Map stored stats to rows by key. Returns DataFrame with same index."""
        stat_names = [
            "count", "median", "mean", "std",
            "q25", "q75", "iqr",
            "skew", "kurtosis",
            "min", "max", "range",
        ]
        prefix = self.entity_name
        result = pd.DataFrame(index=keys.index)

        for stat in stat_names:
            mapping = {k: v.get(stat, np.nan) for k, v in self.stats.items()}
            result[f"{prefix}_{stat}"] = keys.map(mapping)

        return result

    @property
    def n_entities(self) -> int:
        return len(self.stats)


# ══════════════════════════════════════════════════════════════════════════════
# Main builder
# ══════════════════════════════════════════════════════════════════════════════

class AggregateFeatureBuilder:
    """
    Compute per-entity aggregate features from training data.

    Produces features at four entity levels:
        - film_agg_*      : per Film
        - film_iso_agg_*  : per Film @ ISO
        - dd_agg_*        : per dev_dil (Developer + Dilution)
        - dev_agg_*       : per Developer (across all dilutions)

    Plus row-level features:
        - dil_factor     : raw numeric dilution factor
        - is_box_iso     : 1 if shooting at box speed, 0 otherwise

    Note: `stops` and `log_dil_factor` are NOT produced here as they
    come from prepare_data() and the slope estimators respectively.

    Parameters
    ----------
    box_iso_tolerance : float
        Stops tolerance for considering an ISO as "box speed" (default 0.05).
    """

    def __init__(self, box_iso_tolerance: float = 0.05):
        self.box_iso_tolerance = box_iso_tolerance

        # Entity builders
        self._film_stats = _EntityStatsBuilder("film_agg")
        self._film_iso_stats = _EntityStatsBuilder("film_iso_agg")
        self._dd_stats = _EntityStatsBuilder("dd_agg")
        self._dev_stats = _EntityStatsBuilder("dev_agg")

        self._fitted = False

    def fit(self, df_train: pd.DataFrame) -> "AggregateFeatureBuilder":
        """
        Compute all entity-level statistics from training data.

        Parameters
        ----------
        df_train : DataFrame
            Must contain: Film, Developer, Dilution, dev_dil, iso, box_iso, 35mm
        """
        times = df_train["35mm"]

        # Per Film
        self._film_stats.fit(df_train["Film"], times)

        # Per Film @ ISO
        film_iso_keys = df_train["Film"] + " @ " + df_train["iso"].astype(int).astype(str)
        self._film_iso_stats.fit(film_iso_keys, times)

        # Per dev_dil
        self._dd_stats.fit(df_train["dev_dil"], times)

        # Per Developer
        self._dev_stats.fit(df_train["Developer"].astype(str).str.strip(), times)

        self._fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Map entity stats + row-level features to any dataset.

        Returns a new DataFrame containing ONLY the generated feature
        columns (same index as input df). Caller can pd.concat with
        original df as needed.
        """
        if not self._fitted:
            raise RuntimeError("Call .fit() first")

        parts = []

        # ── Entity-level aggregates ──
        parts.append(self._film_stats.transform(df["Film"]))

        film_iso_keys = df["Film"] + " @ " + df["iso"].astype(int).astype(str)
        parts.append(self._film_iso_stats.transform(film_iso_keys))

        parts.append(self._dd_stats.transform(df["dev_dil"]))

        dev_keys = df["Developer"].astype(str).str.strip()
        parts.append(self._dev_stats.transform(dev_keys))

        # ── Row-level features ──
        row_feats = pd.DataFrame(index=df.index)

        # Dilution factor (raw numeric)
        dil_factors = df["Dilution"].apply(_parse_dilution_factor)
        row_feats["dil_factor"] = dil_factors

        # Box ISO flag
        if "stops" in df.columns:
            row_feats["is_box_iso"] = (df["stops"].abs() < self.box_iso_tolerance).astype(float)
        else:
            row_feats["is_box_iso"] = np.nan

        parts.append(row_feats)

        return pd.concat(parts, axis=1)

    def fit_transform(self, df_train: pd.DataFrame) -> pd.DataFrame:
        """Fit on training data and return features for same rows."""
        self.fit(df_train)
        return self.transform(df_train)

    def summary(self) -> dict[str, Any]:
        """Return summary of fitted entity counts."""
        return {
            "n_films": self._film_stats.n_entities,
            "n_film_iso": self._film_iso_stats.n_entities,
            "n_dev_dils": self._dd_stats.n_entities,
            "n_developers": self._dev_stats.n_entities,
            "features_per_entity": 12,
            "entity_levels": 4,
            "row_features": 2,
            "total_features": 12 * 4 + 2,
        }


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    from data_split import DataSplitter, prepare_data

    df = prepare_data()
    splitter = DataSplitter.load("splits.json")

    print(f"\n{'='*90}")
    print("Aggregate Feature Builder — Per-Fold Validation")
    print(f"{'='*90}")

    fold_info = []

    for fold in range(splitter.n_folds):
        train_idx, val_idx = splitter.get_fold(fold)
        test_idx = splitter.test_indices

        df_train = df.loc[train_idx]
        df_val = df.loc[val_idx]
        df_test = df.loc[test_idx]

        agg = AggregateFeatureBuilder()
        feat_train = agg.fit_transform(df_train)
        feat_val = agg.transform(df_val)
        feat_test = agg.transform(df_test)

        info = agg.summary()

        # Coverage: fraction of non-null values per entity level
        def _coverage(feat_df, prefix):
            col = f"{prefix}_count"
            if col in feat_df.columns:
                return feat_df[col].notna().mean() * 100
            return 0.0

        fold_info.append({
            "fold": fold,
            "n_cols": feat_train.shape[1],
            "n_films": info["n_films"],
            "n_film_iso": info["n_film_iso"],
            "n_dd": info["n_dev_dils"],
            "n_dev": info["n_developers"],
            "val_film_cov": _coverage(feat_val, "film_agg"),
            "val_fiso_cov": _coverage(feat_val, "film_iso_agg"),
            "val_dd_cov": _coverage(feat_val, "dd_agg"),
            "val_dev_cov": _coverage(feat_val, "dev_agg"),
            "test_film_cov": _coverage(feat_test, "film_agg"),
            "test_dd_cov": _coverage(feat_test, "dd_agg"),
        })

        print(f"\nFold {fold}:")
        print(f"  Shapes: train={feat_train.shape}  val={feat_val.shape}  test={feat_test.shape}")
        print(f"  Entities: {info['n_films']} films, {info['n_film_iso']} film@ISO, "
              f"{info['n_dev_dils']} dd, {info['n_developers']} devs")
        print(f"  Val coverage — film: {_coverage(feat_val, 'film_agg'):.1f}%  "
              f"film@ISO: {_coverage(feat_val, 'film_iso_agg'):.1f}%  "
              f"dd: {_coverage(feat_val, 'dd_agg'):.1f}%  "
              f"dev: {_coverage(feat_val, 'dev_agg'):.1f}%")

        if fold == 0:
            print(f"\n  All columns ({feat_train.shape[1]}):")
            for c in feat_train.columns:
                nn = feat_train[c].notna().sum()
                print(f"    {c:35s}  {nn}/{len(feat_train)} ({nn/len(feat_train)*100:.1f}%)")

    print(f"\n{'='*90}")
    print("Consistency Across Folds")
    print(f"{'='*90}")
    fi = pd.DataFrame(fold_info)
    print(fi.to_string(index=False, float_format="%.1f"))

    col_counts = fi["n_cols"].unique()
    print(f"\n  Column count consistent: {'✓' if len(col_counts) == 1 else '✗'} ({col_counts})")
    for col in ["n_films", "n_film_iso", "n_dd", "n_dev"]:
        vals = fi[col]
        print(f"  {col:15s}  min={vals.min()}  max={vals.max()}  range={vals.max()-vals.min()}")
