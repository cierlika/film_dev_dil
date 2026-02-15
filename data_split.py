"""
Data Splitting Module
=====================

Creates reproducible train/validation/test splits for the film development
time prediction pipeline.

Schema:
    1. Hold out 20% as a fixed test set (stratified by film)
    2. Remaining 80% split into 5 folds for cross-validation
    3. All assignments saved to a JSON file with the seed, so splits
       are frozen across experiments

Usage:
    # Create splits (once)
    from data_split import DataSplitter
    splitter = DataSplitter(seed=42, n_folds=5, test_fraction=0.2)
    splitter.fit(df)             # df must have an index to track rows
    splitter.save("splits.json")

    # Load splits (every experiment)
    splitter = DataSplitter.load("splits.json")
    train_idx, val_idx = splitter.get_fold(fold=0)
    test_idx = splitter.test_indices
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

import re
import numpy as np   


def _parse_temp_celsius(raw: object) -> float:

    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return float("nan")
    s = str(raw).strip()
    # Fahrenheit
    m_f = re.match(r"^([\d.]+)\s*°?[Ff]$", s)
    if m_f:
        return (float(m_f.group(1)) - 32) * 5 / 9
    # Celsius (explicit suffix)
    m_c = re.match(r"^([\d.]+)\s*°?[Cc]$", s)
    if m_c:
        return float(m_c.group(1))
    # Bare number – assume Celsius
    m_n = re.match(r"^([\d.]+)$", s)
    if m_n:
        return float(m_n.group(1))
    return float("nan")
        
class DataSplitter:
    """
    Stratified train/val/test splitter with persistence.

    Stratifies by film to ensure every film (where possible) appears
    in both train and test. Films with very few rows (< n_folds + 1)
    are grouped into a synthetic stratum so sklearn-style stratification
    doesn't fail.

    Parameters
    ----------
    seed : int
        Random seed for reproducibility.
    n_folds : int
        Number of CV folds within the train/val portion.
    test_fraction : float
        Fraction of data held out for test (default 0.2).
    """

    def __init__(self, seed: int = 42, n_folds: int = 5, test_fraction: float = 0.2):
        self.seed = seed
        self.n_folds = n_folds
        self.test_fraction = test_fraction

        # Populated by fit()
        self.test_indices: list[int] = []
        self.fold_indices: list[list[int]] = []  # fold_indices[i] = indices in fold i
        self.n_total: int = 0

    def fit(self, df: pd.DataFrame, stratify_col: str = "Film") -> "DataSplitter":
        """
        Create stratified test holdout + CV folds.

        Parameters
        ----------
        df : DataFrame
            The full filtered dataset. Uses df.index for row tracking.
        stratify_col : str
            Column to stratify by (default "Film").
        """
        rng = np.random.RandomState(self.seed)
        self.n_total = len(df)
        indices = df.index.tolist()

        # ── Step 1: Stratified test holdout ──
        # Group indices by stratum
        strata = df[stratify_col].values
        stratum_to_idx: dict[str, list[int]] = {}
        for idx, s in zip(indices, strata):
            stratum_to_idx.setdefault(s, []).append(idx)

        test_set = []
        trainval_set = []

        for stratum, s_indices in stratum_to_idx.items():
            rng.shuffle(s_indices)
            n_test = max(1, int(round(len(s_indices) * self.test_fraction)))
            # Films with only 1 row: put in trainval (can't test what we've never seen)
            if len(s_indices) == 1:
                trainval_set.extend(s_indices)
            else:
                test_set.extend(s_indices[:n_test])
                trainval_set.extend(s_indices[n_test:])

        self.test_indices = sorted(test_set)
        trainval_set = sorted(trainval_set)

        # ── Step 2: Stratified K-fold on trainval ──
        # Build strata for trainval rows
        trainval_df = df.loc[trainval_set]
        tv_strata = trainval_df[stratify_col].values
        tv_indices = trainval_df.index.tolist()

        # Group trainval by stratum
        tv_stratum_to_idx: dict[str, list[int]] = {}
        for idx, s in zip(tv_indices, tv_strata):
            tv_stratum_to_idx.setdefault(s, []).append(idx)

        # Assign each row to a fold via round-robin within each stratum
        fold_assignment: dict[int, int] = {}
        for stratum, s_indices in tv_stratum_to_idx.items():
            rng.shuffle(s_indices)
            for i, idx in enumerate(s_indices):
                fold_assignment[idx] = i % self.n_folds

        self.fold_indices = [[] for _ in range(self.n_folds)]
        for idx in trainval_set:
            fold = fold_assignment[idx]
            self.fold_indices[fold].append(idx)

        # Sort for determinism
        for i in range(self.n_folds):
            self.fold_indices[i].sort()

        return self

    def get_fold(self, fold: int) -> tuple[list[int], list[int]]:
        """
        Return (train_indices, val_indices) for a given fold.

        The validation set is fold `fold`; training set is all other folds.
        """
        if fold < 0 or fold >= self.n_folds:
            raise ValueError(f"fold must be 0..{self.n_folds-1}, got {fold}")

        val_idx = self.fold_indices[fold]
        train_idx = []
        for i in range(self.n_folds):
            if i != fold:
                train_idx.extend(self.fold_indices[i])
        train_idx.sort()
        return train_idx, val_idx

    def get_trainval_indices(self) -> list[int]:
        """Return all train+val indices (everything except test)."""
        all_tv = []
        for fold in self.fold_indices:
            all_tv.extend(fold)
        return sorted(all_tv)

    def save(self, path: str | Path) -> None:
        """Save splits to JSON."""
        data = {
            "seed": self.seed,
            "n_folds": self.n_folds,
            "test_fraction": self.test_fraction,
            "n_total": self.n_total,
            "test_indices": self.test_indices,
            "fold_indices": self.fold_indices,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "DataSplitter":
        """Load splits from JSON."""
        with open(path) as f:
            data = json.load(f)
        obj = cls(
            seed=data["seed"],
            n_folds=data["n_folds"],
            test_fraction=data["test_fraction"],
        )
        obj.n_total = data["n_total"]
        obj.test_indices = data["test_indices"]
        obj.fold_indices = data["fold_indices"]
        return obj

    def summary(self, df: pd.DataFrame, stratify_col: str = "Film") -> pd.DataFrame:
        """
        Return a statistical summary of each fold + test set.

        Columns: set, n_rows, n_films, n_dev_dils, pct_total,
                 time_median, time_mean, time_std, time_q25, time_q75,
                 time_skew, time_kurtosis,
                 unique_films_exclusive (films only in this set)
        """
        rows = []

        def _stats(subset_df: pd.DataFrame, set_name: str) -> dict:
            times = subset_df["35mm"]
            films = set(subset_df["Film"].unique())
            dev_dils = set(
                (subset_df["Developer"].astype(str).str.strip()
                 + " " + subset_df["Dilution"].astype(str).str.strip()).unique()
            )
            return {
                "set": set_name,
                "n_rows": len(subset_df),
                "pct_total": len(subset_df) / len(df) * 100,
                "n_films": len(films),
                "n_dev_dils": len(dev_dils),
                "time_median": times.median(),
                "time_mean": times.mean(),
                "time_std": times.std(),
                "time_q25": times.quantile(0.25),
                "time_q75": times.quantile(0.75),
                "time_skew": float(sp_stats.skew(times, nan_policy="omit")),
                "time_kurtosis": float(sp_stats.kurtosis(times, nan_policy="omit")),
                "_films": films,
                "_dev_dils": dev_dils,
            }

        # Test set
        test_stats = _stats(df.loc[self.test_indices], "test")
        rows.append(test_stats)

        # Each fold (as validation)
        for fold in range(self.n_folds):
            train_idx, val_idx = self.get_fold(fold)
            val_stats = _stats(df.loc[val_idx], f"fold_{fold}_val")
            train_stats = _stats(df.loc[train_idx], f"fold_{fold}_train")
            rows.append(train_stats)
            rows.append(val_stats)

        # Compute exclusive films/dev_dils
        test_films = rows[0]["_films"]
        all_trainval_films = set()
        for r in rows[1:]:
            all_trainval_films |= r["_films"]

        for r in rows:
            other_films = set()
            for r2 in rows:
                if r2["set"] != r["set"]:
                    other_films |= r2["_films"]
            r["exclusive_films"] = len(r["_films"] - other_films)

        # Build DataFrame
        summary_df = pd.DataFrame(rows).drop(columns=["_films", "_dev_dils"])
        return summary_df


# ═════════════════════════════════════════════════════════════════════
# Data preparation function (shared filtering logic)
# ═════════════════════════════════════════════════════════════════════

def prepare_data(
    csv_path: str = "film_data.csv",
    min_time: float = 3.0,
    max_time: float = 30.0,
) -> pd.DataFrame:
    """
    Load and apply base filters to the film development dataset.

    Filters applied (in order):
        1. 35mm time: numeric, ≥ min_time, ≤ max_time (default 3–30 min)
        2. Remove unusual processes:
           - Stand development, semi-stand
           - Rotary / Jobo processing
           - High Contrast / Very High Contrast
           - Continuous slow agitation
        3. Valid ISO > 0
        4. Known box ISO (from FilmSlopeEstimator)

    Returns a clean DataFrame with added columns:
        iso, dev_dil, box_iso, stops
    """
    raw = pd.read_csv(csv_path)
    raw["35mm"] = pd.to_numeric(raw["35mm"], errors="coerce")
    n_before = len(raw)
    raw = raw[
        raw["35mm"].notna() & (raw["35mm"] >= min_time) & (raw["35mm"] <= max_time)
    ].copy()
    raw = raw.reset_index(drop=True)
    print(f"  Time filter ({min_time}–{max_time} min): {n_before} → {len(raw)} "
          f"(removed {n_before - len(raw)})")

    # Remove unusual processes
    notes = raw["Notes"].fillna("")
    unusual = notes.str.contains(
        r"Stand development|Semi-stand|rotary|jobo"
        r"|High Contrast|Very High Contrast"
        r"|continuous \(slow",
        case=False,
    )
    n_before = len(raw)
    df = raw[~unusual].copy()
    print(f"  Unusual processes removed: {unusual.sum()} → {len(df)} remaining")

    # Parse ISO
    df["iso"] = pd.to_numeric(df["ASA/ISO"], errors="coerce")
    n_before = len(df)
    df = df[df["iso"].notna() & (df["iso"] > 0)].copy()
    print(f"  Invalid ISO removed: {n_before - len(df)} → {len(df)} remaining")

    # Dev+dil key
    df["dev_dil"] = (
        df["Developer"].astype(str).str.strip()
        + " "
        + df["Dilution"].astype(str).str.strip()
    )

    # Box ISO and stops
    from film_slopes import FilmSlopeEstimator
    film_est = FilmSlopeEstimator().fit(raw)
    df["box_iso"] = df["Film"].map(film_est.box_isos)
    n_before = len(df)
    df = df[df["box_iso"].notna()].copy()
    print(f"  No box ISO removed: {n_before - len(df)} → {len(df)} remaining")
    df["stops"] = np.log2(df["iso"] / df["box_iso"])

    df = df.reset_index(drop=True)
    return df


# ═════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Prepare data
    print("Filtering pipeline:")
    df = prepare_data()
    print(f"\nFinal dataset: {len(df)} rows, {df['Film'].nunique()} films, "
          f"{df['dev_dil'].nunique()} dev_dils")
    print(f"Time range: {df['35mm'].min():.1f} – {df['35mm'].max():.1f} min\n")

    # Create splits
    splitter = DataSplitter(seed=42, n_folds=5, test_fraction=0.2)
    splitter.fit(df, stratify_col="Film")

    # Save
    split_path = "splits.json"
    splitter.save(split_path)
    print(f"Splits saved to {split_path}\n")

    # Verify reload
    splitter2 = DataSplitter.load(split_path)
    assert splitter2.test_indices == splitter.test_indices
    assert splitter2.fold_indices == splitter.fold_indices
    print("✓ Reload verified\n")

    # Summary
    summary = splitter.summary(df)

    # Print nicely
    print(f"{'='*100}")
    print("Split Summary")
    print(f"{'='*100}")
    print(summary.to_string(index=False, float_format="%.2f"))

    # Verify no overlap
    test_set = set(splitter.test_indices)
    all_folds = [set(f) for f in splitter.fold_indices]

    # Test vs folds
    for i, fold in enumerate(all_folds):
        overlap = test_set & fold
        assert len(overlap) == 0, f"Test overlaps with fold {i}: {len(overlap)} rows"

    # Fold vs fold
    for i in range(len(all_folds)):
        for j in range(i + 1, len(all_folds)):
            overlap = all_folds[i] & all_folds[j]
            assert len(overlap) == 0, f"Fold {i} overlaps with fold {j}: {len(overlap)} rows"

    # All rows accounted for
    all_assigned = test_set.copy()
    for fold in all_folds:
        all_assigned |= fold
    assert len(all_assigned) == len(df), (
        f"Mismatch: {len(all_assigned)} assigned vs {len(df)} total"
    )
    print(f"\n✓ No overlaps between test and folds")
    print(f"✓ No overlaps between folds")
    print(f"✓ All {len(df)} rows accounted for")

    # Film coverage check
    print(f"\n{'='*100}")
    print("Film Coverage")
    print(f"{'='*100}")
    test_films = set(df.loc[splitter.test_indices, "Film"].unique())
    trainval_films = set()
    for fold in splitter.fold_indices:
        trainval_films |= set(df.loc[fold, "Film"].unique())

    test_only = test_films - trainval_films
    trainval_only = trainval_films - test_films
    both = test_films & trainval_films

    print(f"  Films in both train+test:  {len(both)}")
    print(f"  Films in test only:        {len(test_only)}")
    print(f"  Films in trainval only:    {len(trainval_only)}")
    if test_only:
        print(f"    Test-only films: {sorted(test_only)[:10]}")

    # Dev_dil coverage
    test_dd = set(df.loc[splitter.test_indices, "dev_dil"].unique())
    trainval_dd = set()
    for fold in splitter.fold_indices:
        trainval_dd |= set(df.loc[fold, "dev_dil"].unique())
    dd_test_only = test_dd - trainval_dd
    dd_trainval_only = trainval_dd - test_dd

    print(f"\n  Dev_dils in both:          {len(test_dd & trainval_dd)}")
    print(f"  Dev_dils in test only:     {len(dd_test_only)}")
    print(f"  Dev_dils in trainval only: {len(dd_trainval_only)}")

    # Per-fold film coverage
    print(f"\n{'='*100}")
    print("Per-Fold Film Coverage (as validation set)")
    print(f"{'='*100}")
    for fold in range(splitter.n_folds):
        train_idx, val_idx = splitter.get_fold(fold)
        train_films = set(df.loc[train_idx, "Film"].unique())
        val_films = set(df.loc[val_idx, "Film"].unique())
        val_only = val_films - train_films
        print(f"  Fold {fold}: train={len(train_films)} films, val={len(val_films)} films, "
              f"val-only={len(val_only)} films")

    # Time distribution comparison
    print(f"\n{'='*100}")
    print("Time Distribution by Set")
    print(f"{'='*100}")
    test_times = df.loc[splitter.test_indices, "35mm"]
    print(f"  Test:     median={test_times.median():.1f}  mean={test_times.mean():.1f}  "
          f"std={test_times.std():.1f}  [Q25={test_times.quantile(0.25):.1f}, "
          f"Q75={test_times.quantile(0.75):.1f}]")
    for fold in range(splitter.n_folds):
        train_idx, val_idx = splitter.get_fold(fold)
        t = df.loc[train_idx, "35mm"]
        v = df.loc[val_idx, "35mm"]
        print(f"  Fold {fold} train: median={t.median():.1f}  mean={t.mean():.1f}  "
              f"std={t.std():.1f}  [Q25={t.quantile(0.25):.1f}, Q75={t.quantile(0.75):.1f}]")
        print(f"  Fold {fold} val:   median={v.median():.1f}  mean={v.mean():.1f}  "
              f"std={v.std():.1f}  [Q25={v.quantile(0.25):.1f}, Q75={v.quantile(0.75):.1f}]")
