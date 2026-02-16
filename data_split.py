"""
data_split.py
=============
Loads the raw film-development dataset, performs data cleaning (including the
temperature filter introduced in the 2026-02 refactor), computes derived scalar
features that don't require training-data statistics (stops, dilution_factor,
temp_celsius), and serialises stratified cross-validation fold indices to
splits.json so that all downstream training scripts share identical splits.

Usage
-----
    python data_split.py                         # default CSV path
    python data_split.py --csv path/to/data.csv  # custom CSV path
    python data_split.py --no-temp-filter        # keep all temperatures

Output
------
    splits.json  –  list of {"train": [...], "val": [...], "test": [...]}
                    one dict per fold, indices into the filtered DataFrame.
"""

from __future__ import annotations

import argparse
import json
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

# ─── Paths ────────────────────────────────────────────────────────────────────
DEFAULT_CSV = Path("data/filmdev_times.csv")
SPLITS_PATH = Path("splits.json")

# ─── Temperature filter bounds ────────────────────────────────────────────────
TEMP_MIN_C: float = 18.0
TEMP_MAX_C: float = 24.0

# ─── Cross-validation settings ───────────────────────────────────────────────
N_FOLDS: int         = 5
TEST_FRACTION: float = 0.2   # fraction of data held out as shared test set
RANDOM_STATE: int    = 42


# ══════════════════════════════════════════════════════════════════════════════
# Temperature parsing
# ══════════════════════════════════════════════════════════════════════════════

def _parse_temp_celsius(raw: object) -> float:
    """
    Parse a raw temperature value to °C.

    Handles these formats (case-insensitive):
      "20"        bare number – assumed Celsius
      "20C"       Celsius with letter suffix
      "20°C"      Celsius with degree symbol
      "20.5 C"    Celsius with space before suffix
      "68F"       Fahrenheit → converted to Celsius
      "68°F"      Fahrenheit with degree symbol
      NaN / None  → returns NaN

    Returns float(NaN) for any value that cannot be parsed.
    """
    if raw is None:
        return float("nan")
    if isinstance(raw, (int, float)):
        return float(raw) if not np.isnan(float(raw)) else float("nan")
    s = str(raw).strip()
    if not s:
        return float("nan")

    # Fahrenheit – must test before Celsius because "f" could appear
    m = re.match(r"^([\d.]+)\s*°?[Ff]$", s)
    if m:
        return (float(m.group(1)) - 32.0) * 5.0 / 9.0

    # Celsius with explicit suffix
    m = re.match(r"^([\d.]+)\s*°?[Cc]$", s)
    if m:
        return float(m.group(1))

    # Bare number – assume Celsius
    m = re.match(r"^([\d.]+)$", s)
    if m:
        return float(m.group(1))

    return float("nan")


# ══════════════════════════════════════════════════════════════════════════════
# Dilution parsing
# ══════════════════════════════════════════════════════════════════════════════

def _parse_dilution_factor(raw: object) -> float:
    """
    Convert a dilution string to a numeric dilution factor (developer parts per
    total volume, i.e. 1/(1+n) for "1+n" notation).

    Examples:
      "stock" / "1+0"  → 1.0
      "1+1"            → 0.5
      "1+3"            → 0.25
      "1+49"           → 0.02
      "1:50"           → 0.02   (same meaning, colon separator)
      "0.5"            → 0.5    (already a fraction)
    """
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return float("nan")
    s = str(raw).strip().lower()

    if s in ("stock", "straight", "undiluted", "1+0", "1:0", "1:1"):
        # "1:1" in some datasheets means stock; treat as 1+0 = stock
        if s == "1:1":
            # ambiguous – could mean 1-part dev to 1-part water (→ 0.5)
            # FilmDev.org uses "1:1" as 1+1 equivalent so dilution = 0.5
            return 0.5
        return 1.0

    # 1+N or 1:N format
    m = re.match(r"^1[+:](\d+(?:\.\d+)?)$", s)
    if m:
        n = float(m.group(1))
        return 1.0 / (1.0 + n)

    # Plain float/integer
    try:
        val = float(s)
        return val
    except ValueError:
        return float("nan")


# ══════════════════════════════════════════════════════════════════════════════
# Main data preparation
# ══════════════════════════════════════════════════════════════════════════════

def prepare_data(
    csv_path: str | Path = DEFAULT_CSV,
    temp_filter: bool = True,
) -> pd.DataFrame:
    """
    Load and clean the raw CSV.  Returns a tidy DataFrame with index reset.

    Derived scalar columns added here (safe – no training-data statistics):
      temp_celsius     – parsed numeric temperature in °C
      dilution_factor  – numeric dilution (0–1 fraction)
      stops            – push/pull in stops: log2(EI / box_iso)
                         box_iso is the box-speed ISO; EI is the exposed index

    Categorical raw columns (Film, Developer, Dilution, Temp) are kept so that
    downstream feature builders can use them for lookup / slope estimation.
    They are removed from the feature matrix inside train_extratrees.py.

    Parameters
    ----------
    csv_path : path to the raw CSV file
    temp_filter : if True, remove rows where temp is known and outside [18, 24]°C

    Notes
    -----
    After changing temp_filter=True (the default since 2026-02), you MUST
    regenerate splits.json by running this script again before training.
    """
    df = pd.read_csv(csv_path)

    # ── Normalise column names ────────────────────────────────────────────────
    df.columns = df.columns.str.strip()

    # ── Required columns ─────────────────────────────────────────────────────
    required = {"Film", "Developer", "Dilution", "Temp", "ISO", "EI", "Time"}
    missing_cols = required - set(df.columns)
    if missing_cols:
        raise ValueError(
            f"CSV is missing required columns: {sorted(missing_cols)}. "
            f"Found: {sorted(df.columns.tolist())}"
        )

    # ── Drop rows missing target or key identifiers ───────────────────────────
    df = df.dropna(subset=["Time", "Film", "Developer"])
    df = df[df["Time"] > 0].copy()

    # ── Temperature: parse to numeric ─────────────────────────────────────────
    df["temp_celsius"] = df["Temp"].map(_parse_temp_celsius)

    if temp_filter:
        temp_known    = df["temp_celsius"].notna()
        temp_in_range = df["temp_celsius"].between(TEMP_MIN_C, TEMP_MAX_C)
        n_before = len(df)
        df = df[~temp_known | temp_in_range].copy()
        n_removed = n_before - len(df)
        if n_removed:
            print(
                f"[temperature filter] removed {n_removed} rows "
                f"with known temp outside [{TEMP_MIN_C}°C, {TEMP_MAX_C}°C] "
                f"({n_removed / n_before:.1%} of data)"
            )

    # ── Dilution: parse to numeric ─────────────────────────────────────────────
    df["dilution_factor"] = df["Dilution"].map(_parse_dilution_factor)

    # ── ISO / stops ───────────────────────────────────────────────────────────
    # ISO = box speed (labelled speed on box)
    # EI  = exposure index actually used (may differ due to push/pull)
    df["ISO"] = pd.to_numeric(df["ISO"], errors="coerce")
    df["EI"]  = pd.to_numeric(df["EI"],  errors="coerce")

    # Where EI is missing, assume box-speed shooting (no push/pull)
    df["EI"] = df["EI"].fillna(df["ISO"])

    # stops > 0 → pushed; stops < 0 → pulled
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df["stops"] = np.log2(df["EI"] / df["ISO"].replace(0, np.nan))
    df["stops"] = df["stops"].fillna(0.0)

    # ── Final cleanup ─────────────────────────────────────────────────────────
    # Strip leading/trailing whitespace from string columns
    for col in ["Film", "Developer", "Dilution"]:
        df[col] = df[col].astype(str).str.strip()

    df = df.reset_index(drop=True)
    print(f"[prepare_data] {len(df)} rows loaded from {csv_path}")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# Split generation
# ══════════════════════════════════════════════════════════════════════════════

def make_splits(
    df: pd.DataFrame,
    n_folds: int = N_FOLDS,
    test_fraction: float = TEST_FRACTION,
    random_state: int = RANDOM_STATE,
) -> dict:
    """
    Generate CV splits using the original DataSplitter logic.

    One shared held-out test set is carved out first (test_fraction of all
    rows). The remaining trainval rows are divided into n_folds equal folds
    using KFold; each fold becomes the validation set for one CV iteration
    while the rest of trainval is used for training.

    Returns a dict (the splits.json schema):
        {
            "seed":          <int>,
            "n_folds":       <int>,
            "test_fraction": <float>,
            "n_total":       <int>,
            "test_indices":  [<int>, ...],          # shared across all folds
            "fold_indices":  [[<int>, ...], ...],   # one list per fold (val set)
        }

    All indices are integer positional indices into `df` (after reset_index).
    """
    n   = len(df)
    rng = np.random.default_rng(random_state)

    # 1. Hold out a shared test set
    n_test   = int(n * test_fraction)
    test_pos = sorted(rng.choice(n, size=n_test, replace=False).tolist())
    tv_pos   = sorted(set(range(n)) - set(test_pos))

    # 2. Divide trainval into n_folds using KFold (each fold → val set)
    kf     = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    tv_arr = np.array(tv_pos)
    fold_indices = []
    for _, val_local in kf.split(tv_arr):
        fold_indices.append(sorted(tv_arr[val_local].tolist()))

    return {
        "seed":          random_state,
        "n_folds":       n_folds,
        "test_fraction": test_fraction,
        "n_total":       n,
        "test_indices":  test_pos,
        "fold_indices":  fold_indices,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare data and generate CV splits")
    parser.add_argument("--csv",           default=str(DEFAULT_CSV), help="Path to raw CSV")
    parser.add_argument("--splits-out",    default=str(SPLITS_PATH), help="Output path for splits.json")
    parser.add_argument("--n-folds",       type=int,   default=N_FOLDS)
    parser.add_argument("--test-fraction", type=float, default=TEST_FRACTION)
    parser.add_argument("--random-state",  type=int,   default=RANDOM_STATE)
    parser.add_argument(
        "--no-temp-filter",
        action="store_true",
        help="Disable temperature range filter (keep all rows)"
    )
    args = parser.parse_args()

    df = prepare_data(
        csv_path=args.csv,
        temp_filter=not args.no_temp_filter,
    )

    splits = make_splits(
        df,
        n_folds=args.n_folds,
        test_fraction=args.test_fraction,
        random_state=args.random_state,
    )

    out_path = Path(args.splits_out)
    with open(out_path, "w") as f:
        json.dump(splits, f, indent=2)

    n_tv = splits["n_total"] - len(splits["test_indices"])
    print(f"[data_split] wrote {splits['n_folds']} folds to {out_path}")
    print(f"  test={len(splits['test_indices']):,}  trainval={n_tv:,}")
    for i, fi in enumerate(splits["fold_indices"]):
        print(f"  fold {i}: val={len(fi):,}  train={n_tv - len(fi):,}")


if __name__ == "__main__":
    main()