"""
Latent Variable Assignment Module
==================================

Runs matrix factorization on a training fold and assigns latent vectors
back to individual rows of the dataset.

Key design decisions:
    - Film LV: Uses the BASE ISO column only (e.g., HP5+ @ 400), then
      propagates the same vector to all ISOs of that film (HP5+ @ 800,
      HP5+ @ 1600, etc.). This captures the film's intrinsic character
      independent of push/pull level.
    - Dev_dil LV: Used as-is, each developer+dilution gets its own vector.
    - Matrix factorization is re-fit per fold on training data only.

Usage:
    from latent_features import LatentFeatureBuilder

    builder = LatentFeatureBuilder(n_factors=30, reg=0.005)
    lv_train = builder.fit_transform(df_train)  # fit MF + assign LVs
    lv_val = builder.transform(df_val)           # assign LVs only
    lv_test = builder.transform(df_test)         # assign LVs only
"""

import numpy as np
import pandas as pd
from matrix_factorization import ALSMatrixFactorization


class LatentFeatureBuilder:
    """
    Build latent features from matrix factorization.

    Parameters
    ----------
    n_factors : int
        Number of latent factors (default 30).
    reg : float
        ALS regularisation (default 0.005).
    n_iter : int
        ALS iterations (default 80).
    min_dd_count : int
        Minimum observations for a dev_dil to be included in the matrix.
    min_col_count : int
        Minimum dev_dils for a film@ISO column to be kept.
    seed : int
        Random seed for ALS.
    """

    def __init__(
        self,
        n_factors: int = 30,
        reg: float = 0.005,
        n_iter: int = 80,
        min_dd_count: int = 3,
        min_col_count: int = 3,
        seed: int = 42,
    ):
        self.n_factors = n_factors
        self.reg = reg
        self.n_iter = n_iter
        self.min_dd_count = min_dd_count
        self.min_col_count = min_col_count
        self.seed = seed

        # Populated by fit()
        self.als: ALSMatrixFactorization | None = None
        self.matrix: pd.DataFrame | None = None
        self.dev_dil_lv: dict = {}       # dd -> {bias, factors}
        self.film_lv: dict = {}          # film_name -> {bias, factors, source_col}
        self.global_mean: float = 0.0
        self.box_isos: dict = {}

    def fit(self, df_train: pd.DataFrame) -> "LatentFeatureBuilder":
        """
        Build matrix from training data, fit ALS, extract latent vectors.

        Parameters
        ----------
        df_train : DataFrame
            Training fold. Must contain columns:
            Film, iso, dev_dil, box_iso, 35mm
        """
        # ── Detect box ISOs from training data ──
        self.box_isos = (
            df_train[["Film", "box_iso"]]
            .drop_duplicates()
            .set_index("Film")["box_iso"]
            .to_dict()
        )

        # ── Build top-3-ISO matrix ──
        self.matrix = self._build_matrix(df_train)
        M_raw = self.matrix.values.astype(float)

        # ── Fit ALS ──
        self.als = ALSMatrixFactorization(
            n_factors=self.n_factors,
            reg=self.reg,
            n_iter=self.n_iter,
            seed=self.seed,
        )
        self.als.fit(M_raw, verbose=False)
        self.global_mean = self.als.global_mean

        # ── Extract latent vectors ──
        self._extract_dev_dil_lv()
        self._extract_film_lv()

        return self

    def fit_transform(self, df_train: pd.DataFrame) -> pd.DataFrame:
        """Fit on training data and return LV features for training rows."""
        self.fit(df_train)
        return self.transform(df_train)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Assign latent vectors to each row.

        Returns a DataFrame (same index as df) with columns:
            lv_global_mean          float
            lv_dd_bias              float (NaN if dev_dil not in matrix)
            lv_film_bias            float (NaN if film not in matrix)
            lv_dd_0 .. lv_dd_{k-1}     dev_dil latent factors
            lv_film_0 .. lv_film_{k-1}  film latent factors (from base ISO)
            lv_available            bool (True if both dd and film have LVs)
            lv_pred_log             float (MF predicted log-time, NaN if unavailable)
            lv_pred_time            float (MF predicted time in minutes)
        """
        n = len(df)
        k = self.n_factors

        dd_bias = np.full(n, np.nan)
        film_bias = np.full(n, np.nan)
        dd_factors = np.full((n, k), np.nan)
        film_factors = np.full((n, k), np.nan)
        available = np.zeros(n, dtype=bool)

        dev_dils = df["dev_dil"].values
        films = df["Film"].values

        for i in range(n):
            dd = dev_dils[i]
            film = films[i]

            has_dd = dd in self.dev_dil_lv
            has_film = film in self.film_lv

            if has_dd:
                dd_bias[i] = self.dev_dil_lv[dd]["bias"]
                dd_factors[i, :] = self.dev_dil_lv[dd]["factors"]

            if has_film:
                film_bias[i] = self.film_lv[film]["bias"]
                film_factors[i, :] = self.film_lv[film]["factors"]

            available[i] = has_dd and has_film

        # Build result
        result = pd.DataFrame(index=df.index)
        result["lv_global_mean"] = self.global_mean
        result["lv_dd_bias"] = dd_bias
        result["lv_film_bias"] = film_bias

        for j in range(k):
            result[f"lv_dd_{j}"] = dd_factors[:, j]
            result[f"lv_film_{j}"] = film_factors[:, j]

        result["lv_available"] = available

        # MF prediction where both vectors available
        dot_product = np.nansum(dd_factors * film_factors, axis=1)
        pred_log = self.global_mean + dd_bias + film_bias + dot_product
        pred_log[~available] = np.nan
        result["lv_pred_log"] = pred_log
        result["lv_pred_time"] = np.where(available, np.exp(pred_log), np.nan)

        return result

    def summary(self) -> dict:
        """Return summary statistics of the fitted model."""
        M = self.matrix.values.astype(float)
        obs = ~np.isnan(M)
        metrics = self.als.score(M)
        return {
            "matrix_shape": self.matrix.shape,
            "n_dev_dils": self.matrix.shape[0],
            "n_film_iso_cols": self.matrix.shape[1],
            "n_films_with_lv": len(self.film_lv),
            "n_dd_with_lv": len(self.dev_dil_lv),
            "density": obs.sum() / M.size,
            "filled_cells": int(obs.sum()),
            "recon_smape": metrics["smape"],
            "recon_r2": metrics["r2"],
            "global_mean_time": np.exp(self.global_mean),
        }

    # ── Private methods ──

    def _build_matrix(self, df_subset: pd.DataFrame) -> pd.DataFrame:
        """Build top-3-ISO-per-film matrix from a data subset."""
        sub = df_subset.copy()
        sub["iso_int"] = sub["iso"].astype(int)

        # Top 3 ISOs per film
        top3 = (
            sub.groupby(["Film", "iso_int"])
            .size()
            .reset_index(name="count")
            .sort_values(["Film", "count"], ascending=[True, False])
            .groupby("Film")
            .head(3)
        )
        valid_pairs = set(zip(top3["Film"], top3["iso_int"]))
        mask = sub.apply(lambda r: (r["Film"], r["iso_int"]) in valid_pairs, axis=1)
        sub = sub[mask].copy()

        sub["film_iso"] = sub["Film"] + " @ " + sub["iso_int"].astype(str)

        # Dev_dil filter
        dd_counts = sub["dev_dil"].value_counts()
        sub = sub[sub["dev_dil"].isin(dd_counts[dd_counts >= self.min_dd_count].index)]

        # Pivot
        matrix = sub.pivot_table(
            index="dev_dil", columns="film_iso", values="35mm", aggfunc="median"
        )
        matrix = matrix.loc[
            matrix.notna().sum(axis=1).sort_values(ascending=False).index
        ]

        def _col_key(c):
            parts = c.rsplit(" @ ", 1)
            return (parts[0], int(parts[1]))

        matrix = matrix[sorted(matrix.columns, key=_col_key)]
        matrix = matrix.loc[:, matrix.notna().sum(axis=0) >= self.min_col_count]
        return matrix

    def _extract_dev_dil_lv(self) -> None:
        """Extract per-dev_dil latent vectors from fitted ALS."""
        self.dev_dil_lv = {}
        for i, dd in enumerate(self.matrix.index):
            self.dev_dil_lv[dd] = {
                "bias": float(self.als.row_bias[i]),
                "factors": self.als.U[i, :].copy(),
            }

    def _extract_film_lv(self) -> None:
        """
        Extract per-film latent vectors using base ISO columns only.

        For each film, finds the column matching the box ISO. Falls back
        to the column with the most observations if no exact match.
        """
        self.film_lv = {}
        col_to_j = {c: j for j, c in enumerate(self.matrix.columns)}

        # Parse columns → {film_name: [(iso, col_name), ...]}
        col_films: dict[str, list[tuple[int, str]]] = {}
        for c in self.matrix.columns:
            parts = c.rsplit(" @ ", 1)
            film_name = parts[0]
            iso = int(parts[1])
            col_films.setdefault(film_name, []).append((iso, c))

        for film_name, iso_cols in col_films.items():
            base = self.box_isos.get(film_name)

            # Try exact base ISO match
            chosen_col = None
            if base is not None:
                base_int = int(base)
                for iso, c in iso_cols:
                    if iso == base_int:
                        chosen_col = c
                        break

            # Fallback: column with most observations
            if chosen_col is None:
                best_count = -1
                for iso, c in iso_cols:
                    j = col_to_j[c]
                    count = int(self.matrix.iloc[:, j].notna().sum())
                    if count > best_count:
                        best_count = count
                        chosen_col = c

            j = col_to_j[chosen_col]
            self.film_lv[film_name] = {
                "bias": float(self.als.col_bias[j]),
                "factors": self.als.V[j, :].copy(),
                "source_col": chosen_col,
            }


# ═════════════════════════════════════════════════════════════════════
# MAIN — validate with cross-validation
# ═════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    from data_split import DataSplitter, prepare_data

    df = prepare_data()
    splitter = DataSplitter.load("splits.json")

    print(f"\n{'='*80}")
    print("Latent Feature Builder — Per-Fold Validation")
    print(f"{'='*80}")

    for fold in range(splitter.n_folds):
        train_idx, val_idx = splitter.get_fold(fold)
        test_idx = splitter.test_indices

        df_train = df.loc[train_idx]
        df_val = df.loc[val_idx]
        df_test = df.loc[test_idx]

        builder = LatentFeatureBuilder(n_factors=30, reg=0.005)
        lv_train = builder.fit_transform(df_train)
        lv_val = builder.transform(df_val)
        lv_test = builder.transform(df_test)

        info = builder.summary()

        print(f"\nFold {fold}:")
        print(f"  Matrix: {info['n_dev_dils']} dd × {info['n_film_iso_cols']} film@ISO, "
              f"density={info['density']:.1%}, recon SMAPE={info['recon_smape']:.2f}%")
        print(f"  Films with LV: {info['n_films_with_lv']}, Dev_dils with LV: {info['n_dd_with_lv']}")

        for name, lv_df, orig_df in [("train", lv_train, df_train),
                                      ("val", lv_val, df_val),
                                      ("test", lv_test, df_test)]:
            cov = lv_df["lv_available"].mean() * 100
            mask = lv_df["lv_available"]
            if mask.sum() > 0:
                actual = orig_df.loc[mask, "35mm"].values
                predicted = lv_df.loc[mask, "lv_pred_time"].values
                smape = np.mean(
                    2 * np.abs(actual - predicted) / (np.abs(actual) + np.abs(predicted))
                ) * 100
                mae = np.mean(np.abs(actual - predicted))
                print(f"  {name:5s}: coverage={cov:5.1f}%  SMAPE={smape:5.2f}%  "
                      f"MAE={mae:.2f} min  ({mask.sum()}/{len(orig_df)} rows)")
            else:
                print(f"  {name:5s}: coverage={cov:5.1f}%  (no predictions)")

        # Show LV column overview
        if fold == 0:
            print(f"\n  LV columns generated ({lv_train.shape[1]} total):")
            cols = lv_train.columns.tolist()
            print(f"    {', '.join(cols[:5])}, ...")
            print(f"    {', '.join(cols[5:10])}, ...")
            print(f"    ... {', '.join(cols[-5:])}")
