"""
film_slopes.py  –  Per-film, per-dev+dil, and per-developer slope estimation
=============================================================================

Three complementary slope estimators:

FilmSlopeEstimator        – one slope per film vs ISO, across all developers
DevDilSlopeEstimator      – one slope per dev+dilution vs ISO, across all films
DevDilutionSlopeEstimator – one slope per developer vs dilution, across all films+ISOs

The first two share the same model in log space:

    log(time) = α_group + β_entity · stops + ε       (stops = log₂(ISO/box_ISO))

The third uses dilution as the x-axis:

    log(time) = α_film_iso + β_developer · log(dil_factor) + ε

All solved via demeaned fixed-effects regression (analytical).

Usage
-----
    from film_slopes import (FilmSlopeEstimator, DevDilSlopeEstimator,
                             DevDilutionSlopeEstimator)

    film_est = FilmSlopeEstimator().fit(df)
    dev_est  = DevDilSlopeEstimator(box_isos=film_est.box_isos).fit(df)
    dil_est  = DevDilutionSlopeEstimator().fit(df)

    df = film_est.transform(df)
    df = dev_est.transform(df)
    df = dil_est.transform(df)
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as sp_stats


# ══════════════════════════════════════════════════════════════════════════════
# Shared helpers
# ══════════════════════════════════════════════════════════════════════════════

def _extract_box_iso_from_name(film_name: str) -> float | None:
    """Try to pull box ISO from the film name."""
    m = re.search(r'(\d{2,5})\s*$', film_name.strip().rstrip('+'))
    if m:
        val = int(m.group(1))
        if 6 <= val <= 100000:
            return float(val)
    return None


def _detect_box_iso(film_group: pd.DataFrame, film_name: str) -> float | None:
    """Determine box ISO for a film.

    Priority: name match in data > most frequent ISO > name parse.
    """
    iso_counts = film_group.groupby("iso").size()
    if len(iso_counts) == 0:
        return None

    most_freq_iso = iso_counts.idxmax()
    max_count = iso_counts.max()
    name_iso = _extract_box_iso_from_name(film_name)

    if name_iso is not None and name_iso in iso_counts.index:
        return name_iso
    if max_count >= 3:
        return most_freq_iso
    if name_iso is not None:
        return name_iso
    return most_freq_iso


# HC-110 letter code → numeric dilution mapping
_HC110_CODES: dict[str, str] = {
    "A": "1+15", "B": "1+31", "C": "1+19", "D": "1+39",
    "E": "1+47", "F": "1+79", "G": "1+119", "H": "1+63", "J": "1+150",
}


def _parse_dilution_factor(dilution: str) -> float | None:
    """Parse a dilution string to a numeric dilution factor.

    stock → 1.0,  1+1 → 2.0,  1+50 → 51.0,  1+2+100 → ~34.3
    HC-110 letter codes (A–J) are also handled.
    """
    d = str(dilution).strip()

    if d.lower() in ("stock", "nan", "", "none"):
        return 1.0

    # HC-110 letter codes
    if d.upper() in _HC110_CODES:
        d = _HC110_CODES[d.upper()]

    # 1+N → (1+N)/1
    m = re.match(r'^(\d+)\+(\d+)$', d)
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        return (a + b) / a if a > 0 else None

    # 1+2+100 (two-part concentrate)
    m = re.match(r'^(\d+)\+(\d+)\+(\d+)$', d)
    if m:
        parts = [int(m.group(i)) for i in (1, 2, 3)]
        concentrate = parts[0] + parts[1]
        total = sum(parts)
        return total / concentrate if concentrate > 0 else None

    return None


def _simple_ols(x: np.ndarray, y: np.ndarray,
                min_n: int = 2) -> dict | None:
    """Simple OLS: y = intercept + slope * x."""
    if len(x) < min_n:
        return None
    if np.ptp(x) < 0.3:
        return None
    try:
        slope, intercept, r, p, se = sp_stats.linregress(x, y)
        if not np.isfinite(slope):
            return None
        y_pred = slope * x + intercept
        resid_std = float(np.std(y - y_pred, ddof=2)) if len(x) > 2 else float('inf')
        return {
            "slope": float(slope),
            "intercept": float(intercept),
            "r2": float(r ** 2),
            "n": len(x),
            "se": float(se),
            "residual_std": resid_std,
        }
    except Exception:
        return None


def _fixed_effects_slope(stops: np.ndarray, log_times: np.ndarray,
                         group_labels: np.ndarray) -> dict | None:
    """
    Fixed-effects regression: find β minimising

        Σ_i ( log_time_i  -  α_{group_i}  -  β · stops_i )²

    Each group gets its own intercept α.  Solved analytically by
    demeaning within groups and running no-intercept OLS.

    Returns dict with slope, se, r2_within, r2_total, n_obs, n_groups, etc.
    """
    unique_groups = np.unique(group_labels)
    if len(unique_groups) < 1:
        return None

    n_total = len(stops)

    # Demean within each group
    stops_dm = np.empty_like(stops, dtype=np.float64)
    lt_dm = np.empty_like(log_times, dtype=np.float64)
    group_means_lt = {}
    group_means_s = {}

    for g in unique_groups:
        mask = group_labels == g
        m_s = stops[mask].mean()
        m_lt = log_times[mask].mean()
        stops_dm[mask] = stops[mask] - m_s
        lt_dm[mask] = log_times[mask] - m_lt
        group_means_lt[g] = m_lt
        group_means_s[g] = m_s

    # OLS on demeaned data (no intercept)
    ss_xx = np.dot(stops_dm, stops_dm)
    if ss_xx < 1e-12:
        return None

    slope = float(np.dot(stops_dm, lt_dm) / ss_xx)

    # Per-group intercepts: α_g = mean(log_time_g) - β · mean(stops_g)
    intercepts = {}
    for g in unique_groups:
        intercepts[g] = group_means_lt[g] - slope * group_means_s[g]

    # Residuals
    predicted = np.array([intercepts[g] for g in group_labels]) + slope * stops
    residuals = log_times - predicted
    sse = float(np.dot(residuals, residuals))
    n_params = len(unique_groups) + 1
    dof = n_total - n_params

    if dof > 0:
        mse = sse / dof
        se_slope = float(np.sqrt(mse / ss_xx))
    else:
        mse = float('inf')
        se_slope = float('inf')

    # R² within (does the slope help beyond group intercepts?)
    ss_tot_within = float(np.dot(lt_dm, lt_dm))
    r2_within = 1.0 - sse / ss_tot_within if ss_tot_within > 1e-12 else 0.0

    # R² total
    grand_mean = log_times.mean()
    ss_tot_total = float(np.sum((log_times - grand_mean) ** 2))
    r2_total = 1.0 - sse / ss_tot_total if ss_tot_total > 1e-12 else 0.0

    return {
        "slope": slope,
        "se": se_slope,
        "r2_within": float(r2_within),
        "r2_total": float(r2_total),
        "n_obs": n_total,
        "n_groups": len(unique_groups),
        "dof": dof,
        "rmse": float(np.sqrt(mse)) if dof > 0 else float('inf'),
        "group_intercepts": intercepts,
    }


def _fixed_effects_quadratic(stops: np.ndarray, log_times: np.ndarray,
                             group_labels: np.ndarray) -> dict | None:
    """
    Fixed-effects quadratic regression:

        log_time_i = α_{group_i} + β₁ · stops_i + β₂ · stops_i² + ε_i

    Solved by demeaning within groups then OLS on [stops_dm, stops²_dm].

    Returns dict with slope (β₁), accel (β₂), se_slope, se_accel,
    r2_within, r2_total, n_obs, n_groups, group_intercepts.
    """
    unique_groups = np.unique(group_labels)
    if len(unique_groups) < 1:
        return None

    n_total = len(stops)
    stops2 = stops ** 2

    # Demean within each group
    stops_dm = np.empty_like(stops, dtype=np.float64)
    stops2_dm = np.empty_like(stops, dtype=np.float64)
    lt_dm = np.empty_like(log_times, dtype=np.float64)
    group_means = {}  # {g: (mean_s, mean_s2, mean_lt)}

    for g in unique_groups:
        mask = group_labels == g
        m_s = stops[mask].mean()
        m_s2 = stops2[mask].mean()
        m_lt = log_times[mask].mean()
        stops_dm[mask] = stops[mask] - m_s
        stops2_dm[mask] = stops2[mask] - m_s2
        lt_dm[mask] = log_times[mask] - m_lt
        group_means[g] = (m_s, m_s2, m_lt)

    # OLS on demeaned data: y_dm = b1*stops_dm + b2*stops2_dm
    X = np.column_stack([stops_dm, stops2_dm])
    XtX = X.T @ X
    det = XtX[0, 0] * XtX[1, 1] - XtX[0, 1] * XtX[1, 0]
    if abs(det) < 1e-12:
        return None

    Xty = X.T @ lt_dm
    beta = np.linalg.lstsq(X, lt_dm, rcond=None)[0]
    b1, b2 = float(beta[0]), float(beta[1])

    # Per-group intercepts: α_g = mean(lt_g) - b1*mean(s_g) - b2*mean(s2_g)
    intercepts = {}
    for g in unique_groups:
        m_s, m_s2, m_lt = group_means[g]
        intercepts[g] = m_lt - b1 * m_s - b2 * m_s2

    # Residuals
    predicted = np.array([intercepts[g] for g in group_labels]) + b1 * stops + b2 * stops2
    residuals = log_times - predicted
    sse = float(np.dot(residuals, residuals))
    n_params = len(unique_groups) + 2  # intercepts + b1 + b2
    dof = n_total - n_params

    if dof > 0:
        mse = sse / dof
        # Standard errors from (X'X)^-1 * mse
        XtX_inv = np.linalg.inv(XtX)
        se_b1 = float(np.sqrt(mse * XtX_inv[0, 0]))
        se_b2 = float(np.sqrt(mse * XtX_inv[1, 1]))
    else:
        mse = float('inf')
        se_b1 = float('inf')
        se_b2 = float('inf')

    # R² within
    ss_tot_within = float(np.dot(lt_dm, lt_dm))
    r2_within = 1.0 - sse / ss_tot_within if ss_tot_within > 1e-12 else 0.0

    # R² total
    grand_mean = log_times.mean()
    ss_tot_total = float(np.sum((log_times - grand_mean) ** 2))
    r2_total = 1.0 - sse / ss_tot_total if ss_tot_total > 1e-12 else 0.0

    return {
        "slope": b1,
        "accel": b2,
        "se": se_b1,
        "se_accel": se_b2,
        "r2_within": float(r2_within),
        "r2_total": float(r2_total),
        "n_obs": n_total,
        "n_groups": len(unique_groups),
        "dof": dof,
        "rmse": float(np.sqrt(mse)) if dof > 0 else float('inf'),
        "group_intercepts": intercepts,
    }


def _outlier_filter_by_slope(entity_slopes: dict[str, dict],
                             sigma: float = 3.0) -> set[str]:
    """Return set of entity keys whose individual slopes are outliers (by MAD)."""
    if len(entity_slopes) < 3:
        return set()
    slopes_arr = np.array([v["slope"] for v in entity_slopes.values()])
    med = np.median(slopes_arr)
    mad = np.median(np.abs(slopes_arr - med)) * 1.4826
    if mad < 1e-6:
        return set()
    return {
        k for k, v in entity_slopes.items()
        if abs(v["slope"] - med) > sigma * mad
    }


def _filter_iso_monotonicity(wdf: pd.DataFrame,
                              group_cols: list[str]) -> tuple[pd.DataFrame, int]:
    """Remove rows from groups where median time is NOT non-decreasing with ISO.

    For each group defined by `group_cols`, compute median time at each ISO.
    If medians aren't monotonically non-decreasing with ISO, drop ALL rows
    from that group.

    Parameters
    ----------
    wdf : DataFrame with 'iso' and 'time' columns
    group_cols : columns that define a combo (e.g. ['Film', 'dev_dil'])

    Returns
    -------
    Filtered DataFrame and count of removed rows.
    """
    # Identify bad group keys
    bad_keys = set()
    for keys, grp in wdf.groupby(group_cols):
        if grp["iso"].nunique() < 2:
            continue
        medians = grp.groupby("iso")["time"].median().sort_index()
        if not np.all(np.diff(medians.values) >= 0):
            bad_keys.add(keys if isinstance(keys, tuple) else (keys,))

    if not bad_keys:
        return wdf, 0

    # Build boolean mask for rows belonging to bad groups
    if len(group_cols) == 1:
        col = group_cols[0]
        bad_vals = {k[0] for k in bad_keys}
        mask = wdf[col].isin(bad_vals)
    else:
        # Multi-column: build a tuple key per row and check membership
        row_keys = list(zip(*(wdf[c] for c in group_cols)))
        mask = pd.Series([tuple(k) in bad_keys for k in row_keys],
                         index=wdf.index)

    n_removed = int(mask.sum())
    return wdf[~mask].copy(), n_removed


def _filter_dilution_monotonicity(wdf: pd.DataFrame,
                                   group_cols: list[str]) -> tuple[pd.DataFrame, int]:
    """Remove rows from groups where median time is NOT non-decreasing with dilution.

    For each group defined by `group_cols`, compute median time at each
    dilution factor.  If medians aren't monotonically non-decreasing,
    drop ALL rows from that group.

    Parameters
    ----------
    wdf : DataFrame with 'dil_factor' and 'time' columns
    group_cols : columns that define a combo (e.g. ['Developer', 'Film', 'iso'])

    Returns
    -------
    Filtered DataFrame and count of removed rows.
    """
    bad_keys = set()
    for keys, grp in wdf.groupby(group_cols):
        if grp["dil_factor"].nunique() < 2:
            continue
        medians = grp.groupby("dil_factor")["time"].median().sort_index()
        if not np.all(np.diff(medians.values) >= 0):
            bad_keys.add(keys if isinstance(keys, tuple) else (keys,))

    if not bad_keys:
        return wdf, 0

    if len(group_cols) == 1:
        col = group_cols[0]
        bad_vals = {k[0] for k in bad_keys}
        mask = wdf[col].isin(bad_vals)
    else:
        row_keys = list(zip(*(wdf[c] for c in group_cols)))
        mask = pd.Series([tuple(k) in bad_keys for k in row_keys],
                         index=wdf.index)

    n_removed = int(mask.sum())
    return wdf[~mask].copy(), n_removed


# ══════════════════════════════════════════════════════════════════════════════
# FilmSlopeEstimator
# ══════════════════════════════════════════════════════════════════════════════

class FilmSlopeEstimator:
    """
    Per-film quadratic slope: log(time) = α_dev + β₁ · stops + β₂ · stops² + ε

    Each developer gets its own intercept; β₁ captures the film's
    intrinsic push/pull rate and β₂ captures the acceleration
    (disproportionate time increase for extreme push/pull).

    Parameters
    ----------
    min_iso_spread_stops : float
        Minimum range of stops to attempt fitting.
    min_groups_for_entity : int
        Minimum developers needed for the fixed-effects fit.
    min_obs_per_group : int
        Minimum rows per developer to include it.
    min_obs_total : int
        Minimum total rows for the fixed-effects fit.
    outlier_sigma : float | None
        MAD-based outlier removal threshold.
    """

    def __init__(
        self,
        min_iso_spread_stops: float = 0.5,
        min_groups_for_entity: int = 1,
        min_obs_per_group: int = 2,
        min_obs_total: int = 3,
        outlier_sigma: float | None = 3.0,
        enforce_monotonicity: bool = True,
    ):
        self.min_iso_spread_stops = min_iso_spread_stops
        self.min_groups_for_entity = min_groups_for_entity
        self.min_obs_per_group = min_obs_per_group
        self.min_obs_total = min_obs_total
        self.outlier_sigma = outlier_sigma
        self.enforce_monotonicity = enforce_monotonicity

        self.box_isos: dict[str, float] = {}
        self.film_slopes: dict[str, dict[str, Any]] = {}
        self.per_dev_slopes: dict[tuple[str, str], dict] = {}
        self.global_slope: float | None = None
        self.global_accel: float = 0.0
        self.global_slope_info: dict | None = None
        self._fitted = False
        self.monotonicity_stats: dict[str, int] = {}

    def fit(self, df: pd.DataFrame) -> "FilmSlopeEstimator":
        wdf = self._prepare(df)
        if len(wdf) == 0:
            warnings.warn("FilmSlopeEstimator: no usable data.")
            self._fitted = True
            return self

        self._fit_per_group_slopes(wdf)
        self._fit_entity_slopes(wdf)
        self._fit_global_slope(wdf)
        self._fitted = True
        return self

    def _prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        wdf = df.copy()
        wdf["iso"] = pd.to_numeric(wdf["ASA/ISO"], errors="coerce")
        wdf["time"] = pd.to_numeric(wdf["35mm"], errors="coerce")
        wdf = wdf[wdf["iso"].notna() & wdf["time"].notna()].copy()
        wdf = wdf[(wdf["iso"] > 0) & (wdf["time"] > 0)].copy()

        wdf["dev_dil"] = (
            wdf["Developer"].astype(str).str.strip()
            + "_"
            + wdf["Dilution"].astype(str).str.strip()
        )

        # Remove combos where higher ISO doesn't mean longer time
        if self.enforce_monotonicity:
            wdf, n_removed = _filter_iso_monotonicity(
                wdf, ["Film", "dev_dil"]
            )
            self.monotonicity_stats["data_rows_removed_iso"] = n_removed

        for film, grp in wdf.groupby("Film"):
            box = _detect_box_iso(grp, film)
            if box is not None:
                self.box_isos[film] = box

        wdf["box_iso"] = wdf["Film"].map(self.box_isos)
        wdf = wdf[wdf["box_iso"].notna() & (wdf["box_iso"] > 0)].copy()
        wdf["stops"] = np.log2(wdf["iso"] / wdf["box_iso"])
        wdf["log_time"] = np.log(wdf["time"])
        return wdf

    def _fit_per_group_slopes(self, wdf: pd.DataFrame) -> None:
        """Fit slope for each (film, dev_dil) pair."""
        self.per_dev_slopes = {}
        n_rejected = 0
        for (film, dd), grp in wdf.groupby(["Film", "dev_dil"]):
            if len(grp) < self.min_obs_per_group:
                continue
            result = _simple_ols(grp["stops"].values, grp["log_time"].values,
                                 min_n=self.min_obs_per_group)
            if result is None:
                continue
            # Higher ISO must → longer time (positive slope)
            if self.enforce_monotonicity and result["slope"] < 0:
                n_rejected += 1
                continue
            result["iso_range"] = (float(grp["iso"].min()), float(grp["iso"].max()))
            result["stops_range"] = (float(grp["stops"].min()), float(grp["stops"].max()))
            self.per_dev_slopes[(film, dd)] = result
        self.monotonicity_stats["per_group_rejected"] = n_rejected

    def _fit_entity_slopes(self, wdf: pd.DataFrame) -> None:
        """Optimise one slope per film across all developers."""
        self.film_slopes = {}
        n_entity_clamped = 0

        for film, film_grp in wdf.groupby("Film"):
            if np.ptp(film_grp["stops"]) < self.min_iso_spread_stops:
                continue

            dev_slopes_here = {
                k[1]: v for k, v in self.per_dev_slopes.items() if k[0] == film
            }

            dev_counts = film_grp.groupby("dev_dil").size()
            valid_devs = dev_counts[dev_counts >= self.min_obs_per_group].index
            fit_data = film_grp[film_grp["dev_dil"].isin(valid_devs)].copy()

            if len(fit_data) < self.min_obs_total:
                continue
            if fit_data["dev_dil"].nunique() < self.min_groups_for_entity:
                continue

            if self.outlier_sigma is not None and len(dev_slopes_here) >= 3:
                outliers = _outlier_filter_by_slope(dev_slopes_here, self.outlier_sigma)
                if outliers:
                    fit_data = fit_data[~fit_data["dev_dil"].isin(outliers)].copy()

            if len(fit_data) < self.min_obs_total:
                continue

            result = _fixed_effects_quadratic(
                fit_data["stops"].values,
                fit_data["log_time"].values,
                fit_data["dev_dil"].values,
            )
            if result is None:
                # Fallback to linear if quadratic is singular
                result = _fixed_effects_slope(
                    fit_data["stops"].values,
                    fit_data["log_time"].values,
                    fit_data["dev_dil"].values,
                )
                if result is None:
                    continue
                result["accel"] = 0.0
                result["se_accel"] = float("inf")

            # Enforce: β₁ ≥ 0 (increasing at box ISO), β₂ ≥ 0 (convex)
            if self.enforce_monotonicity:
                if result["slope"] < 0:
                    n_entity_clamped += 1
                    continue
                result["accel"] = max(0.0, result["accel"])

            result["per_dev_slopes"] = {dd: v["slope"] for dd, v in dev_slopes_here.items()}
            result["per_dev_n"] = {dd: v["n"] for dd, v in dev_slopes_here.items()}
            result["box_iso"] = self.box_isos.get(film)
            result["iso_range"] = (float(film_grp["iso"].min()), float(film_grp["iso"].max()))
            self.film_slopes[film] = result

        self.monotonicity_stats["entity_rejected"] = n_entity_clamped

    def _fit_global_slope(self, wdf: pd.DataFrame) -> None:
        films_with = set(self.film_slopes.keys())
        pool = wdf[wdf["Film"].isin(films_with)] if len(films_with) >= 10 else wdf

        if len(pool) < 5:
            self.global_slope = 0.20
            self.global_accel = 0.0
            self.global_slope_info = {"slope": 0.20, "accel": 0.0, "source": "default"}
            return

        result = _fixed_effects_quadratic(
            pool["stops"].values, pool["log_time"].values,
            pool["dev_dil"].values,
        )
        if result and (not self.enforce_monotonicity or result["slope"] >= 0):
            self.global_slope = result["slope"]
            self.global_accel = max(0.0, result["accel"]) if self.enforce_monotonicity else result["accel"]
            self.global_slope_info = result
        else:
            self.global_slope = 0.20
            self.global_accel = 0.0
            self.global_slope_info = {"slope": 0.20, "accel": 0.0, "source": "default"}

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Attach film slope features.

        New columns: film_slope, film_slope_se, film_slope_r2,
                     film_slope_n_devs, box_iso, stops_from_box,
                     film_slope_pred_log_ratio
        """
        if not self._fitted:
            raise RuntimeError("Call .fit() first")

        out = df.copy()
        out["iso"] = pd.to_numeric(out["ASA/ISO"], errors="coerce")
        out["box_iso"] = out["Film"].map(self.box_isos)

        valid = out["box_iso"].notna() & (out["box_iso"] > 0) & out["iso"].notna() & (out["iso"] > 0)
        out["stops_from_box"] = np.nan
        out.loc[valid, "stops_from_box"] = np.log2(out.loc[valid, "iso"] / out.loc[valid, "box_iso"])

        slope_map = {f: v["slope"] for f, v in self.film_slopes.items()}
        accel_map = {f: v["accel"] for f, v in self.film_slopes.items()}
        se_map = {f: v["se"] for f, v in self.film_slopes.items()}
        r2_map = {f: v["r2_within"] for f, v in self.film_slopes.items()}
        ndev_map = {f: v["n_groups"] for f, v in self.film_slopes.items()}

        out["film_slope"] = out["Film"].map(slope_map)
        out["film_accel"] = out["Film"].map(accel_map)
        out["film_slope_se"] = out["Film"].map(se_map)
        out["film_slope_r2"] = out["Film"].map(r2_map)
        out["film_slope_n_devs"] = out["Film"].map(ndev_map)

        if self.global_slope is not None:
            out["film_slope"] = out["film_slope"].fillna(self.global_slope)
            out["film_accel"] = out["film_accel"].fillna(self.global_accel)
            if self.global_slope_info:
                out["film_slope_se"] = out["film_slope_se"].fillna(
                    self.global_slope_info.get("se", float("inf"))
                )

        out["film_slope_pred_log_ratio"] = (
            out["film_slope"] * out["stops_from_box"]
            + out["film_accel"] * out["stops_from_box"] ** 2
        )
        out.drop(columns=["iso"], inplace=True, errors="ignore")
        return out

    def summary(self) -> pd.DataFrame:
        rows = []
        for film, info in sorted(self.film_slopes.items()):
            rows.append({
                "Film": film,
                "slope": info["slope"],
                "accel": info["accel"],
                "se": info["se"],
                "se_accel": info["se_accel"],
                "r2_within": info["r2_within"],
                "r2_total": info["r2_total"],
                "n_obs": info["n_obs"],
                "n_devs": info["n_groups"],
                "box_iso": info.get("box_iso"),
                "iso_min": info["iso_range"][0],
                "iso_max": info["iso_range"][1],
                "per_dev_slope_median": float(np.median(
                    list(info["per_dev_slopes"].values())
                )) if info["per_dev_slopes"] else np.nan,
                "per_dev_slope_std": float(np.std(
                    list(info["per_dev_slopes"].values())
                )) if len(info["per_dev_slopes"]) > 1 else np.nan,
            })
        return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# DevDilSlopeEstimator
# ══════════════════════════════════════════════════════════════════════════════

class DevDilSlopeEstimator:
    """
    Per-developer+dilution quadratic slope:
        log(time) = α_film + β₁ · stops + β₂ · stops² + ε

    Each film gets its own intercept; β₁ captures how this developer
    at this dilution responds to pushing/pulling, and β₂ captures
    the acceleration for extreme push/pull.

    Parameters
    ----------
    box_isos : dict[str, float] | None
        Pre-computed box ISOs per film (e.g. from FilmSlopeEstimator).
        If None, box ISOs will be detected from the data.
    min_iso_spread_stops : float
        Minimum range of stops within a dev_dil to attempt fitting.
    min_groups_for_entity : int
        Minimum number of films needed for the fixed-effects fit.
    min_obs_per_group : int
        Minimum rows per film to include it as a group.
    min_obs_total : int
        Minimum total rows for the fixed-effects fit.
    outlier_sigma : float | None
        MAD-based outlier removal on per-film slopes before joint fit.
    """

    def __init__(
        self,
        box_isos: dict[str, float] | None = None,
        min_iso_spread_stops: float = 0.5,
        min_groups_for_entity: int = 1,
        min_obs_per_group: int = 2,
        min_obs_total: int = 3,
        outlier_sigma: float | None = 3.0,
        enforce_monotonicity: bool = True,
    ):
        self.min_iso_spread_stops = min_iso_spread_stops
        self.min_groups_for_entity = min_groups_for_entity
        self.min_obs_per_group = min_obs_per_group
        self.min_obs_total = min_obs_total
        self.outlier_sigma = outlier_sigma
        self.enforce_monotonicity = enforce_monotonicity

        self.box_isos: dict[str, float] = dict(box_isos) if box_isos else {}
        self.dev_dil_slopes: dict[str, dict[str, Any]] = {}
        self.per_film_slopes: dict[tuple[str, str], dict] = {}
        self.global_slope: float | None = None
        self.global_accel: float = 0.0
        self.global_slope_info: dict | None = None
        self._fitted = False
        self.monotonicity_stats: dict[str, int] = {}

    def fit(self, df: pd.DataFrame) -> "DevDilSlopeEstimator":
        wdf = self._prepare(df)
        if len(wdf) == 0:
            warnings.warn("DevDilSlopeEstimator: no usable data.")
            self._fitted = True
            return self

        self._fit_per_group_slopes(wdf)
        self._fit_entity_slopes(wdf)
        self._fit_global_slope(wdf)
        self._fitted = True
        return self

    def _prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        wdf = df.copy()
        wdf["iso"] = pd.to_numeric(wdf["ASA/ISO"], errors="coerce")
        wdf["time"] = pd.to_numeric(wdf["35mm"], errors="coerce")
        wdf = wdf[wdf["iso"].notna() & wdf["time"].notna()].copy()
        wdf = wdf[(wdf["iso"] > 0) & (wdf["time"] > 0)].copy()

        wdf["dev_dil"] = (
            wdf["Developer"].astype(str).str.strip()
            + "_"
            + wdf["Dilution"].astype(str).str.strip()
        )

        # Remove combos where higher ISO doesn't mean longer time
        if self.enforce_monotonicity:
            wdf, n_removed = _filter_iso_monotonicity(
                wdf, ["Film", "dev_dil"]
            )
            self.monotonicity_stats["data_rows_removed_iso"] = n_removed

        # Detect box ISOs if not provided
        if not self.box_isos:
            for film, grp in wdf.groupby("Film"):
                box = _detect_box_iso(grp, film)
                if box is not None:
                    self.box_isos[film] = box

        wdf["box_iso"] = wdf["Film"].map(self.box_isos)
        wdf = wdf[wdf["box_iso"].notna() & (wdf["box_iso"] > 0)].copy()
        wdf["stops"] = np.log2(wdf["iso"] / wdf["box_iso"])
        wdf["log_time"] = np.log(wdf["time"])
        return wdf

    def _fit_per_group_slopes(self, wdf: pd.DataFrame) -> None:
        """Fit slope for each (dev_dil, film) pair."""
        self.per_film_slopes = {}
        n_rejected = 0
        for (dd, film), grp in wdf.groupby(["dev_dil", "Film"]):
            if len(grp) < self.min_obs_per_group:
                continue
            result = _simple_ols(grp["stops"].values, grp["log_time"].values,
                                 min_n=self.min_obs_per_group)
            if result is None:
                continue
            if self.enforce_monotonicity and result["slope"] < 0:
                n_rejected += 1
                continue
            result["iso_range"] = (float(grp["iso"].min()), float(grp["iso"].max()))
            self.per_film_slopes[(dd, film)] = result
        self.monotonicity_stats["per_group_rejected"] = n_rejected

    def _fit_entity_slopes(self, wdf: pd.DataFrame) -> None:
        """Optimise one slope per dev_dil across all films."""
        self.dev_dil_slopes = {}
        n_entity_clamped = 0

        for dd, dd_grp in wdf.groupby("dev_dil"):
            if np.ptp(dd_grp["stops"]) < self.min_iso_spread_stops:
                continue

            # Per-film slopes for this dev_dil
            film_slopes_here = {
                k[1]: v for k, v in self.per_film_slopes.items() if k[0] == dd
            }

            # Filter to films with enough observations
            film_counts = dd_grp.groupby("Film").size()
            valid_films = film_counts[film_counts >= self.min_obs_per_group].index
            fit_data = dd_grp[dd_grp["Film"].isin(valid_films)].copy()

            if len(fit_data) < self.min_obs_total:
                continue
            if fit_data["Film"].nunique() < self.min_groups_for_entity:
                continue

            # Outlier film removal
            if self.outlier_sigma is not None and len(film_slopes_here) >= 3:
                outliers = _outlier_filter_by_slope(film_slopes_here, self.outlier_sigma)
                if outliers:
                    fit_data = fit_data[~fit_data["Film"].isin(outliers)].copy()

            if len(fit_data) < self.min_obs_total:
                continue

            result = _fixed_effects_quadratic(
                fit_data["stops"].values,
                fit_data["log_time"].values,
                fit_data["Film"].values,
            )
            if result is None:
                # Fallback to linear if quadratic is singular
                result = _fixed_effects_slope(
                    fit_data["stops"].values,
                    fit_data["log_time"].values,
                    fit_data["Film"].values,
                )
                if result is None:
                    continue
                result["accel"] = 0.0
                result["se_accel"] = float("inf")

            if self.enforce_monotonicity:
                if result["slope"] < 0:
                    n_entity_clamped += 1
                    continue
                result["accel"] = max(0.0, result["accel"])

            result["per_film_slopes"] = {f: v["slope"] for f, v in film_slopes_here.items()}
            result["per_film_n"] = {f: v["n"] for f, v in film_slopes_here.items()}
            result["iso_range"] = (float(dd_grp["iso"].min()), float(dd_grp["iso"].max()))
            result["n_films"] = result.pop("n_groups")
            self.dev_dil_slopes[dd] = result

        self.monotonicity_stats["entity_rejected"] = n_entity_clamped

    def _fit_global_slope(self, wdf: pd.DataFrame) -> None:
        dds_with = set(self.dev_dil_slopes.keys())
        pool = wdf[wdf["dev_dil"].isin(dds_with)] if len(dds_with) >= 10 else wdf

        if len(pool) < 5:
            self.global_slope = 0.20
            self.global_accel = 0.0
            self.global_slope_info = {"slope": 0.20, "accel": 0.0, "source": "default"}
            return

        result = _fixed_effects_quadratic(
            pool["stops"].values, pool["log_time"].values,
            pool["Film"].values,
        )
        if result and (not self.enforce_monotonicity or result["slope"] >= 0):
            self.global_slope = result["slope"]
            self.global_accel = max(0.0, result["accel"]) if self.enforce_monotonicity else result["accel"]
            self.global_slope_info = result
        else:
            self.global_slope = 0.20
            self.global_accel = 0.0
            self.global_slope_info = {"slope": 0.20, "accel": 0.0, "source": "default"}

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Attach dev_dil slope features.

        New columns: dev_dil_slope, dev_dil_slope_se, dev_dil_slope_r2,
                     dev_dil_slope_n_films, dev_dil_slope_pred_log_ratio
        """
        if not self._fitted:
            raise RuntimeError("Call .fit() first")

        out = df.copy()
        out["iso"] = pd.to_numeric(out["ASA/ISO"], errors="coerce")

        # Ensure box_iso and stops exist (may already be there from FilmSlopeEstimator)
        if "box_iso" not in out.columns:
            out["box_iso"] = out["Film"].map(self.box_isos)
        if "stops_from_box" not in out.columns:
            valid = out["box_iso"].notna() & (out["box_iso"] > 0) & out["iso"].notna() & (out["iso"] > 0)
            out["stops_from_box"] = np.nan
            out.loc[valid, "stops_from_box"] = np.log2(
                out.loc[valid, "iso"] / out.loc[valid, "box_iso"]
            )

        # Build dev_dil key
        out["_dd"] = (
            out["Developer"].astype(str).str.strip()
            + "_"
            + out["Dilution"].astype(str).str.strip()
        )

        slope_map = {dd: v["slope"] for dd, v in self.dev_dil_slopes.items()}
        accel_map = {dd: v["accel"] for dd, v in self.dev_dil_slopes.items()}
        se_map = {dd: v["se"] for dd, v in self.dev_dil_slopes.items()}
        r2_map = {dd: v["r2_within"] for dd, v in self.dev_dil_slopes.items()}
        nfilm_map = {dd: v["n_films"] for dd, v in self.dev_dil_slopes.items()}

        out["dev_dil_slope"] = out["_dd"].map(slope_map)
        out["dev_dil_accel"] = out["_dd"].map(accel_map)
        out["dev_dil_slope_se"] = out["_dd"].map(se_map)
        out["dev_dil_slope_r2"] = out["_dd"].map(r2_map)
        out["dev_dil_slope_n_films"] = out["_dd"].map(nfilm_map)

        if self.global_slope is not None:
            out["dev_dil_slope"] = out["dev_dil_slope"].fillna(self.global_slope)
            out["dev_dil_accel"] = out["dev_dil_accel"].fillna(self.global_accel)
            if self.global_slope_info:
                out["dev_dil_slope_se"] = out["dev_dil_slope_se"].fillna(
                    self.global_slope_info.get("se", float("inf"))
                )

        out["dev_dil_slope_pred_log_ratio"] = (
            out["dev_dil_slope"] * out["stops_from_box"]
            + out["dev_dil_accel"] * out["stops_from_box"] ** 2
        )
        out.drop(columns=["iso", "_dd"], inplace=True, errors="ignore")
        return out

    def summary(self) -> pd.DataFrame:
        rows = []
        for dd, info in sorted(self.dev_dil_slopes.items()):
            rows.append({
                "dev_dil": dd,
                "slope": info["slope"],
                "accel": info["accel"],
                "se": info["se"],
                "se_accel": info["se_accel"],
                "r2_within": info["r2_within"],
                "r2_total": info["r2_total"],
                "n_obs": info["n_obs"],
                "n_films": info["n_films"],
                "iso_min": info["iso_range"][0],
                "iso_max": info["iso_range"][1],
                "per_film_slope_median": float(np.median(
                    list(info["per_film_slopes"].values())
                )) if info["per_film_slopes"] else np.nan,
                "per_film_slope_std": float(np.std(
                    list(info["per_film_slopes"].values())
                )) if len(info["per_film_slopes"]) > 1 else np.nan,
            })
        return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# DevDilutionSlopeEstimator
# ══════════════════════════════════════════════════════════════════════════════

class DevDilutionSlopeEstimator:
    """
    Per-developer dilution slope: log(time) = α_{film_iso} + β_dev · log(dil_factor) + ε

    Each film×ISO combination gets its own intercept; the slope captures how
    this developer's time scales with dilution across all films.

    The x-axis is log(dilution_factor), where dilution_factor is:
        stock → 1,  1+1 → 2,  1+50 → 51, etc.

    A positive slope means more dilute → longer time (expected behaviour).

    Parameters
    ----------
    min_dil_spread_log : float
        Minimum range of log(dil_factor) to attempt fitting.
        Default 0.5 (roughly stock vs 1+1).
    min_groups_for_entity : int
        Minimum film×ISO groups for fixed-effects fit.
    min_obs_per_group : int
        Minimum rows per film×ISO group to include it.
    min_obs_total : int
        Minimum total rows for the fixed-effects fit.
    min_dilutions : int
        Minimum distinct dilutions a developer must have.
    outlier_sigma : float | None
        MAD-based outlier removal on per-group slopes.
    """

    def __init__(
        self,
        min_dil_spread_log: float = 0.5,
        min_groups_for_entity: int = 2,
        min_obs_per_group: int = 2,
        min_obs_total: int = 5,
        min_dilutions: int = 2,
        outlier_sigma: float | None = 3.0,
        enforce_monotonicity: bool = True,
    ):
        self.min_dil_spread_log = min_dil_spread_log
        self.min_groups_for_entity = min_groups_for_entity
        self.min_obs_per_group = min_obs_per_group
        self.min_obs_total = min_obs_total
        self.min_dilutions = min_dilutions
        self.outlier_sigma = outlier_sigma
        self.enforce_monotonicity = enforce_monotonicity

        self.dev_dilution_slopes: dict[str, dict[str, Any]] = {}
        self.per_filmiso_slopes: dict[tuple[str, str], dict] = {}
        self.global_slope: float | None = None
        self.global_slope_info: dict | None = None
        self._fitted = False
        self.monotonicity_stats: dict[str, int] = {}

    def fit(self, df: pd.DataFrame) -> "DevDilutionSlopeEstimator":
        wdf = self._prepare(df)
        if len(wdf) == 0:
            warnings.warn("DevDilutionSlopeEstimator: no usable data.")
            self._fitted = True
            return self

        self._fit_per_group_slopes(wdf)
        self._fit_entity_slopes(wdf)
        self._fit_global_slope(wdf)
        self._fitted = True
        return self

    def _prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        wdf = df.copy()
        wdf["time"] = pd.to_numeric(wdf["35mm"], errors="coerce")
        wdf["iso"] = pd.to_numeric(wdf["ASA/ISO"], errors="coerce")
        wdf = wdf[wdf["time"].notna() & wdf["iso"].notna()].copy()
        wdf = wdf[(wdf["time"] > 0) & (wdf["iso"] > 0)].copy()

        wdf["dil_factor"] = wdf["Dilution"].apply(_parse_dilution_factor)
        wdf = wdf[wdf["dil_factor"].notna() & (wdf["dil_factor"] > 0)].copy()

        # Remove combos where more dilution doesn't mean longer time
        if self.enforce_monotonicity:
            wdf, n_removed = _filter_dilution_monotonicity(
                wdf, ["Developer", "Film", "iso"]
            )
            self.monotonicity_stats["data_rows_removed_dil"] = n_removed

        wdf["log_dil"] = np.log(wdf["dil_factor"])
        wdf["log_time"] = np.log(wdf["time"])

        # film_iso key: groups that share a baseline time
        wdf["film_iso"] = (
            wdf["Film"].astype(str).str.strip()
            + "_"
            + wdf["ASA/ISO"].astype(str).str.strip()
        )

        return wdf

    def _fit_per_group_slopes(self, wdf: pd.DataFrame) -> None:
        """Fit slope for each (developer, film_iso) pair."""
        self.per_filmiso_slopes = {}
        n_rejected = 0
        for (dev, fiso), grp in wdf.groupby(["Developer", "film_iso"]):
            if len(grp) < self.min_obs_per_group:
                continue
            result = _simple_ols(grp["log_dil"].values, grp["log_time"].values,
                                 min_n=self.min_obs_per_group)
            if result is None:
                continue
            # More dilute must → longer time (positive slope)
            if self.enforce_monotonicity and result["slope"] < 0:
                n_rejected += 1
                continue
            self.per_filmiso_slopes[(dev, fiso)] = result
        self.monotonicity_stats["per_group_rejected"] = n_rejected

    def _fit_entity_slopes(self, wdf: pd.DataFrame) -> None:
        """Optimise one dilution slope per developer across all film×ISO combos."""
        self.dev_dilution_slopes = {}
        n_entity_clamped = 0

        for dev, dev_grp in wdf.groupby("Developer"):
            # Must have multiple distinct dilutions
            n_dils = dev_grp["Dilution"].nunique()
            if n_dils < self.min_dilutions:
                continue

            if np.ptp(dev_grp["log_dil"]) < self.min_dil_spread_log:
                continue

            # Per-group slopes for diagnostics
            group_slopes_here = {
                k[1]: v for k, v in self.per_filmiso_slopes.items() if k[0] == dev
            }

            # Filter to film_iso groups with enough observations
            group_counts = dev_grp.groupby("film_iso").size()
            valid_groups = group_counts[group_counts >= self.min_obs_per_group].index
            fit_data = dev_grp[dev_grp["film_iso"].isin(valid_groups)].copy()

            if len(fit_data) < self.min_obs_total:
                continue
            if fit_data["film_iso"].nunique() < self.min_groups_for_entity:
                continue

            # Outlier removal
            if self.outlier_sigma is not None and len(group_slopes_here) >= 3:
                outliers = _outlier_filter_by_slope(group_slopes_here, self.outlier_sigma)
                if outliers:
                    fit_data = fit_data[~fit_data["film_iso"].isin(outliers)].copy()

            if len(fit_data) < self.min_obs_total:
                continue

            result = _fixed_effects_slope(
                fit_data["log_dil"].values,
                fit_data["log_time"].values,
                fit_data["film_iso"].values,
            )
            if result is None:
                continue

            # More dilute must → longer time (positive slope)
            if self.enforce_monotonicity and result["slope"] < 0:
                n_entity_clamped += 1
                continue

            result["per_filmiso_slopes"] = {
                fiso: v["slope"] for fiso, v in group_slopes_here.items()
            }
            result["n_dilutions"] = n_dils
            result["dil_range"] = (
                float(dev_grp["dil_factor"].min()),
                float(dev_grp["dil_factor"].max()),
            )
            result["n_film_isos"] = result.pop("n_groups")
            self.dev_dilution_slopes[dev] = result

        self.monotonicity_stats["entity_rejected"] = n_entity_clamped

    def _fit_global_slope(self, wdf: pd.DataFrame) -> None:
        devs_with = set(self.dev_dilution_slopes.keys())
        pool = wdf[wdf["Developer"].isin(devs_with)] if len(devs_with) >= 5 else wdf

        if len(pool) < 5:
            self.global_slope = 0.30
            self.global_slope_info = {"slope": 0.30, "source": "default"}
            return

        result = _fixed_effects_slope(
            pool["log_dil"].values, pool["log_time"].values,
            pool["film_iso"].values,
        )
        if result and (not self.enforce_monotonicity or result["slope"] >= 0):
            self.global_slope = result["slope"]
            self.global_slope_info = result
        else:
            self.global_slope = 0.30
            self.global_slope_info = {"slope": 0.30, "source": "default"}

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Attach developer dilution-slope features.

        New columns: dev_dilution_slope, dev_dilution_slope_se,
                     dev_dilution_slope_r2, dev_dilution_slope_n_film_isos,
                     log_dil_factor, dev_dilution_slope_pred_log_ratio
        """
        if not self._fitted:
            raise RuntimeError("Call .fit() first")

        out = df.copy()

        out["dil_factor"] = out["Dilution"].apply(_parse_dilution_factor)
        out["log_dil_factor"] = np.where(
            out["dil_factor"].notna() & (out["dil_factor"] > 0),
            np.log(out["dil_factor"].fillna(1).clip(lower=1e-6)),
            np.nan,
        )

        slope_map = {d: v["slope"] for d, v in self.dev_dilution_slopes.items()}
        se_map = {d: v["se"] for d, v in self.dev_dilution_slopes.items()}
        r2_map = {d: v["r2_within"] for d, v in self.dev_dilution_slopes.items()}
        nfi_map = {d: v["n_film_isos"] for d, v in self.dev_dilution_slopes.items()}

        out["dev_dilution_slope"] = out["Developer"].map(slope_map)
        out["dev_dilution_slope_se"] = out["Developer"].map(se_map)
        out["dev_dilution_slope_r2"] = out["Developer"].map(r2_map)
        out["dev_dilution_slope_n_film_isos"] = out["Developer"].map(nfi_map)

        if self.global_slope is not None:
            out["dev_dilution_slope"] = out["dev_dilution_slope"].fillna(self.global_slope)
            if self.global_slope_info:
                out["dev_dilution_slope_se"] = out["dev_dilution_slope_se"].fillna(
                    self.global_slope_info.get("se", float("inf"))
                )

        out["dev_dilution_slope_pred_log_ratio"] = (
            out["dev_dilution_slope"] * out["log_dil_factor"]
        )

        out.drop(columns=["dil_factor"], inplace=True, errors="ignore")
        return out

    def summary(self) -> pd.DataFrame:
        rows = []
        for dev, info in sorted(self.dev_dilution_slopes.items()):
            rows.append({
                "Developer": dev,
                "slope": info["slope"],
                "se": info["se"],
                "r2_within": info["r2_within"],
                "r2_total": info["r2_total"],
                "n_obs": info["n_obs"],
                "n_film_isos": info["n_film_isos"],
                "n_dilutions": info["n_dilutions"],
                "dil_min": info["dil_range"][0],
                "dil_max": info["dil_range"][1],
            })
        return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    csv_path = sys.argv[1] if len(sys.argv) > 1 else "film_data.csv"
    print(f"Loading {csv_path}...")
    raw = pd.read_csv(csv_path)
    raw["35mm"] = pd.to_numeric(raw["35mm"], errors="coerce")
    raw = raw[raw["35mm"].notna() & (raw["35mm"] > 0) & (raw["35mm"] <= 30)].copy()
    print(f"Rows: {len(raw)}")

    # ── Film slopes ──
    film_est = FilmSlopeEstimator()
    film_est.fit(raw)
    fs = film_est.summary()
    print(f"\n{'='*70}")
    print(f"FILM SLOPES: {len(fs)} films, global = {film_est.global_slope:.4f}")
    print(f"  Monotonicity: {film_est.monotonicity_stats}")
    print(f"{'='*70}")
    print(fs.sort_values("n_devs", ascending=False).head(10).to_string(
        index=False, float_format="{:.4f}".format))

    # ── Dev+dil slopes (reuse box ISOs) ──
    dev_est = DevDilSlopeEstimator(box_isos=film_est.box_isos)
    dev_est.fit(raw)
    ds = dev_est.summary()
    print(f"\n{'='*70}")
    print(f"DEV+DIL SLOPES: {len(ds)} combos, global = {dev_est.global_slope:.4f}")
    print(f"  Monotonicity: {dev_est.monotonicity_stats}")
    print(f"{'='*70}")
    print(ds.sort_values("n_films", ascending=False).head(10).to_string(
        index=False, float_format="{:.4f}".format))

    # ── Developer dilution slopes ──
    dil_est = DevDilutionSlopeEstimator()
    dil_est.fit(raw)
    dils = dil_est.summary()
    print(f"\n{'='*70}")
    print(f"DEV DILUTION SLOPES: {len(dils)} developers, global = {dil_est.global_slope:.4f}")
    print(f"  Monotonicity: {dil_est.monotonicity_stats}")
    print(f"{'='*70}")
    print(dils.sort_values("n_film_isos", ascending=False).head(15).to_string(
        index=False, float_format="{:.4f}".format))

    print(f"\nDilution slope distribution:")
    print(f"  Mean:   {dils['slope'].mean():.4f}")
    print(f"  Median: {dils['slope'].median():.4f}")
    print(f"  Std:    {dils['slope'].std():.4f}")
    print(f"  Min:    {dils['slope'].min():.4f}  ({dils.loc[dils['slope'].idxmin(), 'Developer']})")
    print(f"  Max:    {dils['slope'].max():.4f}  ({dils.loc[dils['slope'].idxmax(), 'Developer']})")

    # ── Combined transform ──
    df_out = film_est.transform(raw)
    df_out = dev_est.transform(df_out)
    df_out = dil_est.transform(df_out)
    print(f"\n{'='*70}")
    print("COMBINED TRANSFORM (sample)")
    print(f"{'='*70}")
    cols = ["Film", "Developer", "Dilution", "ASA/ISO", "35mm",
            "film_slope", "dev_dil_slope", "dev_dilution_slope",
            "stops_from_box", "log_dil_factor"]
    print(df_out[cols].dropna().head(15).to_string(index=False, float_format="{:.3f}".format))

    # Coverage
    has_film = df_out["Film"].isin(film_est.film_slopes.keys())
    dd_key = df_out["Developer"].astype(str).str.strip() + "_" + df_out["Dilution"].astype(str).str.strip()
    has_dev_dil = dd_key.isin(dev_est.dev_dil_slopes.keys())
    has_dil = df_out["Developer"].isin(dil_est.dev_dilution_slopes.keys())
    print(f"\nCoverage:")
    print(f"  Film slope (specific):      {has_film.sum():>6} / {len(df_out)} ({has_film.mean()*100:.1f}%)")
    print(f"  Dev+dil slope (specific):   {has_dev_dil.sum():>6} / {len(df_out)} ({has_dev_dil.mean()*100:.1f}%)")
    print(f"  Dev dilution slope (spec):  {has_dil.sum():>6} / {len(df_out)} ({has_dil.mean()*100:.1f}%)")
    print(f"  All three specific:         {(has_film & has_dev_dil & has_dil).sum():>6} ({(has_film & has_dev_dil & has_dil).mean()*100:.1f}%)")
