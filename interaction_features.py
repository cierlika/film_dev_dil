"""
interaction_features.py
------------------------
Arithmetic interaction features computed from the fully-featurized DataFrame
(after FilmSlopeEstimator, LatentValueBuilder, AggregateFeatureBuilder, and
metadata builders have all been applied).

Leakage analysis
----------------
ALL features below are safe:

Temperature features
  • temp_celsius is a raw row-level measurement – no leakage.
  • temp_deviation, temp_factor, temp_deviation_sq are deterministic
    transformations of temp_celsius.
  • temp_dev_factor multiplies temp_factor by `stops` – both are row-level.

Slope interactions
  • film_slope and dev_dil_slope are computed by FilmSlopeEstimator, which is
    fitted on training folds only and applied to test via transform().  Safe.
  • avg_slope, slope_delta, slope_times_stops, log_film_slope,
    log_dev_dil_slope are deterministic transforms of those fitted slopes.

ISO / dilution interactions
  • log_iso is a transform of box_iso (from FilmSlopeEstimator, training-only
    fit). Safe.
  • log_dil_times_stops multiplies two row-level features. Safe.

Aggregate interactions
  • dd_agg_median, film_agg_median, film_iso_agg_median are produced by
    AggregateFeatureBuilder, fitted on training folds only. Safe.
  • dd_to_film_median_ratio, film_iso_ratio, combo_popularity_log are
    purely arithmetic combinations of those fitted aggregates.
  • lv_pred_time is from LatentValueBuilder (training-only fit). Safe.

No target values are used anywhere in this file.
"""

import numpy as np
import pandas as pd


# ── Feature recipes ───────────────────────────────────────────────────────────
# Each recipe is a (output_name, lambda_fn) pair.
# The lambda receives the full featurized DataFrame and returns a pd.Series.
# Missing inputs produce NaN gracefully; trees handle NaN natively.

def _safe_div(a: pd.Series, b: pd.Series, fill: float = np.nan) -> pd.Series:
    """Element-wise a/b, replacing zeros-in-denominator with *fill*."""
    with np.errstate(divide="ignore", invalid="ignore"):
        result = a / b.replace(0, np.nan)
    return result.fillna(fill)


def _safe_log(s: pd.Series, offset: float = 1e-8) -> pd.Series:
    return np.log(s.clip(lower=offset))


INTERACTION_RECIPES: list[tuple[str, callable]] = [

    # ── Temperature ───────────────────────────────────────────────────────────
    (
        "temp_deviation",
        lambda df: df["temp_celsius"] - 20.0,
    ),
    (
        "temp_deviation_sq",
        lambda df: (df["temp_celsius"] - 20.0) ** 2,
    ),
    (
        "temp_factor",
        # Arrhenius approximation: development rate ∝ 2^((T-20)/10)
        # temp_factor > 1 → faster at higher temperature
        lambda df: 2.0 ** ((df["temp_celsius"] - 20.0) / 10.0),
    ),
    (
        "temp_dev_factor",
        # Temperature-push interaction: how much the temp shift compounds with push
        lambda df: (2.0 ** ((df["temp_celsius"] - 20.0) / 10.0)) * df["stops"],
    ),

    # ── Slope interactions ────────────────────────────────────────────────────
    (
        "avg_slope",
        lambda df: (df["film_slope"] + df["dev_dil_slope"]) / 2.0,
    ),
    (
        "slope_delta",
        lambda df: df["film_slope"] - df["dev_dil_slope"],
    ),
    (
        "slope_times_stops",
        lambda df: ((df["film_slope"] + df["dev_dil_slope"]) / 2.0) * df["stops"],
    ),
    (
        "log_film_slope",
        lambda df: _safe_log(df["film_slope"]),
    ),
    (
        "log_dev_dil_slope",
        lambda df: _safe_log(df["dev_dil_slope"]),
    ),

    # ── ISO / dilution ────────────────────────────────────────────────────────
    (
        "log_iso",
        lambda df: _safe_log(df["box_iso"]),
    ),
    (
        "log_dil_times_stops",
        lambda df: _safe_log(df["dilution_factor"]) * df["stops"],
    ),

    # ── Aggregate interactions ────────────────────────────────────────────────
    (
        "dd_to_film_median_ratio",
        # Developer speed relative to film – captures how "aggressive" the
        # developer is for this particular film stock.
        lambda df: _safe_div(df["dd_agg_median"], df["film_agg_median"]),
    ),
    (
        "film_iso_ratio",
        # ISO-adjusted film median vs raw film median – captures ISO push penalty
        lambda df: _safe_div(df["film_iso_agg_median"], df["film_agg_median"]),
    ),
    (
        "combo_popularity_log",
        # Geometric mean of film and developer log-counts – proxy for how
        # well-represented this combo is in the training data.
        lambda df: (
            _safe_log(df["film_agg_count"]) + _safe_log(df["dd_agg_count"])
        ) / 2.0,
    ),
    (
        "lv_to_film_iso_ratio",
        # Matrix-factorisation prediction vs ISO-specific aggregate –
        # how much MF adds over the simple ISO aggregate.
        lambda df: _safe_div(df["lv_pred_time"], df["film_iso_agg_median"]),
    ),
    (
        "stops_times_dd_median",
        # Push/pull scaled by typical developer speed
        lambda df: df["stops"] * df["dd_agg_median"],
    ),
    (
        "stops_times_film_median",
        lambda df: df["stops"] * df["film_agg_median"],
    ),
]

# Columns required from the full featurized DataFrame
_REQUIRED_COLS = {
    "temp_celsius", "stops", "film_slope", "dev_dil_slope",
    "box_iso", "dilution_factor",
    "dd_agg_median", "film_agg_median", "film_iso_agg_median",
    "film_agg_count", "dd_agg_count", "lv_pred_time",
}


class InteractionFeatureBuilder:
    """
    Computes arithmetic interaction features from the fully-featurized DataFrame.

    Usage (identical API to the other builders)::

        interaction_builder = InteractionFeatureBuilder()
        # No fit() needed – all operations are leakage-free arithmetic.
        ixn_train = interaction_builder.transform(full_train)
        ixn_test  = interaction_builder.transform(full_test)

    The output DataFrame shares the same index as the input and contains only
    the interaction columns (one per recipe in INTERACTION_RECIPES).
    """

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Returns a NEW DataFrame (same index) with one column per recipe.
        Input columns that are missing produce NaN in the corresponding output.
        """
        out = pd.DataFrame(index=df.index)

        missing = _REQUIRED_COLS - set(df.columns)
        if missing:
            import warnings
            warnings.warn(
                f"InteractionFeatureBuilder: {len(missing)} expected columns are "
                f"missing from the input DataFrame and will produce NaN features: "
                f"{sorted(missing)}",
                stacklevel=2,
            )

        for name, recipe in INTERACTION_RECIPES:
            try:
                out[name] = recipe(df)
            except (KeyError, TypeError):
                out[name] = np.nan

        return out.astype(float)
