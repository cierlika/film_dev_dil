# Film Development Time Prediction

Predicts darkroom development times for analog photography film/developer combinations.

## Conversation Summary

### Problem
Photographers need accurate development times for specific film + developer + dilution + ISO combinations. A 1-minute error on a 3-minute development ruins film; the same error on a 25-minute development is negligible. This makes SMAPE (percentage error) the right metric.

### Dataset
~13,000 records from the Massive Dev Chart. After filtering (3–30 min times, removing stand/rotary/high-contrast processes), **11,226 rows** across 243 films and 627 developer+dilution combos.

### What We Built

**Phase 1 — Slope Estimation** (`film_slopes.py`): Quadratic push/pull models. For each film and each developer+dilution, fits `log(time) = α + β₁·stops + β₂·stops²` using demeaned fixed-effects regression. Also fits developer dilution slopes. These capture physics: pushing film requires disproportionately more development time.

**Phase 2 — Matrix Factorization** (`matrix_factorization.py`, `latent_features.py`): ALS-based collaborative filtering on a dev_dil × film@ISO matrix. Decomposes the sparse matrix into latent factors (k=30) that capture developer-film affinities (e.g., T-grain films with tabular developers). We explored multiple matrix configurations — full range, base+push1, top-3 ISOs per film — settling on top-3 ISOs for best CV performance (12.58% SMAPE on observed entries). Film latent vectors use **base ISO only** and propagate to all ISOs of that film, so they capture intrinsic film character independent of push/pull.

**Phase 3 — Data Splitting** (`data_split.py`, `splits.json`): Stratified by film, 20% test holdout + 5-fold CV on the remaining 80%. Frozen to JSON for reproducibility. All transformations fit on training folds only.

**Phase 4 — Aggregate Features** (`aggregate_features.py`): Per-entity statistics (median, quantiles, skewness, kurtosis, count/popularity) at four levels: per Film, per Film@ISO, per dev_dil, per Developer.

**Phase 5 — Baseline Model** (`train_extratrees.py`): ExtraTreesRegressor with 500 trees. 5-fold CV SMAPE: **14.13% ± 0.24%**. Test SMAPE: **13.11%**. Top features: MF predicted time (29.7%), film@ISO aggregates (26.5%), dev_dil aggregates (18.8%), slope predictions (4.0%), stops/push-pull (3.8%).

## Project Structure

```
film_data.csv              Raw dataset (Massive Dev Chart)
splits.json                Frozen train/val/test split indices

data_split.py              Data loading, filtering, stratified splitting
  ├─ prepare_data()        Load CSV, filter 3–30 min, remove unusual processes
  ├─ DataSplitter          Create/save/load stratified splits
  └─ .summary()            Per-fold statistics

film_slopes.py             Physics-based slope estimation
  ├─ FilmSlopeEstimator        Per-film quadratic push/pull slopes
  ├─ DevDilSlopeEstimator      Per-dev+dilution quadratic slopes
  └─ DevDilutionSlopeEstimator Per-developer dilution slopes

matrix_factorization.py    ALS collaborative filtering
  └─ ALSMatrixFactorization    Fit/predict/score on sparse matrix

latent_features.py         Latent vector assignment to rows
  └─ LatentFeatureBuilder      fit(train) → build matrix, run ALS
                               transform(any) → 66 LV columns per row
                               Film vectors from base ISO, propagated

aggregate_features.py      Per-entity summary statistics
  └─ AggregateFeatureBuilder   fit(train) → compute stats per entity
                               transform(any) → 50 feature columns

train_extratrees.py        Baseline model training + evaluation
  ├─ build_features()      Full pipeline: slopes + LV + aggregates
  ├─ 5-fold CV             Per-fold fit/evaluate
  └─ Final test eval       Train on all trainval, score on test
```

## Feature Pipeline (per fold)

```python
# All fitted on training data only
film_est = FilmSlopeEstimator().fit(df_train)
devdil_est = DevDilSlopeEstimator(box_isos=film_est.box_isos).fit(df_train)
dil_est = DevDilutionSlopeEstimator().fit(df_train)
lv = LatentFeatureBuilder(n_factors=30, reg=0.005).fit(df_train)
agg = AggregateFeatureBuilder().fit(df_train)

# Transform any split
df_out = dil_est.transform(devdil_est.transform(film_est.transform(df_any)))
lv_out = lv.transform(df_any)
agg_out = agg.transform(df_any)
full = pd.concat([df_out, lv_out, agg_out], axis=1)  # 147 columns
```

## Feature Groups (136 numeric features after dropping metadata)

| Group | Cols | Description |
|-------|------|-------------|
| Slopes | 19 | Film/devdil/dilution slopes, accelerations, SE, R², predicted log-ratios |
| Latent | 66 | Global mean, biases, 30 dd factors, 30 film factors, MF predictions |
| Aggregates | 50 | 12 stats × 4 entity levels + dil_factor + is_box_iso |
| Original numeric | 3 | box_iso, stops, iso |

## Current Results

| Metric | CV (5-fold) | Test |
|--------|-------------|------|
| SMAPE | 14.13% ± 0.24% | 13.11% |
| MAE | 1.50 ± 0.04 min | 1.39 min |
| MAPE | 15.01% ± 0.21% | 14.36% |
| R² | 0.730 ± 0.016 | 0.706 |
