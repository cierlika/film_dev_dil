"""
Film Development Time Matrix Factorization
===========================================

Two-stage pipeline:
  1. Build a sparse matrix of development times:
       rows = developer + dilution combos (≥5 observations)
       cols = film × push/pull level (-3 to +3 stops from box ISO)
       values = median development time (minutes)

  2. Complete the matrix via ALS (Alternating Least Squares):
       log(time) ≈ μ + row_bias_i + col_bias_j + U_i · V_j

     ALS is the right algorithm here because:
       - 92% of entries are missing → needs native sparse handling
       - Values are positive → log-space keeps predictions positive
       - Biases capture the bulk of variance (dev speed + film base time)
       - Latent factors capture interactions (some devs work better
         with certain films at certain push levels)

Usage:
    python matrix_factorization.py
"""

import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

# ─── Imports from project ────────────────────────────────────────────
from film_slopes import FilmSlopeEstimator


# ═════════════════════════════════════════════════════════════════════
# STEP 1: BUILD THE DEVELOPMENT TIME MATRIX
# ═════════════════════════════════════════════════════════════════════

def build_dev_time_matrix(
    csv_path: str = "film_data.csv",
    min_dd_count: int = 5,
    min_col_count: int = 3,
    max_time: float = 30.0,
    stops_range: tuple[int, int] = (-3, 3),
    stops_tolerance: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build a sparse matrix of median development times.

    Returns
    -------
    matrix : DataFrame
        Rows = dev_dil, Columns = "Film [±N]", Values = median time (min).
        Sorted by row coverage (desc), then columns by (film, stops).
    df_clean : DataFrame
        The filtered long-form data used to build the matrix.
    """
    raw = pd.read_csv(csv_path)
    raw["35mm"] = pd.to_numeric(raw["35mm"], errors="coerce")
    raw = raw[raw["35mm"].notna() & (raw["35mm"] > 0) & (raw["35mm"] <= max_time)].copy()
    raw = raw.reset_index(drop=True)

    # ── 1a. Remove unusual development processes ──
    notes = raw["Notes"].fillna("")
    unusual = notes.str.contains(
        r"Stand development|Semi-stand|rotary|jobo"
        r"|High Contrast|Very High Contrast"
        r"|continuous \(slow",
        case=False,
    )
    df = raw[~unusual].copy()
    n_removed = unusual.sum()

    # ── 1b. Parse ISO, build dev_dil key ──
    df["iso"] = pd.to_numeric(df["ASA/ISO"], errors="coerce")
    df = df[df["iso"].notna() & (df["iso"] > 0)].copy()
    df["dev_dil"] = (
        df["Developer"].astype(str).str.strip()
        + " "
        + df["Dilution"].astype(str).str.strip()
    )

    # ── 1c. Detect box ISOs, compute stops ──
    film_est = FilmSlopeEstimator().fit(raw)
    df["box_iso"] = df["Film"].map(film_est.box_isos)
    df = df[df["box_iso"].notna()].copy()
    df["stops"] = np.log2(df["iso"] / df["box_iso"])

    # ── 1d. Round to integer stops within tolerance, clip to range ──
    df["stops_int"] = df["stops"].round().astype(int)
    lo, hi = stops_range
    df = df[(df["stops_int"] >= lo) & (df["stops_int"] <= hi)].copy()
    df = df[np.abs(df["stops"] - df["stops_int"]) < stops_tolerance].copy()

    # ── 1e. Column labels: "Film [±N]" ──
    df["film_stops"] = (
        df["Film"]
        + " ["
        + df["stops_int"].apply(lambda s: f"{s:+d}" if s != 0 else "0")
        + "]"
    )

    # ── 1f. Keep dev_dils with enough observations ──
    dd_counts = df["dev_dil"].value_counts()
    df = df[df["dev_dil"].isin(dd_counts[dd_counts >= min_dd_count].index)].copy()

    # ── 1g. Pivot: median time per (dev_dil, film_stops) ──
    matrix = df.pivot_table(
        index="dev_dil", columns="film_stops", values="35mm", aggfunc="median"
    )

    # Sort rows by coverage (most observed first)
    matrix = matrix.loc[
        matrix.notna().sum(axis=1).sort_values(ascending=False).index
    ]

    # Sort columns by (film_name, stops)
    def _col_key(col: str) -> tuple[str, int]:
        parts = col.rsplit(" [", 1)
        return (parts[0], int(parts[1].rstrip("]")))

    matrix = matrix[sorted(matrix.columns, key=_col_key)]

    # Drop very sparse columns
    matrix = matrix.loc[:, matrix.notna().sum(axis=0) >= min_col_count]

    print(f"Matrix built: {matrix.shape[0]} dev_dils × {matrix.shape[1]} film_stops")
    print(f"  Unusual process rows removed: {n_removed}")
    print(f"  Filled cells: {matrix.notna().sum().sum():,} / {matrix.size:,} "
          f"({matrix.notna().sum().sum() / matrix.size * 100:.1f}%)")
    print(f"  Time range: {matrix.min().min():.1f} – {matrix.max().max():.1f} min")

    return matrix, df


# ═════════════════════════════════════════════════════════════════════
# STEP 2: ALS MATRIX FACTORIZATION
# ═════════════════════════════════════════════════════════════════════

class ALSMatrixFactorization:
    """
    Biased ALS for sparse matrix completion in log-space.

    Model:
        log(time_ij) ≈ μ + b_i + c_j + U_i · V_j

    where:
        μ         = global mean log-time
        b_i       = row (developer+dilution) bias
        c_j       = column (film×stops) bias
        U_i, V_j  = k-dimensional latent vectors capturing interactions

    The biases alone form a strong baseline (~15% SMAPE).
    Latent factors capture residual structure — e.g., some developers
    pair particularly well or poorly with specific film emulsions.

    Parameters
    ----------
    n_factors : int
        Latent dimension k.
    reg : float
        L2 regularisation per observed entry.
    n_iter : int
        Number of alternating updates.
    n_bias_init : int
        Iterations for bias initialisation before latent fitting.
    seed : int
        Random seed for reproducibility.
    """

    def __init__(
        self,
        n_factors: int = 20,
        reg: float = 0.01,
        n_iter: int = 60,
        n_bias_init: int = 10,
        seed: int = 42,
    ):
        self.n_factors = n_factors
        self.reg = reg
        self.n_iter = n_iter
        self.n_bias_init = n_bias_init
        self.seed = seed

        # Fitted attributes
        self.global_mean: float = 0.0
        self.row_bias: np.ndarray | None = None
        self.col_bias: np.ndarray | None = None
        self.U: np.ndarray | None = None
        self.V: np.ndarray | None = None
        self.losses: list[float] = []

    def fit(self, M_raw: np.ndarray, verbose: bool = True) -> "ALSMatrixFactorization":
        """
        Fit on an (m × n) array with NaN for missing entries.
        All positive values expected (development times in minutes).
        """
        M = np.log(M_raw)
        m, n = M.shape
        k = self.n_factors
        obs = ~np.isnan(M)

        # ── Bias initialisation ──
        self.global_mean = float(np.nanmean(M))
        self.row_bias = np.zeros(m)
        self.col_bias = np.zeros(n)

        for _ in range(self.n_bias_init):
            for i in range(m):
                mask = obs[i, :]
                if mask.any():
                    self.row_bias[i] = np.mean(
                        M[i, mask] - self.global_mean - self.col_bias[mask]
                    )
            for j in range(n):
                mask = obs[:, j]
                if mask.any():
                    self.col_bias[j] = np.mean(
                        M[mask, j] - self.global_mean - self.row_bias[mask]
                    )

        # Residual matrix (observed entries only)
        R = np.zeros_like(M)
        R[obs] = (
            M[obs]
            - self.global_mean
            - self.row_bias[:, None].repeat(n, axis=1)[obs]
            - self.col_bias[None, :].repeat(m, axis=0)[obs]
        )

        # ── Latent factor initialisation ──
        rng = np.random.RandomState(self.seed)
        self.U = rng.randn(m, k) * 0.01
        self.V = rng.randn(n, k) * 0.01
        reg_I = self.reg * np.eye(k)

        self.losses = []
        for iteration in range(self.n_iter):
            # Update U (row factors)
            for i in range(m):
                mask = obs[i, :]
                if not mask.any():
                    continue
                V_obs = self.V[mask, :]
                r_obs = R[i, mask]
                n_obs = mask.sum()
                A = V_obs.T @ V_obs + reg_I * n_obs
                self.U[i, :] = np.linalg.solve(A, V_obs.T @ r_obs)

            # Update V (column factors)
            for j in range(n):
                mask = obs[:, j]
                if not mask.any():
                    continue
                U_obs = self.U[mask, :]
                r_obs = R[mask, j]
                n_obs = mask.sum()
                A = U_obs.T @ U_obs + reg_I * n_obs
                self.V[j, :] = np.linalg.solve(A, U_obs.T @ r_obs)

            # Compute loss
            pred_log = (
                self.global_mean
                + self.row_bias[:, None]
                + self.col_bias[None, :]
                + self.U @ self.V.T
            )
            loss = float(np.mean((M[obs] - pred_log[obs]) ** 2))
            self.losses.append(loss)

            if verbose and (iteration % 10 == 0 or iteration == self.n_iter - 1):
                rmse_log = np.sqrt(loss)
                pred_time = np.exp(pred_log)
                actual_time = np.exp(M)
                mape = float(np.nanmean(
                    np.abs(pred_time[obs] - actual_time[obs]) / actual_time[obs]
                )) * 100
                print(f"  iter {iteration:3d}: RMSE(log)={rmse_log:.4f}  MAPE={mape:.1f}%")

        return self

    def predict(self) -> np.ndarray:
        """Return full predicted matrix in original time-space."""
        log_pred = (
            self.global_mean
            + self.row_bias[:, None]
            + self.col_bias[None, :]
            + self.U @ self.V.T
        )
        return np.exp(log_pred)

    def score(self, M_raw: np.ndarray) -> dict[str, float]:
        """Compute error metrics on observed entries of M_raw."""
        pred = self.predict()
        obs = ~np.isnan(M_raw)
        actual = M_raw[obs]
        predicted = pred[obs]

        smape = float(np.mean(
            2 * np.abs(actual - predicted) / (np.abs(actual) + np.abs(predicted))
        )) * 100
        mape = float(np.mean(np.abs(actual - predicted) / actual)) * 100
        mae = float(np.mean(np.abs(actual - predicted)))
        rmse = float(np.sqrt(np.mean((actual - predicted) ** 2)))
        ss_res = np.sum((actual - predicted) ** 2)
        ss_tot = np.sum((actual - actual.mean()) ** 2)
        r2 = float(1 - ss_res / ss_tot)

        return {"smape": smape, "mape": mape, "mae": mae, "rmse": rmse, "r2": r2}


# ═════════════════════════════════════════════════════════════════════
# STEP 3: CROSS-VALIDATION
# ═════════════════════════════════════════════════════════════════════

def cross_validate(
    M_raw: np.ndarray,
    n_folds: int = 5,
    configs: list[dict] | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """
    k-fold CV on observed entries. Returns a DataFrame of results.

    Parameters
    ----------
    M_raw : (m, n) array with NaN for missing
    configs : list of dicts with keys matching ALSMatrixFactorization params
              plus a 'name' key for labelling. If None, uses defaults.
    """
    obs = ~np.isnan(M_raw)
    obs_indices = list(zip(*np.where(obs)))
    rng = np.random.RandomState(seed)
    rng.shuffle(obs_indices)

    fold_size = len(obs_indices) // n_folds

    if configs is None:
        configs = [
            {"name": "Bias only",    "n_factors": 0},
            {"name": "k=5  λ=0.01",  "n_factors": 5,  "reg": 0.01},
            {"name": "k=10 λ=0.01",  "n_factors": 10, "reg": 0.01},
            {"name": "k=20 λ=0.01",  "n_factors": 20, "reg": 0.01},
            {"name": "k=20 λ=0.005", "n_factors": 20, "reg": 0.005},
        ]

    results = []
    for cfg in configs:
        name = cfg.pop("name", str(cfg))
        fold_metrics = {"smape": [], "mape": [], "mae": []}

        for fold in range(n_folds):
            test_idx = obs_indices[fold * fold_size : (fold + 1) * fold_size]
            train_idx = (
                obs_indices[: fold * fold_size]
                + obs_indices[(fold + 1) * fold_size :]
            )

            # Build train matrix
            M_train = np.full_like(M_raw, np.nan)
            for i, j in train_idx:
                M_train[i, j] = M_raw[i, j]

            # Handle "bias only" as k=0
            n_factors = cfg.get("n_factors", 20)
            if n_factors == 0:
                # Pure bias model
                M_log = np.log(M_train)
                obs_t = ~np.isnan(M_log)
                gm = float(np.nanmean(M_log))
                rb = np.zeros(M_log.shape[0])
                cb = np.zeros(M_log.shape[1])
                for _ in range(10):
                    for i in range(M_log.shape[0]):
                        mask = obs_t[i, :]
                        if mask.any():
                            rb[i] = np.mean(M_log[i, mask] - gm - cb[mask])
                    for j in range(M_log.shape[1]):
                        mask = obs_t[:, j]
                        if mask.any():
                            cb[j] = np.mean(M_log[mask, j] - gm - rb[mask])
                pred = np.exp(gm + rb[:, None] + cb[None, :])
            else:
                als = ALSMatrixFactorization(**cfg, seed=42)
                als.fit(M_train, verbose=False)
                pred = als.predict()

            actual = np.array([M_raw[i, j] for i, j in test_idx])
            predicted = np.array([pred[i, j] for i, j in test_idx])

            fold_metrics["smape"].append(float(np.mean(
                2 * np.abs(actual - predicted) / (np.abs(actual) + np.abs(predicted))
            )) * 100)
            fold_metrics["mape"].append(float(np.mean(
                np.abs(actual - predicted) / actual
            )) * 100)
            fold_metrics["mae"].append(float(np.mean(np.abs(actual - predicted))))

        results.append({
            "model": name,
            "smape": np.mean(fold_metrics["smape"]),
            "mape": np.mean(fold_metrics["mape"]),
            "mae": np.mean(fold_metrics["mae"]),
        })

        # Restore name for clean output
        cfg["name"] = name

    return pd.DataFrame(results)


# ═════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Build matrix
    matrix, df_clean = build_dev_time_matrix()
    M_raw = matrix.values.astype(float)

    # Cross-validate
    print(f"\n{'='*60}")
    print("5-fold cross-validation")
    print(f"{'='*60}")
    cv_results = cross_validate(M_raw)
    print(cv_results.to_string(index=False, float_format="%.2f"))

    # Fit best model on full data
    print(f"\n{'='*60}")
    print("Final model: ALS k=20, λ=0.01")
    print(f"{'='*60}")
    als = ALSMatrixFactorization(n_factors=20, reg=0.01)
    als.fit(M_raw)
    metrics = als.score(M_raw)
    print(f"\nReconstruction metrics:")
    print(f"  SMAPE: {metrics['smape']:.2f}%")
    print(f"  MAPE:  {metrics['mape']:.2f}%")
    print(f"  MAE:   {metrics['mae']:.2f} min")
    print(f"  R²:    {metrics['r2']:.4f}")

    # Show sample predictions vs actuals
    print(f"\n{'='*60}")
    print("Sample predictions (observed cells)")
    print(f"{'='*60}")
    pred_matrix = pd.DataFrame(
        als.predict(), index=matrix.index, columns=matrix.columns
    )

    # Pick a well-known dev_dil and show a few films
    for dd in ["D-76 stock", "Rodinal 1+50", "HC-110 B"]:
        if dd not in matrix.index:
            continue
        row_actual = matrix.loc[dd]
        row_pred = pred_matrix.loc[dd]
        filled = row_actual.dropna()
        if len(filled) == 0:
            continue
        sample = filled.sample(min(6, len(filled)), random_state=42).sort_index()
        print(f"\n  {dd}:")
        for col in sample.index:
            a = sample[col]
            p = row_pred[col]
            err = abs(a - p) / a * 100
            print(f"    {col:40s}  actual={a:5.1f}  pred={p:5.1f}  err={err:4.1f}%")

    # Show some imputed (missing) values
    print(f"\n{'='*60}")
    print("Imputed values (previously missing cells)")
    print(f"{'='*60}")
    missing_mask = matrix.isna()
    for dd in ["D-76 stock", "Rodinal 1+50"]:
        if dd not in matrix.index:
            continue
        missing_cols = missing_mask.loc[dd]
        missing_films = missing_cols[missing_cols].index.tolist()
        if not missing_films:
            continue
        # Show a few imputed values
        sample_missing = missing_films[:8]
        print(f"\n  {dd}:")
        for col in sample_missing:
            p = pred_matrix.loc[dd, col]
            print(f"    {col:40s}  imputed={p:5.1f} min")
