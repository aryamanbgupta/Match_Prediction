"""
Calibration Module for CricML Match Prediction System.

Provides both ball-level and match-level probability calibration,
plus diagnostic tools (ECE, reliability diagrams, bootstrap CIs).

Usage:
    # Match-level Platt scaling with LOOCV
    from calibration import PlattCalibrator
    cal = PlattCalibrator()
    calibrated = cal.fit_loocv(raw_probs, actual_outcomes)

    # Ball-level diagnostics
    from calibration import BallLevelCalibrationDiagnostics
    diag = BallLevelCalibrationDiagnostics(model, test_data_path)
    diag.print_summary()

    # ECE computation
    from calibration import compute_ece
    ece = compute_ece(predicted, actual, n_bins=10)
"""

import json
import pickle
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path


# ---------------------------------------------------------------------------
# ECE & Reliability Diagram Utilities
# ---------------------------------------------------------------------------

def compute_ece(predicted_probs: np.ndarray, actual_outcomes: np.ndarray,
                n_bins: int = 10, strategy: str = 'uniform') -> float:
    """Compute Expected Calibration Error.

    Args:
        predicted_probs: shape (n,) predicted probabilities in [0, 1].
        actual_outcomes: shape (n,) binary outcomes (0 or 1).
        n_bins: number of bins.
        strategy: 'uniform' (equal-width) or 'quantile' (equal-mass).

    Returns:
        ECE value (lower is better, 0 = perfect calibration).
    """
    predicted_probs = np.asarray(predicted_probs, dtype=float)
    actual_outcomes = np.asarray(actual_outcomes, dtype=float)
    n = len(predicted_probs)
    if n == 0:
        return 0.0

    bin_edges = _get_bin_edges(predicted_probs, n_bins, strategy)
    ece = 0.0
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (predicted_probs >= lo) & (predicted_probs < hi)
        # Include right edge for last bin
        if hi == bin_edges[-1]:
            mask = mask | (predicted_probs == hi)
        count = mask.sum()
        if count == 0:
            continue
        avg_pred = predicted_probs[mask].mean()
        avg_actual = actual_outcomes[mask].mean()
        ece += (count / n) * abs(avg_pred - avg_actual)
    return float(ece)


def compute_mce(predicted_probs: np.ndarray, actual_outcomes: np.ndarray,
                n_bins: int = 10, strategy: str = 'uniform') -> float:
    """Compute Maximum Calibration Error (worst-case bin)."""
    predicted_probs = np.asarray(predicted_probs, dtype=float)
    actual_outcomes = np.asarray(actual_outcomes, dtype=float)
    if len(predicted_probs) == 0:
        return 0.0

    bin_edges = _get_bin_edges(predicted_probs, n_bins, strategy)
    mce = 0.0
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (predicted_probs >= lo) & (predicted_probs < hi)
        if hi == bin_edges[-1]:
            mask = mask | (predicted_probs == hi)
        count = mask.sum()
        if count == 0:
            continue
        avg_pred = predicted_probs[mask].mean()
        avg_actual = actual_outcomes[mask].mean()
        mce = max(mce, abs(avg_pred - avg_actual))
    return float(mce)


def reliability_diagram_data(predicted_probs: np.ndarray, actual_outcomes: np.ndarray,
                             n_bins: int = 10, strategy: str = 'uniform') -> List[Dict]:
    """Compute data for reliability diagram.

    Returns list of dicts, each with:
        bin_lower, bin_upper, bin_center, avg_predicted, avg_actual, count,
        ci_lower, ci_upper (Wilson 95% CI on avg_actual).
    """
    predicted_probs = np.asarray(predicted_probs, dtype=float)
    actual_outcomes = np.asarray(actual_outcomes, dtype=float)
    if len(predicted_probs) == 0:
        return []

    bin_edges = _get_bin_edges(predicted_probs, n_bins, strategy)
    bins_data = []
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (predicted_probs >= lo) & (predicted_probs < hi)
        if hi == bin_edges[-1]:
            mask = mask | (predicted_probs == hi)
        count = int(mask.sum())
        if count == 0:
            continue
        avg_pred = float(predicted_probs[mask].mean())
        avg_actual = float(actual_outcomes[mask].mean())
        ci_lo, ci_hi = _wilson_ci(avg_actual, count)
        bins_data.append({
            'bin_lower': float(lo),
            'bin_upper': float(hi),
            'bin_center': float((lo + hi) / 2),
            'avg_predicted': avg_pred,
            'avg_actual': avg_actual,
            'count': count,
            'ci_lower': ci_lo,
            'ci_upper': ci_hi,
        })
    return bins_data


def bootstrap_ece(predicted_probs: np.ndarray, actual_outcomes: np.ndarray,
                  n_bootstrap: int = 1000, n_bins: int = 10,
                  strategy: str = 'uniform', confidence: float = 0.95) -> Dict:
    """Bootstrap confidence interval for ECE.

    Returns: {'ece': float, 'ci_lower': float, 'ci_upper': float, 'std': float}
    """
    predicted_probs = np.asarray(predicted_probs, dtype=float)
    actual_outcomes = np.asarray(actual_outcomes, dtype=float)
    n = len(predicted_probs)

    point_ece = compute_ece(predicted_probs, actual_outcomes, n_bins, strategy)
    if n < 5:
        return {'ece': point_ece, 'ci_lower': 0.0, 'ci_upper': 1.0, 'std': float('nan')}

    rng = np.random.default_rng(42)
    ece_samples = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        ece_samples[i] = compute_ece(predicted_probs[idx], actual_outcomes[idx],
                                     n_bins, strategy)

    alpha = 1 - confidence
    return {
        'ece': point_ece,
        'ci_lower': float(np.percentile(ece_samples, 100 * alpha / 2)),
        'ci_upper': float(np.percentile(ece_samples, 100 * (1 - alpha / 2))),
        'std': float(np.std(ece_samples)),
    }


def _get_bin_edges(values: np.ndarray, n_bins: int, strategy: str) -> np.ndarray:
    """Return bin edges for equal-width or equal-mass binning."""
    if strategy == 'quantile':
        percentiles = np.linspace(0, 100, n_bins + 1)
        edges = np.percentile(values, percentiles)
        edges = np.unique(edges)  # deduplicate
        if len(edges) < 2:
            edges = np.array([values.min(), values.max()])
    else:  # uniform
        edges = np.linspace(0, 1, n_bins + 1)
    return edges


def _wilson_ci(p: float, n: int, z: float = 1.96) -> Tuple[float, float]:
    """Wilson score confidence interval for a binomial proportion."""
    if n == 0:
        return (0.0, 1.0)
    denom = 1 + z ** 2 / n
    center = (p + z ** 2 / (2 * n)) / denom
    spread = z * np.sqrt((p * (1 - p) + z ** 2 / (4 * n)) / n) / denom
    return (max(0.0, float(center - spread)), min(1.0, float(center + spread)))


# ---------------------------------------------------------------------------
# Platt Calibrator (Match-Level)
# ---------------------------------------------------------------------------

class PlattCalibrator:
    """Match-level Platt scaling calibrator (2-parameter logistic).

    Fits: calibrated_p = sigmoid(a * logit(p) + b)
    where a (slope) and b (intercept) are learned from data.

    Safe for small datasets (44 matches) — only 2 parameters.
    """

    def __init__(self):
        self.a = 1.0   # slope (identity by default)
        self.b = 0.0   # intercept
        self._fitted = False

    def fit(self, predicted_probs: np.ndarray, actual_outcomes: np.ndarray) -> None:
        """Fit Platt scaling parameters by minimizing negative log-likelihood.

        Args:
            predicted_probs: shape (n,) raw probabilities in (0, 1).
            actual_outcomes: shape (n,) binary outcomes (0 or 1).
        """
        from scipy.optimize import minimize
        from scipy.special import logit as sp_logit

        p = np.clip(np.asarray(predicted_probs, dtype=float), 1e-6, 1 - 1e-6)
        y = np.asarray(actual_outcomes, dtype=float)
        z = sp_logit(p)

        def nll(params):
            a, b = params
            logits = a * z + b
            # Numerically stable sigmoid + NLL
            logits_clipped = np.clip(logits, -30, 30)
            sig = 1.0 / (1.0 + np.exp(-logits_clipped))
            sig = np.clip(sig, 1e-15, 1 - 1e-15)
            return -np.sum(y * np.log(sig) + (1 - y) * np.log(1 - sig))

        result = minimize(nll, x0=[1.0, 0.0], method='L-BFGS-B')
        self.a, self.b = result.x
        self._fitted = True

    def predict(self, predicted_probs: np.ndarray) -> np.ndarray:
        """Apply Platt scaling to raw probabilities.

        Returns calibrated probabilities.
        """
        from scipy.special import logit as sp_logit, expit

        p = np.clip(np.asarray(predicted_probs, dtype=float), 1e-6, 1 - 1e-6)
        z = sp_logit(p)
        calibrated = expit(self.a * z + self.b)
        return np.asarray(calibrated, dtype=float)

    def fit_loocv(self, predicted_probs: np.ndarray,
                  actual_outcomes: np.ndarray) -> np.ndarray:
        """Leave-one-out cross-validated Platt calibration.

        For each sample i: train on all j != i, predict for i.
        Returns array of calibrated probabilities (same shape as input).
        """
        p = np.asarray(predicted_probs, dtype=float)
        y = np.asarray(actual_outcomes, dtype=float)
        n = len(p)
        calibrated = np.empty(n)

        for i in range(n):
            mask = np.ones(n, dtype=bool)
            mask[i] = False
            temp = PlattCalibrator()
            temp.fit(p[mask], y[mask])
            calibrated[i] = temp.predict(p[i:i + 1])[0]

        # Also fit on full data and store params for future use
        self.fit(p, y)
        return calibrated

    def save(self, path: str) -> None:
        """Save calibrator parameters to JSON."""
        data = {'a': float(self.a), 'b': float(self.b), 'fitted': self._fitted}
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str) -> 'PlattCalibrator':
        """Load calibrator from JSON."""
        with open(path, 'r') as f:
            data = json.load(f)
        cal = cls()
        cal.a = data['a']
        cal.b = data['b']
        cal._fitted = data.get('fitted', True)
        return cal

    def __repr__(self) -> str:
        status = "fitted" if self._fitted else "unfitted"
        return f"PlattCalibrator(a={self.a:.4f}, b={self.b:.4f}, {status})"


# ---------------------------------------------------------------------------
# Isotonic Calibrator (Match-Level)
# ---------------------------------------------------------------------------

class IsotonicCalibrator:
    """Match-level isotonic regression calibrator.

    Non-parametric monotonic step function. More flexible than Platt
    but requires more data to avoid overfitting.
    """

    MIN_RECOMMENDED_SAMPLES = 200

    def __init__(self):
        self._ir = None
        self._fitted = False

    def fit(self, predicted_probs: np.ndarray, actual_outcomes: np.ndarray) -> None:
        """Fit isotonic regression.

        Warns if sample size < MIN_RECOMMENDED_SAMPLES.
        """
        from sklearn.isotonic import IsotonicRegression

        p = np.asarray(predicted_probs, dtype=float)
        y = np.asarray(actual_outcomes, dtype=float)

        if len(p) < self.MIN_RECOMMENDED_SAMPLES:
            print(f"  Warning: Isotonic regression with {len(p)} samples "
                  f"(recommended >= {self.MIN_RECOMMENDED_SAMPLES}). "
                  f"Risk of overfitting. Consider PlattCalibrator instead.")

        self._ir = IsotonicRegression(y_min=0.01, y_max=0.99, out_of_bounds='clip')
        self._ir.fit(p, y)
        self._fitted = True

    def predict(self, predicted_probs: np.ndarray) -> np.ndarray:
        """Apply isotonic calibration."""
        if not self._fitted:
            raise RuntimeError("IsotonicCalibrator has not been fitted.")
        p = np.asarray(predicted_probs, dtype=float)
        return self._ir.predict(p)

    def fit_loocv(self, predicted_probs: np.ndarray,
                  actual_outcomes: np.ndarray) -> np.ndarray:
        """Leave-one-out cross-validated isotonic calibration."""
        p = np.asarray(predicted_probs, dtype=float)
        y = np.asarray(actual_outcomes, dtype=float)
        n = len(p)
        calibrated = np.empty(n)

        for i in range(n):
            mask = np.ones(n, dtype=bool)
            mask[i] = False
            temp = IsotonicCalibrator()
            temp.fit(p[mask], y[mask])
            calibrated[i] = temp.predict(p[i:i + 1])[0]

        # Fit on full data for future use
        self.fit(p, y)
        return calibrated

    def save(self, path: str) -> None:
        """Save calibrator to pickle."""
        with open(path, 'wb') as f:
            pickle.dump({'ir': self._ir, 'fitted': self._fitted}, f)

    @classmethod
    def load(cls, path: str) -> 'IsotonicCalibrator':
        """Load calibrator from pickle."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        cal = cls()
        cal._ir = data['ir']
        cal._fitted = data.get('fitted', True)
        return cal

    def __repr__(self) -> str:
        status = "fitted" if self._fitted else "unfitted"
        return f"IsotonicCalibrator({status})"


# ---------------------------------------------------------------------------
# Ball-Level Calibrator (Per-Class Isotonic Regression)
# ---------------------------------------------------------------------------

class PriorCorrectionCalibrator:
    """Undo `balanced` class-weighted training at inference time (E5, 2026-06-09).

    xgboost_v2.py trains with sklearn `compute_class_weight('balanced')`
    sample weights, so the booster approximates the *weighted* posterior
    p_w(c|x) ∝ w_c · p(c|x) with w_c = n / (k · n_c). The sim's
    XGBoostModelV2 samples these tilted probabilities raw, which is the
    root cause of the systematic tail-event overshoot documented in
    reports/e2_prop_fair_baselines.md (P(wicket) ≈ 2× actual per ball,
    boundary classes ≈ +0.05 absolute each; see
    reports/e5_teacher_forced_bias.md).

    The correction is the standard prior re-weighting:
        p(c|x) ∝ p_w(c|x) / w_c ∝ p_w(c|x) · n_c
    Stateless given the train-split class frequencies; no retraining,
    no extra fit data. Same `calibrate_probs` interface as
    BallLevelCalibrator so it drops into the existing wrapper hook.
    """

    # train-split class frequencies (dot, one, two, four, six, wicket),
    # computed from data/xgb_data_v3/cricket_data_v3_train.parquet.
    DEFAULT_PRIORS = (0.303601, 0.413227, 0.076086, 0.107686, 0.045362,
                      0.054037)

    def __init__(self, class_priors=None):
        p = np.asarray(class_priors if class_priors is not None
                       else self.DEFAULT_PRIORS, dtype=float)
        self._priors = p / p.sum()

    def calibrate_probs(self, raw_probs: np.ndarray) -> np.ndarray:
        single = raw_probs.ndim == 1
        probs = raw_probs.reshape(1, -1) if single else raw_probs
        corrected = probs * self._priors
        corrected = corrected / np.maximum(
            corrected.sum(axis=1, keepdims=True), 1e-12)
        return corrected[0] if single else corrected


class VectorScalingCalibrator:
    """Per-class multiplicative correction fit on validation (E5, 2026-06-09).

    The theoretical prior correction (PriorCorrectionCalibrator)
    over-corrects: the early-stopped booster does not reach the full
    `balanced`-weight tilt. This calibrator estimates the *actual* tilt
    empirically: find a 6-vector v such that the corrected probabilities'
    marginal matches the validation class frequencies, via fixed-point
    iterative scaling
        v_c ← v_c · (actual_freq_c / corrected_pred_freq_c).
    6 parameters fit on ~120k validation balls; fit on probabilities
    produced under the SIM's input distribution (venue_encoded = 0) so the
    correction matches deployment. Same `calibrate_probs` interface.
    """

    def __init__(self, weights=None):
        self._v = None if weights is None else np.asarray(weights, float)

    def fit(self, raw_probs: np.ndarray, labels: np.ndarray,
            n_iter: int = 50, tol: float = 1e-8):
        raw_probs = np.asarray(raw_probs, float)
        k = raw_probs.shape[1]
        actual = np.bincount(np.asarray(labels), minlength=k) / len(labels)
        v = np.ones(k)
        for _ in range(n_iter):
            corr = raw_probs * v
            corr /= corr.sum(axis=1, keepdims=True)
            pred = corr.mean(axis=0)
            ratio = actual / np.maximum(pred, 1e-12)
            v = v * ratio
            v = v / v.sum()
            if np.max(np.abs(ratio - 1)) < tol:
                break
        self._v = v
        return self

    def calibrate_probs(self, raw_probs: np.ndarray) -> np.ndarray:
        single = raw_probs.ndim == 1
        probs = raw_probs.reshape(1, -1) if single else raw_probs
        corrected = probs * self._v
        corrected = corrected / np.maximum(
            corrected.sum(axis=1, keepdims=True), 1e-12)
        return corrected[0] if single else corrected


def _fit_scaling_vector(raw_probs: np.ndarray, labels: np.ndarray,
                        n_iter: int = 50, tol: float = 1e-8) -> np.ndarray:
    """Iterative marginal-matching scaling vector (same fixed point as
    VectorScalingCalibrator.fit), factored out so the phase-conditional
    calibrator can reuse it per phase bucket without touching the landed
    VectorScalingCalibrator code path."""
    raw_probs = np.asarray(raw_probs, float)
    k = raw_probs.shape[1]
    actual = np.bincount(np.asarray(labels), minlength=k) / len(labels)
    v = np.ones(k)
    for _ in range(n_iter):
        corr = raw_probs * v
        corr /= corr.sum(axis=1, keepdims=True)
        pred = corr.mean(axis=0)
        ratio = actual / np.maximum(pred, 1e-12)
        v = v * ratio
        v = v / v.sum()
        if np.max(np.abs(ratio - 1)) < tol:
            break
    return v


class PhaseVectorScalingCalibrator:
    """Phase-conditional vector scaling (A8, 2026-07-11; follow-up to E5).

    Three independent 6-vectors — powerplay / middle / death — each fit by
    the same iterative marginal-matching as `VectorScalingCalibrator` but on
    the validation balls of one phase only. E5 (`reports/e5_class_weight_fix.md`)
    found the single global 6-vector fixes *marginal* class rates yet
    under-corrects boundary-heavy contexts, because the `balanced`
    class-weight tilt is not uniform across phases: count/tail families
    (`bowler_wkts`, `pp_total`) got fixed while boundary-*clustering* families
    (`first_over`, `highest_over`) regressed. Per-phase scaling is the filed
    next step.

    Phase buckets follow the sim's own definition
    (`XGBoostModelV2.extract_features`): pp = balls < 36, mid = 36<=balls<96,
    death = balls >= 96.

    Interface: `calibrate_probs(raw_probs, phase=...)`. `phase_aware = True`
    signals the sim wrapper (`XGBoostModelV2.predict_next_ball`) to pass the
    current ball's phase. If `phase` is None/unknown it falls back to a global
    vector so it degrades gracefully to the E5 behaviour rather than crashing.
    """

    phase_aware = True
    PHASES = ('pp', 'mid', 'death')

    def __init__(self, weights=None, global_weights=None):
        # weights: {phase: 6-vector}; global_weights: 6-vector fallback.
        self._v = ({} if weights is None
                   else {k: np.asarray(w, float) for k, w in weights.items()})
        self._global = (None if global_weights is None
                        else np.asarray(global_weights, float))

    def fit_phase(self, phase: str, raw_probs: np.ndarray, labels: np.ndarray,
                  n_iter: int = 50, tol: float = 1e-8) -> np.ndarray:
        v = _fit_scaling_vector(raw_probs, labels, n_iter, tol)
        self._v[phase] = v
        return v

    def set_global(self, raw_probs: np.ndarray, labels: np.ndarray,
                   n_iter: int = 50, tol: float = 1e-8) -> np.ndarray:
        self._global = _fit_scaling_vector(raw_probs, labels, n_iter, tol)
        return self._global

    def _vector_for(self, phase):
        v = self._v.get(phase) if phase is not None else None
        if v is None:
            v = (self._global if self._global is not None
                 else next(iter(self._v.values())))
        return v

    def calibrate_probs(self, raw_probs: np.ndarray, phase=None) -> np.ndarray:
        single = raw_probs.ndim == 1
        probs = raw_probs.reshape(1, -1) if single else raw_probs
        v = self._vector_for(phase)
        corrected = probs * v
        corrected = corrected / np.maximum(
            corrected.sum(axis=1, keepdims=True), 1e-12)
        return corrected[0] if single else corrected


class BallLevelCalibrator:
    """Per-class isotonic regression for ball-level 6-class predictions.

    Wraps 6 independent IsotonicRegression instances (one per outcome class).
    Fitted on validation data (hundreds of thousands of balls).
    """

    CLASS_NAMES = ['dot', 'one', 'two', 'four', 'six', 'wicket']

    def __init__(self):
        self._calibrators = {}  # class_idx -> IsotonicRegression
        self._fitted = False

    def fit(self, raw_probs: np.ndarray, true_labels: np.ndarray) -> None:
        """Fit per-class isotonic regression.

        Args:
            raw_probs: shape (n_balls, 6) predicted probabilities from model.
            true_labels: shape (n_balls,) integer class labels (0-5).
        """
        from sklearn.isotonic import IsotonicRegression

        raw_probs = np.asarray(raw_probs)
        true_labels = np.asarray(true_labels)
        n_classes = raw_probs.shape[1]

        for k in range(n_classes):
            binary_labels = (true_labels == k).astype(float)
            ir = IsotonicRegression(y_min=1e-4, y_max=1.0 - 1e-4, out_of_bounds='clip')
            ir.fit(raw_probs[:, k], binary_labels)
            self._calibrators[k] = ir
            name = self.CLASS_NAMES[k] if k < len(self.CLASS_NAMES) else f"class_{k}"
            print(f"  Fitted ball-level calibrator for class {k} ({name}): "
                  f"{len(binary_labels)} samples, {binary_labels.mean():.3f} base rate")

        self._fitted = True

    def calibrate_probs(self, raw_probs: np.ndarray) -> np.ndarray:
        """Transform 6-class probability array and re-normalize.

        Args:
            raw_probs: shape (6,) or (n, 6) raw model probabilities.

        Returns:
            Calibrated probabilities (same shape), re-normalized to sum to 1.
        """
        if not self._fitted:
            raise RuntimeError("BallLevelCalibrator has not been fitted.")

        single = raw_probs.ndim == 1
        if single:
            raw_probs = raw_probs.reshape(1, -1)

        n_classes = raw_probs.shape[1]
        calibrated = np.empty_like(raw_probs)
        for k in range(n_classes):
            if k not in self._calibrators:
                calibrated[:, k] = raw_probs[:, k]  # Pass through if no calibrator
                continue
            calibrated[:, k] = self._calibrators[k].predict(raw_probs[:, k])

        # Re-normalize each row to sum to 1
        row_sums = calibrated.sum(axis=1, keepdims=True)
        row_sums = np.maximum(row_sums, 1e-10)
        calibrated = calibrated / row_sums

        if single:
            return calibrated[0]
        return calibrated

    def save(self, path: str) -> None:
        """Save to pickle."""
        with open(path, 'wb') as f:
            pickle.dump({'calibrators': self._calibrators, 'fitted': self._fitted}, f)

    @classmethod
    def load(cls, path: str) -> 'BallLevelCalibrator':
        """Load from pickle."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        cal = cls()
        cal._calibrators = data['calibrators']
        cal._fitted = data.get('fitted', True)
        return cal

    def __repr__(self) -> str:
        status = "fitted" if self._fitted else "unfitted"
        n = len(self._calibrators)
        return f"BallLevelCalibrator({n} classes, {status})"


# ---------------------------------------------------------------------------
# Ball-Level Diagnostics (Read-Only)
# ---------------------------------------------------------------------------

def _apply_encoders_to_df(df, feature_columns: List[str],
                          encoder_dir: Optional[str] = None) -> None:
    """Apply label encoding to a DataFrame in-place for columns that the
    model expects but the raw parquet doesn't contain.

    Encoding map:
        batter_id  → batter_encoded    (via batter_encoder_v3.pkl)
        bowler_id  → bowler_encoded    (via bowler_encoder_v3.pkl)
        venue      → venue_encoded     (via venue_encoder_v3.pkl)
        matchup_type → matchup_type_encoded (via matchup_encoder_v3.pkl)
    """
    import joblib
    from pathlib import Path

    if encoder_dir is None:
        return

    enc_dir = Path(encoder_dir)
    encode_map = {
        'batter_encoded':       ('batter_id',   'batter_encoder'),
        'bowler_encoded':       ('bowler_id',   'bowler_encoder'),
        'venue_encoded':        ('venue',       'venue_encoder'),
        'matchup_type_encoded': ('matchup_type', 'matchup_encoder'),
    }

    for encoded_col, (source_col, encoder_prefix) in encode_map.items():
        if encoded_col in feature_columns and encoded_col not in df.columns:
            if source_col not in df.columns:
                raise KeyError(f"Source column '{source_col}' not in DataFrame")

            # Try to find saved encoder
            enc_files = list(enc_dir.glob(f'{encoder_prefix}*.pkl'))
            raw = df[source_col].astype(str)
            if enc_files:
                le = joblib.load(enc_files[0])
                # Vectorized safe transform: unknown labels get -1
                class_to_int = {c: i for i, c in enumerate(le.classes_)}
                df[encoded_col] = raw.map(class_to_int).fillna(-1).astype(int)
            else:
                # No saved encoder — fit from data (e.g. venue_encoder)
                from sklearn.preprocessing import LabelEncoder as LE
                le = LE()
                le.fit(raw)
                df[encoded_col] = le.transform(raw)


class BallLevelCalibrationDiagnostics:
    """Read-only analysis of XGBoost's 6-class ball predictions.

    Loads the model and test parquet, runs predict_proba, and computes
    per-class ECE and reliability data.
    """

    CLASS_NAMES = ['dot', 'one', 'two', 'four', 'six', 'wicket']
    # XGBoost class remapping: original {0,1,2,4,6,7} → sequential {0,1,2,3,4,5}
    OUTCOME_REMAP = {0: 0, 1: 1, 2: 2, 4: 3, 6: 4, 7: 5}

    def __init__(self, model, test_data_path: str, feature_columns: List[str],
                 encoder_dir: Optional[str] = None):
        """
        Args:
            model: Fitted XGBoost model with predict_proba method.
            test_data_path: Path to test parquet file.
            feature_columns: List of feature column names used by the model.
            encoder_dir: Directory containing label encoder .pkl files.
                If None, assumes encoded columns already exist in the parquet.
        """
        self.model = model
        self.test_data_path = test_data_path
        self.feature_columns = feature_columns
        self.encoder_dir = encoder_dir

    def compute_all(self, max_samples: Optional[int] = None) -> Dict:
        """Run full ball-level diagnostics.

        Returns dict with:
            per_class_ece: {class_name: ece_value}
            overall_ece: weighted average ECE
            per_class_reliability: {class_name: [reliability_diagram_data]}
            n_samples: total test samples
        """
        import pandas as pd

        print("Loading test data for ball-level diagnostics...")
        df = pd.read_parquet(self.test_data_path)

        if max_samples and len(df) > max_samples:
            df = df.sample(n=max_samples, random_state=42)

        # Apply label encoding if needed
        _apply_encoders_to_df(df, self.feature_columns, self.encoder_dir)

        # Prepare target: remap to sequential 0-5
        target = df['ball_outcome'].copy()
        target = target.replace({-1: 7})  # wicket
        target = target.map(self.OUTCOME_REMAP)
        valid_mask = target.notna()
        df = df[valid_mask]
        target = target[valid_mask].astype(int).values

        # Get features
        X = df[self.feature_columns].values
        print(f"  Running predict_proba on {len(X):,} balls...")
        raw_probs = self.model.predict_proba(X)

        n_classes = raw_probs.shape[1]
        per_class_ece = {}
        per_class_reliability = {}

        for k in range(n_classes):
            name = self.CLASS_NAMES[k] if k < len(self.CLASS_NAMES) else f"class_{k}"
            binary_labels = (target == k).astype(float)
            ece_val = compute_ece(raw_probs[:, k], binary_labels, n_bins=10)
            rel_data = reliability_diagram_data(raw_probs[:, k], binary_labels, n_bins=10)
            per_class_ece[name] = ece_val
            per_class_reliability[name] = rel_data

        # Weighted average ECE (by class frequency)
        class_counts = np.bincount(target, minlength=n_classes)
        class_weights = class_counts / class_counts.sum()
        overall_ece = sum(
            per_class_ece[self.CLASS_NAMES[k]] * class_weights[k]
            for k in range(min(n_classes, len(self.CLASS_NAMES)))
        )

        return {
            'per_class_ece': per_class_ece,
            'overall_ece': float(overall_ece),
            'per_class_reliability': per_class_reliability,
            'n_samples': len(target),
            'class_distribution': {
                self.CLASS_NAMES[k]: int(class_counts[k])
                for k in range(min(n_classes, len(self.CLASS_NAMES)))
            },
        }

    def print_summary(self, results: Optional[Dict] = None) -> None:
        """Print formatted ball-level calibration summary."""
        if results is None:
            results = self.compute_all()

        print("\n" + "=" * 60)
        print("BALL-LEVEL CALIBRATION DIAGNOSTICS")
        print("=" * 60)
        print(f"Total balls analyzed: {results['n_samples']:,}")
        print(f"Overall ECE (weighted): {results['overall_ece']:.4f}")
        print()

        # Class distribution
        print("Class distribution:")
        for name, count in results['class_distribution'].items():
            pct = count / results['n_samples'] * 100
            print(f"  {name:>7s}: {count:>8,} ({pct:5.1f}%)")
        print()

        # Per-class ECE
        print("Per-class ECE:")
        for name, ece_val in results['per_class_ece'].items():
            quality = "good" if ece_val < 0.03 else "fair" if ece_val < 0.06 else "poor"
            print(f"  {name:>7s}: {ece_val:.4f}  ({quality})")
        print()

        # Reliability per class (top-level summary)
        print("Reliability summary (predicted vs actual):")
        for name, rel_data in results['per_class_reliability'].items():
            if not rel_data:
                continue
            diffs = [abs(b['avg_predicted'] - b['avg_actual']) for b in rel_data]
            max_diff = max(diffs) if diffs else 0
            print(f"  {name:>7s}: max bin error = {max_diff:.3f}, "
                  f"{len(rel_data)} non-empty bins")

        print("=" * 60)


# ---------------------------------------------------------------------------
# Convenience: fit ball-level calibrator from parquet data
# ---------------------------------------------------------------------------

def fit_ball_calibrator_from_data(model, data_path: str,
                                  feature_columns: List[str],
                                  max_samples: Optional[int] = None,
                                  encoder_dir: Optional[str] = None) -> BallLevelCalibrator:
    """Fit a BallLevelCalibrator using model predictions on a parquet dataset.

    Args:
        model: Fitted XGBoost model with predict_proba method.
        data_path: Path to validation parquet file.
        feature_columns: Feature column names.
        max_samples: Cap samples for speed (None = use all).
        encoder_dir: Directory containing label encoder .pkl files.

    Returns:
        Fitted BallLevelCalibrator.
    """
    import pandas as pd

    print(f"Fitting ball-level calibrator from {data_path}...")
    df = pd.read_parquet(data_path)

    if max_samples and len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=42)
        print(f"  Sampled {max_samples:,} balls for calibration fitting")

    # Apply label encoding if needed
    _apply_encoders_to_df(df, feature_columns, encoder_dir)

    # Prepare target
    target = df['ball_outcome'].copy()
    target = target.replace({-1: 7})
    remap = {0: 0, 1: 1, 2: 2, 4: 3, 6: 4, 7: 5}
    target = target.map(remap)
    valid_mask = target.notna()
    df = df[valid_mask]
    target = target[valid_mask].astype(int).values

    X = df[feature_columns].values
    print(f"  Running predict_proba on {len(X):,} balls...")
    raw_probs = model.predict_proba(X)

    cal = BallLevelCalibrator()
    cal.fit(raw_probs, target)
    return cal
