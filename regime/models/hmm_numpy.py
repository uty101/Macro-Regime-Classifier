"""Pure-numpy forward filter for a Gaussian HMM (``forward_filter``). Built in step 3.2.

This exists because hmmlearn has no filter. ``predict_proba`` is the smoothed
posterior P(state_t | y_1..y_T), which at every row but the last uses
observations after t (convention 6). The whole point of the expanding protocol
is that the probability at decision date t may use nothing dated after t, so
the filter is written here and validated against hmmlearn twice: the
log-likelihood against ``model.score`` (the forward pass alone determines it),
and the final row against ``model.predict_proba``, where filtered and smoothed
coincide because there is nothing after T to condition on.

The recursion is scaled, not log-space: at each row the unnormalised weights
are divided by their sum and the log of that sum is accumulated, with the row's
maximum log-density factored out first so the exponential cannot underflow.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import multivariate_normal


def log_emissions(y: np.ndarray, means: np.ndarray, covars: np.ndarray) -> np.ndarray:
    """(T, K) log Gaussian densities: column k is ``logpdf(y; means[k], covars[k])``."""
    K = means.shape[0]
    out = np.empty((y.shape[0], K), dtype="float64")
    for k in range(K):
        out[:, k] = multivariate_normal.logpdf(y, mean=means[k], cov=covars[k], allow_singular=False)
    return out


def forward_filter(
    y: np.ndarray,
    startprob: np.ndarray,
    transmat: np.ndarray,
    means: np.ndarray,
    covars: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Scaled forward recursion.

    Returns ``(alpha, loglik)``: ``alpha`` is (T, K) with ``alpha[t, k] =
    P(state_t = k | y_0..y_t)`` — filtered, never smoothed — and ``loglik`` is
    ``log P(y_0..y_{T-1})`` under the same parameters. Row 0 predicts from
    ``startprob``; every later row predicts from ``alpha[t-1] @ transmat``.
    ``covars`` is (K, d, d).
    """
    y = np.asarray(y, dtype="float64")
    if y.ndim != 2:
        raise ValueError(f"y must be (T, d), got shape {y.shape}")
    T, K = y.shape[0], means.shape[0]

    logb = log_emissions(y, means, covars)
    alpha = np.empty((T, K), dtype="float64")
    loglik = 0.0
    for t in range(T):
        predicted = startprob if t == 0 else alpha[t - 1] @ transmat
        shift = logb[t].max()
        weights = predicted * np.exp(logb[t] - shift)
        total = weights.sum()
        if total <= 0.0 or not np.isfinite(total):
            raise FloatingPointError(f"forward filter lost all probability mass at row {t}")
        alpha[t] = weights / total
        loglik += float(np.log(total)) + float(shift)
    return alpha, loglik
