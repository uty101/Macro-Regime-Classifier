"""Step 3.2: the numpy forward filter against hmmlearn, in the only two places they may be compared.

The log-likelihood is a pure forward-pass quantity, so ``model.score`` and
``forward_filter`` must agree exactly. ``predict_proba`` is smoothed, so only
its final row — where the smoother has no future left to condition on and
collapses onto the filter — is compared (convention 6). No other row of
``predict_proba`` is compared to, or used as, a filtered probability anywhere
in this repo.
"""

from pathlib import Path

import numpy as np
from hmmlearn.hmm import GaussianHMM

from regime.config import load_config
from regime.models.hmm_numpy import forward_filter

ROOT = Path(__file__).resolve().parents[1]
CFG = load_config(str(ROOT / "config.toml"))


def _synthetic(cfg) -> np.ndarray:
    """T = 400 rows of d = 3 from a 3-state Gaussian HMM, drawn with the config seed."""
    rng = np.random.default_rng(cfg.run_seed)
    T, d, K = 400, 3, 3
    means = np.array([[-2.0, 0.0, 1.0], [0.0, 2.0, -1.0], [2.0, -2.0, 0.0]])
    transmat = np.array([[0.90, 0.07, 0.03], [0.05, 0.90, 0.05], [0.03, 0.07, 0.90]])
    state, y = 0, np.empty((T, d))
    for t in range(T):
        if t:
            state = rng.choice(K, p=transmat[state])
        y[t] = means[state] + rng.normal(scale=0.7, size=d)
    return y


def _fitted(y: np.ndarray, cfg) -> GaussianHMM:
    model = GaussianHMM(
        n_components=3,
        covariance_type="full",
        random_state=cfg.run_seed,
        n_iter=cfg.hmm_n_iter,
        tol=cfg.hmm_tol,
        min_covar=cfg.hmm_min_covar,
    )
    model.fit(y)
    return model


def test_loglik_matches_hmmlearn_score() -> None:
    y = _synthetic(CFG)
    model = _fitted(y, CFG)

    _, loglik = forward_filter(y, model.startprob_, model.transmat_, model.means_, model.covars_)

    assert abs(loglik - model.score(y)) < 1e-6, (loglik, model.score(y))


def test_final_row_matches_predict_proba() -> None:
    y = _synthetic(CFG)
    model = _fitted(y, CFG)

    alpha, _ = forward_filter(y, model.startprob_, model.transmat_, model.means_, model.covars_)
    smoothed = model.predict_proba(y)

    # Convention 6: the final row only. predict_proba is the smoothed
    # posterior; at t = T - 1 there is no future observation, so it equals the
    # filtered probability. Comparing any earlier row would assert something
    # false, and using one as a filtered probability would be lookahead.
    assert np.allclose(alpha[-1], smoothed[-1], atol=1e-8)
    assert np.allclose(alpha.sum(axis=1), 1.0, atol=1e-12)
