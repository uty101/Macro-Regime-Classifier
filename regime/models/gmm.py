"""Expanding-window Gaussian mixture (``run_expanding_gmm``). Built in step 3.6.

The comparison model. A mixture has no transition matrix, so its probability
at t depends only on the features at t: it is the same expanding refit
protocol as the HMM with the persistence removed. That is the point — the
difference between the two says how much of the HMM's labelling comes from the
Markov structure rather than from where the month sits in feature space. The
"filtered" probability here is ``predict_proba`` of the single row t, which
uses nothing dated after t, so it is real-time despite sharing a name with the
smoother used elsewhere.
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture

from regime.config import Config
from regime.models.anchor import chain_permutation
from regime.models.hmm import refit_dates

log = logging.getLogger("regime")

RESTART_COLUMNS = ("restart", "seed", "lower_bound", "n_iter", "converged")


def fit_gmm(
    z: np.ndarray, K: int, cfg: Config, refit_date: pd.Timestamp
) -> tuple[GaussianMixture, pd.DataFrame]:
    """``cfg.gmm_n_restarts`` seeded fits; the highest ``lower_bound_`` is kept."""
    rows, kept, best = [], None, -np.inf
    for i in range(cfg.gmm_n_restarts):
        seed = cfg.run_seed + i
        model = GaussianMixture(
            n_components=K,
            covariance_type=cfg.gmm_covariance_type,
            n_init=cfg.gmm_n_init,
            max_iter=cfg.gmm_max_iter,
            tol=cfg.gmm_tol,
            reg_covar=cfg.gmm_reg_covar,
            random_state=seed,
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model.fit(z)
        for w in caught:
            log.warning(
                "gmm fit K=%d refit_date=%s restart=%d seed=%d: %s: %s",
                K, pd.Timestamp(refit_date).date(), i, seed, w.category.__name__, w.message,
            )
        rows.append(
            {
                "restart": i,
                "seed": seed,
                "lower_bound": float(model.lower_bound_),
                "n_iter": int(model.n_iter_),
                "converged": bool(model.converged_),
            }
        )
        if model.lower_bound_ > best:
            kept, best = model, float(model.lower_bound_)

    restarts = pd.DataFrame(rows, columns=list(RESTART_COLUMNS))
    if not bool(restarts.loc[restarts["lower_bound"].idxmax(), "converged"]):
        log.warning(
            "gmm fit K=%d refit_date=%s: the kept restart did not converge in %d iterations",
            K, pd.Timestamp(refit_date).date(), cfg.gmm_max_iter,
        )
    return kept, restarts


def chain_gmm(model: GaussianMixture, prev_means: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Reorder a fitted mixture's components in place to follow ``prev_means``; returns perm and distances.

    The same ``chain_permutation`` the HMM uses (convention 16), so state k
    means the same thing in both frames. The GMM's first refit chains to the
    HMM's first anchored refit means, which is what ties the two numberings
    together; every later GMM refit chains to the previous GMM refit.

    ``precisions_cholesky_`` is permuted with the rest because
    ``predict_proba`` reads it, not ``covariances_``.
    """
    perm, distances = chain_permutation(prev_means, model.means_)
    model.means_ = model.means_[perm]
    model.covariances_ = model.covariances_[perm]
    model.weights_ = model.weights_[perm]
    model.precisions_cholesky_ = model.precisions_cholesky_[perm]
    return perm, distances


def run_expanding_gmm(
    z: pd.DataFrame, K: int, cfg: Config, chain_to: np.ndarray
) -> pd.DataFrame:
    """The HMM's refit dates and protocol with the transition matrix removed.

    At each refit date D the mixture is fitted on ``[features_from, D]`` and
    chained. For ``D <= t < next D`` the probability at t is ``predict_proba``
    of row t alone under the parameters in force.

    ``chain_to`` is the HMM's first anchored refit means: the GMM's first
    refit chains to them and every later GMM refit chains to the previous GMM
    refit (convention 16), so the two models' state numbers refer to the same
    regimes and the confusion table between them is readable.

    Writes ``outputs/tables/gmm_restarts_<D>.csv`` per refit and
    ``cfg.outputs_gmm_filtered_probs``.
    """
    tables_dir = Path(cfg.outputs_tables_dir)
    tables_dir.mkdir(parents=True, exist_ok=True)
    start = pd.Timestamp(cfg.sample_features_from)
    dates = [d for d in refit_dates(cfg) if d <= z.index[-1]]

    frames, previous = [], np.asarray(chain_to, dtype="float64")
    for i, D in enumerate(dates):
        train = z.loc[(z.index >= start) & (z.index <= D)]
        model, restarts = fit_gmm(train.to_numpy(dtype="float64"), K, cfg, D)
        restarts.to_csv(tables_dir / f"gmm_restarts_{D:%Y-%m-%d}.csv", index=False)
        _perm, _distances = chain_gmm(model, previous)
        previous = model.means_

        next_D = dates[i + 1] if i + 1 < len(dates) else None
        in_force = z.loc[(z.index >= D) & ((z.index < next_D) if next_D is not None else True)]
        if len(in_force):
            probs = pd.DataFrame(
                model.predict_proba(in_force.to_numpy(dtype="float64")),
                index=in_force.index.copy(),
                columns=[f"p{k}" for k in range(K)],
            )
            probs["refit_date"] = D
            frames.append(probs)
        log.info(
            "gmm refit %s: %d training rows, kept lower_bound %.6f, %d/%d restarts converged",
            D.date(), len(train), restarts["lower_bound"].max(),
            int(restarts["converged"].sum()), len(restarts),
        )

    out_frame = pd.concat(frames)
    out_frame.index.name = "date"
    out = Path(cfg.outputs_gmm_filtered_probs)
    out.parent.mkdir(parents=True, exist_ok=True)
    out_frame.to_csv(out, index=True, date_format="%Y-%m-%d")
    return out_frame
