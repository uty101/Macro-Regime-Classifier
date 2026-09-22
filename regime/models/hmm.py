"""Expanding-window Gaussian HMM: select_k, fit_hmm, run_expanding_hmm, run_smoothed_hmm. Built in steps 3.3 to 3.5.

Every fit is ``cfg.hmm_n_restarts`` restarts of ``GaussianHMM`` seeded
``cfg.run_seed + i``, kept by the highest training log-likelihood. Nothing here
retries, widens ``n_iter`` or relaxes ``tol`` when a fit misbehaves: a
non-converged kept restart, a log-likelihood that decreased and a degenerate
covariance are all recorded and reported, because whether the model is stable
at a given K is one of the section's findings, not an obstacle to it.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM

from regime.config import Config

log = logging.getLogger("regime")


class _CollectHmmlearn(logging.Handler):
    """Collect hmmlearn's own log records so each can be re-reported with its K and restart.

    hmmlearn reports non-convergence and a decreasing log-likelihood through
    ``logging``, not through ``warnings``, and its message names neither the K
    nor the restart that produced it. Records are collected here, hmmlearn's
    propagation is silenced for the duration so nothing prints twice, and every
    record is re-emitted with the fit it belongs to.
    """

    def __init__(self) -> None:
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)

    def __enter__(self) -> "_CollectHmmlearn":
        self._logger = logging.getLogger("hmmlearn")
        self._propagate = self._logger.propagate
        self._logger.addHandler(self)
        self._logger.propagate = False
        return self

    def __exit__(self, *exc) -> None:
        self._logger.removeHandler(self)
        self._logger.propagate = self._propagate

RESTART_COLUMNS = ("restart", "seed", "loglik", "n_iter", "converged")
BIC_COLUMNS = ("K", "loglik", "m", "T", "bic", "converged")


@dataclass(frozen=True)
class HMMParams:
    """One fitted (and, from step 3.4, anchored) parameter set, with its provenance."""

    startprob: np.ndarray
    transmat: np.ndarray
    means: np.ndarray
    covars: np.ndarray
    K: int
    refit_date: pd.Timestamp
    loglik: float
    converged: bool


def n_params(K: int, d: int) -> int:
    """Free parameters of a K-state full-covariance Gaussian HMM in d dimensions.

    ``K(K-1)`` transition rows + ``Kd`` means + ``Kd(d+1)/2`` covariance
    entries. The initial distribution is not counted: it is estimated from a
    single sequence and contributes no independent information at these sample
    sizes.
    """
    return K * (K - 1) + K * d + K * d * (d + 1) // 2


def bic(loglik: float, m: int, T: int) -> float:
    """``-2 loglik + m log T``. Lower is better."""
    return -2.0 * loglik + m * float(np.log(T))


def fit_hmm(
    z: np.ndarray, K: int, cfg: Config, refit_date: pd.Timestamp
) -> tuple[HMMParams, pd.DataFrame]:
    """``cfg.hmm_n_restarts`` seeded fits on ``z``; the highest-log-likelihood restart is kept.

    Returns the kept restart's parameters and the restarts table (columns
    ``restart, seed, loglik, n_iter, converged``). Any warning hmmlearn raises
    during a restart is logged with its K, refit date and restart index and is
    not suppressed; ``n_iter``, ``tol``, ``min_covar`` and the training window
    are never changed in response to one.
    """
    z = np.asarray(z, dtype="float64")
    rows, kept_model, kept_loglik = [], None, -np.inf
    for i in range(cfg.hmm_n_restarts):
        seed = cfg.run_seed + i
        model = GaussianHMM(
            n_components=K,
            covariance_type=cfg.hmm_covariance_type,
            n_iter=cfg.hmm_n_iter,
            tol=cfg.hmm_tol,
            min_covar=cfg.hmm_min_covar,
            init_params=cfg.hmm_init_params,
            params=cfg.hmm_params,
            random_state=seed,
        )
        with warnings.catch_warnings(record=True) as caught, _CollectHmmlearn() as collected:
            warnings.simplefilter("always")
            model.fit(z)
            loglik = float(model.score(z))
        for w in caught:
            log.warning(
                "hmm fit K=%d refit_date=%s restart=%d seed=%d: %s: %s",
                K, pd.Timestamp(refit_date).date(), i, seed, w.category.__name__, w.message,
            )
        for record in collected.records:
            log.warning(
                "hmm fit K=%d refit_date=%s restart=%d seed=%d: hmmlearn %s: %s",
                K, pd.Timestamp(refit_date).date(), i, seed,
                record.levelname.lower(), record.getMessage(),
            )
        rows.append(
            {
                "restart": i,
                "seed": seed,
                "loglik": loglik,
                "n_iter": int(model.monitor_.iter),
                "converged": bool(model.monitor_.converged),
            }
        )
        if loglik > kept_loglik:
            kept_model, kept_loglik = model, loglik

    restarts = pd.DataFrame(rows, columns=list(RESTART_COLUMNS))
    kept = restarts.loc[restarts["loglik"].idxmax()]
    if not bool(kept["converged"]):
        log.warning(
            "hmm fit K=%d refit_date=%s: kept restart %d did not converge in %d iterations",
            K, pd.Timestamp(refit_date).date(), int(kept["restart"]), cfg.hmm_n_iter,
        )
    params = HMMParams(
        startprob=np.asarray(kept_model.startprob_, dtype="float64"),
        transmat=np.asarray(kept_model.transmat_, dtype="float64"),
        means=np.asarray(kept_model.means_, dtype="float64"),
        covars=np.asarray(kept_model.covars_, dtype="float64"),
        K=K,
        refit_date=pd.Timestamp(refit_date),
        loglik=kept_loglik,
        converged=bool(kept["converged"]),
    )
    return params, restarts


def primary_k_from_bic(table: pd.DataFrame, cfg: Config) -> int:
    """``argmin`` BIC over ``cfg.hmm_k_candidates`` only, not over the whole grid.

    The grid is fitted so the BIC curve can be seen; the candidates are the K
    values the project will run, fixed in config before any fit. A global
    minimum outside the candidate set is reported, never adopted.
    """
    candidates = table.loc[table["K"].isin(cfg.hmm_k_candidates)]
    if candidates.empty:
        raise ValueError(f"no K in {cfg.hmm_k_candidates} present in the BIC table")
    chosen = int(candidates.loc[candidates["bic"].idxmin(), "K"])
    global_min = int(table.loc[table["bic"].idxmin(), "K"])
    if global_min != chosen:
        log.info(
            "BIC is globally minimised at K=%d, outside hmm.k_candidates %s; primary_K=%d",
            global_min, list(cfg.hmm_k_candidates), chosen,
        )
    return chosen


def select_k(z_first_window: pd.DataFrame, cfg: Config) -> tuple[int, pd.DataFrame]:
    """Fit every K in ``cfg.hmm_k_grid`` on the first window and pick ``primary_K`` by BIC.

    Writes ``outputs/tables/bic_by_k.csv``, ``cfg.outputs_primary_k`` and one
    ``outputs/tables/hmm_restarts_selectk_K<K>.csv`` per K. The first window is
    the rows given, which is exactly the window standardisation treats as
    in-sample (convention 2), so this adds no lookahead.
    """
    tables_dir = Path(cfg.outputs_tables_dir)
    tables_dir.mkdir(parents=True, exist_ok=True)
    z = z_first_window.to_numpy(dtype="float64")
    T, d = z.shape
    refit_date = pd.Timestamp(z_first_window.index[-1])

    rows = []
    for K in cfg.hmm_k_grid:
        params, restarts = fit_hmm(z, K, cfg, refit_date)
        restarts.to_csv(tables_dir / f"hmm_restarts_selectk_K{K}.csv", index=False)
        m = n_params(K, d)
        rows.append(
            {
                "K": K,
                "loglik": params.loglik,
                "m": m,
                "T": T,
                "bic": bic(params.loglik, m, T),
                "converged": params.converged,
            }
        )
        log.info(
            "select_k K=%d: loglik %.4f, m %d, T %d, bic %.4f, %d/%d restarts converged",
            K, params.loglik, m, T, rows[-1]["bic"],
            int(restarts["converged"].sum()), len(restarts),
        )

    table = pd.DataFrame(rows, columns=list(BIC_COLUMNS))
    table.to_csv(tables_dir / "bic_by_k.csv", index=False)
    primary_K = primary_k_from_bic(table, cfg)
    primary_path = Path(cfg.outputs_primary_k)
    primary_path.parent.mkdir(parents=True, exist_ok=True)
    primary_path.write_text(f"{primary_K}\n", encoding="utf-8")
    return primary_K, table
