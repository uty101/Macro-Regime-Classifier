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
from regime.models.hmm_numpy import forward_filter

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


def refit_dates(cfg: Config) -> list[pd.Timestamp]:
    """``first_window_end`` then every ``cfg.hmm_refit_month`` months, up to ``sample_end``."""
    first, end = pd.Timestamp(cfg.sample_first_window_end), pd.Timestamp(cfg.sample_end)
    dates, d = [], first
    while d <= end:
        dates.append(d)
        d = (d + pd.DateOffset(months=cfg.hmm_refit_month)) + pd.offsets.MonthEnd(0)
    return dates


def model_from_params(params: HMMParams, cfg: Config) -> GaussianHMM:
    """A ``GaussianHMM`` carrying ``params`` verbatim, for ``predict_proba`` under anchored parameters.

    Nothing is fitted here: the arrays are assigned onto an unfitted model so
    hmmlearn's own smoother can be called on the training rows. ``covars_`` is
    set through ``covars_`` (full-shaped for every covariance type), which
    hmmlearn converts back into its internal representation.
    """
    model = GaussianHMM(n_components=params.K, covariance_type=cfg.hmm_covariance_type)
    model.startprob_ = params.startprob
    model.transmat_ = params.transmat
    model.means_ = params.means
    model.covars_ = params.covars
    return model


def hard_labels(probs: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """``label`` = argmax state, ``assigned`` = max probability > ``cfg.hmm_assigned_threshold``.

    Index ``date``; the probability columns are every ``p<k>`` column of
    ``probs``, so the same function serves the HMM filtered, HMM smoothed and
    GMM frames.
    """
    cols = [c for c in probs.columns if c.startswith("p") and c[1:].isdigit()]
    p = probs[cols].to_numpy(dtype="float64")
    out = pd.DataFrame(
        {"label": p.argmax(axis=1).astype("int64"), "assigned": p.max(axis=1) > cfg.hmm_assigned_threshold},
        index=probs.index,
    )
    out.index.name = "date"
    return out


def _state_counts(params: HMMParams, train: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Training rows per anchored state, and the degenerate flag, for one refit."""
    assign = model_from_params(params, cfg).predict_proba(train.to_numpy(dtype="float64")).argmax(axis=1)
    counts = pd.Series(assign).value_counts().reindex(range(params.K), fill_value=0).sort_index()
    return pd.DataFrame(
        {
            "refit_date": params.refit_date.date(),
            "state": counts.index.astype(int),
            "n_rows": counts.to_numpy().astype(int),
            "degenerate": counts.to_numpy() < cfg.hmm_min_state_rows,
        }
    )


def run_expanding_hmm(
    z: pd.DataFrame, K: int, cfg: Config
) -> tuple[pd.DataFrame, list[HMMParams]]:
    """The real-time protocol: refit annually, filter forward, never look past t.

    At each refit date D the model is fitted on the model-input rows
    ``[features_from, D]`` and anchored. Its parameters are in force for
    ``D <= t < next D`` (convention 4). The probability at t is the last row of
    a fresh ``forward_filter`` over ``[features_from, t]`` under those
    parameters (convention 5) — not a state carried across the boundary, and
    never ``predict_proba``, which would use rows after t.

    Writes ``outputs/tables/hmm_restarts_<D>.csv`` per refit,
    ``state_counts.csv``, ``anchor_agreement.csv`` and
    ``cfg.outputs_filtered_probs``. Returns the probability frame and the
    anchored parameter sets in refit order.
    """
    from regime.models.anchor import anchor, hungarian_agreement

    tables_dir = Path(cfg.outputs_tables_dir)
    tables_dir.mkdir(parents=True, exist_ok=True)
    features = list(z.columns)
    start = pd.Timestamp(cfg.sample_features_from)
    dates = [d for d in refit_dates(cfg) if d <= z.index[-1]]

    kept, counts, agreements = [], [], []
    for D in dates:
        train = z.loc[(z.index >= start) & (z.index <= D)]
        params, restarts = fit_hmm(train.to_numpy(dtype="float64"), K, cfg, D)
        params, _perm = anchor(params, features, cfg)
        restarts.to_csv(tables_dir / f"hmm_restarts_{D:%Y-%m-%d}.csv", index=False)
        counts.append(_state_counts(params, train, cfg))
        if kept:
            agrees, assignment = hungarian_agreement(kept[-1].means, params.means)
            agreements.append(
                {
                    "refit_date": D.date(),
                    "agrees": agrees,
                    "assignment": " ".join(str(int(a)) for a in assignment),
                }
            )
            if not agrees:
                log.warning(
                    "anchor disagreement at refit %s: assignment %s against the previous refit",
                    D.date(), list(map(int, assignment)),
                )
        kept.append(params)
        log.info(
            "refit %s: %d training rows, kept loglik %.4f, n_iter %d, converged %s, %d/%d restarts converged",
            D.date(), len(train), params.loglik,
            int(restarts.loc[restarts["loglik"].idxmax(), "n_iter"]), params.converged,
            int(restarts["converged"].sum()), len(restarts),
        )

    pd.concat(counts, ignore_index=True).to_csv(tables_dir / "state_counts.csv", index=False)
    pd.DataFrame(
        agreements, columns=["refit_date", "agrees", "assignment"]
    ).to_csv(tables_dir / "anchor_agreement.csv", index=False)

    rows, index = [], []
    for i, D in enumerate(dates):
        next_D = dates[i + 1] if i + 1 < len(dates) else None
        in_force = z.index[(z.index >= D) & ((z.index < next_D) if next_D is not None else True)]
        params = kept[i]
        for t in in_force:
            history = z.loc[(z.index >= start) & (z.index <= t)].to_numpy(dtype="float64")
            alpha, _ = forward_filter(
                history, params.startprob, params.transmat, params.means, params.covars
            )
            rows.append(np.append(alpha[-1], i))
            index.append(t)

    probs = pd.DataFrame(
        [r[:K] for r in rows], index=pd.DatetimeIndex(index, name="date"), columns=[f"p{k}" for k in range(K)]
    )
    probs["refit_date"] = [dates[int(r[K])] for r in rows]
    out = Path(cfg.outputs_filtered_probs)
    out.parent.mkdir(parents=True, exist_ok=True)
    probs.to_csv(out, index=True, date_format="%Y-%m-%d")
    return probs, kept


def run_smoothed_hmm(z: pd.DataFrame, K: int, cfg: Config) -> tuple[pd.DataFrame, HMMParams]:
    """One fit on the whole sample, smoothed over the whole sample: the hindsight benchmark.

    This is what the regimes look like to someone who already knows how the
    sample ended — one fit on ``[features_from, sample_end]`` and
    ``predict_proba`` over the full sequence, so every row uses every other
    row. It is deliberate lookahead and exists only to bound what perfect
    regime knowledge would have been worth. Nothing in the timed strategy may
    read it (step 3.8 enforces that for ``strategy.py``).

    Writes ``outputs/tables/hmm_restarts_smoothed.csv`` and
    ``cfg.outputs_smoothed_probs``.
    """
    from regime.models.anchor import anchor

    tables_dir = Path(cfg.outputs_tables_dir)
    tables_dir.mkdir(parents=True, exist_ok=True)
    start, end = pd.Timestamp(cfg.sample_features_from), pd.Timestamp(cfg.sample_end)
    train = z.loc[(z.index >= start) & (z.index <= end)]

    params, restarts = fit_hmm(train.to_numpy(dtype="float64"), K, cfg, end)
    params, _perm = anchor(params, list(z.columns), cfg)
    restarts.to_csv(tables_dir / "hmm_restarts_smoothed.csv", index=False)
    log.info(
        "smoothed fit on %d rows to %s: kept loglik %.4f, n_iter %d, converged %s",
        len(train), end.date(), params.loglik,
        int(restarts.loc[restarts["loglik"].idxmax(), "n_iter"]), params.converged,
    )

    smoothed = model_from_params(params, cfg).predict_proba(train.to_numpy(dtype="float64"))
    probs = pd.DataFrame(smoothed, index=train.index.copy(), columns=[f"p{k}" for k in range(K)])
    probs.index.name = "date"
    out = Path(cfg.outputs_smoothed_probs)
    out.parent.mkdir(parents=True, exist_ok=True)
    probs.to_csv(out, index=True, date_format="%Y-%m-%d")
    return probs, params
