"""Steps 3.3 to 3.5: the BIC arithmetic, the candidate restriction, and seeded restarts."""

import dataclasses
import math
from pathlib import Path

import numpy as np
import pandas as pd

from regime.config import load_config
from regime.models.hmm import bic, fit_hmm, n_params, primary_k_from_bic

ROOT = Path(__file__).resolve().parents[1]
CFG = load_config(str(ROOT / "config.toml"))


def synthetic_z(cfg, T: int = 120, d: int = 3, K: int = 3) -> pd.DataFrame:
    """A persistent K-state Gaussian sequence on a month-end index, drawn with the config seed."""
    rng = np.random.default_rng(cfg.run_seed)
    means = rng.normal(scale=2.0, size=(K, d))
    transmat = np.full((K, K), 0.05)
    np.fill_diagonal(transmat, 1.0 - 0.05 * (K - 1))
    state, y = 0, np.empty((T, d))
    for t in range(T):
        if t:
            state = rng.choice(K, p=transmat[state])
        y[t] = means[state] + rng.normal(scale=0.6, size=d)
    index = pd.date_range("1991-01-31", periods=T, freq="ME")
    frame = pd.DataFrame(y, index=index, columns=[f"f{j}" for j in range(d)])
    frame.index.name = "date"
    return frame


def test_bic_parameter_count() -> None:
    # K(K-1) transitions + Kd means + Kd(d+1)/2 covariance entries.
    assert n_params(K=3, d=8) == 138        # 6 + 24 + 108
    assert n_params(K=4, d=8) == 188        # 12 + 32 + 144
    assert abs(bic(loglik=-1000.0, m=138, T=168) - (2000.0 + 138 * math.log(168))) < 1e-9


def test_select_k_argmin_over_candidates_only() -> None:
    # A stubbed table: K = 2 is the global minimum and is not a candidate;
    # among the candidates {3, 4}, K = 4 is lower. primary_K must be 4.
    table = pd.DataFrame(
        {
            "K": [2, 3, 4, 5],
            "loglik": [-900.0, -880.0, -870.0, -865.0],
            "m": [30, 138, 188, 240],
            "T": [168] * 4,
            "bic": [1000.0, 1200.0, 1100.0, 1500.0],
            "converged": [True] * 4,
        }
    )
    assert CFG.hmm_k_candidates == (3, 4)
    assert int(table.loc[table["bic"].idxmin(), "K"]) == 2
    assert primary_k_from_bic(table, CFG) == 4


def test_fit_hmm_restarts_are_seeded_and_deterministic() -> None:
    cfg = dataclasses.replace(CFG, hmm_n_restarts=3, hmm_n_iter=20)
    z = synthetic_z(cfg).to_numpy()

    params_a, restarts_a = fit_hmm(z, K=2, cfg=cfg, refit_date=pd.Timestamp("2004-12-31"))
    params_b, restarts_b = fit_hmm(z, K=2, cfg=cfg, refit_date=pd.Timestamp("2004-12-31"))

    assert list(restarts_a.columns) == ["restart", "seed", "loglik", "n_iter", "converged"]
    assert len(restarts_a) == cfg.hmm_n_restarts
    assert list(restarts_a["seed"]) == [cfg.run_seed + i for i in restarts_a["restart"]]
    pd.testing.assert_series_equal(restarts_a["loglik"], restarts_b["loglik"])
    assert params_a.loglik == restarts_a["loglik"].max() == params_b.loglik
    np.testing.assert_array_equal(params_a.means, params_b.means)
    assert params_a.covars.shape == (2, z.shape[1], z.shape[1])


def _short_cfg(tmp_path, index, first_window_end):
    """The real config on a short synthetic sample, writing every output into tmp_path."""
    return dataclasses.replace(
        CFG,
        hmm_n_restarts=2,
        hmm_n_iter=20,
        sample_features_from=str(index[0].date()),
        sample_first_window_end=str(pd.Timestamp(first_window_end).date()),
        sample_end=str(index[-1].date()),
        hmm_anchor_feature="f0",
        hmm_anchor_tiebreak_feature="f1",
        outputs_tables_dir=str(tmp_path / "tables"),
        outputs_filtered_probs=str(tmp_path / "filtered_probs.csv"),
    )


def test_refit_parameters_apply_from_refit_date_inclusive(tmp_path) -> None:
    from regime.models.hmm import run_expanding_hmm

    z = synthetic_z(CFG, T=60)
    # Refit dates every twelve rows from row 35: rows 35, 47 and 59 (the last
    # row of the sample), so two boundaries to check.
    cfg = _short_cfg(tmp_path, z.index, z.index[35])
    probs, kept = run_expanding_hmm(z, K=2, cfg=cfg)

    D0, D1, D2 = z.index[35], z.index[47], z.index[59]
    assert [p.refit_date for p in kept] == [D0, D1, D2]
    assert probs.index[0] == D0                       # nothing before the first refit
    assert probs.loc[D0, "refit_date"] == D0          # in force from D inclusive
    assert probs.loc[D1, "refit_date"] == D1
    assert probs.loc[D2, "refit_date"] == D2
    assert probs.loc[z.index[46], "refit_date"] == D0  # the row before D1 is still the old fit
    assert probs.loc[z.index[58], "refit_date"] == D1  # and the row before D2 the fit before it
    assert probs.loc[z.index[36], "refit_date"] == D0


def test_filtered_is_fresh_full_history_pass(tmp_path) -> None:
    from regime.models.hmm import run_expanding_hmm
    from regime.models.hmm_numpy import forward_filter

    z = synthetic_z(CFG, T=60)
    cfg = _short_cfg(tmp_path, z.index, z.index[35])
    probs, kept = run_expanding_hmm(z, K=2, cfg=cfg)

    # The row immediately after the second refit boundary: recompute the whole
    # forward pass from features_from under the parameters in force and take
    # its last row. Nothing is carried across the boundary, so this must match
    # to machine precision.
    t = z.index[48]
    params = kept[1]
    assert probs.loc[t, "refit_date"] == params.refit_date
    history = z.loc[z.index <= t].to_numpy()
    alpha, _ = forward_filter(history, params.startprob, params.transmat, params.means, params.covars)

    np.testing.assert_allclose(probs.loc[t, ["p0", "p1"]].to_numpy(dtype=float), alpha[-1], atol=1e-12)
