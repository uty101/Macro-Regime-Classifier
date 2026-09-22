"""Step 3.6: the mixture's restarts are seeded, and its probability at t uses only row t."""

import dataclasses
from pathlib import Path

import numpy as np
import pandas as pd

from regime.config import load_config
from regime.models.gmm import anchor_gmm, fit_gmm, run_expanding_gmm
from tests.test_hmm import synthetic_z

ROOT = Path(__file__).resolve().parents[1]
CFG = load_config(str(ROOT / "config.toml"))


def _cfg(tmp_path, index, first_window_end):
    return dataclasses.replace(
        CFG,
        gmm_n_restarts=3,
        sample_features_from=str(index[0].date()),
        sample_first_window_end=str(pd.Timestamp(first_window_end).date()),
        sample_end=str(index[-1].date()),
        hmm_anchor_feature="f0",
        hmm_anchor_tiebreak_feature="f1",
        outputs_tables_dir=str(tmp_path / "tables"),
        outputs_gmm_filtered_probs=str(tmp_path / "gmm_filtered_probs.csv"),
    )


def test_gmm_restarts_seeded_and_deterministic(tmp_path) -> None:
    z = synthetic_z(CFG, T=60)
    cfg = _cfg(tmp_path, z.index, z.index[35])

    _, restarts_a = fit_gmm(z.to_numpy(), K=2, cfg=cfg, refit_date=z.index[35])
    _, restarts_b = fit_gmm(z.to_numpy(), K=2, cfg=cfg, refit_date=z.index[35])

    assert list(restarts_a.columns) == ["restart", "seed", "lower_bound", "n_iter", "converged"]
    assert len(restarts_a) == cfg.gmm_n_restarts
    assert list(restarts_a["seed"]) == [cfg.run_seed + i for i in restarts_a["restart"]]
    pd.testing.assert_frame_equal(restarts_a, restarts_b)


def test_gmm_probability_is_predict_proba_under_params_in_force(tmp_path) -> None:
    z = synthetic_z(CFG, T=60)
    cfg = _cfg(tmp_path, z.index, z.index[35])
    probs = run_expanding_gmm(z, K=2, cfg=cfg)

    # Refit the second refit date independently, anchor it the same way, and
    # score one row of the window that fit governs. A mixture has no
    # transition matrix, so the probability at t must be that one row's
    # predict_proba and nothing else — no history, no carry-over.
    D, t = z.index[47], z.index[50]
    assert probs.loc[t, "refit_date"] == D

    model, _ = fit_gmm(z.loc[z.index <= D].to_numpy(), K=2, cfg=cfg, refit_date=D)
    anchor_gmm(model, list(z.columns), cfg)
    expected = model.predict_proba(z.loc[[t]].to_numpy())[0]

    np.testing.assert_allclose(probs.loc[t, ["p0", "p1"]].to_numpy(dtype=float), expected, atol=1e-12)
