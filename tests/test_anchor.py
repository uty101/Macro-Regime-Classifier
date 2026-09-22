"""Step 3.4: the anchor ordering, its tolerance band, and Hungarian agreement between refits."""

import dataclasses
from pathlib import Path

import numpy as np
import pandas as pd

from regime.config import load_config
from regime.models.anchor import anchor, anchor_permutation, hungarian_agreement
from regime.models.hmm import HMMParams

ROOT = Path(__file__).resolve().parents[1]
CFG = load_config(str(ROOT / "config.toml"))
FEATURES = list(CFG.features_core)
ANCHOR_COL = FEATURES.index(CFG.hmm_anchor_feature)
TIEBREAK_COL = FEATURES.index(CFG.hmm_anchor_tiebreak_feature)


def _params(means: np.ndarray, rng) -> HMMParams:
    K, d = means.shape
    transmat = rng.random((K, K))
    transmat /= transmat.sum(axis=1, keepdims=True)
    startprob = rng.random(K)
    startprob /= startprob.sum()
    covars = np.stack([np.eye(d) * (k + 1) for k in range(K)])
    return HMMParams(
        startprob=startprob, transmat=transmat, means=means, covars=covars,
        K=K, refit_date=pd.Timestamp("2004-12-31"), loglik=-1.0, converged=True,
    )


def test_anchor_orders_by_dgs10_chg12_and_permutes_consistently() -> None:
    rng = np.random.default_rng(CFG.run_seed)
    means = rng.normal(size=(3, len(FEATURES)))
    means[:, ANCHOR_COL] = [2.0, -1.0, 0.5]          # unsorted, well outside the tolerance
    params = _params(means, rng)

    anchored, perm = anchor(params, FEATURES, CFG)

    assert list(perm) == [1, 2, 0]
    assert list(anchored.means[:, ANCHOR_COL]) == [-1.0, 0.5, 2.0]
    # The same perm on every array, and transmat on both axes.
    np.testing.assert_array_equal(anchored.means, params.means[perm])
    np.testing.assert_array_equal(anchored.startprob, params.startprob[perm])
    np.testing.assert_array_equal(anchored.covars, params.covars[perm])
    np.testing.assert_array_equal(anchored.transmat, params.transmat[perm][:, perm])
    # A row of transmat is still a distribution, and entry (j, l) is still the
    # old (perm[j], perm[l]) probability.
    np.testing.assert_allclose(anchored.transmat.sum(axis=1), 1.0)
    for j in range(3):
        for l in range(3):
            assert anchored.transmat[j, l] == params.transmat[perm[j], perm[l]]


def test_anchor_tiebreak_within_tolerance() -> None:
    rng = np.random.default_rng(CFG.run_seed)
    means = rng.normal(size=(2, len(FEATURES)))
    assert CFG.hmm_anchor_tie_tolerance == 0.1
    means[:, ANCHOR_COL] = [0.00, 0.05]              # 0.05 apart: inside the band
    means[:, TIEBREAK_COL] = [1.0, -1.0]             # so cpi_3m_ann decides
    params = _params(means, rng)

    anchored, perm = anchor(params, FEATURES, CFG)

    assert list(perm) == [1, 0]
    assert list(anchored.means[:, TIEBREAK_COL]) == [-1.0, 1.0]

    # The same pair 0.5 apart is outside the band, so the anchor feature decides.
    means[:, ANCHOR_COL] = [0.0, 0.5]
    _, perm_wide = anchor(_params(means, rng), FEATURES, CFG)
    assert list(perm_wide) == [0, 1]


def test_hungarian_identity_and_swap() -> None:
    means = np.array([[-1.0, 0.0], [0.0, 1.0], [2.0, -1.0]])

    agrees, assignment = hungarian_agreement(means, means.copy())
    assert agrees is True
    assert list(assignment) == [0, 1, 2]

    swapped = means[[1, 0, 2]]
    agrees, assignment = hungarian_agreement(means, swapped)
    assert agrees is False
    assert list(assignment) == [1, 0, 2]
