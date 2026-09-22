"""Step 3.4, revised: the first refit is sorted, every later refit is chained (convention 16)."""

import dataclasses
from pathlib import Path

import numpy as np
import pandas as pd

from regime.config import load_config
from regime.models.anchor import anchor, anchor_permutation, chain, chain_permutation
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


def test_chain_permutation_recovers_a_shuffle() -> None:
    # Three well-separated states, shuffled. Chaining must undo the shuffle
    # exactly, whatever the anchor feature says, and report zero distance
    # because the means are identical up to the reordering.
    prev = np.array([[-1.0, 0.0, 2.0], [0.0, 1.0, -3.0], [2.0, -1.0, 0.5]])
    shuffle = [2, 0, 1]
    new_means = prev[shuffle]

    perm, distances = chain_permutation(prev, new_means)

    # perm is the inverse of the shuffle, because perm maps a slot to the new
    # state that belongs in it: new state perm[j] is previous state j.
    assert list(perm) == [1, 2, 0]
    np.testing.assert_array_equal(new_means[perm], prev)
    np.testing.assert_allclose(distances, 0.0, atol=1e-12)

    # And with the states merely nudged, not permuted, the match is the
    # identity: chaining does not invent a reordering that is not there.
    perm_identity, distances_identity = chain_permutation(prev, prev + 0.01)
    assert list(perm_identity) == [0, 1, 2]
    assert distances_identity.max() < 0.05


def test_chain_permutation_applied_consistently() -> None:
    rng = np.random.default_rng(CFG.run_seed)
    prev = np.array([[-1.0, 0.0], [0.0, 1.0], [2.0, -1.0]])
    means = prev[[1, 2, 0]] + rng.normal(scale=0.01, size=(3, 2))
    params = _params(means, rng)

    chained, perm, distances = chain(params, prev)

    assert list(perm) == [2, 0, 1]
    np.testing.assert_array_equal(chained.means, params.means[perm])
    np.testing.assert_array_equal(chained.startprob, params.startprob[perm])
    np.testing.assert_array_equal(chained.covars, params.covars[perm])
    np.testing.assert_array_equal(chained.transmat, params.transmat[perm][:, perm])
    np.testing.assert_allclose(chained.transmat.sum(axis=1), 1.0)
    for j in range(3):
        for l in range(3):
            assert chained.transmat[j, l] == params.transmat[perm[j], perm[l]]
    assert distances.shape == (3,)
    assert distances.max() < 0.1


def test_chaining_survives_a_feature_the_sort_rule_cannot_separate() -> None:
    # This is the case that forced the revision: the anchor feature has almost
    # no spread across states, so the sort rule orders noise, while another
    # feature separates them cleanly. Chaining tracks the real structure.
    names = FEATURES
    prev = np.zeros((3, len(names)))
    prev[:, ANCHOR_COL] = [0.00, 0.01, 0.02]        # no spread: inside the tie band
    prev[:, names.index("log_vix")] = [-2.0, 0.0, 2.0]   # the real separation

    # The new fit finds the same three states, with the anchor feature's
    # near-zero means jittered into a different order.
    new_means = prev.copy()
    new_means[:, ANCHOR_COL] = [0.02, 0.00, 0.01]
    shuffle = [2, 0, 1]
    new_means = new_means[shuffle]

    perm, distances = chain_permutation(prev, new_means)
    np.testing.assert_allclose(new_means[perm][:, names.index("log_vix")], [-2.0, 0.0, 2.0])
    # The only residual is the jitter in the anchor column itself.
    assert distances.max() < 0.05

    # The old sort rule, on the same fit, would not have recovered it.
    sort_perm = anchor_permutation(new_means, names, CFG)
    assert list(sort_perm) != list(perm)
