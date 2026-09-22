"""State ordering by anchor feature so state numbers mean the same thing across refits (``anchor``). Built in step 3.4.

An HMM's state labels are arbitrary: refit the model a year later and the
state that was 0 can come back as 2. Nothing downstream would notice — the
conditional statistics would silently average two different regimes together.
So every fitted parameter set is put into a fixed order before it is used:
ascending state mean of ``cfg.hmm_anchor_feature``, with a tolerance band in
which a pair is ordered by ``cfg.hmm_anchor_tiebreak_feature`` instead, so two
states whose anchor means are indistinguishable do not swap places on noise.

That is a labelling convention, not a fix. Whether consecutive refits actually
describe the same states is a separate question, answered by
``hungarian_agreement`` and reported per refit; a disagreement is recorded, not
corrected.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

from regime.config import Config
from regime.models.hmm import HMMParams


def anchor_permutation(means: np.ndarray, feature_names: list[str], cfg: Config) -> np.ndarray:
    """``perm`` with new state j = old state ``perm[j]``.

    States ascending by their mean of ``cfg.hmm_anchor_feature``; then, walking
    the sorted list once, any adjacent pair whose anchor means differ by less
    than ``cfg.hmm_anchor_tie_tolerance`` is ordered by ascending mean of
    ``cfg.hmm_anchor_tiebreak_feature``.
    """
    names = list(feature_names)
    anchor_col = names.index(cfg.hmm_anchor_feature)
    tiebreak_col = names.index(cfg.hmm_anchor_tiebreak_feature)
    anchor, tiebreak = means[:, anchor_col], means[:, tiebreak_col]

    perm = list(np.argsort(anchor, kind="stable"))
    for j in range(len(perm) - 1):
        a, b = perm[j], perm[j + 1]
        if abs(anchor[b] - anchor[a]) < cfg.hmm_anchor_tie_tolerance and tiebreak[a] > tiebreak[b]:
            perm[j], perm[j + 1] = b, a
    return np.asarray(perm, dtype=int)


def anchor(params: HMMParams, feature_names: list[str], cfg: Config) -> tuple[HMMParams, np.ndarray]:
    """Relabel ``params`` into anchor order. Returns the relabelled parameters and the permutation."""
    import dataclasses

    perm = anchor_permutation(params.means, feature_names, cfg)
    relabelled = dataclasses.replace(
        params,
        startprob=params.startprob[perm],
        transmat=params.transmat[perm][:, perm],
        means=params.means[perm],
        covars=params.covars[perm],
    )
    return relabelled, perm


def hungarian_agreement(prev_means: np.ndarray, new_means: np.ndarray) -> tuple[bool, np.ndarray]:
    """Match two anchored mean sets by Euclidean distance; agreement is the identity assignment.

    If the optimal matching of this refit's anchored states to the previous
    refit's is anything but the identity, the anchor ordering has moved a state
    relative to the last fit and the two refits are not describing the same
    states in the same slots.
    """
    cost = np.linalg.norm(prev_means[:, None, :] - new_means[None, :, :], axis=2)
    rows, cols = linear_sum_assignment(cost)
    assignment = np.asarray(cols, dtype=int)
    return bool(np.array_equal(assignment, np.arange(len(assignment)))), assignment
