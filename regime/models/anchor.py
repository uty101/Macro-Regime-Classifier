"""State identity across refits: sort the first fit, chain every later one (convention 16).

An HMM's state labels are arbitrary: refit the model a year later and the
state that was 0 can come back as 2. Nothing downstream would notice — the
conditional statistics would silently average two different regimes together.

The first section 3 run showed that sorting alone does not fix this. Ordering
by ascending mean ``dgs10_chg12`` needs that feature to separate the states,
and it does not: the anchored state means span only -0.15 to +0.5 z, so the
sort was ordering noise, and the Hungarian check disagreed with the previous
refit at 13 of 21 refits. 15 of the 26 filtered label changes from 2005 landed
exactly on a refit date — the classifier was relabelling, not detecting.

So the ordering rule is now two rules. The **first** refit is sorted, by
``anchor_permutation``, which fixes an origin for the numbering and nothing
else. **Every later** refit is chained: ``chain_permutation`` matches the new
states to the previous refit's anchored means by minimum total Euclidean
distance over all model-input columns, so state k stays the state nearest to
what state k was a year ago. The GMM chains within its own refits from the
HMM's first anchored fit, and the smoothed fit chains to the last expanding
refit, so every frame in the project numbers its states the same way.

Chaining is a labelling rule, not a correction: it does not make a state stable,
it makes the *numbering* follow whatever stability there is. How far each state
actually moved is recorded per refit as ``matched_distance`` in
``anchor_chain.csv`` and reported, never acted on.
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


def chain_permutation(prev_means: np.ndarray, new_means: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Match a new fit's states to the previous refit's anchored means (convention 16).

    ``linear_sum_assignment`` on the Euclidean distance matrix between
    ``prev_means`` and ``new_means``, over all model-input columns. Returns
    ``(perm, distances)``: new state j is the old state ``perm[j]``, which is
    the state matched to previous anchored state j, and ``distances[j]`` is
    that matched pair's distance. ``linear_sum_assignment`` is deterministic,
    so no further tie-break is applied.

    A large ``distances[j]`` means state j moved a long way between refits. It
    is reported, never acted on: the alternative — refusing the match and
    falling back to a sort — is what the first section 3 run showed does not
    work.
    """
    cost = np.linalg.norm(
        np.asarray(prev_means, dtype="float64")[:, None, :]
        - np.asarray(new_means, dtype="float64")[None, :, :],
        axis=2,
    )
    prev_states, matched = linear_sum_assignment(cost)
    perm = np.asarray(matched, dtype=int)
    return perm, cost[prev_states, matched]


def chain(
    params: HMMParams, prev_means: np.ndarray
) -> tuple[HMMParams, np.ndarray, np.ndarray]:
    """Relabel ``params`` to follow ``prev_means``. Returns the parameters, the perm and the distances."""
    import dataclasses

    perm, distances = chain_permutation(prev_means, params.means)
    relabelled = dataclasses.replace(
        params,
        startprob=params.startprob[perm],
        transmat=params.transmat[perm][:, perm],
        means=params.means[perm],
        covars=params.covars[perm],
    )
    return relabelled, perm, distances
