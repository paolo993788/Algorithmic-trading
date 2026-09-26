"""Cross-validation for financial time series with overlapping labels: purging, embargo and combinatorial splits.

Why ordinary k-fold fails. A label (target) of observation i often depends on information over an interval
[t0_i, t1_i], for example the return over the next 20 days. If a training observation's interval overlaps the
test period, the model is trained on information that lies inside the test period (leakage): test scores are
optimistic. Serial correlation adds a second channel: training observations immediately after the test period carry
information about its last labels.

Purging removes from the training set every observation whose label interval [t0, t1] overlaps the span of the
test labels [min t0_test, max t1_test]. The embargo also removes the h training observations that immediately
follow the test block (h = ceil(embargo x T)). Both follow Lopez de Prado (2018), chapter 7.

Combinatorial purged cross-validation (Lopez de Prado, 2018, chapter 12): the T observations are split into N
contiguous groups and every choice of k groups is used once as the test set (C(N, k) splits), each purged and
embargoed around every test group. Each group is tested in C(N-1, k-1) splits, and the splits can be assembled into
phi = k C(N, k) / N complete backtest paths, which gives a distribution of out-of-sample performance rather than
a single path.

Inputs: t0 and t1 are integer positions or timestamps (one per observation, sorted by t0, with t0 <= t1).
Outputs: lists of (train, test) arrays of integer positions.
"""

from __future__ import annotations

import itertools
import math

import numpy as np


def _as_int(values) -> np.ndarray:
    v = np.asarray(values)
    if np.issubdtype(v.dtype, np.datetime64):
        return v.astype("datetime64[ns]").astype(np.int64)
    return v.astype(np.int64)


def _check(t0, t1):
    t0, t1 = _as_int(t0), _as_int(t1)
    if t0.shape != t1.shape or t0.ndim != 1:
        raise ValueError("t0 and t1 must be 1-D arrays of equal length")
    if np.any(t1 < t0):
        raise ValueError("each label must end on or after its start (t1 >= t0)")
    if np.any(np.diff(t0) < 0):
        raise ValueError("observations must be sorted by t0")
    return t0, t1


def _purged_train(t0, t1, test_blocks, embargo_obs: int) -> np.ndarray:
    n = t0.size
    keep = np.ones(n, dtype=bool)
    for block in test_blocks:
        lo, hi = t0[block].min(), t1[block].max()
        keep &= (t1 < lo) | (t0 > hi)                          # purge: no overlap with the test label span
        end = block.max()
        keep[end + 1:min(n, end + 1 + embargo_obs)] = False    # embargo after the test block
    for block in test_blocks:
        keep[block] = False
    return np.flatnonzero(keep)


def purged_kfold(t0, t1, n_splits: int = 5, embargo: float = 0.0):
    """k-fold with contiguous test folds, purging and embargo; returns [(train, test), ...]."""
    t0, t1 = _check(t0, t1)
    n = t0.size
    if not 2 <= n_splits <= n:
        raise ValueError("need 2 <= n_splits <= number of observations")
    h = int(math.ceil(embargo * n))
    folds = np.array_split(np.arange(n), n_splits)
    return [(_purged_train(t0, t1, [test], h), test) for test in folds]


def combinatorial_purged_splits(t0, t1, n_groups: int = 6, n_test_groups: int = 2, embargo: float = 0.0):
    """Every choice of `n_test_groups` of `n_groups` contiguous groups as the test set, purged and embargoed.

    Returns (splits, groups): splits is a list of (train, test, test_group_ids) and groups the list of index arrays.
    """
    t0, t1 = _check(t0, t1)
    n = t0.size
    if not 1 <= n_test_groups < n_groups <= n:
        raise ValueError("need 1 <= n_test_groups < n_groups <= number of observations")
    h = int(math.ceil(embargo * n))
    groups = np.array_split(np.arange(n), n_groups)
    splits = []
    for combo in itertools.combinations(range(n_groups), n_test_groups):
        blocks = [groups[g] for g in combo]
        splits.append((_purged_train(t0, t1, blocks, h), np.concatenate(blocks), combo))
    return splits, groups


def number_of_paths(n_groups: int, n_test_groups: int) -> int:
    """phi(N, k) = k C(N, k) / N complete backtest paths from combinatorial purged cross-validation."""
    return n_test_groups * math.comb(n_groups, n_test_groups) // n_groups
