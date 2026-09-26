"""Purged k-fold and combinatorial purged cross-validation: no training label overlaps a test label, the embargo
holds, test folds partition the sample, and the combinatorics match the formulas."""

import math

import numpy as np
import pytest

from backtest_engine import cross_validation as cv


def labels(n=600, max_horizon=20, seed=0):
    t0 = np.arange(n)
    t1 = np.minimum(t0 + np.random.default_rng(seed).integers(0, max_horizon, n), n - 1)
    return t0, t1


def assert_clean(t0, t1, train, test_blocks, embargo_obs):
    assert np.intersect1d(train, np.concatenate(test_blocks)).size == 0
    for block in test_blocks:
        lo, hi = t0[block].min(), t1[block].max()
        overlap = (t1[train] >= lo) & (t0[train] <= hi)
        assert not overlap.any()                                            # purged
        after = np.arange(block.max() + 1, block.max() + 1 + embargo_obs)
        assert np.intersect1d(train, after).size == 0                       # embargoed


def test_purged_kfold():
    t0, t1 = labels()
    splits = cv.purged_kfold(t0, t1, n_splits=5, embargo=0.01)
    assert np.array_equal(np.sort(np.concatenate([test for _, test in splits])), t0)   # folds partition the sample
    for train, test in splits:
        assert_clean(t0, t1, train, [test], math.ceil(0.01 * t0.size))
        assert train.size > 0.6 * t0.size
    # Without overlapping labels and without embargo, purged k-fold is ordinary contiguous k-fold.
    plain = cv.purged_kfold(t0, t0, n_splits=4)
    for train, test in plain:
        assert np.array_equal(np.sort(np.concatenate([train, test])), t0)


def test_purging_removes_exactly_the_overlapping_labels():
    t0 = np.arange(10)
    t1 = t0 + 2                                                             # each label spans three periods
    (train, test), *_ = cv.purged_kfold(t0, np.minimum(t1, 9), n_splits=2)
    assert test.tolist() == [0, 1, 2, 3, 4] and train.tolist() == [7, 8, 9]  # 5 and 6 start inside the test label span [0, 6]
    (_, _), (train2, test2) = cv.purged_kfold(t0, np.minimum(t1, 9), n_splits=2)
    assert test2.tolist() == [5, 6, 7, 8, 9] and train2.tolist() == [0, 1, 2]  # labels of 3 and 4 end in the test span


def test_combinatorial_purged_splits():
    t0, t1 = labels(n=900, seed=1)
    splits, groups = cv.combinatorial_purged_splits(t0, t1, n_groups=6, n_test_groups=2, embargo=0.01)
    assert len(splits) == math.comb(6, 2) == 15 and cv.number_of_paths(6, 2) == 5
    appearances = np.zeros(900, dtype=int)
    for train, test, combo in splits:
        assert_clean(t0, t1, train, [groups[g] for g in combo], math.ceil(0.01 * 900))
        appearances[test] += 1
    assert np.all(appearances == math.comb(5, 1))                           # each group tested C(N-1, k-1) times
    with pytest.raises(ValueError):
        cv.combinatorial_purged_splits(t0, t1, n_groups=4, n_test_groups=4)
    with pytest.raises(ValueError):
        cv.purged_kfold(t0[::-1], t1[::-1])
