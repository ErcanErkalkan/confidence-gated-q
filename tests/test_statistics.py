import numpy as np
import pandas as pd

from hybrid_q.statistics import (
    bootstrap_mean_interval,
    cohen_dz,
    empirical_lower_cvar,
    holm_adjust,
    lower_quantile,
    maximum_drawdown,
    paired_differences,
    pairwise,
    t_interval,
    win_loss_tie,
    worst_checkpoint,
    worst_decile_mean,
)


def test_holm_adjustment_is_bounded():
    adjusted = holm_adjust([0.01, 0.04, 0.2])
    assert all(0 <= value <= 1 for value in adjusted)
    assert adjusted[0] <= adjusted[1] <= adjusted[2]


def test_effect_size_and_interval():
    values = np.array([1.0, 2.0, 3.0, 4.0])
    low, high = t_interval(values)
    assert low < values.mean() < high
    assert cohen_dz(values) > 0


def test_zero_differences_have_unit_p_values():
    rows = []
    for agent in ("a", "b"):
        for seed in (0, 1):
            rows.append(
                {
                    "environment": "env",
                    "agent": agent,
                    "seed": seed,
                    "mean_return": 0.0,
                    "success_rate": 0.0,
                    "return_auc": 0.0,
                }
            )
    result = pairwise(pd.DataFrame(rows))
    assert (result["paired_t_p"] == 1.0).all()
    assert (result["paired_t_holm_p"] == 1.0).all()
    assert (result["wilcoxon_holm_p"] == 1.0).all()
    assert (result["sign_test_holm_p"] == 1.0).all()


def test_bootstrap_interval_is_reproducible():
    values = np.array([-2.0, 1.0, 3.0, 7.0])
    assert bootstrap_mean_interval(values, seed=9, samples=500) == (
        bootstrap_mean_interval(values, seed=9, samples=500)
    )


def test_win_loss_tie_and_median_difference():
    assert win_loss_tie(np.array([2.0, -1.0, 0.0, 4.0])) == (2, 1, 1)
    rows = []
    for seed, left, right in ((0, 2.0, 1.0), (1, 5.0, 1.0), (2, 3.0, 3.0)):
        rows.append(
            {
                "environment": "env",
                "agent": "left",
                "seed": seed,
                "mean_return": left,
                "success_rate": left,
                "return_auc": left,
            }
        )
        rows.append(
            {
                "environment": "env",
                "agent": "right",
                "seed": seed,
                "mean_return": right,
                "success_rate": right,
                "return_auc": right,
            }
        )
    result = pairwise(pd.DataFrame(rows))
    comparison = result[
        (result["left"] == "left") & (result["right"] == "right")
    ].iloc[0]
    assert comparison["median_difference"] == 1.0
    assert (comparison["wins"], comparison["losses"], comparison["ties"]) == (
        2,
        0,
        1,
    )


def test_paired_input_validation_rejects_mismatched_seeds():
    left = pd.DataFrame({"seed": [0, 1], "score": [1.0, 2.0]})
    right = pd.DataFrame({"seed": [0, 2], "score": [1.0, 2.0]})
    with np.testing.assert_raises(ValueError):
        paired_differences(left, right, "score")


def test_empirical_lower_cvar_uses_locked_tail_count_and_ties():
    values = np.arange(1.0, 12.0)
    assert empirical_lower_cvar(values, 0.10) == 1.5
    assert empirical_lower_cvar([1.0, 1.0, 4.0, 8.0], 0.50) == 1.0
    assert worst_decile_mean(values) == empirical_lower_cvar(values, 0.10)


def test_tail_statistics_filter_nonfinite_values_and_handle_singletons():
    values = [np.nan, np.inf, -np.inf, 1.0, 3.0]
    assert empirical_lower_cvar(values, 0.50) == 1.0
    assert lower_quantile(values, 0.05) == 1.1
    assert empirical_lower_cvar([7.0], 0.05) == 7.0
    assert lower_quantile([7.0], 0.05) == 7.0
    assert worst_checkpoint(values) == 1.0


def test_tail_statistics_reject_invalid_or_insufficient_finite_input():
    for alpha in (0.0, -0.1, 1.1, np.nan):
        with np.testing.assert_raises(ValueError):
            empirical_lower_cvar([1.0, 2.0], alpha)
    for q in (-0.1, 1.1, np.inf):
        with np.testing.assert_raises(ValueError):
            lower_quantile([1.0, 2.0], q)
    for values in ([], [np.nan, np.inf]):
        with np.testing.assert_raises(ValueError):
            empirical_lower_cvar(values, 0.10)
    with np.testing.assert_raises(ValueError):
        lower_quantile([[1.0, 2.0]], 0.05)


def test_maximum_drawdown_sorts_checkpoints_and_handles_nonfinite_pairs():
    checkpoints = [3.0, 0.0, 2.0, 1.0, np.nan]
    values = [0.0, 1.0, 2.0, 3.0, -100.0]
    assert maximum_drawdown(checkpoints, values) == 3.0
    assert maximum_drawdown([10.0], [-2.0]) == 0.0
    assert maximum_drawdown([0.0, 1.0], [2.0, 2.0]) == 0.0


def test_maximum_drawdown_rejects_misaligned_or_duplicate_checkpoints():
    with np.testing.assert_raises(ValueError):
        maximum_drawdown([0.0], [1.0, 2.0])
    with np.testing.assert_raises(ValueError):
        maximum_drawdown([0.0, 0.0], [1.0, 2.0])
    with np.testing.assert_raises(ValueError):
        maximum_drawdown([np.nan], [1.0])
