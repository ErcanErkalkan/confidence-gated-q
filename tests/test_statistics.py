import numpy as np
import pandas as pd

from hybrid_q.statistics import (
    bootstrap_mean_interval,
    cohen_dz,
    holm_adjust,
    paired_differences,
    pairwise,
    t_interval,
    win_loss_tie,
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
