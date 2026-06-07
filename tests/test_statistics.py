import numpy as np
import pandas as pd

from hybrid_q.statistics import cohen_dz, holm_adjust, pairwise, t_interval


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
