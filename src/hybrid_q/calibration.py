from __future__ import annotations

from typing import Iterable

import numpy as np


def _binary_inputs(
    scores: Iterable[float], targets: Iterable[int | bool]
) -> tuple[np.ndarray, np.ndarray]:
    score_array = np.asarray(list(scores), dtype=float)
    target_array = np.asarray(list(targets), dtype=float)
    if score_array.shape != target_array.shape:
        raise ValueError("scores and targets must have the same shape")
    finite = np.isfinite(score_array) & np.isfinite(target_array)
    score_array = score_array[finite]
    target_array = target_array[finite]
    if not np.isin(target_array, [0.0, 1.0]).all():
        raise ValueError("targets must be binary")
    return score_array, target_array.astype(int)


def binary_auroc(
    scores: Iterable[float], targets: Iterable[int | bool]
) -> float:
    """Compute tie-aware AUROC, or NaN when either class is absent."""

    score_array, target_array = _binary_inputs(scores, targets)
    positives = int(target_array.sum())
    negatives = int(target_array.size - positives)
    if positives == 0 or negatives == 0:
        return float("nan")
    order = np.argsort(score_array, kind="mergesort")
    sorted_scores = score_array[order]
    ranks = np.empty(score_array.size, dtype=float)
    start = 0
    while start < sorted_scores.size:
        end = start + 1
        while end < sorted_scores.size and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + 1 + end)
        start = end
    positive_rank_sum = float(ranks[target_array == 1].sum())
    return (
        positive_rank_sum - positives * (positives + 1) / 2.0
    ) / (positives * negatives)


def average_precision(
    scores: Iterable[float], targets: Iterable[int | bool]
) -> float:
    """Compute non-interpolated average precision; NaN without positives."""

    score_array, target_array = _binary_inputs(scores, targets)
    positives = int(target_array.sum())
    if positives == 0:
        return float("nan")
    order = np.argsort(-score_array, kind="mergesort")
    sorted_targets = target_array[order]
    cumulative = np.cumsum(sorted_targets)
    positive_positions = np.flatnonzero(sorted_targets == 1)
    precisions = cumulative[positive_positions] / (positive_positions + 1)
    return float(precisions.mean())


def balanced_accuracy_at_half(
    scores: Iterable[float], targets: Iterable[int | bool]
) -> float:
    """Balanced accuracy at 0.5, or NaN for a one-class target."""

    score_array, target_array = _binary_inputs(scores, targets)
    positives = target_array == 1
    negatives = target_array == 0
    if not positives.any() or not negatives.any():
        return float("nan")
    predictions = score_array >= 0.5
    sensitivity = float(predictions[positives].mean())
    specificity = float((~predictions[negatives]).mean())
    return 0.5 * (sensitivity + specificity)


def brier_score(
    scores: Iterable[float], targets: Iterable[int | bool]
) -> float:
    """Mean squared probability error for a defined binary target."""

    score_array, target_array = _binary_inputs(scores, targets)
    if score_array.size == 0:
        return float("nan")
    if ((score_array < 0.0) | (score_array > 1.0)).any():
        raise ValueError("scores must lie in [0, 1] for Brier score")
    return float(np.mean((score_array - target_array) ** 2))


def first_sustained_detection(
    checkpoints: Iterable[int],
    correct: Iterable[int],
    totals: Iterable[int],
    *,
    shift_step: int,
    accuracy_threshold: float = 0.5,
    minimum_rows: int = 5,
    consecutive_checkpoints: int = 2,
) -> tuple[float, float]:
    """Return first sustained correct-detection checkpoint and delay.

    A checkpoint is detected when its informative-row accuracy is strictly
    above ``accuracy_threshold`` and it meets ``minimum_rows``. Detection must
    persist for ``consecutive_checkpoints`` observed post-shift checkpoints.
    """

    checkpoint_array = np.asarray(list(checkpoints), dtype=int)
    correct_array = np.asarray(list(correct), dtype=int)
    total_array = np.asarray(list(totals), dtype=int)
    if not (
        checkpoint_array.shape == correct_array.shape == total_array.shape
    ):
        raise ValueError("checkpoint, correct, and total arrays must align")
    if consecutive_checkpoints < 1 or minimum_rows < 1:
        raise ValueError("minimum_rows and consecutive_checkpoints must be positive")
    order = np.argsort(checkpoint_array)
    checkpoint_array = checkpoint_array[order]
    correct_array = correct_array[order]
    total_array = total_array[order]
    detected = (
        (checkpoint_array >= shift_step)
        & (total_array >= minimum_rows)
        & (correct_array / np.maximum(total_array, 1) > accuracy_threshold)
    )
    run = 0
    for index, flag in enumerate(detected):
        run = run + 1 if flag else 0
        if run >= consecutive_checkpoints:
            first_index = index - consecutive_checkpoints + 1
            step = int(checkpoint_array[first_index])
            return float(step), float(step - shift_step)
    return float("nan"), float("nan")
