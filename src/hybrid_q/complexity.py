from __future__ import annotations

from dataclasses import dataclass

from torch import nn


@dataclass(frozen=True)
class MappingOperationEstimate:
    """Implementation-aligned scalar-operation estimate for a gate mapping.

    ``arithmetic_flops`` counts scalar additions, subtractions,
    multiplications, and divisions as one FLOP each. Comparisons, clipping,
    absolute values, exponentials, indexing, and memory traffic are reported
    separately and are not folded into the FLOP estimate.
    """

    arithmetic_flops: int
    comparisons: int
    special_functions: int
    definition: str


def trainable_parameter_count(model: nn.Module) -> int:
    """Return the exact number of trainable scalar parameters in ``model``."""

    return sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )


def approximate_inference_macs(model: nn.Module) -> int:
    """Count dense-layer MACs for one unbatched inference pass.

    The count is exact for the weights of ``nn.Linear`` layers and excludes
    bias additions, activations, reductions, comparisons, and memory traffic.
    """

    return sum(
        module.in_features * module.out_features
        for module in model.modules()
        if isinstance(module, nn.Linear)
    )


def approximate_inference_flops(model: nn.Module) -> int:
    """Return the conventional approximate dense FLOP count, ``2 * MACs``.

    A multiply and its accumulation are counted as two FLOPs. The same
    exclusions as :func:`approximate_inference_macs` apply.
    """

    return 2 * approximate_inference_macs(model)


def mapping_operation_estimate(mapping: str) -> MappingOperationEstimate:
    """Estimate mapping-only overhead from the same two normalized inputs.

    The fuzzy estimate follows the triangular three-membership, five-rule
    weighted-average implementation. The crisp estimate follows the
    same-input two-threshold branch. Upstream support/reliability estimation
    and downstream Q-value mixing are excluded from both rows.
    """

    if mapping == "fuzzy_triangular_five_rule":
        return MappingOperationEstimate(
            arithmetic_flops=26,
            comparisons=8,
            special_functions=1,
            definition=(
                "Mapping-only estimate after two normalized scalar inputs: "
                "three triangular memberships, five rule weights, and a "
                "weighted-average consequent; special function is abs."
            ),
        )
    if mapping == "fuzzy_shoulder_five_rule":
        return MappingOperationEstimate(
            arithmetic_flops=25,
            comparisons=4,
            special_functions=5,
            definition=(
                "Mapping-only estimate after two normalized scalar inputs: "
                "logistic shoulder memberships, five rule weights, and a "
                "weighted-average consequent; special functions are four "
                "exponentials and one maximum."
            ),
        )
    if mapping == "same_input_crisp_threshold":
        return MappingOperationEstimate(
            arithmetic_flops=0,
            comparisons=2,
            special_functions=0,
            definition=(
                "Mapping-only estimate after the same two normalized scalar "
                "inputs: two threshold comparisons and a Boolean branch."
            ),
        )
    raise ValueError(f"Unknown mapping: {mapping}")
