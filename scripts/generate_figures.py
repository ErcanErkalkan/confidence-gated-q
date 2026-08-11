from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "figures"


def fuzzy_memberships() -> None:
    """Generate the public fuzzy-membership diagnostic figure."""
    support = np.linspace(0.0, 1.0, 401)
    low = np.clip(1.0 - 2.0 * support, 0.0, 1.0)
    medium = np.clip(1.0 - np.abs(2.0 * support - 1.0), 0.0, 1.0)
    high = np.clip(2.0 * support - 1.0, 0.0, 1.0)
    uncertainty = np.linspace(0.0, 1.0, 401)

    figure, axes = plt.subplots(1, 2, figsize=(9.2, 3.6))
    axes[0].plot(support, low, label="Low support")
    axes[0].plot(support, medium, label="Medium support")
    axes[0].plot(support, high, label="High support")
    axes[0].set_xlabel("Normalized support")
    axes[0].set_ylabel("Membership")
    axes[0].legend(frameon=False)
    axes[0].grid(alpha=0.25)

    axes[1].plot(uncertainty, 1.0 - uncertainty, label="Low uncertainty")
    axes[1].plot(uncertainty, uncertainty, label="High uncertainty")
    axes[1].set_xlabel("TD-residual uncertainty proxy")
    axes[1].set_ylabel("Membership")
    axes[1].legend(frameon=False)
    axes[1].grid(alpha=0.25)

    figure.tight_layout()
    FIGURES.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURES / "fig_fuzzy_memberships.pdf", bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    fuzzy_memberships()
    print("Generated public artifact figures.")


if __name__ == "__main__":
    main()
