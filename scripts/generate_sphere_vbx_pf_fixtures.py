# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "scipy"]
# ///
"""Generate deterministic SphereVBx-PF numerical fixtures from paper equations 5-8."""

from pathlib import Path

import numpy as np
from scipy.special import ive, logsumexp


FA = 1.5
FB = 6.0
MAX_ITERS = 4
RESPONSIBILITY_TOLERANCE = 1e-12


def mean_length(dimension: int, kappa: np.ndarray) -> np.ndarray:
    """Return I_(d/2)(kappa) / I_(d/2-1)(kappa) with scaled Bessel values."""
    order = dimension / 2 - 1
    result = np.zeros_like(kappa)
    positive = kappa > 0
    result[positive] = ive(order + 1, kappa[positive]) / ive(order, kappa[positive])
    return result


def run(features: np.ndarray, gamma_init: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Apply SphereVBx-PF equations 5-8 from supplied responsibilities."""
    features = features / np.linalg.norm(features, axis=1, keepdims=True)
    gamma = gamma_init / gamma_init.sum(axis=1, keepdims=True)
    pi = gamma.sum(axis=0)
    pi /= pi.sum()

    for _ in range(MAX_ITERS):
        natural = (FA / FB) * gamma.T @ features
        kappa = np.linalg.norm(natural, axis=1)
        expected = np.zeros_like(natural)
        positive = kappa > 0
        expected[positive] = (
            natural[positive]
            * (mean_length(features.shape[1], kappa[positive]) / kappa[positive])[
                :, None
            ]
        )

        log_scores = FA * features @ expected.T
        log_prior = np.full_like(pi, -np.inf)
        log_prior[pi > 0] = np.log(pi[pi > 0])
        normalized = log_scores + log_prior
        next_gamma = np.exp(normalized - logsumexp(normalized, axis=1, keepdims=True))
        pi = next_gamma.sum(axis=0)
        pi /= pi.sum()
        maximum_change = np.max(np.abs(next_gamma - gamma))
        gamma = next_gamma
        if maximum_change <= RESPONSIBILITY_TOLERANCE:
            break

    return gamma, pi


def main() -> None:
    fixtures = Path(__file__).resolve().parents[1] / "fixtures"
    features = np.array(
        [
            [1.0, 0.5, 0.0, 0.0],
            [0.75, 0.5, 0.25, 0.0],
            [0.5, 0.75, 0.25, 0.0],
            [0.5, 1.0, 0.0, 0.25],
            [0.25, 0.75, 0.5, 0.25],
            [0.0, 0.5, 1.0, 0.25],
        ],
        dtype=np.float64,
    )
    gamma_init = np.array(
        [
            [0.75, 0.25],
            [0.65, 0.35],
            [0.55, 0.45],
            [0.45, 0.55],
            [0.35, 0.65],
            [0.25, 0.75],
        ],
        dtype=np.float64,
    )
    gamma, pi = run(features, gamma_init)

    np.save(fixtures / "sphere_vbx_pf_features.npy", features)
    np.save(fixtures / "sphere_vbx_pf_initial_gamma.npy", gamma_init)
    np.save(fixtures / "sphere_vbx_pf_gamma.npy", gamma)
    np.save(fixtures / "sphere_vbx_pf_pi.npy", pi)


if __name__ == "__main__":
    main()
