# -*- coding: utf-8 -*-
"""
Created on Sat Nov 22 09:05:02 2025

@author: chris and christine
"""

"""
Hamiltonian Belief Dynamics
============================

Test whether Hamiltonian formulation of belief dynamics with symplectic
structure can induce Lorentzian signature.

Background
----------
In Hamiltonian mechanics, phase space has natural symplectic structure:
    ω = dp ∧ dq

The metric in phase space (q, p) can have signature (-,+,+,...) if we use
the natural pairing from the Hamiltonian.

For beliefs, we can try:
- q = μ (belief mean, position)
- p = conjugate momentum (related to Σ or dμ/dt)
- H = Free energy functional

Key Question: Does the symplectic structure + Hamiltonian flow induce
a natural Lorentzian metric on belief space?

Author: Chris
Date: November 2025
"""

import numpy as np
from typing import Tuple, Dict, Optional, Callable
from dataclasses import dataclass

from geometry.signature_analysis import analyze_metric_signature, MetricSignature


def symplectic_form_2d() -> np.ndarray:
    """
    Standard symplectic form ω in 2D phase space (q, p).

    Returns:
        ω: (2, 2) antisymmetric matrix [[0, 1], [-1, 0]]
    """
    return np.array([[0, 1], [-1, 0]])


def canonical_symplectic_form(n: int) -> np.ndarray:
    """
    Canonical symplectic form ω in 2n-dimensional phase space.

    Phase space coords: (q_1, ..., q_n, p_1, ..., p_n)

    Returns:
        ω: (2n, 2n) block matrix [[0, I_n], [-I_n, 0]]
    """
    I_n = np.eye(n)
    zero_n = np.zeros((n, n))
    return np.block([[zero_n, I_n], [-I_n, zero_n]])


def metric_from_symplectic(
    omega: np.ndarray,
    J: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Construct metric from symplectic form using almost complex structure.

    For a symplectic manifold (M, ω), an almost complex structure J
    compatible with ω defines a Riemannian metric:
        g(X, Y) = ω(X, JY)

    However, if we use the Hamiltonian to define a different pairing,
    we might get an indefinite metric.

    Args:
        omega: (2n, 2n) symplectic form
        J: (2n, 2n) almost complex structure (default: standard J)

    Returns:
        g: (2n, 2n) metric tensor

    Note:
        Standard construction gives Riemannian metric.
        For Lorentzian, we need modified construction.
    """
    n = omega.shape[0]

    if J is None:
        # Standard almost complex structure: J² = -I
        # In canonical coords: J = [[0, -I], [I, 0]]
        m = n // 2
        I_m = np.eye(m)
        zero_m = np.zeros((m, m))
        J = np.block([[zero_m, -I_m], [I_m, zero_m]])

    # Standard metric: g(X, Y) = ω(X, JY)
    # This is always Riemannian for compatible J
    g = omega @ J

    return g


def hamiltonian_metric(
    q: np.ndarray,
    p: np.ndarray,
    hamiltonian: Callable,
    use_energy_time: bool = False
) -> np.ndarray:
    """
    Construct metric on phase space using Hamiltonian.

    Two approaches:
    1. Standard: Use symplectic form + complex structure (Riemannian)
    2. Energy-time: Use Hamiltonian as "time" component (Lorentzian)

    Args:
        q: (n,) position coordinates
        p: (n,) momentum coordinates
        hamiltonian: Function H(q, p) -> float
        use_energy_time: If True, use Hamiltonian to define time component

    Returns:
        g: (2n, 2n) metric on phase space (q, p)

    Key insight for Lorentzian signature:
    If we treat H as "time" coordinate and (q,p) as "space",
    we get metric like ds² = -dH² + dq² + dp² (Lorentzian!)
    """
    n = len(q)
    phase_dim = 2 * n

    if not use_energy_time:
        # Standard symplectic metric (Riemannian)
        omega = canonical_symplectic_form(n)
        g = metric_from_symplectic(omega)
        return g

    # Energy-time construction (attempt at Lorentzian)
    # Idea: Embed (q, p, H) as (2n+1)-dimensional spacetime
    # with metric ds² = -dH² + g_phase(dq, dp)

    # Compute Hamiltonian gradient
    eps = 1e-5
    dH_dq = np.zeros(n)
    dH_dp = np.zeros(n)

    for i in range(n):
        q_plus = q.copy()
        q_plus[i] += eps
        q_minus = q.copy()
        q_minus[i] -= eps
        dH_dq[i] = (hamiltonian(q_plus, p) - hamiltonian(q_minus, p)) / (2 * eps)

        p_plus = p.copy()
        p_plus[i] += eps
        p_minus = p.copy()
        p_minus[i] -= eps
        dH_dp[i] = (hamiltonian(q, p_plus) - hamiltonian(q, p_minus)) / (2 * eps)

    # Phase space metric (spatial part, positive)
    g_spatial = np.eye(phase_dim)  # Simplified

    # Modify to include Hamiltonian direction
    # g = g_spatial - (dH ⊗ dH) / H  (makes one eigenvalue negative)
    dH = np.concatenate([dH_dq, dH_dp])  # (2n,)
    H_val = hamiltonian(q, p)

    if np.abs(H_val) > 1e-10:
        # Add negative contribution from energy
        g = g_spatial - np.outer(dH, dH) / (H_val + 1e-10)
    else:
        g = g_spatial

    return g


@dataclass
class HamiltonianBeliefExperiment:
    """
    Experiment: Test Hamiltonian belief dynamics for Lorentzian signature.

    Attributes:
        hamiltonian: Energy functional H(μ, π) where π is conjugate momentum
        initial_mu: Initial belief mean (position)
        initial_p: Initial momentum
        use_energy_time: Whether to use energy-time construction
    """
    hamiltonian: Callable
    initial_mu: np.ndarray
    initial_p: np.ndarray
    use_energy_time: bool = False
    result: Optional[Dict] = None

    def run(self):
        """
        Compute metric signature at initial point.

        Returns:
            result: Dict with metric, signature, eigenvalues
        """
        print(f"\n{'='*70}")
        print(f"HAMILTONIAN BELIEF DYNAMICS EXPERIMENT")
        print(f"{'='*70}")
        print(f"Initial μ: {self.initial_mu}")
        print(f"Initial p: {self.initial_p}")
        print(f"Use energy-time construction: {self.use_energy_time}")
        print()

        # Compute Hamiltonian value
        H_val = self.hamiltonian(self.initial_mu, self.initial_p)
        print(f"Hamiltonian value H(μ, p) = {H_val:.6f}")

        # Compute metric
        g = hamiltonian_metric(
            self.initial_mu,
            self.initial_p,
            self.hamiltonian,
            use_energy_time=self.use_energy_time
        )

        # Analyze signature
        sig = analyze_metric_signature(g)

        print(f"\nMetric signature: {sig}")
        print(f"Eigenvalues: {sig.eigenvalues}")

        if sig.signature == MetricSignature.LORENTZIAN:
            print(f"\n🎉 LORENTZIAN SIGNATURE DETECTED!")
            print(f"Timelike direction: {sig.timelike_direction}")

        self.result = {
            'metric': g,
            'signature': sig,
            'hamiltonian_value': H_val
        }

        print(f"{'='*70}\n")
        return self.result


def free_energy_hamiltonian(mu: np.ndarray, p: np.ndarray) -> float:
    """
    Simple free energy Hamiltonian.

    H = (1/2) p^T p + V(μ)

    where V(μ) is potential (e.g., KL divergence to prior).

    Args:
        mu: (n,) belief mean (position)
        p: (n,) conjugate momentum

    Returns:
        H: Hamiltonian value
    """
    # Kinetic energy
    T = 0.5 * np.dot(p, p)

    # Potential energy (prior alignment)
    # V(μ) = (1/2) ||μ - μ_prior||²
    mu_prior = np.zeros_like(mu)
    V = 0.5 * np.linalg.norm(mu - mu_prior)**2

    return T + V


def harmonic_oscillator_hamiltonian(mu: np.ndarray, p: np.ndarray) -> float:
    """
    Harmonic oscillator Hamiltonian (separable).

    H = (1/2) (p^T p + μ^T μ)

    This is integrable and has closed periodic orbits.

    Args:
        mu: (n,) position
        p: (n,) momentum

    Returns:
        H: Energy
    """
    return 0.5 * (np.dot(p, p) + np.dot(mu, mu))


def run_hamiltonian_test():
    """
    Test Hamiltonian belief dynamics for Lorentzian signature.
    """
    # 2D belief space
    mu = np.array([1.0, 0.5])
    p = np.array([0.0, 1.0])

    print("Test 1: Standard symplectic metric (expect Riemannian)")
    exp1 = HamiltonianBeliefExperiment(
        hamiltonian=free_energy_hamiltonian,
        initial_mu=mu,
        initial_p=p,
        use_energy_time=False
    )
    exp1.run()

    print("\nTest 2: Energy-time construction (expect Lorentzian?)")
    exp2 = HamiltonianBeliefExperiment(
        hamiltonian=free_energy_hamiltonian,
        initial_mu=mu,
        initial_p=p,
        use_energy_time=True
    )
    exp2.run()

    print("\nTest 3: Harmonic oscillator with energy-time")
    exp3 = HamiltonianBeliefExperiment(
        hamiltonian=harmonic_oscillator_hamiltonian,
        initial_mu=mu,
        initial_p=p,
        use_energy_time=True
    )
    exp3.run()

    return [exp1, exp2, exp3]


if __name__ == "__main__":
    run_hamiltonian_test()