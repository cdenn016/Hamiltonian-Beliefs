# -*- coding: utf-8 -*-
"""
Created on Sat Nov 22 09:05:24 2025

@author: chris and christine
"""

"""
Lorentz Gauge Field Experiments
================================

Test whether using SO(1,3) gauge fields (Lorentz group) instead of SO(3)
can induce Lorentzian signature in the pullback metric.

Background
----------
Currently, gauge fields φ ∈ so(3) transform under SO(3) rotations.

The Lorentz group SO(1,3) has:
- Signature (-,+,+,+) in defining representation
- Generators: 3 rotations (J_i) + 3 boosts (K_i)
- Algebra: [J_i, J_j] = ε_ijk J_k, [J_i, K_j] = ε_ijk K_k, [K_i, K_j] = -ε_ijk J_k

Key idea: The Lorentz Killing form (metric on algebra) is indefinite.
This might induce Lorentzian signature in the gauge contribution to
the pullback metric.

Author: Chris
Date: November 2025
"""

import numpy as np
from typing import Tuple, Dict, Optional
from dataclasses import dataclass

from geometry.signature_analysis import analyze_metric_signature, MetricSignature


def so3_generators() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    SO(3) generators (rotations) in 3D representation.

    Returns:
        J_1, J_2, J_3: (3, 3) antisymmetric matrices
    """
    J1 = np.array([[0, 0, 0], [0, 0, -1], [0, 1, 0]])
    J2 = np.array([[0, 0, 1], [0, 0, 0], [-1, 0, 0]])
    J3 = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 0]])
    return J1, J2, J3


def lorentz_generators() -> Tuple[Tuple, Tuple]:
    """
    SO(1,3) generators in 4D representation (Minkowski space).

    Rotations:
        J_i: Spatial rotations (i,j,k cyclic)

    Boosts:
        K_i: Boosts in i-direction

    Returns:
        rotations: (J_1, J_2, J_3)
        boosts: (K_1, K_2, K_3)

    Each is a (4, 4) matrix in Minkowski signature convention:
        η = diag(-1, 1, 1, 1)
    """
    # Rotations (same as SO(3) but embedded in 4D)
    J1 = np.array([
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, -1],
        [0, 0, 1, 0]
    ])

    J2 = np.array([
        [0, 0, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 0, 0],
        [0, -1, 0, 0]
    ])

    J3 = np.array([
        [0, 0, 0, 0],
        [0, 0, -1, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 0]
    ])

    # Boosts (mix time and space)
    K1 = np.array([
        [0, 1, 0, 0],
        [1, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0]
    ])

    K2 = np.array([
        [0, 0, 1, 0],
        [0, 0, 0, 0],
        [1, 0, 0, 0],
        [0, 0, 0, 0]
    ])

    K3 = np.array([
        [0, 0, 0, 1],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [1, 0, 0, 0]
    ])

    rotations = (J1, J2, J3)
    boosts = (K1, K2, K3)

    return rotations, boosts


def killing_form(
    X: np.ndarray,
    Y: np.ndarray,
    generators: Tuple
) -> float:
    """
    Killing form κ(X, Y) = Tr(ad_X ∘ ad_Y).

    For matrix Lie algebras: κ(X, Y) = Tr(XY)

    The Killing form is:
    - Positive definite for compact groups (SO(3))
    - Indefinite for non-compact groups (SO(1,3))

    Args:
        X: (d, d) Lie algebra element
        Y: (d, d) Lie algebra element
        generators: Tuple of generators (for structure constants)

    Returns:
        κ(X, Y): Killing form value
    """
    # For matrix algebras, simplified: κ(X, Y) ∝ Tr(XY)
    return np.trace(X @ Y)


def gauge_metric_tensor(
    phi: np.ndarray,
    gauge_group: str = "SO(3)"
) -> np.ndarray:
    """
    Compute metric on gauge field space induced by Killing form.

    For φ ∈ Lie algebra, the metric is:
        g_ij = κ(T_i, T_j)
    where T_i are generators and κ is the Killing form.

    Args:
        phi: Gauge field (coordinates in Lie algebra basis)
        gauge_group: "SO(3)" or "SO(1,3)"

    Returns:
        g: Metric tensor on gauge algebra

    Key insight:
    - SO(3): Killing form positive definite → Riemannian metric
    - SO(1,3): Killing form indefinite → Could be Lorentzian!
    """
    if gauge_group == "SO(3)":
        generators = so3_generators()
        dim = 3
    elif gauge_group == "SO(1,3)":
        rotations, boosts = lorentz_generators()
        generators = rotations + boosts
        dim = 6  # 3 rotations + 3 boosts
    else:
        raise ValueError(f"Unknown gauge group: {gauge_group}")

    # Compute Killing metric
    g = np.zeros((dim, dim))
    for i in range(dim):
        for j in range(dim):
            g[i, j] = killing_form(generators[i], generators[j], generators)

    return g


@dataclass
class LorentzGaugeExperiment:
    """
    Experiment: Compare SO(3) vs SO(1,3) gauge field metrics.

    Attributes:
        phi_so3: (3,) gauge field in so(3)
        phi_lorentz: (6,) gauge field in so(1,3) (3 rotations + 3 boosts)
        results: Dict with signature analysis
    """
    phi_so3: np.ndarray
    phi_lorentz: np.ndarray
    results: Optional[Dict] = None

    def run(self):
        """
        Compute and compare gauge metrics.

        Returns:
            results: Dict with metrics and signatures
        """
        print(f"\n{'='*70}")
        print(f"LORENTZ GAUGE FIELD EXPERIMENT")
        print(f"{'='*70}")
        print(f"SO(3) field φ: {self.phi_so3}")
        print(f"SO(1,3) field φ: {self.phi_lorentz}")
        print()

        # SO(3) gauge metric
        print("Computing SO(3) gauge metric...")
        g_so3 = gauge_metric_tensor(self.phi_so3, gauge_group="SO(3)")
        sig_so3 = analyze_metric_signature(g_so3)
        print(f"  Signature: {sig_so3}")
        print(f"  Eigenvalues: {sig_so3.eigenvalues}")

        # SO(1,3) gauge metric
        print("\nComputing SO(1,3) gauge metric...")
        g_lorentz = gauge_metric_tensor(self.phi_lorentz, gauge_group="SO(1,3)")
        sig_lorentz = analyze_metric_signature(g_lorentz)
        print(f"  Signature: {sig_lorentz}")
        print(f"  Eigenvalues: {sig_lorentz.eigenvalues}")

        if sig_lorentz.signature == MetricSignature.LORENTZIAN:
            print(f"\n🎉 LORENTZIAN SIGNATURE IN SO(1,3) GAUGE METRIC!")
            print(f"Timelike direction: {sig_lorentz.timelike_direction}")
        elif sig_lorentz.signature == MetricSignature.INDEFINITE:
            print(f"\n⚠️  Indefinite signature (multiple negative eigenvalues)")
            print(f"   This suggests Lorentz group structure is affecting metric!")

        self.results = {
            'g_so3': g_so3,
            'sig_so3': sig_so3,
            'g_lorentz': g_lorentz,
            'sig_lorentz': sig_lorentz
        }

        print(f"\n{'='*70}")
        print(f"COMPARISON")
        print(f"{'='*70}")
        print(f"SO(3):    {sig_so3.signature.value:12s}  ({sig_so3.signature_tuple})")
        print(f"SO(1,3):  {sig_lorentz.signature.value:12s}  ({sig_lorentz.signature_tuple})")
        print(f"{'='*70}\n")

        return self.results

    def visualize_algebras(self, out_path: Optional[str] = None):
        """
        Visualize structure of SO(3) vs SO(1,3) generators.
        """
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # SO(3) Killing form
        ax1 = axes[0]
        g_so3 = self.results['g_so3']
        im1 = ax1.imshow(g_so3, cmap='RdBu_r', aspect='auto')
        ax1.set_title('SO(3) Killing Form')
        ax1.set_xlabel('Generator index')
        ax1.set_ylabel('Generator index')
        plt.colorbar(im1, ax=ax1)

        # SO(1,3) Killing form
        ax2 = axes[1]
        g_lorentz = self.results['g_lorentz']
        im2 = ax2.imshow(g_lorentz, cmap='RdBu_r', aspect='auto')
        ax2.set_title('SO(1,3) Killing Form')
        ax2.set_xlabel('Generator index (0-2: rotations, 3-5: boosts)')
        ax2.set_ylabel('Generator index')
        plt.colorbar(im2, ax=ax2)

        plt.suptitle('Gauge Group Killing Forms', fontsize=14, y=1.0)
        plt.tight_layout()

        if out_path:
            plt.savefig(out_path, dpi=150, bbox_inches='tight')
            print(f"✓ Saved {out_path}")
        else:
            plt.show()

        plt.close()


def verify_lorentz_algebra():
    """
    Verify Lorentz algebra commutation relations.

    [J_i, J_j] = ε_ijk J_k
    [J_i, K_j] = ε_ijk K_k
    [K_i, K_j] = -ε_ijk J_k  (note the minus sign!)
    """
    rotations, boosts = lorentz_generators()
    J1, J2, J3 = rotations
    K1, K2, K3 = boosts

    print("\nVerifying Lorentz algebra commutation relations:")
    print("=" * 60)

    # [J_1, J_2] = J_3
    comm_J1_J2 = J1 @ J2 - J2 @ J1
    print(f"[J_1, J_2] - J_3 norm: {np.linalg.norm(comm_J1_J2 - J3):.6e}")

    # [J_1, K_1] = 0 (rotation and boost in same direction commute)
    comm_J1_K1 = J1 @ K1 - K1 @ J1
    print(f"[J_1, K_1] norm: {np.linalg.norm(comm_J1_K1):.6e}")

    # [J_1, K_2] = K_3
    comm_J1_K2 = J1 @ K2 - K2 @ J1
    print(f"[J_1, K_2] - K_3 norm: {np.linalg.norm(comm_J1_K2 - K3):.6e}")

    # [K_1, K_2] = -J_3 (boosts don't commute!)
    comm_K1_K2 = K1 @ K2 - K2 @ K1
    print(f"[K_1, K_2] + J_3 norm: {np.linalg.norm(comm_K1_K2 + J3):.6e}")

    print("=" * 60)


def run_lorentz_gauge_test():
    """
    Test SO(3) vs SO(1,3) gauge metrics.
    """
    # Verify algebra first
    verify_lorentz_algebra()

    # Small random gauge fields
    phi_so3 = np.random.randn(3) * 0.1
    phi_lorentz = np.random.randn(6) * 0.1  # 3 rotations + 3 boosts

    exp = LorentzGaugeExperiment(
        phi_so3=phi_so3,
        phi_lorentz=phi_lorentz
    )

    results = exp.run()
    exp.visualize_algebras(out_path="_results/lorentz_gauge_test.png")

    return exp


if __name__ == "__main__":
    run_lorentz_gauge_test()