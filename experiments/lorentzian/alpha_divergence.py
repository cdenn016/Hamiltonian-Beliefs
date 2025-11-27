# -*- coding: utf-8 -*-
"""
Created on Sat Nov 22 09:04:42 2025

@author: chris and christine
"""

"""
Alpha-Divergence Experiments
=============================

Test whether α-divergences with α < 0 can induce indefinite metrics
that might reveal Lorentzian structure.

Background
----------
The α-divergence between distributions p and q is:

    D_α(p||q) = (4/(1-α²)) ∫ [1 - p^((1+α)/2) q^((1-α)/2)] dx

Special cases:
- α = 0: Hellinger divergence
- α → 1: KL divergence D(p||q)
- α → -1: Reverse KL D(q||p)

Key insight: For α < 0, the induced metric can be indefinite, potentially
giving Lorentzian signature in certain regions.

Author: Chris
Date: November 2025
"""

import numpy as np
from typing import Tuple, Dict, Optional
from dataclasses import dataclass

from geometry.signature_analysis import analyze_metric_signature, MetricSignature


def alpha_divergence(
    p: np.ndarray,
    q: np.ndarray,
    alpha: float
) -> float:
    """
    Compute α-divergence D_α(p||q).

    Args:
        p: (N,) distribution (must be normalized)
        q: (N,) distribution (must be normalized)
        alpha: Divergence parameter

    Returns:
        D_α(p||q)

    Note:
        - For |α| = 1, use limiting KL divergence
        - Assumes discrete distributions
    """
    eps = 1e-10

    if np.abs(alpha - 1.0) < eps:
        # KL divergence D(p||q)
        return np.sum(p * np.log((p + eps) / (q + eps)))

    if np.abs(alpha + 1.0) < eps:
        # Reverse KL divergence D(q||p)
        return np.sum(q * np.log((q + eps) / (p + eps)))

    # General α-divergence
    p_safe = np.maximum(p, eps)
    q_safe = np.maximum(q, eps)

    integrand = 1.0 - p_safe**((1 + alpha) / 2) * q_safe**((1 - alpha) / 2)
    return (4.0 / (1.0 - alpha**2)) * np.sum(integrand)


def alpha_metric_tensor(
    mu: np.ndarray,
    Sigma: np.ndarray,
    alpha: float,
    eps: float = 1e-4
) -> np.ndarray:
    """
    Compute α-divergence induced metric tensor for Gaussian.

    For Gaussian N(μ, Σ), the α-metric is derived from second derivatives
    of D_α(N(μ,Σ) || N(μ+δμ, Σ+δΣ)).

    Args:
        mu: (K,) mean
        Sigma: (K, K) covariance
        alpha: Divergence parameter
        eps: Finite difference step

    Returns:
        g: (2K, 2K) metric tensor (μ and Sigma parameters combined)
            First K coords: μ
            Last K coords: diag(Σ) (simplified)

    Note:
        This is a simplified implementation. Full α-metric for Gaussians
        is more complex. See Amari (2016) for details.
    """
    K = len(mu)

    # For simplicity, parameterize by (μ, diag(Σ))
    # Full implementation would use all Σ parameters
    d = 2 * K  # Dimension of parameter space

    g = np.zeros((d, d))

    # Finite difference approximation of metric
    # g_ij = ∂²D_α / ∂θ_i ∂θ_j

    # Helper: Gaussian pdf
    def gaussian_pdf(x, mu_val, Sigma_val):
        det = np.linalg.det(Sigma_val)
        inv = np.linalg.inv(Sigma_val)
        diff = x - mu_val
        exponent = -0.5 * diff @ inv @ diff
        return (2 * np.pi)**(-K/2) * det**(-0.5) * np.exp(exponent)

    # Sample points for numerical integration
    n_samples = 1000
    samples = np.random.multivariate_normal(mu, Sigma, size=n_samples)

    # Base distribution
    p_base = np.array([gaussian_pdf(x, mu, Sigma) for x in samples])
    p_base = p_base / np.sum(p_base)  # Normalize

    # Compute metric by perturbing parameters
    for i in range(d):
        for j in range(i, d):
            # Perturb parameter i
            mu_i, Sigma_i = mu.copy(), Sigma.copy()
            if i < K:
                mu_i[i] += eps
            else:
                Sigma_i[i - K, i - K] += eps

            # Perturb parameter j
            mu_j, Sigma_j = mu.copy(), Sigma.copy()
            if j < K:
                mu_j[j] += eps
            else:
                Sigma_j[j - K, j - K] += eps

            # Perturb both
            mu_ij, Sigma_ij = mu.copy(), Sigma.copy()
            if i < K:
                mu_ij[i] += eps
            else:
                Sigma_ij[i - K, i - K] += eps
            if j < K:
                mu_ij[j] += eps
            else:
                Sigma_ij[j - K, j - K] += eps

            # Compute divergences
            q_i = np.array([gaussian_pdf(x, mu_i, Sigma_i) for x in samples])
            q_i = q_i / np.sum(q_i)

            q_j = np.array([gaussian_pdf(x, mu_j, Sigma_j) for x in samples])
            q_j = q_j / np.sum(q_j)

            q_ij = np.array([gaussian_pdf(x, mu_ij, Sigma_ij) for x in samples])
            q_ij = q_ij / np.sum(q_ij)

            # Second derivative: [D(0,0) - D(i,0) - D(0,j) + D(i,j)] / eps²
            D_00 = alpha_divergence(p_base, p_base, alpha)
            D_i0 = alpha_divergence(p_base, q_i, alpha)
            D_0j = alpha_divergence(p_base, q_j, alpha)
            D_ij = alpha_divergence(p_base, q_ij, alpha)

            g[i, j] = (D_00 - D_i0 - D_0j + D_ij) / (eps**2)
            g[j, i] = g[i, j]  # Symmetry

    return g


@dataclass
class AlphaDivergenceExperiment:
    """
    Experiment configuration for α-divergence testing.

    Attributes:
        alpha_values: List of α values to test
        mu: Mean of Gaussian
        Sigma: Covariance of Gaussian
        results: Signature analysis for each α
    """
    alpha_values: np.ndarray
    mu: np.ndarray
    Sigma: np.ndarray
    results: Optional[Dict] = None

    def run(self):
        """
        Run experiment: compute metric signature for each α.

        Returns:
            results: Dict with keys 'alpha', 'signature', 'eigenvalues'
        """
        print(f"\n{'='*70}")
        print(f"ALPHA-DIVERGENCE SIGNATURE EXPERIMENT")
        print(f"{'='*70}")
        print(f"Testing α values: {self.alpha_values}")
        print(f"Gaussian: μ={self.mu}, Σ=diag({np.diag(self.Sigma)})")
        print()

        signatures = []
        eigenvalues_list = []
        n_negative_list = []

        for alpha in self.alpha_values:
            print(f"α = {alpha:6.2f} ... ", end='')

            # Compute α-metric
            g = alpha_metric_tensor(self.mu, self.Sigma, alpha)

            # Analyze signature
            sig = analyze_metric_signature(g)

            signatures.append(sig.signature.value)
            eigenvalues_list.append(sig.eigenvalues)
            n_negative_list.append(sig.signature_tuple[0])

            print(f"{sig.signature.value:12s}  ({sig.signature_tuple[0]} negative eigenvalues)")

        self.results = {
            'alpha': self.alpha_values,
            'signature': signatures,
            'eigenvalues': np.array(eigenvalues_list),
            'n_negative': np.array(n_negative_list)
        }

        print(f"\n{'='*70}")
        print(f"SUMMARY")
        print(f"{'='*70}")
        n_lorentzian = np.sum(np.array(n_negative_list) == 1)
        n_indefinite = np.sum(np.array(n_negative_list) > 1)
        print(f"Lorentzian signatures: {n_lorentzian}/{len(self.alpha_values)}")
        print(f"Indefinite signatures: {n_indefinite}/{len(self.alpha_values)}")
        print(f"{'='*70}\n")

        return self.results

    def plot_results(self, out_path: Optional[str] = None):
        """Plot signature vs α."""
        if self.results is None:
            raise ValueError("Must run experiment before plotting")

        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 1, figsize=(10, 8))

        # Top: Number of negative eigenvalues
        ax1 = axes[0]
        ax1.plot(self.results['alpha'], self.results['n_negative'],
                 marker='o', linewidth=2, markersize=6)
        ax1.axhline(1, color='red', linestyle='--', alpha=0.5,
                    label='Lorentzian (1 negative)')
        ax1.set_xlabel('α')
        ax1.set_ylabel('# Negative Eigenvalues')
        ax1.set_title('Signature vs α-Divergence Parameter')
        ax1.grid(alpha=0.3)
        ax1.legend()

        # Bottom: Eigenvalues
        ax2 = axes[1]
        eigenvalues = self.results['eigenvalues']
        for i in range(eigenvalues.shape[1]):
            ax2.plot(self.results['alpha'], eigenvalues[:, i],
                     label=f'λ_{i}', linewidth=2)
        ax2.axhline(0, color='black', linestyle='--', alpha=0.3)
        ax2.set_xlabel('α')
        ax2.set_ylabel('Eigenvalue')
        ax2.set_title('Metric Eigenvalues vs α')
        ax2.legend()
        ax2.grid(alpha=0.3)

        plt.tight_layout()

        if out_path:
            plt.savefig(out_path, dpi=150, bbox_inches='tight')
            print(f"✓ Saved {out_path}")
        else:
            plt.show()

        plt.close()


def run_alpha_divergence_test():
    """
    Quick test: Scan α from -2 to +2 and check for Lorentzian signatures.
    """
    # Simple 2D Gaussian
    mu = np.array([0.0, 0.0])
    Sigma = np.eye(2)

    # Test α values (including negative)
    alpha_values = np.linspace(-2.0, 2.0, 21)

    exp = AlphaDivergenceExperiment(
        alpha_values=alpha_values,
        mu=mu,
        Sigma=Sigma
    )

    results = exp.run()
    exp.plot_results(out_path="_results/alpha_divergence_test.png")

    return exp


if __name__ == "__main__":
    run_alpha_divergence_test()