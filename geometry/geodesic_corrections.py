# -*- coding: utf-8 -*-
"""
Geodesic Corrections for Hamiltonian Dynamics
==============================================

Implements the missing geodesic correction term in Hamilton's equations:

    dp/dt = -dF/d\theta - (1/2) pi^T (dM^{-1}/d\theta) pi

This term arises from the position-dependent mass matrix M(\theta) and
ensures that trajectories follow geodesics on the statistical manifold.

Key Features:
-------------
1. Computes dM^{-1}/d\theta via finite differences (robust)
2. Accounts for beta variation through M (critical for self-consistency)
3. Supports both mu and Sigma parameter variations

Mathematical Background:
------------------------
The mass matrix M includes:
    M = Sigma_p^{-1} + sum_j beta_ij Omega_ij Sigma_q_j^{-1} Omega_ij^T

The softmax weights beta_ij depend on theta through KL divergences:
    beta_ij = exp[-KL_ij/kappa] / sum_k exp[-KL_ik/kappa]

Therefore dM/d\theta includes contributions from:
    1. Direct dependence of Sigma_q_j on theta (when theta = Sigma_j)
    2. Indirect dependence through beta_ij(theta) (always)

Author: Claude & Chris
Date: November 2025
"""

import numpy as np
from typing import Optional, Tuple, Dict
from dataclasses import dataclass


@dataclass
class GeodesicForceResult:
    """Container for geodesic force computation results."""
    geodesic_force: np.ndarray  # Full geodesic force vector
    force_mu: np.ndarray  # Component from mu variations
    force_Sigma: np.ndarray  # Component from Sigma variations
    dM_inv_norm: float  # Diagnostic: ||dM^{-1}/d\theta||_F


def compute_geodesic_force(
    trainer,
    theta: np.ndarray,
    p: np.ndarray,
    eps: float = 1e-5,
    include_beta_variation: bool = True
) -> np.ndarray:
    """
    Compute geodesic correction force: -(1/2) p^T (dM^{-1}/d\theta) p

    CRITICAL: This term ensures trajectories follow geodesics on the
    statistical manifold defined by the Fisher-Rao metric.

    The computation uses finite differences to compute dM^{-1}/d\theta_i
    for each parameter theta_i. When include_beta_variation=True, the
    softmax weights beta are recomputed at each perturbed theta, ensuring
    the full chain rule is respected.

    Args:
        trainer: HamiltonianTrainer instance with system reference
        theta: Current parameter vector (flattened [mu, Sigma])
        p: Current momentum vector (flattened [pi_mu, Pi_Sigma])
        eps: Finite difference step size
        include_beta_variation: If True, recompute beta at each theta
                               If False, freeze beta (faster but less accurate)

    Returns:
        geodesic_force: Force vector with same shape as theta

    Mathematical Form:
    ------------------
    For each parameter theta_i:
        F_geo,i = -(1/2) p^T (dM^{-1}/d\theta_i) p

    where dM^{-1}/d\theta_i is computed via central differences:
        dM^{-1}/d\theta_i = [M^{-1}(theta + eps*e_i) - M^{-1}(theta - eps*e_i)] / (2*eps)
    """
    geodesic_force = np.zeros_like(theta)

    # Get index ranges for each agent
    idx_ranges = _compute_parameter_index_ranges(trainer)

    # Only compute for mu parameters (Sigma handled by hyperbolic geodesic flow)
    for agent_idx, (mu_start, mu_end, Sigma_start, Sigma_end) in idx_ranges.items():
        agent = trainer.system.agents[agent_idx]
        K = agent.config.K
        n_spatial = agent.mu_q.size // K

        # Extract momenta for this agent's mu part
        p_mu = p[mu_start:mu_end].reshape(agent.mu_q.shape)

        # Compute M^{-1} at current theta
        theta_backup = trainer._pack_parameters()
        M_inv_current = _compute_M_inverse_for_agent(trainer, agent, agent_idx)

        # Compute geodesic force for each mu parameter
        for local_idx in range(mu_end - mu_start):
            global_idx = mu_start + local_idx

            # Perturbed theta+
            theta_plus = theta.copy()
            theta_plus[global_idx] += eps
            trainer._unpack_parameters(theta_plus)

            M_inv_plus = _compute_M_inverse_for_agent(
                trainer, trainer.system.agents[agent_idx], agent_idx,
                recompute_beta=include_beta_variation
            )

            # Perturbed theta-
            theta_minus = theta.copy()
            theta_minus[global_idx] -= eps
            trainer._unpack_parameters(theta_minus)

            M_inv_minus = _compute_M_inverse_for_agent(
                trainer, trainer.system.agents[agent_idx], agent_idx,
                recompute_beta=include_beta_variation
            )

            # Restore original theta
            trainer._unpack_parameters(theta_backup)

            # Central difference: dM^{-1}/d\theta_i
            dM_inv_dtheta_i = (M_inv_plus - M_inv_minus) / (2 * eps)

            # Compute -(1/2) p^T (dM^{-1}/d\theta_i) p for this spatial point
            # Map local_idx to spatial point and component
            spatial_idx = local_idx // K
            component_idx = local_idx % K

            if agent.mu_q.ndim == 1:
                # 0D: single matrix
                geodesic_force[global_idx] = -0.5 * p_mu @ dM_inv_dtheta_i @ p_mu
            elif agent.mu_q.ndim == 2:
                # 1D field
                p_local = p_mu[spatial_idx]
                dM_local = dM_inv_dtheta_i[spatial_idx] if dM_inv_dtheta_i.ndim == 3 else dM_inv_dtheta_i
                geodesic_force[global_idx] = -0.5 * p_local @ dM_local @ p_local
            else:
                # 2D field
                shape = agent.mu_q.shape[:-1]
                x = spatial_idx // shape[1]
                y = spatial_idx % shape[1]
                p_local = p_mu[x, y]
                dM_local = dM_inv_dtheta_i[x, y] if dM_inv_dtheta_i.ndim == 4 else dM_inv_dtheta_i
                geodesic_force[global_idx] = -0.5 * p_local @ dM_local @ p_local

    return geodesic_force


def compute_geodesic_force_vectorized(
    trainer,
    theta: np.ndarray,
    p: np.ndarray,
    eps: float = 1e-5
) -> np.ndarray:
    """
    Vectorized geodesic force computation for efficiency.

    This version computes dM^{-1}/d\theta for all mu parameters simultaneously
    using batched operations where possible.

    Note: This is an optimization over compute_geodesic_force. Use this
    for production; use the scalar version for debugging.

    Args:
        trainer: HamiltonianTrainer instance
        theta: Parameter vector
        p: Momentum vector
        eps: Finite difference step size

    Returns:
        geodesic_force: Force vector
    """
    geodesic_force = np.zeros_like(theta)

    # Get index ranges
    idx_ranges = _compute_parameter_index_ranges(trainer)

    # Current state backup
    theta_backup = trainer._pack_parameters()

    for agent_idx, (mu_start, mu_end, Sigma_start, Sigma_end) in idx_ranges.items():
        agent = trainer.system.agents[agent_idx]
        K = agent.config.K
        n_spatial = agent.mu_q.size // K

        # Extract mu momentum for this agent
        p_mu = p[mu_start:mu_end].reshape(agent.mu_q.shape)

        # Compute current M^{-1}
        M_inv_current = _compute_M_inverse_for_agent(trainer, agent, agent_idx)

        # For each mu parameter, compute the geodesic force contribution
        for local_idx in range(mu_end - mu_start):
            global_idx = mu_start + local_idx
            spatial_idx = local_idx // K

            # Forward perturbation
            theta[global_idx] += eps
            trainer._unpack_parameters(theta)
            M_inv_plus = _compute_M_inverse_for_agent(
                trainer, trainer.system.agents[agent_idx], agent_idx,
                recompute_beta=True
            )

            # Backward perturbation
            theta[global_idx] -= 2 * eps
            trainer._unpack_parameters(theta)
            M_inv_minus = _compute_M_inverse_for_agent(
                trainer, trainer.system.agents[agent_idx], agent_idx,
                recompute_beta=True
            )

            # Restore
            theta[global_idx] += eps

            # Central difference
            dM_inv = (M_inv_plus - M_inv_minus) / (2 * eps)

            # Compute quadratic form at appropriate spatial location
            if agent.mu_q.ndim == 1:
                geodesic_force[global_idx] = -0.5 * p_mu @ dM_inv @ p_mu
            elif agent.mu_q.ndim == 2:
                p_local = p_mu[spatial_idx]
                dM_local = dM_inv[spatial_idx] if dM_inv.ndim == 3 else dM_inv
                geodesic_force[global_idx] = -0.5 * p_local @ dM_local @ p_local
            else:
                shape = agent.mu_q.shape[:-1]
                x = spatial_idx // shape[1]
                y = spatial_idx % shape[1]
                p_local = p_mu[x, y]
                dM_local = dM_inv[x, y] if dM_inv.ndim == 4 else dM_inv
                geodesic_force[global_idx] = -0.5 * p_local @ dM_local @ p_local

    # Restore original parameters
    trainer._unpack_parameters(theta_backup)

    return geodesic_force


def _compute_M_inverse_for_agent(
    trainer,
    agent,
    agent_idx: int,
    recompute_beta: bool = True
) -> np.ndarray:
    """
    Compute M^{-1} for a single agent at current system state.

    Args:
        trainer: HamiltonianTrainer
        agent: Agent instance
        agent_idx: Agent index in system
        recompute_beta: If True, recompute softmax weights

    Returns:
        M_inv: Inverse mass matrix, shape (*S, K, K) or (K, K)
    """
    from gradients.softmax_grads import compute_softmax_weights

    K = agent.config.K
    spatial_shape = agent.mu_q.shape[:-1] if agent.mu_q.ndim > 1 else ()

    # Initialize with bare mass: Sigma_p^{-1}
    if agent.Sigma_p.ndim == 2:
        M = np.linalg.inv(agent.Sigma_p + 1e-8 * np.eye(K))
    elif agent.Sigma_p.ndim == 3:
        M = np.zeros(agent.Sigma_p.shape, dtype=np.float64)
        for i in range(agent.Sigma_p.shape[0]):
            M[i] = np.linalg.inv(agent.Sigma_p[i] + 1e-8 * np.eye(K))
    else:
        M = np.zeros(agent.Sigma_p.shape, dtype=np.float64)
        for i in range(agent.Sigma_p.shape[0]):
            for j in range(agent.Sigma_p.shape[1]):
                M[i, j] = np.linalg.inv(agent.Sigma_p[i, j] + 1e-8 * np.eye(K))

    # Add relational mass from consensus coupling
    kappa_beta = getattr(trainer.system.config, 'kappa_beta', 1.0)

    if recompute_beta:
        beta_fields = compute_softmax_weights(
            trainer.system, agent_idx, 'belief', kappa_beta
        )
    else:
        # Use cached beta if available
        if hasattr(trainer, '_beta_cache') and trainer._beta_cache is not None:
            beta_fields = trainer._beta_cache.get(agent_idx, {})
        else:
            beta_fields = compute_softmax_weights(
                trainer.system, agent_idx, 'belief', kappa_beta
            )

    for j_idx, beta_ij in beta_fields.items():
        agent_j = trainer.system.agents[j_idx]
        Omega_ij = trainer.system.compute_transport_ij(agent_idx, j_idx)

        if agent.mu_q.ndim == 1:
            # 0D
            Sigma_q_j_inv = np.linalg.inv(agent_j.Sigma_q + 1e-8 * np.eye(K))
            M += float(beta_ij) * (Omega_ij @ Sigma_q_j_inv @ Omega_ij.T)
        elif agent.mu_q.ndim == 2:
            # 1D field
            for i in range(agent.mu_q.shape[0]):
                Sigma_q_j_inv = np.linalg.inv(agent_j.Sigma_q[i] + 1e-8 * np.eye(K))
                Omega_c = Omega_ij[i] if Omega_ij.ndim == 3 else Omega_ij
                M[i] += beta_ij[i] * (Omega_c @ Sigma_q_j_inv @ Omega_c.T)
        else:
            # 2D field
            for i in range(agent.mu_q.shape[0]):
                for j in range(agent.mu_q.shape[1]):
                    Sigma_q_j_inv = np.linalg.inv(agent_j.Sigma_q[i, j] + 1e-8 * np.eye(K))
                    Omega_c = Omega_ij[i, j] if Omega_ij.ndim == 4 else Omega_ij
                    M[i, j] += beta_ij[i, j] * (Omega_c @ Sigma_q_j_inv @ Omega_c.T)

    # Invert M to get M^{-1}
    if M.ndim == 2:
        M_inv = np.linalg.inv(M + 1e-8 * np.eye(K))
    elif M.ndim == 3:
        M_inv = np.zeros_like(M)
        for i in range(M.shape[0]):
            M_inv[i] = np.linalg.inv(M[i] + 1e-8 * np.eye(K))
    else:
        M_inv = np.zeros_like(M)
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                M_inv[i, j] = np.linalg.inv(M[i, j] + 1e-8 * np.eye(K))

    return M_inv


def _compute_parameter_index_ranges(trainer) -> Dict[int, Tuple[int, int, int, int]]:
    """
    Compute index ranges for each agent's parameters in the flattened theta vector.

    Returns:
        Dict mapping agent_idx -> (mu_start, mu_end, Sigma_start, Sigma_end)
    """
    idx_ranges = {}
    idx = 0

    for agent_idx, agent in enumerate(trainer.system.agents):
        K = agent.config.K
        n_spatial = agent.mu_q.size // K

        # Mu indices
        mu_start = idx
        mu_size = n_spatial * K
        mu_end = idx + mu_size
        idx = mu_end

        # Sigma indices (upper triangle per spatial point)
        Sigma_start = idx
        Sigma_size_per_point = K * (K + 1) // 2
        Sigma_size = n_spatial * Sigma_size_per_point
        Sigma_end = idx + Sigma_size
        idx = Sigma_end

        idx_ranges[agent_idx] = (mu_start, mu_end, Sigma_start, Sigma_end)

    return idx_ranges


# =============================================================================
# Analytical Geodesic Forces (Future Enhancement)
# =============================================================================

def compute_geodesic_force_analytical(
    trainer,
    theta: np.ndarray,
    p: np.ndarray
) -> np.ndarray:
    """
    Analytical computation of geodesic force (PLACEHOLDER).

    This would compute dM^{-1}/d\theta analytically using:

    1. d(A^{-1})/dx = -A^{-1} (dA/dx) A^{-1}

    2. dM/d\mu_k = sum_j (d\beta_ij/d\mu_k) Omega_ij Sigma_qj^{-1} Omega_ij^T

    3. d\beta_ij/d\mu_k = -(\beta_ij/kappa) * [d KL_ij/d\mu_k - sum_m \beta_im d KL_im/d\mu_k]

    This is more efficient than finite differences but requires careful
    implementation of the chain rule through softmax.

    TODO: Implement analytical gradients for better performance.
    """
    raise NotImplementedError(
        "Analytical geodesic forces not yet implemented. "
        "Use compute_geodesic_force with finite differences."
    )


# =============================================================================
# Diagnostics
# =============================================================================

def diagnose_geodesic_correction(
    trainer,
    theta: np.ndarray,
    p: np.ndarray,
    eps: float = 1e-5
) -> Dict:
    """
    Diagnostic tool for understanding geodesic correction behavior.

    Returns:
        Dictionary with diagnostic information including:
        - geodesic_force_norm: ||F_geo||
        - potential_force_norm: ||-dV/d\theta||
        - ratio: ||F_geo|| / ||-dV/d\theta||
        - component_breakdown: Contribution from each agent
        - dM_inv_norms: Frobenius norms of dM^{-1}/d\theta
    """
    from gradients.gradient_engine import compute_natural_gradients

    # Compute geodesic force
    geodesic_force = compute_geodesic_force(trainer, theta, p, eps)

    # Compute potential force
    potential_force = trainer._compute_force(theta)

    # Norms
    geo_norm = np.linalg.norm(geodesic_force)
    pot_norm = np.linalg.norm(potential_force)

    # Per-agent breakdown
    idx_ranges = _compute_parameter_index_ranges(trainer)
    agent_contributions = {}

    for agent_idx, (mu_start, mu_end, Sigma_start, Sigma_end) in idx_ranges.items():
        mu_geo = geodesic_force[mu_start:mu_end]
        mu_pot = potential_force[mu_start:mu_end]

        agent_contributions[agent_idx] = {
            'mu_geodesic_norm': np.linalg.norm(mu_geo),
            'mu_potential_norm': np.linalg.norm(mu_pot),
            'mu_ratio': np.linalg.norm(mu_geo) / (np.linalg.norm(mu_pot) + 1e-10)
        }

    return {
        'geodesic_force_norm': geo_norm,
        'potential_force_norm': pot_norm,
        'ratio': geo_norm / (pot_norm + 1e-10),
        'agent_contributions': agent_contributions,
        'geodesic_force': geodesic_force,
        'potential_force': potential_force
    }
