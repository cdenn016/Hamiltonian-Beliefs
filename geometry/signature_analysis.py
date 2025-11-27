"""
Metric Signature Analysis Tools
================================

Tools for analyzing the signature of metric tensors to detect emergent
Lorentzian structure in statistical manifolds.

The goal is to find regions where the pullback metric g_pullback has
signature (-,+,+,+) instead of the usual Riemannian (+,+,+,+).

Author: Chris
Date: November 2025
"""

import numpy as np
from typing import Tuple, Dict, List, Optional
from dataclasses import dataclass
from enum import Enum


class MetricSignature(Enum):
    """Metric signature types."""
    RIEMANNIAN = "riemannian"          # All positive eigenvalues
    LORENTZIAN = "lorentzian"          # One negative, rest positive
    EUCLIDEAN = "euclidean"            # Flat (+,+,+,+)
    MINKOWSKI = "minkowski"            # Flat (-,+,+,+)
    DEGENERATE = "degenerate"          # Has zero eigenvalues
    INDEFINITE = "indefinite"          # Multiple negative eigenvalues
    UNKNOWN = "unknown"


@dataclass
class SignatureAnalysis:
    """Results of metric signature analysis."""
    eigenvalues: np.ndarray           # Sorted eigenvalues
    eigenvectors: np.ndarray          # Corresponding eigenvectors
    signature: MetricSignature        # Classified signature type
    signature_tuple: Tuple[int, ...]  # (n_negative, n_zero, n_positive)
    timelike_direction: Optional[np.ndarray] = None  # If Lorentzian
    spacelike_directions: Optional[np.ndarray] = None

    def __str__(self):
        """Human-readable description."""
        sig_str = f"({'-' * self.signature_tuple[0]}"
        sig_str += f"{'0' * self.signature_tuple[1]}"
        sig_str += f"{'+' * self.signature_tuple[2]})"

        return f"Signature: {sig_str} ({self.signature.value})"


def analyze_metric_signature(
    metric: np.ndarray,
    tol: float = 1e-10
) -> SignatureAnalysis:
    """
    Analyze the signature of a metric tensor.

    Args:
        metric: (d, d) metric tensor g_ij
        tol: Tolerance for zero eigenvalues

    Returns:
        SignatureAnalysis object with full signature information

    Example:
        >>> g = np.diag([-1, 1, 1, 1])  # Minkowski metric
        >>> sig = analyze_metric_signature(g)
        >>> print(sig)
        Signature: (-+++) (lorentzian)
    """
    # Ensure symmetric
    metric_sym = 0.5 * (metric + metric.T)

    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(metric_sym)

    # Sort by eigenvalue magnitude (largest absolute value first)
    idx = np.argsort(np.abs(eigenvalues))[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Classify eigenvalues
    n_negative = np.sum(eigenvalues < -tol)
    n_zero = np.sum(np.abs(eigenvalues) <= tol)
    n_positive = np.sum(eigenvalues > tol)

    signature_tuple = (n_negative, n_zero, n_positive)

    # Determine signature type
    if n_zero > 0:
        signature = MetricSignature.DEGENERATE
    elif n_negative == 0:
        signature = MetricSignature.RIEMANNIAN
    elif n_negative == 1:
        signature = MetricSignature.LORENTZIAN
        # Check if approximately flat
        if np.allclose(np.abs(eigenvalues), 1.0, rtol=0.1):
            signature = MetricSignature.MINKOWSKI
    elif n_negative > 1:
        signature = MetricSignature.INDEFINITE
    else:
        signature = MetricSignature.UNKNOWN

    # Extract special directions for Lorentzian metrics
    timelike_direction = None
    spacelike_directions = None

    if signature == MetricSignature.LORENTZIAN or signature == MetricSignature.MINKOWSKI:
        # First eigenvector (largest absolute eigenvalue) is timelike
        timelike_direction = eigenvectors[:, 0] if eigenvalues[0] < 0 else None

        # Find the negative eigenvalue index
        neg_idx = np.where(eigenvalues < -tol)[0][0]
        timelike_direction = eigenvectors[:, neg_idx]

        # Remaining are spacelike
        spacelike_mask = np.arange(len(eigenvalues)) != neg_idx
        spacelike_directions = eigenvectors[:, spacelike_mask]

    return SignatureAnalysis(
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        signature=signature,
        signature_tuple=signature_tuple,
        timelike_direction=timelike_direction,
        spacelike_directions=spacelike_directions
    )


def compute_pullback_metric(
    agent,
    point_idx: Optional[int] = None,
    include_internal: bool = True,
    include_dark: bool = True
) -> np.ndarray:
    """
    Compute the pullback metric at a point in the base manifold.

    The pullback metric is induced by the Fisher information metric
    on the statistical manifold:

        g_pullback = g_spatial + g_dark + g_internal

    where:
        - g_spatial: Induced by varying position in base manifold
        - g_dark: Contribution from gauge field φ
        - g_internal: Contribution from varying μ, Σ

    Args:
        agent: Agent with μ_q, Sigma_q, phi fields
        point_idx: Index of point in flattened base manifold (None = center)
        include_internal: Include internal metric contribution
        include_dark: Include dark (gauge) metric contribution

    Returns:
        metric: Pullback metric tensor g_αβ
    """
    # Get agent's belief parameters
    mu_q = agent.mu_q  # Shape: (*S, K) where S is spatial grid, K is latent dim
    Sigma_q = agent.Sigma_q  # Shape: (*S, K, K)
    phi = agent.phi if hasattr(agent, 'phi') else None  # Shape: (*S, 3) for SO(3)

    # Get spatial structure
    if mu_q.ndim == 1:
        # Single point - no spatial variation, just return Fisher metric
        K = len(mu_q)
        try:
            Sigma_inv = np.linalg.inv(Sigma_q)
        except np.linalg.LinAlgError:
            Sigma_inv = np.linalg.inv(Sigma_q + 1e-6 * np.eye(len(Sigma_q)))
        return Sigma_inv  # Fisher metric g_μμ = Σ^{-1}

    # Extract spatial dimensions
    if mu_q.ndim == 2:
        S, K = mu_q.shape
        spatial_shape = (S,)
    elif mu_q.ndim == 3:
        H, W, K = mu_q.shape
        spatial_shape = (H, W)
        S = H * W
    else:
        raise ValueError(f"Unexpected mu_q shape: {mu_q.shape}")

    # Determine point index
    if point_idx is None:
        # Use center of support
        if hasattr(agent, 'support') and agent.support is not None:
            point_idx = agent.support.center_idx
        else:
            point_idx = S // 2  # Middle of grid

    # Get coordinates in spatial grid
    if len(spatial_shape) == 1:
        i = point_idx
        coords = (i,)
        spatial_dim = 1
    else:
        H, W = spatial_shape
        i = point_idx // W
        j = point_idx % W
        coords = (i, j)
        spatial_dim = 2

    # Extract parameters at this point
    mu_c = mu_q[coords]  # (K,)
    Sigma_c = Sigma_q[coords]  # (K, K)

    # Compute Fisher metric inverse (for Gaussian: g_μμ = Σ^{-1})
    try:
        Sigma_inv = np.linalg.inv(Sigma_c)
    except np.linalg.LinAlgError:
        # Singular covariance - add regularization
        Sigma_inv = np.linalg.inv(Sigma_c + 1e-6 * np.eye(len(Sigma_c)))

    # ===================================================================
    # SPATIAL METRIC: g_spatial_αβ = (∂_α μ)^T Σ^{-1} (∂_β μ)
    # ===================================================================
    # This measures how belief mean μ varies with position in base manifold
    # weighted by the Fisher metric (precision matrix Σ^{-1})

    eps = 1.0  # Grid spacing (assuming unit grid)
    dmu_dc = np.zeros((spatial_dim, K))  # (spatial_dim, K) gradient of μ

    if spatial_dim == 1:
        # 1D spatial grid
        if i > 0 and i < S - 1:
            # Central difference
            mu_plus = mu_q[i + 1]
            mu_minus = mu_q[i - 1]
            dmu_dc[0] = (mu_plus - mu_minus) / (2 * eps)
        elif i == 0 and S > 1:
            # Forward difference
            mu_plus = mu_q[i + 1]
            dmu_dc[0] = (mu_plus - mu_c) / eps
        elif i == S - 1 and S > 1:
            # Backward difference
            mu_minus = mu_q[i - 1]
            dmu_dc[0] = (mu_c - mu_minus) / eps

    else:
        # 2D spatial grid
        # x-direction (i-direction)
        if i > 0 and i < H - 1:
            mu_plus = mu_q[i + 1, j]
            mu_minus = mu_q[i - 1, j]
            dmu_dc[0] = (mu_plus - mu_minus) / (2 * eps)
        elif i == 0 and H > 1:
            mu_plus = mu_q[i + 1, j]
            dmu_dc[0] = (mu_plus - mu_c) / eps
        elif i == H - 1 and H > 1:
            mu_minus = mu_q[i - 1, j]
            dmu_dc[0] = (mu_c - mu_minus) / eps

        # y-direction (j-direction)
        if j > 0 and j < W - 1:
            mu_plus = mu_q[i, j + 1]
            mu_minus = mu_q[i, j - 1]
            dmu_dc[1] = (mu_plus - mu_minus) / (2 * eps)
        elif j == 0 and W > 1:
            mu_plus = mu_q[i, j + 1]
            dmu_dc[1] = (mu_plus - mu_c) / eps
        elif j == W - 1 and W > 1:
            mu_minus = mu_q[i, j - 1]
            dmu_dc[1] = (mu_c - mu_minus) / eps

    # Compute spatial metric: g_αβ = (∂_α μ)^T Σ^{-1} (∂_β μ)
    # This is the pullback of the Fisher metric to the base manifold
    g_spatial = dmu_dc @ Sigma_inv @ dmu_dc.T  # (spatial_dim, spatial_dim)

    # ===================================================================
    # DARK METRIC: Gauge field contribution
    # ===================================================================
    # This measures kinetic energy of the gauge field φ
    # g_dark_αβ = (∂_α φ)^T (∂_β φ)

    g_dark = np.zeros((spatial_dim, spatial_dim))

    if include_dark and phi is not None:
        phi_c = phi[coords]  # (3,) for SO(3)
        dphi_dc = np.zeros((spatial_dim, len(phi_c)))

        if spatial_dim == 1:
            # 1D case
            if i > 0 and i < S - 1:
                phi_plus = phi[i + 1]
                phi_minus = phi[i - 1]
                dphi_dc[0] = (phi_plus - phi_minus) / (2 * eps)
            elif i == 0 and S > 1:
                phi_plus = phi[i + 1]
                dphi_dc[0] = (phi_plus - phi_c) / eps
            elif i == S - 1 and S > 1:
                phi_minus = phi[i - 1]
                dphi_dc[0] = (phi_c - phi_minus) / eps

        else:
            # 2D case
            # x-direction
            if i > 0 and i < H - 1:
                phi_plus = phi[i + 1, j]
                phi_minus = phi[i - 1, j]
                dphi_dc[0] = (phi_plus - phi_minus) / (2 * eps)
            elif i == 0 and H > 1:
                phi_plus = phi[i + 1, j]
                dphi_dc[0] = (phi_plus - phi_c) / eps
            elif i == H - 1 and H > 1:
                phi_minus = phi[i - 1, j]
                dphi_dc[0] = (phi_c - phi_minus) / eps

            # y-direction
            if j > 0 and j < W - 1:
                phi_plus = phi[i, j + 1]
                phi_minus = phi[i, j - 1]
                dphi_dc[1] = (phi_plus - phi_minus) / (2 * eps)
            elif j == 0 and W > 1:
                phi_plus = phi[i, j + 1]
                dphi_dc[1] = (phi_plus - phi_c) / eps
            elif j == W - 1 and W > 1:
                phi_minus = phi[i, j - 1]
                dphi_dc[1] = (phi_c - phi_minus) / eps

        # Dark metric: kinetic term for gauge field
        g_dark = dphi_dc @ dphi_dc.T  # (spatial_dim, spatial_dim)

    # ===================================================================
    # COMBINE METRICS
    # ===================================================================

    g_pullback = g_spatial.copy()

    if include_dark:
        g_pullback += g_dark

    if include_internal:
        # Add small regularization to avoid degeneracy
        # This represents "internal" fluctuations at fixed position
        g_pullback += 0.01 * np.eye(spatial_dim)

    return g_pullback


def signature_field_2d(
    system,
    agent_idx: int = 0,
    grid_size: Tuple[int, int] = (20, 20)
) -> Dict[str, np.ndarray]:
    """
    Compute metric signature across a 2D grid in the base manifold.

    This maps out regions where the metric signature might change,
    potentially revealing Lorentzian regions.

    Args:
        system: MultiAgentSystem
        agent_idx: Which agent to analyze
        grid_size: (H, W) resolution of grid

    Returns:
        Dictionary with:
            - 'signatures': (H, W) array of signature classifications
            - 'eigenvalues': (H, W, d) array of eigenvalues
            - 'n_negative': (H, W) count of negative eigenvalues
            - 'n_positive': (H, W) count of positive eigenvalues
            - 'lorentzian_mask': (H, W) boolean mask of Lorentzian regions
    """
    agent = system.agents[agent_idx]
    H, W = grid_size

    # Get base manifold shape
    if hasattr(agent, 'base_manifold'):
        base_shape = agent.base_manifold.shape
    else:
        base_shape = (H, W)

    # Initialize output arrays
    signatures = np.zeros((H, W), dtype=int)
    eigenvalues_field = np.zeros((H, W, 4))  # Assume max 4D
    n_negative = np.zeros((H, W), dtype=int)
    n_positive = np.zeros((H, W), dtype=int)

    # Compute signature at each grid point
    for i in range(H):
        for j in range(W):
            # Map grid point to base manifold index
            point_idx = i * W + j

            # Compute pullback metric
            g = compute_pullback_metric(agent, point_idx=point_idx)

            # Analyze signature
            sig_analysis = analyze_metric_signature(g)

            # Store results
            d = len(sig_analysis.eigenvalues)
            eigenvalues_field[i, j, :d] = sig_analysis.eigenvalues
            n_negative[i, j] = sig_analysis.signature_tuple[0]
            n_positive[i, j] = sig_analysis.signature_tuple[2]

            # Encode signature as integer
            if sig_analysis.signature == MetricSignature.LORENTZIAN:
                signatures[i, j] = -1
            elif sig_analysis.signature == MetricSignature.RIEMANNIAN:
                signatures[i, j] = 1
            else:
                signatures[i, j] = 0

    # Create Lorentzian mask
    lorentzian_mask = (signatures == -1)

    return {
        'signatures': signatures,
        'eigenvalues': eigenvalues_field,
        'n_negative': n_negative,
        'n_positive': n_positive,
        'lorentzian_mask': lorentzian_mask,
        'grid_size': grid_size
    }


def compute_light_cone_structure(
    metric: np.ndarray,
    point: np.ndarray
) -> Dict[str, np.ndarray]:
    """
    Compute light cone structure at a point if metric is Lorentzian.

    Args:
        metric: (d, d) metric tensor (must be Lorentzian)
        point: (d,) point in manifold

    Returns:
        Dictionary with:
            - 'timelike_vectors': Vectors inside light cone
            - 'null_vectors': Vectors on light cone boundary
            - 'spacelike_vectors': Vectors outside light cone
            - 'light_cone_angle': Opening angle of light cone
    """
    sig = analyze_metric_signature(metric)

    if sig.signature != MetricSignature.LORENTZIAN:
        raise ValueError(f"Metric must be Lorentzian, got {sig.signature}")

    # For a Lorentzian metric g with signature (-,+,+,+),
    # a vector v is:
    #   - Timelike if g(v,v) < 0
    #   - Null if g(v,v) = 0
    #   - Spacelike if g(v,v) > 0

    # Light cone is defined by g(v,v) = 0

    # Generate sample vectors
    d = len(metric)
    n_samples = 100
    vectors = np.random.randn(n_samples, d)

    # Compute g(v,v) for each vector
    inner_products = np.array([
        v @ metric @ v for v in vectors
    ])

    # Classify vectors
    timelike_mask = inner_products < -1e-6
    null_mask = np.abs(inner_products) < 1e-6
    spacelike_mask = inner_products > 1e-6

    # Compute light cone opening angle
    # For Minkowski metric, this is 45 degrees
    # For general Lorentzian metric, depends on curvature
    timelike_dir = sig.timelike_direction
    if timelike_dir is not None:
        # Angle between timelike direction and null directions
        # (This is approximate - proper calculation requires full geometry)
        light_cone_angle = np.pi / 4  # Placeholder
    else:
        light_cone_angle = None

    return {
        'timelike_vectors': vectors[timelike_mask],
        'null_vectors': vectors[null_mask],
        'spacelike_vectors': vectors[spacelike_mask],
        'light_cone_angle': light_cone_angle,
        'timelike_direction': timelike_dir
    }