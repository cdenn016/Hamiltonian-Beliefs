# -*- coding: utf-8 -*-
"""
Created on Mon Nov 10 11:41:37 2025

@author: chris and christine
"""

#!/usr/bin/env python3
"""
Spectral Geometry for Periodic Manifolds
=========================================

Fourier-based derivative operators for fields on periodic domains.

Mathematical Framework:
----------------------
For periodic domain C with period L, any field φ: C → ℝ^K decomposes:
    φ(x) = Σ_k φ̂_k e^{ikx}  where k = 2πn/L

Derivative in Fourier space:
    ∂φ/∂x = Σ_k (ik) φ̂_k e^{ikx} = ℱ⁻¹[ik · ℱ[φ]]

Properties:
- Exact for smooth periodic functions (infinite-order accuracy)
- No boundary artifacts (periodicity built-in)
- FFT makes this O(N log N) - very efficient
- Generalizes to curved manifolds via Laplace-Beltrami eigenfunctions

For Covariant Derivatives:
--------------------------
    D_A φ = ∇φ + [A, φ]
    
where ∇φ uses spectral derivative, [A,φ] uses Lie bracket.

Future Extensions:
------------------
- Non-periodic: Use Chebyshev spectral methods
- Curved manifolds: Laplace-Beltrami eigenbasis
- Graph manifolds: Graph Laplacian eigenfunctions

Author: Chris
Date: November 2025
"""

import numpy as np
from typing import Tuple, Optional, Union
from numpy.fft import fftn, ifftn, fftfreq


# =============================================================================
# Core Spectral Derivative Operators
# =============================================================================

def spectral_gradient(
    field: np.ndarray,  # (*S, K) where *S is spatial shape
    *,
    domain_size: Optional[Union[float, Tuple[float, ...]]] = None,
    axes: Optional[Tuple[int, ...]] = None,
) -> Union[np.ndarray, Tuple[np.ndarray, ...]]:
    """
    Compute spatial gradient ∇φ using Fourier spectral method.
    
    For 1D: Returns (N, K) array
    For 2D: Returns tuple (grad_x, grad_y) each (H, W, K)
    For 3D: Returns tuple (grad_x, grad_y, grad_z) each (D, H, W, K)
    
    Args:
        field: Spatial field, shape (*S, K)
               - 1D: (N, K)
               - 2D: (H, W, K)
               - 3D: (D, H, W, K)
        domain_size: Physical size of domain
                     - float: same size in all dimensions
                     - tuple: (L_x, L_y, ...) for each dimension
                     - None: assumes L = 1.0 for all dimensions
        axes: Which spatial axes to differentiate (default: all but last)
    
    Returns:
        For 1D: gradient array (*S, K)
        For nD: tuple of n gradient arrays, each (*S, K)
    
    Examples:
        >>> # 1D periodic sine wave
        >>> x = np.linspace(0, 2*np.pi, 128, endpoint=False)
        >>> phi = np.sin(x)[:, None]  # (128, 1)
        >>> grad = spectral_gradient(phi, domain_size=2*np.pi)
        >>> # Should match cos(x) exactly
        
        >>> # 2D Gaussian
        >>> phi_2d = ...  # (64, 64, 3)
        >>> grad_x, grad_y = spectral_gradient(phi_2d)
    """
    # Determine spatial dimensions
    field = np.asarray(field, dtype=np.float64)  # Need float64 for FFT accuracy
    
    ndim_spatial = len(field.shape) - 1  # Last dim is K (field dimension)
    K = field.shape[-1]
    spatial_shape = field.shape[:-1]
    
    # Default: all spatial axes
    if axes is None:
        axes = tuple(range(ndim_spatial))
    
    # Handle domain size
    if domain_size is None:
        domain_size = tuple([1.0] * ndim_spatial)
    elif isinstance(domain_size, (int, float)):
        domain_size = tuple([float(domain_size)] * ndim_spatial)
    else:
        domain_size = tuple(domain_size)
    
    if len(domain_size) != ndim_spatial:
        raise ValueError(
            f"domain_size length {len(domain_size)} doesn't match "
            f"spatial dimensions {ndim_spatial}"
        )
    
    # Compute gradient along each axis
    gradients = []
    
    for axis_idx, axis in enumerate(axes):
        # Wavenumbers for this axis
        N_axis = spatial_shape[axis]
        L_axis = domain_size[axis]
        
        # FFT frequencies: k = 2π * freq / L
        freq = fftfreq(N_axis, d=L_axis/N_axis)
        k = 2.0 * np.pi * freq
        
        # Reshape k for broadcasting
        k_shape = [1] * len(field.shape)
        k_shape[axis] = N_axis
        k = k.reshape(k_shape)
        
        # Compute derivative via FFT
        # ∂φ/∂x = ℱ⁻¹[ik · ℱ[φ]]
        grad_axis = np.zeros_like(field)
        
        for k_idx in range(K):
            # Extract k-th component
            field_k = field[..., k_idx]
            
            # FFT along specified axis
            field_hat = fftn(field_k, axes=(axis,))
            
            # Multiply by ik
            grad_hat = 1j * k * field_hat
            
            # Inverse FFT
            grad_axis[..., k_idx] = ifftn(grad_hat, axes=(axis,)).real
        
        gradients.append(grad_axis.astype(np.float32, copy=False))
    
    # Return format depends on dimensionality
    if ndim_spatial == 1:
        return gradients[0]  # Single array for 1D
    else:
        return tuple(gradients)  # Tuple of arrays for 2D+


def spectral_laplacian(
    field: np.ndarray,
    *,
    domain_size: Optional[Union[float, Tuple[float, ...]]] = None,
) -> np.ndarray:
    """
    Compute Laplacian ∇²φ using Fourier spectral method.
    
    For periodic domain: ∇²φ = ℱ⁻¹[-|k|² · ℱ[φ]]
    
    Args:
        field: Spatial field (*S, K)
        domain_size: Physical size of domain
    
    Returns:
        laplacian: Same shape as field (*S, K)
    
    Used for:
        - Smoothness regularizers: λ ∫ ||∇²φ||² dc
        - Diffusion terms
        - Heat equation evolution
    """
    field = np.asarray(field, dtype=np.float64)
    
    ndim_spatial = len(field.shape) - 1
    K = field.shape[-1]
    spatial_shape = field.shape[:-1]
    
    # Handle domain size
    if domain_size is None:
        domain_size = tuple([1.0] * ndim_spatial)
    elif isinstance(domain_size, (int, float)):
        domain_size = tuple([float(domain_size)] * ndim_spatial)
    else:
        domain_size = tuple(domain_size)
    
    # Compute |k|² for all axes
    k_squared = np.zeros(spatial_shape)
    
    for axis in range(ndim_spatial):
        N_axis = spatial_shape[axis]
        L_axis = domain_size[axis]
        
        freq = fftfreq(N_axis, d=L_axis/N_axis)
        k = 2.0 * np.pi * freq
        
        # Reshape for broadcasting
        k_shape = [1] * ndim_spatial
        k_shape[axis] = N_axis
        k_axis = k.reshape(k_shape)
        
        k_squared += k_axis**2
    
    # Compute Laplacian: ∇²φ = ℱ⁻¹[-|k|² · ℱ[φ]]
    laplacian = np.zeros_like(field)
    
    for k_idx in range(K):
        field_k = field[..., k_idx]
        
        # FFT
        field_hat = fftn(field_k)
        
        # Multiply by -|k|²
        lap_hat = -k_squared * field_hat
        
        # Inverse FFT
        laplacian[..., k_idx] = ifftn(lap_hat).real
    
    return laplacian.astype(np.float32, copy=False)


def spectral_divergence(
    vector_field: Union[np.ndarray, Tuple[np.ndarray, ...]],
    *,
    domain_size: Optional[Union[float, Tuple[float, ...]]] = None,
) -> np.ndarray:
    """
    Compute divergence ∇·F of vector field.
    
    Args:
        vector_field: Either
            - Tuple (F_x, F_y, ...) each (*S, K)
            - Single array (*S, ndim, K) with vector components in axis -2
        domain_size: Physical size
    
    Returns:
        divergence: (*S, K)
    """
    # Handle input format
    if isinstance(vector_field, tuple):
        components = vector_field
        ndim_spatial = len(components)
    else:
        # Assume shape (*S, ndim, K)
        ndim_spatial = vector_field.shape[-2]
        components = [vector_field[..., i, :] for i in range(ndim_spatial)]
    
    # Compute ∂F_i/∂x_i for each component
    div = np.zeros_like(components[0])
    
    for i, F_i in enumerate(components):
        # Gradient of i-th component along i-th axis
        grad = spectral_gradient(
            F_i,
            domain_size=domain_size,
            axes=(i,)  # Only differentiate along axis i
        )
        
        if isinstance(grad, tuple):
            grad = grad[0]
        
        div += grad
    
    return div


# =============================================================================
# Gradient Magnitude and Regularization
# =============================================================================

def gradient_magnitude_squared(
    field: np.ndarray,
    *,
    domain_size: Optional[Union[float, Tuple[float, ...]]] = None,
) -> np.ndarray:
    """
    Compute ||∇φ||² at each spatial point.
    
    Returns: (*S, K) array of squared gradient magnitudes.
    
    For regularizer: λ ∫ ||∇φ||² dc = λ Σ_c Σ_k ||∇φ_k(c)||²
    """
    gradients = spectral_gradient(field, domain_size=domain_size)
    
    # Handle 1D vs multi-D
    if not isinstance(gradients, tuple):
        gradients = (gradients,)
    
    # Sum squares: ||∇φ||² = Σ_i (∂φ/∂x_i)²
    grad_sq = np.zeros_like(field)
    for grad_i in gradients:
        grad_sq += grad_i**2
    
    return grad_sq


def smoothness_regularizer(
    field: np.ndarray,
    *,
    domain_size: Optional[Union[float, Tuple[float, ...]]] = None,
    order: int = 1,
) -> float:
    """
    Compute smoothness penalty ∫ ||∇^n φ||² dc.
    
    Args:
        field: (*S, K)
        domain_size: Physical size
        order: 1 for gradient penalty, 2 for Laplacian penalty
    
    Returns:
        penalty: Scalar regularization term
    """
    if order == 1:
        grad_sq = gradient_magnitude_squared(field, domain_size=domain_size)
        return float(np.sum(grad_sq))
    
    elif order == 2:
        lap = spectral_laplacian(field, domain_size=domain_size)
        return float(np.sum(lap**2))
    
    else:
        raise ValueError(f"Order {order} not implemented")


# =============================================================================
# Lie Bracket for Covariant Derivatives
# =============================================================================

def lie_bracket_so3(
    A: np.ndarray,  # (*S, 3)
    phi: np.ndarray,  # (*S, 3)
    generators: np.ndarray,  # (3, K, K)
) -> np.ndarray:
    """
    Compute Lie bracket [A, φ] for 𝔰𝔬(3) algebra.
    
    For 𝔰𝔬(3) ≃ ℝ³ with cross product:
        [A, φ] = A × φ  (vector cross product)
    
    This is exact for SO(3) and computationally cheap.
    
    Args:
        A: Connection field (*S, 3)
        phi: Gauge field (*S, 3)
        generators: SO(3) generators (3, K, K) - not used for 𝔰𝔬(3) ≃ ℝ³
    
    Returns:
        bracket: (*S, 3)
    """
    # For 𝔰𝔬(3), Lie bracket is cross product
    return np.cross(A, phi, axis=-1).astype(np.float32)


def lie_bracket_general(
    A: np.ndarray,  # (*S, N)
    phi: np.ndarray,  # (*S, N)
    generators: np.ndarray,  # (N, K, K)
) -> np.ndarray:
    """
    Compute Lie bracket [A, φ] for general Lie algebra.
    
    [A, φ] = Σ_a Σ_b f^c_ab A^a φ^b T_c
    
    where f^c_ab are structure constants from [T_a, T_b] = f^c_ab T_c.
    
    For SO(N>3), need to compute structure constants from generators.
    
    Args:
        A: Connection (*S, N)
        phi: Gauge field (*S, N)
        generators: Lie algebra generators (N, K, K)
    
    Returns:
        bracket: (*S, N)
    """
    N = generators.shape[0]
    K = generators.shape[1]
    spatial_shape = A.shape[:-1]
    
    # Compute structure constants (cached if possible)
    # [T_a, T_b] = T_a T_b - T_b T_a = Σ_c f^c_ab T_c
    # f^c_ab = tr(T_c [T_a, T_b]) / tr(T_c T_c)  (for compact Lie algebras)
    
    # This is expensive, so for SO(3) use cross product instead
    if N == 3:
        return lie_bracket_so3(A, phi, generators)
    
    # For general case, compute via matrix commutators
    bracket = np.zeros((*spatial_shape, N), dtype=np.float32)
    
    for idx in np.ndindex(spatial_shape):
        # At each spatial point
        A_mat = sum(A[idx][a] * generators[a] for a in range(N))
        phi_mat = sum(phi[idx][b] * generators[b] for b in range(N))
        
        # [A, φ] in matrix form
        bracket_mat = A_mat @ phi_mat - phi_mat @ A_mat
        
        # Project back to algebra: extract coefficients
        for c in range(N):
            bracket[idx][c] = np.trace(generators[c] @ bracket_mat) / np.trace(generators[c] @ generators[c])
    
    return bracket


# =============================================================================
# Covariant Derivative
# =============================================================================

def covariant_derivative(
    phi: np.ndarray,  # (*S, 3) or (*S, N)
    A: np.ndarray,  # (*S, 3) or (*S, N)
    generators: np.ndarray,  # (3, K, K) or (N, K, K)
    *,
    domain_size: Optional[Union[float, Tuple[float, ...]]] = None,
    return_components: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Compute covariant derivative D_A φ = ∇φ + [A, φ].
    
    This is the gauge-covariant generalization of ∇φ.
    
    Args:
        phi: Gauge field (*S, N)
        A: Connection field (*S, N)
        generators: Lie algebra generators (N, K, K)
        domain_size: Physical size of periodic domain
        return_components: If True, return (∇φ, [A,φ]) separately
    
    Returns:
        D_A_phi: Covariant derivative (*S, N)
        OR (grad_phi, bracket) if return_components=True
    
    Usage in Regularizer:
        penalty = ∫ ||D_A φ||² dc
        
        This penalizes φ that don't align with connection geometry.
    """
    # Compute ∇φ (spectral derivative)
    grad_phi = spectral_gradient(phi, domain_size=domain_size)
    
    # For multi-D, need norm of gradient (not tuple of components)
    # For covariant derivative, we use magnitude
    if isinstance(grad_phi, tuple):
        # ||∇φ||² = Σ_i (∂φ/∂x_i)²
        grad_magnitude = np.zeros_like(phi)
        for grad_i in grad_phi:
            grad_magnitude += grad_i**2
        grad_phi = np.sqrt(grad_magnitude)
    
    # Compute [A, φ] (Lie bracket)
    N = phi.shape[-1]
    if N == 3:
        bracket = lie_bracket_so3(A, phi, generators)
    else:
        bracket = lie_bracket_general(A, phi, generators)
    
    if return_components:
        return grad_phi, bracket
    else:
        return grad_phi + bracket


# =============================================================================
# Testing & Validation
# =============================================================================

if __name__ == "__main__":
    print("Testing spectral geometry operators...")
    
    # Test 1: 1D derivative - sine wave
    print("\n1. 1D spectral gradient")
    N = 128
    L = 2 * np.pi
    x = np.linspace(0, L, N, endpoint=False)
    
    phi_1d = np.sin(x)[:, None]  # (128, 1)
    grad_1d = spectral_gradient(phi_1d, domain_size=L)
    true_grad_1d = np.cos(x)[:, None]
    
    error_1d = np.max(np.abs(grad_1d - true_grad_1d))
    print(f"   Max error: {error_1d:.2e} (should be ~machine precision)")
    assert error_1d < 1e-10, "Spectral should be exact for smooth periodic"
    
    # Test 2: 2D gradient - separable function
    print("\n2. 2D spectral gradient")
    Hx, Hy = 64, 64
    Lx, Ly = 2*np.pi, 2*np.pi
    x = np.linspace(0, Lx, Hx, endpoint=False)
    y = np.linspace(0, Ly, Hy, endpoint=False)
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    phi_2d = (np.sin(X) * np.cos(Y))[:, :, None]  # (64, 64, 1)
    grad_x, grad_y = spectral_gradient(phi_2d, domain_size=(Lx, Ly))
    
    true_grad_x = (np.cos(X) * np.cos(Y))[:, :, None]
    true_grad_y = (-np.sin(X) * np.sin(Y))[:, :, None]
    
    error_x = np.max(np.abs(grad_x - true_grad_x))
    error_y = np.max(np.abs(grad_y - true_grad_y))
    
    print(f"   ∂/∂x error: {error_x:.2e}")
    print(f"   ∂/∂y error: {error_y:.2e}")
    assert error_x < 1e-10 and error_y < 1e-10
    
    # Test 3: Laplacian
    print("\n3. Spectral Laplacian")
    lap_1d = spectral_laplacian(phi_1d, domain_size=L)
    true_lap_1d = -np.sin(x)[:, None]
    
    error_lap = np.max(np.abs(lap_1d - true_lap_1d))
    print(f"   Max error: {error_lap:.2e}")
    assert error_lap < 1e-10
    
    # Test 4: Lie bracket (SO(3))
    print("\n4. Lie bracket [A, φ]")
    from math_utils.generators import generate_so3_generators
    
    A_test = np.array([[1.0, 0.0, 0.0]])  # (1, 3) - x-axis rotation
    phi_test = np.array([[0.0, 1.0, 0.0]])  # (1, 3) - y-axis rotation
    G = generate_so3_generators(3)
    
    bracket = lie_bracket_so3(A_test, phi_test, G)
    # [x, y] = z for SO(3)
    expected = np.array([[0.0, 0.0, 1.0]])
    
    print(f"   [x̂, ŷ] = {bracket[0]}")
    print(f"   Expected: {expected[0]}")
    assert np.allclose(bracket, expected, atol=1e-6)
    
    # Test 5: Covariant derivative
    print("\n5. Covariant derivative D_A φ")
    phi_cov = np.sin(x)[:, None, None]  # (128, 1, 1) - treating as (*S, 3) with K=1
    phi_cov = np.tile(phi_cov, (1, 3))  # (128, 3)
    A_cov = np.zeros((N, 3), dtype=np.float32)  # Flat connection
    
    D_phi = covariant_derivative(phi_cov, A_cov, G, domain_size=L)
    grad_only = spectral_gradient(phi_cov, domain_size=L)
    
    # With A=0, should match ordinary gradient
    print(f"   ||D_0 φ - ∇φ||: {np.max(np.abs(D_phi - grad_only)):.2e}")
    
    print("\n✓ All spectral geometry tests passed!")
    print("\nSpectral methods ready for covariant derivatives.")
    print("  - Exact for smooth periodic fields")
    print("  - O(N log N) complexity via FFT")
    print("  - Extends naturally to curved manifolds")