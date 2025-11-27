# -*- coding: utf-8 -*-
"""
Created on Sat Nov 22 09:05:49 2025

@author: chris and christine
"""

"""
Metric Visualization Tools
==========================

Visualization functions for analyzing pullback metrics on the base manifold.

This module provides tools to visualize:
- Metric tensor fields on 2D grids
- Eigenvalue/eigenvector fields (principal stretching directions)
- Metric determinant (volume distortion)
- Signature classification (Riemannian vs Lorentzian regions)
- Connection coefficients (Christoffel symbols)

Author: Chris
Date: November 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from matplotlib.colors import Normalize, TwoSlopeNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from typing import Tuple, Dict, Optional, Callable
from pathlib import Path

from geometry.signature_analysis import (
    compute_pullback_metric,
    analyze_metric_signature,
    MetricSignature
)


def plot_pullback_metric_field(
    system,
    agent_idx: int = 0,
    grid_size: Tuple[int, int] = (20, 20),
    component: str = "determinant",
    out_path: Optional[Path] = None,
    show: bool = False
):
    """
    Visualize pullback metric field on 2D base manifold.

    Args:
        system: MultiAgentSystem
        agent_idx: Which agent to analyze
        grid_size: (H, W) resolution
        component: What to plot:
            - "determinant": det(g) (volume distortion)
            - "trace": tr(g) (total variance)
            - "g00", "g01", "g11": Individual components
        out_path: Save path (if None, just show)
        show: Whether to display plot

    Example:
        >>> plot_pullback_metric_field(system, component="determinant")
    """
    agent = system.agents[agent_idx]
    H, W = grid_size

    # Compute metric at each grid point
    metric_field = np.zeros((H, W))

    for i in range(H):
        for j in range(W):
            point_idx = i * W + j
            g = compute_pullback_metric(agent, point_idx=point_idx)

            if component == "determinant":
                metric_field[i, j] = np.linalg.det(g)
            elif component == "trace":
                metric_field[i, j] = np.trace(g)
            elif component.startswith("g"):
                # Parse "g00", "g01", etc.
                idx1 = int(component[1])
                idx2 = int(component[2])
                metric_field[i, j] = g[idx1, idx2]
            else:
                raise ValueError(f"Unknown component: {component}")

    # Create visualization
    fig, ax = plt.subplots(figsize=(8, 7))

    im = ax.imshow(
        metric_field,
        origin='lower',
        cmap='viridis',
        aspect='auto'
    )

    # Colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    plt.colorbar(im, cax=cax, label=component)

    ax.set_xlabel("Base Manifold X")
    ax.set_ylabel("Base Manifold Y")
    ax.set_title(f"Pullback Metric: {component} (Agent {agent_idx})")

    plt.tight_layout()

    if out_path:
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved {out_path}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_eigenvalue_field(
    system,
    agent_idx: int = 0,
    grid_size: Tuple[int, int] = (20, 20),
    out_path: Optional[Path] = None,
    show: bool = False
):
    """
    Visualize eigenvalue field of pullback metric.

    Shows how the metric stretches space in different directions.
    Separate plots for each eigenvalue (λ_0, λ_1, ...).

    Args:
        system: MultiAgentSystem
        agent_idx: Which agent to analyze
        grid_size: (H, W) resolution
        out_path: Save path
        show: Whether to display
    """
    agent = system.agents[agent_idx]
    H, W = grid_size

    # Compute eigenvalues at each point
    # First, determine metric dimension
    test_g = compute_pullback_metric(agent, point_idx=0)
    d = test_g.shape[0]

    eigenvalue_fields = np.zeros((H, W, d))

    for i in range(H):
        for j in range(W):
            point_idx = i * W + j
            g = compute_pullback_metric(agent, point_idx=point_idx)
            sig = analyze_metric_signature(g)
            eigenvalue_fields[i, j, :] = sig.eigenvalues[:d]

    # Create subplots for each eigenvalue
    ncols = min(d, 3)
    nrows = (d + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows))
    if d == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if nrows > 1 else axes

    for idx in range(d):
        ax = axes[idx]

        # Use diverging colormap centered at 0 to show sign
        norm = TwoSlopeNorm(vmin=eigenvalue_fields[:, :, idx].min(),
                            vcenter=0.0,
                            vmax=eigenvalue_fields[:, :, idx].max())

        im = ax.imshow(
            eigenvalue_fields[:, :, idx],
            origin='lower',
            cmap='RdBu_r',  # Red=negative, Blue=positive
            norm=norm,
            aspect='auto'
        )

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        plt.colorbar(im, cax=cax, label=f"λ_{idx}")

        ax.set_xlabel("Base X")
        ax.set_ylabel("Base Y")
        ax.set_title(f"Eigenvalue λ_{idx}")

    # Hide unused subplots
    for idx in range(d, len(axes)):
        axes[idx].axis('off')

    plt.suptitle(f"Metric Eigenvalues (Agent {agent_idx})", fontsize=14, y=1.02)
    plt.tight_layout()

    if out_path:
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved {out_path}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_eigenvector_field(
    system,
    agent_idx: int = 0,
    grid_size: Tuple[int, int] = (10, 10),  # Coarser for arrow clarity
    eigenvalue_idx: int = 0,
    out_path: Optional[Path] = None,
    show: bool = False
):
    """
    Visualize eigenvector field (principal stretching directions).

    Shows the direction of maximum (or minimum) metric stretching.

    Args:
        system: MultiAgentSystem
        agent_idx: Which agent to analyze
        grid_size: (H, W) resolution (use coarser grid for arrow plots)
        eigenvalue_idx: Which eigenvector to plot (0 = largest |eigenvalue|)
        out_path: Save path
        show: Whether to display
    """
    agent = system.agents[agent_idx]
    H, W = grid_size

    # Compute eigenvectors at each point
    test_g = compute_pullback_metric(agent, point_idx=0)
    d = test_g.shape[0]

    if eigenvalue_idx >= d:
        raise ValueError(f"eigenvalue_idx={eigenvalue_idx} >= metric dimension {d}")

    # Store eigenvector components
    vector_field = np.zeros((H, W, d))
    eigenvalue_field = np.zeros((H, W))

    for i in range(H):
        for j in range(W):
            point_idx = i * W + j
            g = compute_pullback_metric(agent, point_idx=point_idx)
            sig = analyze_metric_signature(g)
            vector_field[i, j, :] = sig.eigenvectors[:d, eigenvalue_idx]
            eigenvalue_field[i, j] = sig.eigenvalues[eigenvalue_idx]

    # Create visualization
    fig, ax = plt.subplots(figsize=(10, 9))

    # Background: eigenvalue magnitude
    norm = TwoSlopeNorm(vmin=eigenvalue_field.min(),
                        vcenter=0.0,
                        vmax=eigenvalue_field.max())

    im = ax.imshow(
        eigenvalue_field,
        origin='lower',
        cmap='RdBu_r',
        norm=norm,
        alpha=0.7,
        aspect='auto'
    )

    # Arrows: eigenvector directions
    # Only plot in 2D (project to first 2 components if d > 2)
    X, Y = np.meshgrid(np.arange(W), np.arange(H))
    U = vector_field[:, :, 0]  # x-component
    V = vector_field[:, :, 1] if d > 1 else np.zeros_like(U)  # y-component

    ax.quiver(
        X, Y, U, V,
        scale=20,
        width=0.003,
        headwidth=4,
        headlength=5,
        color='black',
        alpha=0.8
    )

    # Colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    plt.colorbar(im, cax=cax, label=f"λ_{eigenvalue_idx}")

    ax.set_xlabel("Base Manifold X")
    ax.set_ylabel("Base Manifold Y")
    ax.set_title(f"Eigenvector Field (λ_{eigenvalue_idx}, Agent {agent_idx})")

    plt.tight_layout()

    if out_path:
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved {out_path}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_signature_classification(
    system,
    agent_idx: int = 0,
    grid_size: Tuple[int, int] = (20, 20),
    out_path: Optional[Path] = None,
    show: bool = False
):
    """
    Visualize metric signature classification across base manifold.

    Color-codes regions by signature type:
    - Blue: Riemannian (+,+,...)
    - Red: Lorentzian (-,+,+,...)
    - Green: Minkowski (flat -,+,+,+)
    - Yellow: Indefinite (multiple negative)
    - Gray: Degenerate (zero eigenvalues)

    Args:
        system: MultiAgentSystem
        agent_idx: Which agent to analyze
        grid_size: (H, W) resolution
        out_path: Save path
        show: Whether to display
    """
    agent = system.agents[agent_idx]
    H, W = grid_size

    # Signature classification at each point
    signature_field = np.zeros((H, W), dtype=int)
    n_negative_field = np.zeros((H, W), dtype=int)

    # Define signature color codes
    SIG_CODES = {
        MetricSignature.RIEMANNIAN: 1,
        MetricSignature.LORENTZIAN: 2,
        MetricSignature.MINKOWSKI: 3,
        MetricSignature.INDEFINITE: 4,
        MetricSignature.DEGENERATE: 5,
        MetricSignature.UNKNOWN: 0
    }

    for i in range(H):
        for j in range(W):
            point_idx = i * W + j
            g = compute_pullback_metric(agent, point_idx=point_idx)
            sig = analyze_metric_signature(g)

            signature_field[i, j] = SIG_CODES[sig.signature]
            n_negative_field[i, j] = sig.signature_tuple[0]

    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: Signature classification
    cmap = plt.cm.get_cmap('tab10', 6)
    im1 = ax1.imshow(
        signature_field,
        origin='lower',
        cmap=cmap,
        vmin=0,
        vmax=5,
        aspect='auto'
    )

    ax1.set_xlabel("Base Manifold X")
    ax1.set_ylabel("Base Manifold Y")
    ax1.set_title("Metric Signature Classification")

    # Custom colorbar labels
    divider1 = make_axes_locatable(ax1)
    cax1 = divider1.append_axes("right", size="5%", pad=0.1)
    cbar1 = plt.colorbar(im1, cax=cax1)
    cbar1.set_ticks([0, 1, 2, 3, 4, 5])
    cbar1.set_ticklabels(['Unknown', 'Riemannian', 'Lorentzian',
                          'Minkowski', 'Indefinite', 'Degenerate'])

    # Plot 2: Number of negative eigenvalues
    im2 = ax2.imshow(
        n_negative_field,
        origin='lower',
        cmap='hot',
        aspect='auto'
    )

    ax2.set_xlabel("Base Manifold X")
    ax2.set_ylabel("Base Manifold Y")
    ax2.set_title("Number of Negative Eigenvalues")

    divider2 = make_axes_locatable(ax2)
    cax2 = divider2.append_axes("right", size="5%", pad=0.1)
    plt.colorbar(im2, cax=cax2, label="# Negative λ")

    plt.suptitle(f"Signature Analysis (Agent {agent_idx})", fontsize=14, y=1.02)
    plt.tight_layout()

    if out_path:
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved {out_path}")

    if show:
        plt.show()
    else:
        plt.close()

    # Print summary statistics
    total_points = H * W
    n_lorentzian = np.sum(signature_field == SIG_CODES[MetricSignature.LORENTZIAN])
    n_minkowski = np.sum(signature_field == SIG_CODES[MetricSignature.MINKOWSKI])
    n_riemannian = np.sum(signature_field == SIG_CODES[MetricSignature.RIEMANNIAN])
    n_indefinite = np.sum(signature_field == SIG_CODES[MetricSignature.INDEFINITE])

    print(f"\n📊 Signature Statistics (Agent {agent_idx}):")
    print(f"  Riemannian: {n_riemannian}/{total_points} ({100*n_riemannian/total_points:.1f}%)")
    print(f"  Lorentzian: {n_lorentzian}/{total_points} ({100*n_lorentzian/total_points:.1f}%)")
    print(f"  Minkowski:  {n_minkowski}/{total_points} ({100*n_minkowski/total_points:.1f}%)")
    print(f"  Indefinite: {n_indefinite}/{total_points} ({100*n_indefinite/total_points:.1f}%)")


def plot_metric_dashboard(
    system,
    agent_idx: int = 0,
    grid_size: Tuple[int, int] = (20, 20),
    out_path: Optional[Path] = None,
    show: bool = False
):
    """
    Comprehensive dashboard showing all metric properties.

    Creates 2x3 grid:
    - Row 1: Determinant, Trace, Signature
    - Row 2: Eigenvalue λ_0, Eigenvalue λ_1, # Negative eigenvalues

    Args:
        system: MultiAgentSystem
        agent_idx: Which agent to analyze
        grid_size: (H, W) resolution
        out_path: Save path
        show: Whether to display
    """
    agent = system.agents[agent_idx]
    H, W = grid_size

    # Compute all metric properties
    test_g = compute_pullback_metric(agent, point_idx=0)
    d = test_g.shape[0]

    determinant = np.zeros((H, W))
    trace = np.zeros((H, W))
    signature = np.zeros((H, W), dtype=int)
    eigenvalues = np.zeros((H, W, d))
    n_negative = np.zeros((H, W), dtype=int)

    SIG_CODES = {
        MetricSignature.RIEMANNIAN: 1,
        MetricSignature.LORENTZIAN: 2,
        MetricSignature.MINKOWSKI: 3,
        MetricSignature.INDEFINITE: 4,
        MetricSignature.DEGENERATE: 5,
        MetricSignature.UNKNOWN: 0
    }

    for i in range(H):
        for j in range(W):
            point_idx = i * W + j
            g = compute_pullback_metric(agent, point_idx=point_idx)
            sig = analyze_metric_signature(g)

            determinant[i, j] = np.linalg.det(g)
            trace[i, j] = np.trace(g)
            signature[i, j] = SIG_CODES[sig.signature]
            eigenvalues[i, j, :] = sig.eigenvalues[:d]
            n_negative[i, j] = sig.signature_tuple[0]

    # Create dashboard
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

    # Row 1, Col 1: Determinant
    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.imshow(determinant, origin='lower', cmap='viridis', aspect='auto')
    ax1.set_title("det(g) - Volume Distortion")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    # Row 1, Col 2: Trace
    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.imshow(trace, origin='lower', cmap='plasma', aspect='auto')
    ax2.set_title("tr(g) - Total Variance")
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    # Row 1, Col 3: Signature
    ax3 = fig.add_subplot(gs[0, 2])
    cmap = plt.cm.get_cmap('tab10', 6)
    im3 = ax3.imshow(signature, origin='lower', cmap=cmap, vmin=0, vmax=5, aspect='auto')
    ax3.set_title("Signature Classification")
    ax3.set_xlabel("X")
    ax3.set_ylabel("Y")
    cbar3 = plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
    cbar3.set_ticks([0, 1, 2, 3, 4, 5])
    cbar3.set_ticklabels(['?', 'R', 'L', 'M', 'I', 'D'])

    # Row 2, Col 1: Eigenvalue 0
    ax4 = fig.add_subplot(gs[1, 0])
    norm4 = TwoSlopeNorm(vmin=eigenvalues[:, :, 0].min(),
                         vcenter=0.0,
                         vmax=eigenvalues[:, :, 0].max())
    im4 = ax4.imshow(eigenvalues[:, :, 0], origin='lower', cmap='RdBu_r', norm=norm4, aspect='auto')
    ax4.set_title("Eigenvalue λ_0")
    ax4.set_xlabel("X")
    ax4.set_ylabel("Y")
    plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)

    # Row 2, Col 2: Eigenvalue 1 (if exists)
    ax5 = fig.add_subplot(gs[1, 1])
    if d > 1:
        norm5 = TwoSlopeNorm(vmin=eigenvalues[:, :, 1].min(),
                             vcenter=0.0,
                             vmax=eigenvalues[:, :, 1].max())
        im5 = ax5.imshow(eigenvalues[:, :, 1], origin='lower', cmap='RdBu_r', norm=norm5, aspect='auto')
        ax5.set_title("Eigenvalue λ_1")
    else:
        im5 = ax5.imshow(np.zeros((H, W)), origin='lower', cmap='gray', aspect='auto')
        ax5.set_title("Eigenvalue λ_1 (N/A)")
    ax5.set_xlabel("X")
    ax5.set_ylabel("Y")
    plt.colorbar(im5, ax=ax5, fraction=0.046, pad=0.04)

    # Row 2, Col 3: Number of negative eigenvalues
    ax6 = fig.add_subplot(gs[1, 2])
    im6 = ax6.imshow(n_negative, origin='lower', cmap='hot', aspect='auto')
    ax6.set_title("# Negative Eigenvalues")
    ax6.set_xlabel("X")
    ax6.set_ylabel("Y")
    plt.colorbar(im6, ax=ax6, fraction=0.046, pad=0.04)

    plt.suptitle(f"Pullback Metric Dashboard (Agent {agent_idx})", fontsize=16, y=0.995)

    if out_path:
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved {out_path}")

    if show:
        plt.show()
    else:
        plt.close()


def compute_connection_coefficients(
    system,
    agent_idx: int = 0,
    point_idx: int = 0,
    eps: float = 1e-4
) -> np.ndarray:
    """
    Compute Christoffel symbols Γ^α_βγ at a point using finite differences.

    The Christoffel symbols of the Levi-Civita connection are:
        Γ^α_βγ = (1/2) g^αδ (∂_β g_δγ + ∂_γ g_βδ - ∂_δ g_βγ)

    Args:
        system: MultiAgentSystem
        agent_idx: Which agent
        point_idx: Where to compute
        eps: Finite difference step size

    Returns:
        Gamma: (d, d, d) array of connection coefficients
    """
    agent = system.agents[agent_idx]

    # Compute metric at point
    g = compute_pullback_metric(agent, point_idx=point_idx)
    d = g.shape[0]
    g_inv = np.linalg.inv(g)

    # Compute metric derivatives using finite differences
    # This is approximate - proper implementation needs coordinate system
    dg = np.zeros((d, d, d))  # dg[α, β, γ] = ∂_α g_βγ

    # Placeholder: finite differences in embedding space
    # (Full implementation requires coordinate chart)
    for alpha in range(d):
        # Perturb in direction α
        g_plus = compute_pullback_metric(agent, point_idx=point_idx)  # Would need actual shift
        g_minus = compute_pullback_metric(agent, point_idx=point_idx)
        dg[alpha] = (g_plus - g_minus) / (2 * eps)

    # Compute Christoffel symbols
    Gamma = np.zeros((d, d, d))
    for alpha in range(d):
        for beta in range(d):
            for gamma in range(d):
                Gamma[alpha, beta, gamma] = 0.5 * sum(
                    g_inv[alpha, delta] * (
                        dg[beta, delta, gamma] +
                        dg[gamma, beta, delta] -
                        dg[delta, beta, gamma]
                    )
                    for delta in range(d)
                )

    return Gamma


# Convenience function to generate all plots
def generate_all_metric_plots(
    system,
    agent_idx: int = 0,
    out_dir: Optional[Path] = None,
    grid_size: Tuple[int, int] = (20, 20)
):
    """
    Generate complete suite of metric visualizations.

    Creates:
    - Pullback metric determinant
    - Pullback metric trace
    - Eigenvalue fields
    - Eigenvector field (largest eigenvalue)
    - Signature classification
    - Complete dashboard

    Args:
        system: MultiAgentSystem
        agent_idx: Which agent to analyze
        out_dir: Output directory (creates if doesn't exist)
        grid_size: Resolution for plots
    """
    if out_dir is None:
        out_dir = Path("_results/metric_analysis")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"GENERATING METRIC VISUALIZATIONS (Agent {agent_idx})")
    print(f"{'='*70}")
    print(f"Output directory: {out_dir}")
    print(f"Grid resolution: {grid_size}")
    print()

    # Dashboard (comprehensive overview)
    plot_metric_dashboard(
        system, agent_idx, grid_size,
        out_path=out_dir / f"metric_dashboard_agent{agent_idx}.png"
    )

    # Signature classification
    plot_signature_classification(
        system, agent_idx, grid_size,
        out_path=out_dir / f"signature_classification_agent{agent_idx}.png"
    )

    # Individual components
    plot_pullback_metric_field(
        system, agent_idx, grid_size, component="determinant",
        out_path=out_dir / f"metric_determinant_agent{agent_idx}.png"
    )

    plot_pullback_metric_field(
        system, agent_idx, grid_size, component="trace",
        out_path=out_dir / f"metric_trace_agent{agent_idx}.png"
    )

    # Eigenvalue field
    plot_eigenvalue_field(
        system, agent_idx, grid_size,
        out_path=out_dir / f"eigenvalue_field_agent{agent_idx}.png"
    )

    # Eigenvector field (for largest |eigenvalue|)
    plot_eigenvector_field(
        system, agent_idx,
        grid_size=(10, 10),  # Coarser for arrow clarity
        eigenvalue_idx=0,
        out_path=out_dir / f"eigenvector_field_agent{agent_idx}.png"
    )

    print(f"\n{'='*70}")
    print(f"✓ METRIC VISUALIZATION COMPLETE")
    print(f"{'='*70}\n")