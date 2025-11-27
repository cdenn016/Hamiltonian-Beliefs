# -*- coding: utf-8 -*-
"""
Created on Sat Nov 22 09:03:16 2025

@author: chris and christine
"""

"""
Belief Trajectory Visualization
================================

Tools for visualizing how agent beliefs evolve through the statistical manifold.

The key insight: Belief trajectories (μ(t), Σ(t), φ(t)) naturally define a
"time direction" (the tangent vector), which when combined with the pullback
metric could reveal emergent Lorentzian structure.

For a fixed point c in the base manifold, we track:
- μ_q(t, c): Belief mean evolution
- Σ_q(t, c): Covariance evolution
- φ(t, c): Gauge field evolution
- Tangent vectors: dμ/dt, dΣ/dt, dφ/dt
- Metric signature along trajectory

Author: Chris
Date: November 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import Axes3D, proj3d
from mpl_toolkits.axes_grid1 import make_axes_locatable
from typing import Optional, Tuple, List
from pathlib import Path

from geometry.signature_analysis import (
    compute_pullback_metric,
    analyze_metric_signature,
    MetricSignature
)
from dynamics.trajectory_metrics import (
    compute_trajectory_pullback_metrics,
    analyze_signature_evolution
)


class Arrow3D(FancyArrowPatch):
    """3D arrow for matplotlib 3D plots."""
    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        return np.min(zs)


def extract_trajectory_at_point(
    history,
    point_idx: int = 0,
    agent_idx: int = 0
) -> dict:
    """
    Extract belief trajectory at a fixed base manifold point.

    Args:
        history: Training history (must have mu_tracker)
        point_idx: Which point in base manifold
        agent_idx: Which agent

    Returns:
        Dictionary with:
            - steps: Array of time steps
            - mu: (T, K) belief means
            - mu_norms: (T,) norms of mu
            - tangent_mu: (T-1, K) tangent vectors dμ/dt
    """
    from analysis.core.loaders import get_mu_tracker

    mu_tracker = get_mu_tracker(history)
    if mu_tracker is None:
        raise ValueError("History must have mu_tracker for trajectory analysis")

    steps = np.array(mu_tracker.steps)
    T = len(steps)

    # Extract mu components for this agent at this point
    # mu_tracker.mu_components[agent_idx] is a list of (S,K) arrays
    # We want the trajectory at point_idx
    mu_trajectory = []
    for t in range(T):
        mu_t = mu_tracker.mu_components[agent_idx][t]  # (S, K)
        if mu_t.ndim == 1:
            # Single point case
            mu_trajectory.append(mu_t)
        else:
            # Multiple points - extract point_idx
            mu_trajectory.append(mu_t[point_idx, :])

    mu_trajectory = np.array(mu_trajectory)  # (T, K)
    K = mu_trajectory.shape[1]

    # Compute tangent vectors (finite differences)
    tangent_mu = np.diff(mu_trajectory, axis=0)  # (T-1, K)

    # Compute norms
    mu_norms = np.linalg.norm(mu_trajectory, axis=1)  # (T,)

    return {
        'steps': steps,
        'mu': mu_trajectory,
        'mu_norms': mu_norms,
        'tangent_mu': tangent_mu,
        'K': K,
        'T': T
    }


def plot_belief_trajectory_3d(
    history,
    out_dir: Path,
    point_idx: int = 0,
    agent_idx: int = 0,
    show_tangents: bool = True,
    subsample: int = 10
):
    """
    Plot belief trajectory in 3D (μ₁, μ₂, μ₃) space.

    Args:
        history: Training history
        out_dir: Output directory
        point_idx: Which base manifold point
        agent_idx: Which agent
        show_tangents: Whether to show tangent vectors
        subsample: Show tangent every N steps (for clarity)
    """
    traj = extract_trajectory_at_point(history, point_idx, agent_idx)

    if traj['K'] < 3:
        print(f"⚠️  Latent dim K={traj['K']} < 3, skipping 3D trajectory plot")
        return

    mu = traj['mu']
    tangent = traj['tangent_mu']
    steps = traj['steps']

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Plot trajectory
    ax.plot(
        mu[:, 0], mu[:, 1], mu[:, 2],
        linewidth=2, color='C0', alpha=0.7,
        label='Belief Trajectory'
    )

    # Mark start and end
    ax.scatter(
        mu[0, 0], mu[0, 1], mu[0, 2],
        s=100, c='green', marker='o',
        label='Start', edgecolors='black', linewidths=1.5
    )
    ax.scatter(
        mu[-1, 0], mu[-1, 1], mu[-1, 2],
        s=100, c='red', marker='s',
        label='End', edgecolors='black', linewidths=1.5
    )

    # Show tangent vectors (subsampled)
    if show_tangents:
        for i in range(0, len(tangent), subsample):
            base = mu[i]
            direction = tangent[i]
            # Normalize for visualization
            direction = direction / (np.linalg.norm(direction) + 1e-10) * 0.5

            arrow = Arrow3D(
                [base[0], base[0] + direction[0]],
                [base[1], base[1] + direction[1]],
                [base[2], base[2] + direction[2]],
                mutation_scale=20,
                lw=1.5,
                arrowstyle='-|>',
                color='red',
                alpha=0.6
            )
            ax.add_artist(arrow)

    ax.set_xlabel('μ₁', fontsize=12)
    ax.set_ylabel('μ₂', fontsize=12)
    ax.set_zlabel('μ₃', fontsize=12)
    ax.set_title(f'Belief Trajectory (Agent {agent_idx}, Point {point_idx})', fontsize=14)
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    out_path = out_dir / f"trajectory_3d_agent{agent_idx}_point{point_idx}.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved {out_path}")


def plot_belief_trajectory_projections(
    history,
    out_dir: Path,
    point_idx: int = 0,
    agent_idx: int = 0
):
    """
    Plot 2D projections of belief trajectory.

    Creates 3 subplots: (μ₁, μ₂), (μ₁, μ₃), (μ₂, μ₃).

    Args:
        history: Training history
        out_dir: Output directory
        point_idx: Which base manifold point
        agent_idx: Which agent
    """
    traj = extract_trajectory_at_point(history, point_idx, agent_idx)
    mu = traj['mu']
    K = traj['K']

    if K < 2:
        print(f"⚠️  Latent dim K={K} < 2, skipping trajectory projections")
        return

    # Create subplots for projections
    n_plots = min(3, K * (K - 1) // 2)  # Max 3 projections
    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 5))

    if n_plots == 1:
        axes = [axes]

    # (μ₁, μ₂)
    if K >= 2:
        ax = axes[0]
        ax.plot(mu[:, 0], mu[:, 1], linewidth=2, alpha=0.7, color='C0')
        ax.scatter(mu[0, 0], mu[0, 1], s=100, c='green', marker='o',
                   label='Start', edgecolors='black', linewidths=1.5, zorder=10)
        ax.scatter(mu[-1, 0], mu[-1, 1], s=100, c='red', marker='s',
                   label='End', edgecolors='black', linewidths=1.5, zorder=10)
        ax.set_xlabel('μ₁', fontsize=12)
        ax.set_ylabel('μ₂', fontsize=12)
        ax.set_title('Projection: (μ₁, μ₂)')
        ax.grid(alpha=0.3)
        ax.legend()

    # (μ₁, μ₃)
    if K >= 3 and n_plots >= 2:
        ax = axes[1]
        ax.plot(mu[:, 0], mu[:, 2], linewidth=2, alpha=0.7, color='C1')
        ax.scatter(mu[0, 0], mu[0, 2], s=100, c='green', marker='o',
                   edgecolors='black', linewidths=1.5, zorder=10)
        ax.scatter(mu[-1, 0], mu[-1, 2], s=100, c='red', marker='s',
                   edgecolors='black', linewidths=1.5, zorder=10)
        ax.set_xlabel('μ₁', fontsize=12)
        ax.set_ylabel('μ₃', fontsize=12)
        ax.set_title('Projection: (μ₁, μ₃)')
        ax.grid(alpha=0.3)

    # (μ₂, μ₃)
    if K >= 3 and n_plots >= 3:
        ax = axes[2]
        ax.plot(mu[:, 1], mu[:, 2], linewidth=2, alpha=0.7, color='C2')
        ax.scatter(mu[0, 1], mu[0, 2], s=100, c='green', marker='o',
                   edgecolors='black', linewidths=1.5, zorder=10)
        ax.scatter(mu[-1, 1], mu[-1, 2], s=100, c='red', marker='s',
                   edgecolors='black', linewidths=1.5, zorder=10)
        ax.set_xlabel('μ₂', fontsize=12)
        ax.set_ylabel('μ₃', fontsize=12)
        ax.set_title('Projection: (μ₂, μ₃)')
        ax.grid(alpha=0.3)

    plt.suptitle(f'Belief Trajectory Projections (Agent {agent_idx}, Point {point_idx})',
                 fontsize=14, y=1.02)
    plt.tight_layout()

    out_path = out_dir / f"trajectory_projections_agent{agent_idx}_point{point_idx}.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved {out_path}")


def plot_trajectory_tangent_vectors(
    history,
    out_dir: Path,
    point_idx: int = 0,
    agent_idx: int = 0
):
    """
    Visualize tangent vectors dμ/dt along trajectory.

    The tangent vector defines a natural "time" direction in the manifold.

    Args:
        history: Training history
        out_dir: Output directory
        point_idx: Which base manifold point
        agent_idx: Which agent
    """
    traj = extract_trajectory_at_point(history, point_idx, agent_idx)

    tangent = traj['tangent_mu']  # (T-1, K)
    steps = traj['steps'][:-1]  # Match tangent length
    K = traj['K']

    # Compute tangent norms and directions
    tangent_norms = np.linalg.norm(tangent, axis=1)  # (T-1,)

    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    # Top: Tangent norm evolution
    ax1 = axes[0]
    ax1.plot(steps, tangent_norms, linewidth=2, color='C0')
    ax1.set_xlabel('Training Step')
    ax1.set_ylabel('||dμ/dt||')
    ax1.set_title('Tangent Vector Magnitude (Speed in Manifold)')
    ax1.grid(alpha=0.3)

    # Bottom: Tangent components
    ax2 = axes[1]
    for k in range(min(K, 5)):  # Plot up to 5 components
        ax2.plot(steps, tangent[:, k], label=f'dμ_{k}/dt', alpha=0.7)
    ax2.set_xlabel('Training Step')
    ax2.set_ylabel('Tangent Components')
    ax2.set_title('Tangent Vector Components')
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.suptitle(f'Tangent Vectors - Time Direction (Agent {agent_idx}, Point {point_idx})',
                 fontsize=14, y=1.0)
    plt.tight_layout()

    out_path = out_dir / f"trajectory_tangents_agent{agent_idx}_point{point_idx}.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved {out_path}")


def plot_trajectory_metric_signature(
    history,
    system,
    out_dir: Path,
    point_idx: int = 0,
    agent_idx: int = 0
):
    """
    Analyze metric signature along belief trajectory.

    This is the KEY ANALYSIS: Does the pullback metric change signature
    as beliefs evolve? Do we find Lorentzian regions?

    Args:
        history: Training history (must have agent_snapshots if available)
        system: MultiAgentSystem (for computing pullback metric)
        out_dir: Output directory
        point_idx: Which base manifold point
        agent_idx: Which agent
    """
    traj = extract_trajectory_at_point(history, point_idx, agent_idx)
    steps = traj['steps']
    T = len(steps)

    agent = system.agents[agent_idx]

    # Check if we have snapshots for time-varying metric analysis
    has_snapshots = (hasattr(history, 'agent_snapshots') and
                     len(history.agent_snapshots) > 0)

    signatures = []
    eigenvalues_list = []
    n_negative_list = []

    if has_snapshots:
        # ✨ NEW: Compute time-varying pullback metrics from snapshots
        print(f"  Computing time-varying pullback metrics from {len(history.agent_snapshots)} snapshots...")

        try:
            traj_metrics = compute_trajectory_pullback_metrics(
                history,
                agent_idx=agent_idx,
                point_idx=point_idx,
                metric_type="belief"
            )

            # Extract eigenvalue trajectories and signatures
            eigenvalues_array = traj_metrics.get_eigenvalue_trajectories()  # (T, d)
            signature_sequence = traj_metrics.get_signature_sequence()

            # Convert signatures to counts of negative eigenvalues
            signature_map = {
                "riemannian": 0,
                "lorentzian": 1,
                "indefinite": 2,  # Simplified - could be more
                "degenerate": -1
            }
            n_negative_array = np.array([
                signature_map.get(sig, -1) for sig in signature_sequence
            ])

            # Use snapshot steps (may be subset of all steps)
            steps = np.array(traj_metrics.steps)

            print(f"  ✓ Computed {len(traj_metrics.metrics)} time-varying metrics")

        except Exception as e:
            print(f"  ⚠️  Failed to compute trajectory metrics: {e}")
            print(f"  Falling back to static metric computation...")
            has_snapshots = False  # Fall through to static computation

    if not has_snapshots:
        # Fallback: Static metric (computed once at final state)
        print(f"  ⚠️  No snapshots available - using static metric (repeated for all steps)")

        g = compute_pullback_metric(agent, point_idx=point_idx)
        sig = analyze_metric_signature(g)

        # Repeat static metric for all timesteps
        for t in range(T):
            signatures.append(sig.signature.value)
            eigenvalues_list.append(sig.eigenvalues)
            n_negative_list.append(sig.signature_tuple[0])

        eigenvalues_array = np.array(eigenvalues_list)  # (T, d)
        n_negative_array = np.array(n_negative_list)  # (T,)

    # Create visualization
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    # Top: Eigenvalues over time
    ax1 = axes[0]
    d = eigenvalues_array.shape[1]
    for i in range(d):
        ax1.plot(steps, eigenvalues_array[:, i], label=f'λ_{i}', linewidth=2)
    ax1.axhline(0, color='black', linestyle='--', alpha=0.3, linewidth=1)
    ax1.set_xlabel('Training Step')
    ax1.set_ylabel('Eigenvalue')
    ax1.set_title('Metric Eigenvalues Along Trajectory')
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Bottom: Number of negative eigenvalues
    ax2 = axes[1]
    ax2.plot(steps, n_negative_array, linewidth=2, color='red', marker='o', markersize=4)
    ax2.set_xlabel('Training Step')
    ax2.set_ylabel('# Negative Eigenvalues')
    ax2.set_title('Signature Evolution (0 = Riemannian, 1 = Lorentzian, >1 = Indefinite)')
    ax2.set_ylim(-0.5, max(n_negative_array) + 0.5)
    ax2.grid(alpha=0.3)

    plt.suptitle(f'Metric Signature Along Trajectory (Agent {agent_idx}, Point {point_idx})',
                 fontsize=14, y=1.0)
    plt.tight_layout()

    out_path = out_dir / f"trajectory_signature_agent{agent_idx}_point{point_idx}.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved {out_path}")

    # Print summary
    print(f"\n📊 Signature Summary (Agent {agent_idx}, Point {point_idx}):")
    if has_snapshots:
        # Time-varying metrics summary
        from collections import Counter
        sig_counts = Counter(signature_sequence)
        print(f"  Signature distribution:")
        for sig_type, count in sig_counts.items():
            frac = count / len(signature_sequence)
            print(f"    {sig_type}: {count}/{len(signature_sequence)} ({frac:.1%})")
        print(f"  Final eigenvalues: {eigenvalues_array[-1]}")
        if signature_sequence:
            print(f"  Final signature: {signature_sequence[-1]}")
    else:
        # Static metric summary
        print(f"  Final signature: {sig.signature.value}")
        print(f"  Final eigenvalues: {sig.eigenvalues}")

    print(f"  Lorentzian steps: {np.sum(n_negative_array == 1)}/{len(n_negative_array)}")


def plot_trajectory_phase_space(
    history,
    out_dir: Path,
    point_idx: int = 0,
    agent_idx: int = 0,
    dims: Tuple[int, int] = (0, 1)
):
    """
    Plot phase space (μ, dμ/dt) for selected dimensions.

    This combines position and velocity, showing the full dynamical state.

    Args:
        history: Training history
        out_dir: Output directory
        point_idx: Which base manifold point
        agent_idx: Which agent
        dims: Which (μ_i, μ_j) dimensions to plot
    """
    traj = extract_trajectory_at_point(history, point_idx, agent_idx)

    mu = traj['mu'][:-1]  # (T-1, K) - match tangent length
    tangent = traj['tangent_mu']  # (T-1, K)
    steps = traj['steps'][:-1]

    i, j = dims

    if i >= traj['K'] or j >= traj['K']:
        print(f"⚠️  Dimension {max(i,j)} >= K={traj['K']}, skipping phase space plot")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: (μ_i, dμ_i/dt)
    ax1 = axes[0]
    scatter = ax1.scatter(
        mu[:, i], tangent[:, i],
        c=steps, cmap='viridis',
        s=20, alpha=0.7
    )
    ax1.set_xlabel(f'μ_{i}', fontsize=12)
    ax1.set_ylabel(f'dμ_{i}/dt', fontsize=12)
    ax1.set_title(f'Phase Space: (μ_{i}, dμ_{i}/dt)')
    ax1.grid(alpha=0.3)
    plt.colorbar(scatter, ax=ax1, label='Step')

    # Right: (μ_j, dμ_j/dt)
    ax2 = axes[1]
    scatter = ax2.scatter(
        mu[:, j], tangent[:, j],
        c=steps, cmap='viridis',
        s=20, alpha=0.7
    )
    ax2.set_xlabel(f'μ_{j}', fontsize=12)
    ax2.set_ylabel(f'dμ_{j}/dt', fontsize=12)
    ax2.set_title(f'Phase Space: (μ_{j}, dμ_{j}/dt)')
    ax2.grid(alpha=0.3)
    plt.colorbar(scatter, ax=ax2, label='Step')

    plt.suptitle(f'Phase Space Analysis (Agent {agent_idx}, Point {point_idx})',
                 fontsize=14, y=1.02)
    plt.tight_layout()

    out_path = out_dir / f"trajectory_phase_space_agent{agent_idx}_point{point_idx}.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved {out_path}")


def plot_trajectory_dashboard(
    history,
    system,
    out_dir: Path,
    point_idx: int = 0,
    agent_idx: int = 0
):
    """
    Comprehensive trajectory analysis dashboard.

    Creates 2x2 grid:
    - Top left: 3D trajectory or 2D projection
    - Top right: Tangent norm evolution
    - Bottom left: Eigenvalues along trajectory
    - Bottom right: Phase space (μ₁, dμ₁/dt)

    Args:
        history: Training history
        system: MultiAgentSystem
        out_dir: Output directory
        point_idx: Which base manifold point
        agent_idx: Which agent
    """
    traj = extract_trajectory_at_point(history, point_idx, agent_idx)
    mu = traj['mu']
    tangent = traj['tangent_mu']
    steps = traj['steps']
    K = traj['K']

    agent = system.agents[agent_idx]

    # Check if we have snapshots for time-varying metric analysis
    has_snapshots = (hasattr(history, 'agent_snapshots') and
                     len(history.agent_snapshots) > 0)

    if has_snapshots:
        try:
            traj_metrics = compute_trajectory_pullback_metrics(
                history,
                agent_idx=agent_idx,
                point_idx=point_idx,
                metric_type="belief"
            )
            eigenvalues_array = traj_metrics.get_eigenvalue_trajectories()
            metric_steps = np.array(traj_metrics.steps)
        except:
            has_snapshots = False

    if not has_snapshots:
        # Fallback: static metric
        g = compute_pullback_metric(agent, point_idx=point_idx)
        sig = analyze_metric_signature(g)

    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    # Top left: Trajectory projection (μ₁, μ₂)
    ax1 = fig.add_subplot(gs[0, 0])
    if K >= 2:
        ax1.plot(mu[:, 0], mu[:, 1], linewidth=2, alpha=0.7, color='C0')
        ax1.scatter(mu[0, 0], mu[0, 1], s=100, c='green', marker='o',
                   label='Start', edgecolors='black', linewidths=1.5)
        ax1.scatter(mu[-1, 0], mu[-1, 1], s=100, c='red', marker='s',
                   label='End', edgecolors='black', linewidths=1.5)
        ax1.set_xlabel('μ₁', fontsize=12)
        ax1.set_ylabel('μ₂', fontsize=12)
        ax1.set_title('Belief Trajectory (μ₁, μ₂)')
        ax1.legend()
    ax1.grid(alpha=0.3)

    # Top right: Tangent norm
    ax2 = fig.add_subplot(gs[0, 1])
    tangent_norms = np.linalg.norm(tangent, axis=1)
    ax2.plot(steps[:-1], tangent_norms, linewidth=2, color='C1')
    ax2.set_xlabel('Training Step')
    ax2.set_ylabel('||dμ/dt||')
    ax2.set_title('Tangent Vector Magnitude')
    ax2.grid(alpha=0.3)

    # Bottom left: Eigenvalues (time-varying or static)
    ax3 = fig.add_subplot(gs[1, 0])
    if has_snapshots:
        # Time-varying eigenvalues
        d = eigenvalues_array.shape[1]
        for i in range(d):
            ax3.plot(metric_steps, eigenvalues_array[:, i], label=f'λ_{i}',
                     linewidth=2)
        ax3.set_xlabel('Training Step')
        ax3.set_title('Metric Eigenvalues Evolution')
    else:
        # Static eigenvalues (horizontal lines)
        d = len(sig.eigenvalues)
        for i in range(d):
            ax3.axhline(sig.eigenvalues[i], label=f'λ_{i} = {sig.eigenvalues[i]:.3f}',
                        linewidth=2)
        ax3.set_title(f'Metric Eigenvalues ({sig.signature.value})')

    ax3.axhline(0, color='black', linestyle='--', alpha=0.3, linewidth=1)
    ax3.set_ylabel('Eigenvalue')
    ax3.legend()
    ax3.grid(alpha=0.3)

    # Bottom right: Phase space (μ₁, dμ₁/dt)
    ax4 = fig.add_subplot(gs[1, 1])
    scatter = ax4.scatter(
        mu[:-1, 0], tangent[:, 0],
        c=steps[:-1], cmap='viridis',
        s=30, alpha=0.7
    )
    ax4.set_xlabel('μ₁', fontsize=12)
    ax4.set_ylabel('dμ₁/dt', fontsize=12)
    ax4.set_title('Phase Space (μ₁, dμ₁/dt)')
    ax4.grid(alpha=0.3)
    plt.colorbar(scatter, ax=ax4, label='Step')

    plt.suptitle(f'Belief Trajectory Dashboard (Agent {agent_idx}, Point {point_idx})',
                 fontsize=16, y=0.995)

    out_path = out_dir / f"trajectory_dashboard_agent{agent_idx}_point{point_idx}.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved {out_path}")


def generate_all_trajectory_plots(
    history,
    system,
    out_dir: Optional[Path] = None,
    point_idx: int = 0,
    agent_idx: int = 0
):
    """
    Generate complete suite of belief trajectory visualizations.

    Args:
        history: Training history (must have mu_tracker)
        system: MultiAgentSystem
        out_dir: Output directory
        point_idx: Which base manifold point to analyze
        agent_idx: Which agent to analyze
    """
    if out_dir is None:
        out_dir = Path("_results/trajectory_analysis")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"GENERATING TRAJECTORY VISUALIZATIONS")
    print(f"{'='*70}")
    print(f"Agent: {agent_idx}")
    print(f"Base manifold point: {point_idx}")
    print(f"Output directory: {out_dir}")
    print()

    # Dashboard (comprehensive overview)
    plot_trajectory_dashboard(history, system, out_dir, point_idx, agent_idx)

    # 3D trajectory
    plot_belief_trajectory_3d(history, out_dir, point_idx, agent_idx)

    # 2D projections
    plot_belief_trajectory_projections(history, out_dir, point_idx, agent_idx)

    # Tangent vectors
    plot_trajectory_tangent_vectors(history, out_dir, point_idx, agent_idx)

    # Metric signature analysis
    plot_trajectory_metric_signature(history, system, out_dir, point_idx, agent_idx)

    # Phase space
    plot_trajectory_phase_space(history, out_dir, point_idx, agent_idx, dims=(0, 1))

    print(f"\n{'='*70}")
    print(f"✓ TRAJECTORY VISUALIZATION COMPLETE")
    print(f"{'='*70}\n")