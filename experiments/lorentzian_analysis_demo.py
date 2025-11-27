# -*- coding: utf-8 -*-
"""
Created on Sat Nov 22 09:03:48 2025

@author: chris and christine
"""

#!/usr/bin/env python3
"""
Lorentzian Signature Analysis - Complete Demo
==============================================

Comprehensive demonstration of all Lorentzian signature analysis tools:

1. Metric signature detection and classification
2. Pullback metric visualization on base manifold
3. Belief trajectory analysis with tangent vectors
4. Light cone structure for Lorentzian regions
5. Experimental mechanisms (α-divergence, Hamiltonian, Lorentz gauge)

Usage:
------
    # Run complete demo on existing training results
    python examples/lorentzian_analysis_demo.py --run-dir _results/_playground

    # Run just metric visualization
    python examples/lorentzian_analysis_demo.py --mode metric

    # Run just trajectory analysis
    python examples/lorentzian_analysis_demo.py --mode trajectory

    # Run experimental mechanisms
    python examples/lorentzian_analysis_demo.py --mode experiments

Author: Chris
Date: November 2025
"""

import argparse
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from analysis.core.loaders import load_history, load_system

# Signature analysis
from geometry.signature_analysis import (
    analyze_metric_signature,
    compute_pullback_metric,
    signature_field_2d,
    compute_light_cone_structure,
    MetricSignature
)

# Metric visualization
from geometry.metric_visualization import (
    plot_pullback_metric_field,
    plot_eigenvalue_field,
    plot_eigenvector_field,
    plot_signature_classification,
    plot_metric_dashboard,
    generate_all_metric_plots
)

# Trajectory visualization
from analysis.plots.belief_trajectories import (
    plot_belief_trajectory_3d,
    plot_belief_trajectory_projections,
    plot_trajectory_tangent_vectors,
    plot_trajectory_metric_signature,
    plot_trajectory_phase_space,
    plot_trajectory_dashboard,
    generate_all_trajectory_plots
)

# Experimental mechanisms
from experiments.lorentzian.alpha_divergence import run_alpha_divergence_test
from experiments.lorentzian.hamiltonian_beliefs import run_hamiltonian_test
from experiments.lorentzian.lorentz_gauge import run_lorentz_gauge_test


def demo_signature_detection(system, out_dir: Path):
    """
    Demo: Basic signature detection and classification.
    """
    print("\n" + "="*70)
    print("DEMO 1: SIGNATURE DETECTION")
    print("="*70)

    agent = system.agents[0]
    point_idx = 0  # Center point

    # Compute pullback metric
    g = compute_pullback_metric(agent, point_idx=point_idx)
    print(f"\nPullback metric at point {point_idx}:")
    print(g)

    # Analyze signature
    sig = analyze_metric_signature(g)
    print(f"\n{sig}")
    print(f"Eigenvalues: {sig.eigenvalues}")
    print(f"Signature tuple: {sig.signature_tuple}")

    if sig.signature == MetricSignature.LORENTZIAN:
        print("\n🎉 LORENTZIAN SIGNATURE DETECTED!")
        print(f"Timelike direction: {sig.timelike_direction}")
        print(f"Spacelike directions shape: {sig.spacelike_directions.shape}")

        # Compute light cone structure
        try:
            light_cone = compute_light_cone_structure(g, np.zeros(len(g)))
            print(f"\nLight cone structure:")
            print(f"  Timelike vectors: {len(light_cone['timelike_vectors'])}")
            print(f"  Null vectors: {len(light_cone['null_vectors'])}")
            print(f"  Spacelike vectors: {len(light_cone['spacelike_vectors'])}")
        except Exception as e:
            print(f"  Light cone computation failed: {e}")
    else:
        print(f"\nSignature: {sig.signature.value} (not Lorentzian)")

    print("="*70)


def demo_metric_visualization(system, out_dir: Path):
    """
    Demo: Comprehensive metric visualization on base manifold.
    """
    print("\n" + "="*70)
    print("DEMO 2: METRIC VISUALIZATION")
    print("="*70)

    metric_dir = out_dir / "metric_analysis"
    metric_dir.mkdir(parents=True, exist_ok=True)

    agent_idx = 0
    grid_size = (20, 20)

    print(f"\nGenerating metric visualizations...")
    print(f"Agent: {agent_idx}")
    print(f"Grid size: {grid_size}")

    # Generate all metric plots
    generate_all_metric_plots(
        system,
        agent_idx=agent_idx,
        out_dir=metric_dir,
        grid_size=grid_size
    )

    print(f"\n✓ Metric plots saved to: {metric_dir}")
    print("="*70)


def demo_trajectory_analysis(history, system, out_dir: Path):
    """
    Demo: Belief trajectory analysis with tangent vectors.
    """
    print("\n" + "="*70)
    print("DEMO 3: TRAJECTORY ANALYSIS")
    print("="*70)

    from analysis.core.loaders import get_mu_tracker

    mu_tracker = get_mu_tracker(history)
    if mu_tracker is None:
        print("⚠️  No mu_tracker in history - skipping trajectory analysis")
        print("   (Train with mu tracking enabled to use this feature)")
        print("="*70)
        return

    traj_dir = out_dir / "trajectory_analysis"
    traj_dir.mkdir(parents=True, exist_ok=True)

    agent_idx = 0
    point_idx = 0

    print(f"\nGenerating trajectory visualizations...")
    print(f"Agent: {agent_idx}")
    print(f"Base manifold point: {point_idx}")

    # Generate all trajectory plots
    generate_all_trajectory_plots(
        history,
        system,
        out_dir=traj_dir,
        point_idx=point_idx,
        agent_idx=agent_idx
    )

    print(f"\n✓ Trajectory plots saved to: {traj_dir}")
    print("="*70)


def demo_signature_field(system, out_dir: Path):
    """
    Demo: Map signature across 2D base manifold.
    """
    print("\n" + "="*70)
    print("DEMO 4: SIGNATURE FIELD MAPPING")
    print("="*70)

    agent_idx = 0
    grid_size = (20, 20)

    print(f"\nComputing signature field across base manifold...")
    print(f"Grid size: {grid_size}")

    # Compute signature field
    sig_field = signature_field_2d(system, agent_idx=agent_idx, grid_size=grid_size)

    # Summary statistics
    total_points = grid_size[0] * grid_size[1]
    n_lorentzian = np.sum(sig_field['lorentzian_mask'])

    print(f"\nResults:")
    print(f"  Total points: {total_points}")
    print(f"  Lorentzian regions: {n_lorentzian} ({100*n_lorentzian/total_points:.1f}%)")
    print(f"  Signature distribution:")

    unique, counts = np.unique(sig_field['signatures'], return_counts=True)
    for sig_val, count in zip(unique, counts):
        sig_name = {-1: 'Lorentzian', 0: 'Other', 1: 'Riemannian'}
        print(f"    {sig_name.get(sig_val, 'Unknown'):12s}: {count:4d} ({100*count/total_points:.1f}%)")

    if n_lorentzian > 0:
        print("\n🎉 LORENTZIAN REGIONS FOUND IN BASE MANIFOLD!")

    print("="*70)


def demo_experimental_mechanisms(out_dir: Path):
    """
    Demo: Run experimental mechanism testbeds.
    """
    print("\n" + "="*70)
    print("DEMO 5: EXPERIMENTAL MECHANISMS")
    print("="*70)

    exp_dir = out_dir / "experiments"
    exp_dir.mkdir(parents=True, exist_ok=True)

    print("\n[1/3] Testing α-divergence mechanism...")
    try:
        alpha_exp = run_alpha_divergence_test()
        print("  ✓ α-divergence test complete")
    except Exception as e:
        print(f"  ✗ α-divergence test failed: {e}")

    print("\n[2/3] Testing Hamiltonian dynamics...")
    try:
        ham_exps = run_hamiltonian_test()
        print("  ✓ Hamiltonian test complete")
    except Exception as e:
        print(f"  ✗ Hamiltonian test failed: {e}")

    print("\n[3/3] Testing Lorentz gauge fields...")
    try:
        lorentz_exp = run_lorentz_gauge_test()
        print("  ✓ Lorentz gauge test complete")
    except Exception as e:
        print(f"  ✗ Lorentz gauge test failed: {e}")

    print(f"\n✓ Experimental results saved to: {exp_dir}")
    print("="*70)


def main():
    parser = argparse.ArgumentParser(
        description="Complete demo of Lorentzian signature analysis tools"
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        default="_results/_playground",
        help="Path to training run directory"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["all", "signature", "metric", "trajectory", "field", "experiments"],
        default="all",
        help="Which demo to run"
    )
    parser.add_argument(
        "--agent-idx",
        type=int,
        default=0,
        help="Which agent to analyze"
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    out_dir = run_dir / "lorentzian_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*70)
    print("LORENTZIAN SIGNATURE ANALYSIS - COMPLETE DEMO")
    print("="*70)
    print(f"Run directory: {run_dir}")
    print(f"Output directory: {out_dir}")
    print(f"Mode: {args.mode}")
    print("="*70)

    # Load data
    if args.mode != "experiments":
        print("\nLoading training data...")
        history = load_history(run_dir)
        system = load_system(run_dir)

        if system is None:
            print("✗ Could not load system - check run directory")
            return

        print(f"✓ Loaded system with {system.n_agents} agents")

    # Run selected demos
    if args.mode in ["all", "signature"]:
        if system is not None:
            demo_signature_detection(system, out_dir)

    if args.mode in ["all", "metric"]:
        if system is not None:
            demo_metric_visualization(system, out_dir)

    if args.mode in ["all", "trajectory"]:
        if history is not None and system is not None:
            demo_trajectory_analysis(history, system, out_dir)

    if args.mode in ["all", "field"]:
        if system is not None:
            demo_signature_field(system, out_dir)

    if args.mode in ["all", "experiments"]:
        demo_experimental_mechanisms(out_dir)

    print("\n" + "="*70)
    print("✓ DEMO COMPLETE")
    print("="*70)
    print(f"All outputs saved to: {out_dir}")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()