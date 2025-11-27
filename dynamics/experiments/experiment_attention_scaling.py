# -*- coding: utf-8 -*-
"""
Created on Thu Nov 27 09:43:28 2025

@author: chris and christine
"""

#!/usr/bin/env python3
"""
Experiment: Power-Law Analysis of Attention Patterns
=====================================================

Studies the statistical properties of the REAL attention weights:

    β_ij = softmax(-KL(q_i || Ω_ij[q_j]) / κ)

Key questions:
1. What is the distribution of β_ij values?
2. Power-law vs exponential vs other?
3. How does scaling change with κ (temperature)?
4. Distance dependence (if any)?

This uses the actual KL-based softmax formula, NOT a fake exponential.

Usage:
    python dynamics/experiments/experiment_attention_scaling.py
    python dynamics/experiments/experiment_attention_scaling.py --n_agents 10 --kappa 0.5

Author: Chris & Claude
Date: November 2025
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from simulation_config import SimulationConfig
from simulation_runner import build_manifold, build_supports, build_agents, build_system
from analysis.core.power_law_attention import (
    extract_attention_weights,
    analyze_attention_distribution,
    scan_temperature_dependence,
    compute_rank_frequency,
    plot_attention_distribution,
    plot_rank_frequency,
)


def create_test_system(
    n_agents: int = 25,
    spatial_shape: tuple = (16, 16),
    K: int = 5,
    kappa_beta: float = 1.0,
    seed: int = 42
):
    """
    Create a MultiAgentSystem for testing attention patterns.

    Uses overlapping circular supports so agents have neighbors.
    """
    cfg = SimulationConfig(
        experiment_name="attention_scaling",
        spatial_shape=spatial_shape,
        n_agents=n_agents,
        K_latent=K,
        kappa_beta=kappa_beta,
        kappa_gamma=kappa_beta,
        seed=seed,

        # Use circular supports with overlap
        support_pattern="circles_2d",
        agent_placement_2d="random",
        agent_radius=max(spatial_shape) / 3,  # Large enough for overlap

        # Energy weights
        lambda_self=1.0,
        lambda_belief_align=1.0,
        lambda_prior_align=0.0,
        lambda_obs=0.0,

        # No emergence for simple test
        enable_emergence=False,

        # Mask settings
        mask_type="gaussian",
        overlap_threshold=0.01,
    )

    rng = np.random.default_rng(seed)
    manifold = build_manifold(cfg)
    supports = build_supports(manifold, cfg, rng)
    agents = build_agents(manifold, supports, cfg, rng)
    system = build_system(agents, cfg, rng)

    return system, cfg


def run_attention_analysis(
    n_agents: int = 25,
    spatial_shape: tuple = (16, 16),
    K: int = 5,
    kappa_beta: float = 1.0,
    seed: int = 42,
    output_dir: str = "_experiments/attention_scaling"
):
    """
    Run complete attention scaling analysis.
    """
    print("\n" + "="*70)
    print("ATTENTION SCALING ANALYSIS")
    print("="*70)
    print(f"\nSetup:")
    print(f"  Agents: {n_agents}")
    print(f"  Spatial shape: {spatial_shape}")
    print(f"  Belief dimension K: {K}")
    print(f"  Temperature κ: {kappa_beta}")

    # Create system
    print("\nCreating multi-agent system...")
    system, cfg = create_test_system(
        n_agents=n_agents,
        spatial_shape=spatial_shape,
        K=K,
        kappa_beta=kappa_beta,
        seed=seed
    )

    # Check overlaps
    n_overlaps = len(system.overlap_masks)
    print(f"  Active overlaps: {n_overlaps}")

    if n_overlaps == 0:
        print("\nWARNING: No overlapping agents! Cannot compute β_ij.")
        print("Try increasing agent_radius or n_agents.")
        return

    # Extract attention weights
    print("\nExtracting attention weights β_ij...")
    beta_values, weight_dict = extract_attention_weights(system, 'belief', kappa_beta)

    print(f"  Total β_ij values: {len(beta_values)}")
    print(f"  Non-zero values: {np.sum(beta_values > 1e-10)}")
    print(f"  Mean β_ij: {np.mean(beta_values):.6f}")
    print(f"  Max β_ij: {np.max(beta_values):.6f}")

    # Full analysis
    print("\nAnalyzing distribution...")
    analysis = analyze_attention_distribution(system, 'belief', kappa_beta)

    print(f"\nDistribution Statistics:")
    print(f"  Mean: {analysis.mean:.6f}")
    print(f"  Std: {analysis.std:.6f}")
    print(f"  Median: {analysis.median:.6f}")

    if analysis.power_law_fit is not None:
        pl = analysis.power_law_fit
        print(f"\nPower-law fit:")
        print(f"  Exponent α: {pl.exponent:.3f}")
        print(f"  x_min: {pl.x_min:.6f}")
        print(f"  KS statistic: {pl.ks_statistic:.4f}")
        print(f"  p-value: {pl.ks_pvalue:.4f}")

    if analysis.exponential_fit is not None:
        exp = analysis.exponential_fit
        print(f"\nExponential fit:")
        print(f"  Rate λ: {exp.rate:.3f}")
        print(f"  KS statistic: {exp.ks_statistic:.4f}")
        print(f"  p-value: {exp.ks_pvalue:.4f}")

    if analysis.preferred_model is not None:
        print(f"\nPreferred model: {analysis.preferred_model.upper()}")
        print(f"Log-likelihood ratio: {analysis.likelihood_ratio:.2f}")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Generate plots
    print("\nGenerating plots...")
    create_analysis_figure(analysis, system, output_path, kappa_beta)

    # Temperature scan
    print("\nScanning temperature dependence...")
    kappa_values = np.logspace(-1, 1, 9)  # 0.1 to 10
    temp_results = scan_temperature_dependence(system, kappa_values, 'belief')

    create_temperature_figure(temp_results, kappa_values, output_path)

    print(f"\n✓ Results saved to {output_path}")
    print("="*70)

    return analysis, temp_results


def create_analysis_figure(analysis, system, output_path, kappa):
    """Create comprehensive analysis figure."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Panel A: Histogram (log-log)
    plot_attention_distribution(analysis, axes[0, 0], log_scale=True)
    axes[0, 0].set_title('A: Distribution (log-log)')

    # Panel B: Histogram (linear)
    plot_attention_distribution(analysis, axes[0, 1], log_scale=False)
    axes[0, 1].set_title('B: Distribution (linear)')

    # Panel C: Rank-frequency (Zipf)
    plot_rank_frequency(analysis, axes[0, 2])
    axes[0, 2].set_title('C: Rank-Frequency (Zipf plot)')

    # Panel D: CDF
    beta = analysis.beta_values[analysis.beta_values > 1e-10]
    sorted_beta = np.sort(beta)
    cdf = np.arange(1, len(sorted_beta) + 1) / len(sorted_beta)
    axes[1, 0].plot(sorted_beta, cdf, 'b-', lw=2)
    axes[1, 0].set_xlabel('β_ij')
    axes[1, 0].set_ylabel('CDF')
    axes[1, 0].set_title('D: Cumulative Distribution')
    axes[1, 0].grid(True, alpha=0.3)

    # Panel E: CCDF (complementary CDF) - log-log
    ccdf = 1 - cdf
    mask = ccdf > 0
    axes[1, 1].loglog(sorted_beta[mask], ccdf[mask], 'b-', lw=2)
    axes[1, 1].set_xlabel('β_ij')
    axes[1, 1].set_ylabel('P(X > x)')
    axes[1, 1].set_title('E: CCDF (log-log)')
    axes[1, 1].grid(True, alpha=0.3)

    # Panel F: Summary statistics
    axes[1, 2].axis('off')
    summary_text = [
        f"ATTENTION ANALYSIS SUMMARY",
        f"{'='*40}",
        f"",
        f"System: {system.n_agents} agents",
        f"Temperature κ = {kappa:.2f}",
        f"",
        f"STATISTICS:",
        f"  N values: {len(analysis.beta_values):,}",
        f"  Mean: {analysis.mean:.6f}",
        f"  Std: {analysis.std:.6f}",
        f"  Median: {analysis.median:.6f}",
        f"  Max: {analysis.max_val:.6f}",
        f"",
    ]

    if analysis.power_law_fit is not None:
        pl = analysis.power_law_fit
        summary_text.extend([
            f"POWER-LAW FIT:",
            f"  α = {pl.exponent:.3f}",
            f"  x_min = {pl.x_min:.6f}",
            f"  KS p-value = {pl.ks_pvalue:.3f}",
            f"",
        ])

    if analysis.exponential_fit is not None:
        exp = analysis.exponential_fit
        summary_text.extend([
            f"EXPONENTIAL FIT:",
            f"  λ = {exp.rate:.3f}",
            f"  KS p-value = {exp.ks_pvalue:.3f}",
            f"",
        ])

    if analysis.preferred_model is not None:
        summary_text.extend([
            f"PREFERRED: {analysis.preferred_model.upper()}",
            f"LR = {analysis.likelihood_ratio:.2f}",
        ])

    axes[1, 2].text(0.05, 0.95, '\n'.join(summary_text),
                    transform=axes[1, 2].transAxes,
                    fontsize=10, verticalalignment='top',
                    family='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

    fig.suptitle(f'Attention Weight Distribution β_ij (κ={kappa:.2f})',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    # Save
    plt.savefig(output_path / f"attention_distribution_kappa{kappa:.2f}.png",
                dpi=300, bbox_inches='tight')
    plt.savefig(output_path / f"attention_distribution_kappa{kappa:.2f}.pdf",
                bbox_inches='tight')
    plt.close()

    print(f"  ✓ Saved distribution plots")


def create_temperature_figure(results, kappa_values, output_path):
    """Create figure showing temperature dependence."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Extract metrics
    means = [r.mean for r in results]
    stds = [r.std for r in results]
    exponents = [r.power_law_fit.exponent if r.power_law_fit else np.nan for r in results]
    exp_rates = [r.exponential_fit.rate if r.exponential_fit else np.nan for r in results]

    # Panel A: Mean vs κ
    axes[0, 0].semilogx(kappa_values, means, 'bo-', lw=2, markersize=8)
    axes[0, 0].set_xlabel('Temperature κ')
    axes[0, 0].set_ylabel('Mean β_ij')
    axes[0, 0].set_title('A: Mean Attention vs Temperature')
    axes[0, 0].grid(True, alpha=0.3)

    # Panel B: Std vs κ
    axes[0, 1].semilogx(kappa_values, stds, 'ro-', lw=2, markersize=8)
    axes[0, 1].set_xlabel('Temperature κ')
    axes[0, 1].set_ylabel('Std β_ij')
    axes[0, 1].set_title('B: Spread vs Temperature')
    axes[0, 1].grid(True, alpha=0.3)

    # Panel C: Power-law exponent vs κ
    valid_exp = ~np.isnan(exponents)
    if np.any(valid_exp):
        axes[1, 0].semilogx(np.array(kappa_values)[valid_exp],
                           np.array(exponents)[valid_exp],
                           'go-', lw=2, markersize=8)
    axes[1, 0].set_xlabel('Temperature κ')
    axes[1, 0].set_ylabel('Power-law exponent α')
    axes[1, 0].set_title('C: Power-law Exponent vs Temperature')
    axes[1, 0].grid(True, alpha=0.3)

    # Panel D: Exponential rate vs κ
    valid_rate = ~np.isnan(exp_rates)
    if np.any(valid_rate):
        axes[1, 1].loglog(np.array(kappa_values)[valid_rate],
                         np.array(exp_rates)[valid_rate],
                         'mo-', lw=2, markersize=8)
    axes[1, 1].set_xlabel('Temperature κ')
    axes[1, 1].set_ylabel('Exponential rate λ')
    axes[1, 1].set_title('D: Exponential Rate vs Temperature')
    axes[1, 1].grid(True, alpha=0.3)

    fig.suptitle('Temperature Dependence of Attention Distribution',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    plt.savefig(output_path / "temperature_dependence.png",
                dpi=300, bbox_inches='tight')
    plt.savefig(output_path / "temperature_dependence.pdf",
                bbox_inches='tight')
    plt.close()

    print(f"  ✓ Saved temperature dependence plots")


def main():
    parser = argparse.ArgumentParser(description='Analyze attention scaling')
    parser.add_argument('--n_agents', type=int, default=50,
                        help='Number of agents')
    parser.add_argument('--spatial', type=int, nargs=2, default=[15, 15],
                        help='Spatial shape (H W)')
    parser.add_argument('--K', type=int, default=15,
                        help='Belief dimension')
    parser.add_argument('--kappa', type=float, default=1.0,
                        help='Attention temperature')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--output', type=str, default='_experiments/attention_scaling',
                        help='Output directory')

    args = parser.parse_args()

    run_attention_analysis(
        n_agents=args.n_agents,
        spatial_shape=tuple(args.spatial),
        K=args.K,
        kappa_beta=args.kappa,
        seed=args.seed,
        output_dir=args.output
    )


if __name__ == "__main__":
    main()