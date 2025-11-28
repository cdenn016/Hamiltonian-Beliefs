#!/usr/bin/env python3
"""
Hamiltonian vs Gradient VFE Comparison
=======================================

Systematic comparison of Hamiltonian dynamics vs gradient flow for
Variational Free Energy minimization.

Key experiments:
1. Local Minima Escape: Can Hamiltonian dynamics escape traps?
2. Convergence Speed: Which reaches consensus faster?
3. Memory Persistence: Do limit cycles encode stable memory?
4. Phase Transition: How does friction affect dynamics?

Usage:
    python experiments/hamiltonian_vs_gradient.py

Author: Research comparison suite
Date: November 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import pickle
import time
from copy import deepcopy

# Local imports
from agent.agents import Agent, AgentConfig
from agent.system import MultiAgentSystem
from agent.trainer import Trainer
from agent.hamiltonian_trainer import HamiltonianTrainer
from config import TrainingConfig
from geometry.geometry_base import BaseManifold, TopologyType
from gradients.free_energy_clean import compute_total_free_energy


# =============================================================================
# Experiment Configuration
# =============================================================================

@dataclass
class ExperimentConfig:
    """Configuration for comparison experiments."""

    # System parameters
    n_agents: int = 4
    K: int = 3  # Latent dimension (must be odd for SO(3))
    spatial_size: int = 50  # 1D spatial points

    # Training parameters
    n_steps: int = 500
    dt: float = 0.01  # For Hamiltonian

    # Hamiltonian parameters to sweep
    friction_values: List[float] = field(default_factory=lambda: [0.0, 0.01, 0.1, 0.5, 1.0])
    mass_scale: float = 1.0

    # Gradient trainer learning rates
    lr_mu: float = 0.01
    lr_sigma: float = 0.001
    lr_phi: float = 0.01

    # Output
    output_dir: Path = field(default_factory=lambda: Path("./comparison_results"))
    seed: int = 42
    n_trials: int = 5  # Repeat experiments for statistics


@dataclass
class ExperimentResult:
    """Container for experiment results."""

    method: str  # "gradient" or "hamiltonian_γ=X"
    trial: int

    # Trajectories
    energy_trajectory: List[float] = field(default_factory=list)
    time_trajectory: List[float] = field(default_factory=list)

    # Final metrics
    final_energy: float = 0.0
    convergence_step: Optional[int] = None  # Step where energy stabilized

    # For Hamiltonian only
    energy_conservation: Optional[float] = None
    has_oscillations: bool = False
    oscillation_period: Optional[float] = None

    # Phase space data (for visualization)
    mu_trajectory: Optional[np.ndarray] = None
    momentum_trajectory: Optional[np.ndarray] = None


# =============================================================================
# System Creation
# =============================================================================

def create_test_system(config: ExperimentConfig, rng: np.random.Generator) -> MultiAgentSystem:
    """
    Create a multi-agent system for testing.

    Uses overlapping support regions to create interesting dynamics.
    """
    from agent.masking import MaskConfig, SupportRegionSmooth
    from math_utils.sigma import CovarianceFieldInitializer

    # Create 1D base manifold
    manifold = BaseManifold(
        shape=(config.spatial_size,),
        topology=TopologyType.PERIODIC,
        spacing=1.0
    )

    agents = []
    for i in range(config.n_agents):
        # Create overlapping support regions
        # Agent i is centered at position i * spatial_size / n_agents
        center = int((i + 0.5) * config.spatial_size / config.n_agents)

        mask_cfg = MaskConfig(
            min_mask_for_normal_cov=0.1,
        )

        # Create smooth support mask centered at 'center'
        x = np.arange(config.spatial_size)
        distances = np.minimum(
            np.abs(x - center),
            config.spatial_size - np.abs(x - center)  # Periodic
        )
        width = config.spatial_size / (2 * config.n_agents)
        mask_continuous = np.exp(-0.5 * (distances / width) ** 2).astype(np.float32)
        mask_binary = mask_continuous > 0.1  # Threshold for binary mask

        support = SupportRegionSmooth(
            mask_binary=mask_binary,
            base_shape=(config.spatial_size,),
            config=mask_cfg,
            mask_continuous=mask_continuous,
        )

        # Agent config
        agent_cfg = AgentConfig(
            K=config.K,
            agent_id=f"agent_{i}",
            n_spatial=config.spatial_size,
        )

        # Initialize beliefs with some randomness
        mu_q = 0.5 * rng.standard_normal((config.spatial_size, config.K)).astype(np.float32)
        mu_p = np.zeros((config.spatial_size, config.K), dtype=np.float32)

        # Covariance initialization
        cov_init = CovarianceFieldInitializer(strategy="constant")
        Sigma_q = cov_init.generate((config.spatial_size,), config.K, scale=1.0, rng=rng)
        Sigma_p = cov_init.generate((config.spatial_size,), config.K, scale=2.0, rng=rng)

        agent = Agent(
            config=agent_cfg,
            base_manifold=manifold,
            support=support,
            mu_q=mu_q,
            Sigma_q=Sigma_q,
            mu_p=mu_p,
            Sigma_p=Sigma_p,
        )
        agents.append(agent)

    # Create system
    from simulation_config import SimulationConfig
    sys_config = SimulationConfig(
        n_agents=config.n_agents,
        K=config.K,
    )

    system = MultiAgentSystem(agents, config=sys_config)
    return system


def clone_system(system: MultiAgentSystem) -> MultiAgentSystem:
    """Deep copy a system for fair comparison."""
    return deepcopy(system)


# =============================================================================
# Experiment Runners
# =============================================================================

def run_gradient_experiment(
    system: MultiAgentSystem,
    config: ExperimentConfig,
    trial: int
) -> ExperimentResult:
    """Run gradient-based VFE minimization."""

    result = ExperimentResult(method="gradient", trial=trial)

    # Configure trainer
    train_cfg = TrainingConfig(
        n_steps=config.n_steps,
        lr_mu_q=config.lr_mu,
        lr_sigma_q=config.lr_sigma,
        lr_phi=config.lr_phi,
        log_every=config.n_steps + 1,  # Suppress logging
        save_history=True,
    )

    trainer = Trainer(system, config=train_cfg)

    # Record initial energy
    initial_energy = compute_total_free_energy(system).total
    result.energy_trajectory.append(initial_energy)
    result.time_trajectory.append(0.0)

    # Training loop with energy recording
    start_time = time.perf_counter()
    for step in range(config.n_steps):
        energies = trainer.step()
        result.energy_trajectory.append(energies.total)
        result.time_trajectory.append(time.perf_counter() - start_time)

    result.final_energy = result.energy_trajectory[-1]
    result.convergence_step = _find_convergence_step(result.energy_trajectory)

    return result


def run_hamiltonian_experiment(
    system: MultiAgentSystem,
    config: ExperimentConfig,
    friction: float,
    trial: int
) -> ExperimentResult:
    """Run Hamiltonian VFE dynamics."""

    result = ExperimentResult(method=f"hamiltonian_γ={friction}", trial=trial)

    # Configure trainer
    train_cfg = TrainingConfig(
        n_steps=config.n_steps,
        log_every=config.n_steps + 1,  # Suppress logging
        save_history=True,
    )

    trainer = HamiltonianTrainer(
        system,
        config=train_cfg,
        friction=friction,
        mass_scale=config.mass_scale,
        track_phase_space=True,
    )

    # Record initial energy
    initial_H = trainer.history.total_hamiltonian[0] if trainer.history.total_hamiltonian else 0
    result.energy_trajectory.append(initial_H)
    result.time_trajectory.append(0.0)

    # Training loop
    start_time = time.perf_counter()
    for step in range(config.n_steps):
        trainer.step(dt=config.dt)
        if trainer.history.total_hamiltonian:
            result.energy_trajectory.append(trainer.history.total_hamiltonian[-1])
        result.time_trajectory.append(time.perf_counter() - start_time)

    result.final_energy = result.energy_trajectory[-1]
    result.convergence_step = _find_convergence_step(result.energy_trajectory)

    # Hamiltonian-specific metrics
    if len(trainer.history.total_hamiltonian) > 1:
        H0 = trainer.history.total_hamiltonian[0]
        H_final = trainer.history.total_hamiltonian[-1]
        result.energy_conservation = abs(H_final - H0) / (abs(H0) + 1e-8)

    # Check for oscillations
    result.has_oscillations, result.oscillation_period = _detect_oscillations(
        result.energy_trajectory
    )

    # Store phase space trajectory (first agent's mu)
    if trainer.phase_space_tracker and trainer.phase_space_tracker.snapshots:
        snapshots = trainer.phase_space_tracker.snapshots
        mu_traj = [s['agents'][0]['mu_center'] for s in snapshots if 'agents' in s]
        if mu_traj:
            result.mu_trajectory = np.array(mu_traj)

    return result


def _find_convergence_step(energy_trajectory: List[float], threshold: float = 1e-4) -> Optional[int]:
    """Find step where energy change falls below threshold."""
    if len(energy_trajectory) < 10:
        return None

    energies = np.array(energy_trajectory)
    for i in range(10, len(energies)):
        recent_change = np.std(energies[i-10:i]) / (np.abs(np.mean(energies[i-10:i])) + 1e-8)
        if recent_change < threshold:
            return i
    return None


def _detect_oscillations(energy_trajectory: List[float], min_peaks: int = 3) -> Tuple[bool, Optional[float]]:
    """Detect if energy trajectory has oscillations."""
    if len(energy_trajectory) < 20:
        return False, None

    energies = np.array(energy_trajectory)

    # Remove trend
    from scipy.signal import detrend
    try:
        detrended = detrend(energies)
    except:
        detrended = energies - np.mean(energies)

    # Find peaks
    from scipy.signal import find_peaks
    peaks, _ = find_peaks(detrended)

    if len(peaks) >= min_peaks:
        # Estimate period from peak spacing
        peak_diffs = np.diff(peaks)
        period = np.mean(peak_diffs) if len(peak_diffs) > 0 else None
        return True, period

    return False, None


# =============================================================================
# Main Comparison Suite
# =============================================================================

def run_comparison_experiment(config: ExperimentConfig) -> Dict[str, List[ExperimentResult]]:
    """
    Run full comparison: gradient vs Hamiltonian at various friction values.
    """
    config.output_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "gradient": [],
    }
    for gamma in config.friction_values:
        results[f"hamiltonian_γ={gamma}"] = []

    print("=" * 70)
    print("HAMILTONIAN vs GRADIENT VFE COMPARISON")
    print("=" * 70)
    print(f"Agents: {config.n_agents}, K: {config.K}, Steps: {config.n_steps}")
    print(f"Friction values: {config.friction_values}")
    print(f"Trials per condition: {config.n_trials}")
    print("=" * 70)

    for trial in range(config.n_trials):
        print(f"\n--- Trial {trial + 1}/{config.n_trials} ---")

        # Create fresh RNG for reproducibility
        rng = np.random.default_rng(config.seed + trial)

        # Create base system
        base_system = create_test_system(config, rng)

        # Run gradient experiment
        print("  Running gradient VFE...")
        system_copy = clone_system(base_system)
        grad_result = run_gradient_experiment(system_copy, config, trial)
        results["gradient"].append(grad_result)
        print(f"    Final energy: {grad_result.final_energy:.4f}")

        # Run Hamiltonian experiments at each friction
        for gamma in config.friction_values:
            print(f"  Running Hamiltonian (γ={gamma})...")
            system_copy = clone_system(base_system)
            ham_result = run_hamiltonian_experiment(system_copy, config, gamma, trial)
            results[f"hamiltonian_γ={gamma}"].append(ham_result)
            print(f"    Final energy: {ham_result.final_energy:.4f}, "
                  f"Conservation: {ham_result.energy_conservation:.2e}, "
                  f"Oscillations: {ham_result.has_oscillations}")

    return results


def plot_comparison_results(results: Dict[str, List[ExperimentResult]], config: ExperimentConfig):
    """Generate comparison figures."""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Color scheme
    colors = plt.cm.viridis(np.linspace(0, 1, len(results)))
    method_colors = {method: colors[i] for i, method in enumerate(results.keys())}
    method_colors["gradient"] = "red"  # Make gradient stand out

    # ===== Plot 1: Energy Trajectories =====
    ax1 = axes[0, 0]
    for method, result_list in results.items():
        # Average over trials
        all_energies = [r.energy_trajectory for r in result_list]
        min_len = min(len(e) for e in all_energies)
        energies_array = np.array([e[:min_len] for e in all_energies])

        mean_energy = np.mean(energies_array, axis=0)
        std_energy = np.std(energies_array, axis=0)

        steps = np.arange(len(mean_energy))
        label = "Gradient" if method == "gradient" else method.replace("hamiltonian_", "H-VFE ")

        ax1.plot(steps, mean_energy, label=label, color=method_colors[method], linewidth=2)
        ax1.fill_between(steps, mean_energy - std_energy, mean_energy + std_energy,
                        alpha=0.2, color=method_colors[method])

    ax1.set_xlabel("Step")
    ax1.set_ylabel("Energy")
    ax1.set_title("Energy Trajectories")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # ===== Plot 2: Final Energy Comparison =====
    ax2 = axes[0, 1]
    methods = list(results.keys())
    final_energies = [np.mean([r.final_energy for r in results[m]]) for m in methods]
    final_stds = [np.std([r.final_energy for r in results[m]]) for m in methods]

    x_pos = np.arange(len(methods))
    bars = ax2.bar(x_pos, final_energies, yerr=final_stds, capsize=5,
                   color=[method_colors[m] for m in methods])

    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([m.replace("hamiltonian_", "H-").replace("gradient", "Grad")
                        for m in methods], rotation=45, ha='right', fontsize=8)
    ax2.set_ylabel("Final Energy")
    ax2.set_title("Final Energy Comparison")
    ax2.grid(True, alpha=0.3, axis='y')

    # ===== Plot 3: Convergence Speed =====
    ax3 = axes[1, 0]

    # Box plot of convergence steps
    conv_data = []
    conv_labels = []
    for method in methods:
        conv_steps = [r.convergence_step for r in results[method] if r.convergence_step is not None]
        if conv_steps:
            conv_data.append(conv_steps)
            conv_labels.append(method.replace("hamiltonian_", "H-").replace("gradient", "Grad"))

    if conv_data:
        bp = ax3.boxplot(conv_data, labels=conv_labels, patch_artist=True)
        for i, patch in enumerate(bp['boxes']):
            patch.set_facecolor(method_colors[methods[i]])
            patch.set_alpha(0.7)
        ax3.set_ylabel("Convergence Step")
        ax3.set_title("Convergence Speed (lower = faster)")
        ax3.tick_params(axis='x', rotation=45)

    # ===== Plot 4: Oscillation Detection =====
    ax4 = axes[1, 1]

    # For Hamiltonian methods, show oscillation prevalence
    ham_methods = [m for m in methods if "hamiltonian" in m]
    if ham_methods:
        osc_rates = []
        osc_labels = []
        for method in ham_methods:
            osc_count = sum(1 for r in results[method] if r.has_oscillations)
            osc_rate = osc_count / len(results[method])
            osc_rates.append(osc_rate * 100)
            osc_labels.append(method.replace("hamiltonian_", ""))

        x_pos = np.arange(len(ham_methods))
        ax4.bar(x_pos, osc_rates, color=[method_colors[m] for m in ham_methods])
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels(osc_labels, fontsize=9)
        ax4.set_ylabel("% Trials with Oscillations")
        ax4.set_title("Oscillation Prevalence (Memory Indicator)")
        ax4.set_ylim(0, 100)
        ax4.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    # Save figure
    fig_path = config.output_dir / "comparison_results.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved comparison figure: {fig_path}")

    plt.show()


def generate_summary_table(results: Dict[str, List[ExperimentResult]]) -> str:
    """Generate markdown summary table."""

    lines = [
        "| Method | Final Energy | Convergence Step | Oscillations | Energy Conservation |",
        "|--------|-------------|------------------|--------------|---------------------|",
    ]

    for method, result_list in results.items():
        final_mean = np.mean([r.final_energy for r in result_list])
        final_std = np.std([r.final_energy for r in result_list])

        conv_steps = [r.convergence_step for r in result_list if r.convergence_step]
        conv_str = f"{np.mean(conv_steps):.0f}±{np.std(conv_steps):.0f}" if conv_steps else "N/A"

        if "hamiltonian" in method:
            osc_rate = sum(1 for r in result_list if r.has_oscillations) / len(result_list) * 100
            osc_str = f"{osc_rate:.0f}%"

            conservations = [r.energy_conservation for r in result_list if r.energy_conservation is not None]
            cons_str = f"{np.mean(conservations):.2e}" if conservations else "N/A"
        else:
            osc_str = "N/A"
            cons_str = "N/A"

        method_label = method.replace("hamiltonian_", "H-VFE ").replace("gradient", "Gradient")
        lines.append(f"| {method_label} | {final_mean:.3f}±{final_std:.3f} | {conv_str} | {osc_str} | {cons_str} |")

    return "\n".join(lines)


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    """Run comparison experiments."""

    config = ExperimentConfig(
        n_agents=4,
        K=3,
        spatial_size=50,
        n_steps=500,
        dt=0.01,
        friction_values=[0.0, 0.01, 0.1, 0.5],
        n_trials=3,
        seed=42,
        output_dir=Path("./comparison_results"),
    )

    # Run experiments
    results = run_comparison_experiment(config)

    # Save raw results
    results_path = config.output_dir / "results.pkl"
    with open(results_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"\n✓ Saved raw results: {results_path}")

    # Generate plots
    plot_comparison_results(results, config)

    # Generate summary table
    summary = generate_summary_table(results)
    print("\n" + "=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)
    print(summary)

    # Save summary
    summary_path = config.output_dir / "summary.md"
    with open(summary_path, 'w') as f:
        f.write("# Hamiltonian vs Gradient VFE Comparison\n\n")
        f.write(summary)
    print(f"\n✓ Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
