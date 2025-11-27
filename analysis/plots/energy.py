"""
Energy Component Analysis Plots
================================

Plotting functions for visualizing free energy components over time.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any


def plot_energy_components(history: Dict[str, Any], save_path: Path):
    """
    Plot energy components in separate subplots for clarity.

    Args:
        history: Dictionary with energy time series
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Energy Component Evolution', fontsize=16, fontweight='bold')

    steps = history.get('step', [])
    if not steps:
        print("No step data in history")
        return

    # Total energy
    ax = axes[0, 0]
    if 'total' in history:
        ax.plot(steps, history['total'], 'k-', linewidth=2)
        ax.set_title('Total Free Energy')
        ax.set_ylabel('F_total')
        ax.grid(True, alpha=0.3)

    # Self energy
    ax = axes[0, 1]
    if 'self' in history:
        ax.plot(steps, history['self'], 'b-', linewidth=2)
        ax.set_title('Self Energy')
        ax.set_ylabel('α·F_self')
        ax.grid(True, alpha=0.3)

    # Belief alignment
    ax = axes[0, 2]
    if 'belief_align' in history:
        ax.plot(steps, history['belief_align'], 'r-', linewidth=2)
        ax.set_title('Belief Alignment')
        ax.set_ylabel('β·F_belief')
        ax.grid(True, alpha=0.3)

    # Prior alignment
    ax = axes[1, 0]
    if 'prior_align' in history:
        ax.plot(steps, history['prior_align'], 'g-', linewidth=2)
        ax.set_title('Prior Alignment')
        ax.set_ylabel('F_prior')
        ax.set_xlabel('Step')
        ax.grid(True, alpha=0.3)

    # Observations
    ax = axes[1, 1]
    if 'observations' in history:
        ax.plot(steps, history['observations'], 'm-', linewidth=2)
        ax.set_title('Observations')
        ax.set_ylabel('F_obs')
        ax.set_xlabel('Step')
        ax.grid(True, alpha=0.3)

    # Combined view
    ax = axes[1, 2]
    if 'total' in history:
        ax.plot(steps, history['total'], 'k-', linewidth=2, label='Total', alpha=0.8)
    if 'self' in history:
        ax.plot(steps, history['self'], 'b--', linewidth=1.5, label='Self', alpha=0.6)
    if 'belief_align' in history:
        ax.plot(steps, history['belief_align'], 'r--', linewidth=1.5, label='Belief', alpha=0.6)
    ax.set_title('Combined View')
    ax.set_ylabel('Energy')
    ax.set_xlabel('Step')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f"  ✓ Saved energy plot: {save_path}")