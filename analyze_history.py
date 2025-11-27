#!/usr/bin/env python3
"""Analyze hierarchical history to understand detector behavior."""

import numpy as np
from pathlib import Path

# Load history
hist_path = Path("_results/_playground/hierarchical_history.npz")
data = np.load(hist_path, allow_pickle=True)

print("\n" + "="*70)
print("HIERARCHICAL HISTORY ANALYSIS")
print("="*70 + "\n")

print("Available keys:", list(data.keys()))
print()

# Show key metrics over time
print("Step-by-step breakdown:")
print("-" * 70)
print(f"{'Step':<6} {'Energy':<12} {'#Scales':<10} {'#Active':<10} {'#Cond':<10}")
print("-" * 70)

steps = data['step']
energies = data['total_energy']
n_scales = data['n_scales']
n_active = data['n_active_agents']
n_cond = data['n_condensations']

for i in range(len(steps)):
    step = steps[i]
    energy = energies[i]
    scales = n_scales[i]
    active = n_active[i]
    cond = n_cond[i]

    cond_str = f"{cond}" if cond > 0 else "-"
    marker = " ← EMERGENCE!" if cond > 0 else ""

    print(f"{step:<6} {energy:<12.4f} {scales:<10} {active:<10} {cond_str:<10}{marker}")

print("-" * 70)
print()

# Find when condensations happened
emergence_steps = [int(steps[i]) for i in range(len(steps)) if n_cond[i] > 0]
print(f"Emergence events at steps: {emergence_steps}")
print()

# Check if detector ran at steps 10, 15, 20, etc. (with interval=5)
expected_checks = [s for s in range(0, max(steps)+1, 5)]
print(f"Expected detector checks (interval=5): {expected_checks}")
print(f"Actual emergences: {emergence_steps}")
print()

# Missing checks where no emergence happened
no_emergence_checks = [s for s in expected_checks if s not in emergence_steps and s <= max(steps)]
print(f"Steps where detector ran but found nothing: {no_emergence_checks}")
print()

# Show energy trend
print("Energy analysis:")
print(f"  Initial energy: {energies[0]:.4f}")
print(f"  Final energy:   {energies[-1]:.4f}")
print(f"  Change:         {energies[-1] - energies[0]:+.4f}")

# Check if energy increases after step 5 (would indicate divergence)
if len(energies) > 10:
    energy_at_5 = energies[5]
    energy_at_10 = energies[10] if len(energies) > 10 else energies[-1]
    if energy_at_10 > energy_at_5:
        print(f"\n  ⚠️  Energy increased from step 5 to 10:")
        print(f"      Step  5: {energy_at_5:.4f}")
        print(f"      Step 10: {energy_at_10:.4f}")
        print(f"      Change: +{energy_at_10 - energy_at_5:.4f} (DIVERGING!)")

print()
print("="*70)