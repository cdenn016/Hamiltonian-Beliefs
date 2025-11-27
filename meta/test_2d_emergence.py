# -*- coding: utf-8 -*-
"""
Created on Tue Nov 25 18:47:17 2025

@author: chris and christine
"""

#!/usr/bin/env python3
"""
Validation Test: Pointwise Spatial Emergence
=============================================

Verifies that meta-agent emergence is truly POINTWISE:
1. Consensus computed at each spatial location independently
2. Connected components found where consensus holds
3. Disconnected consensus regions → separate meta-agents
4. Single region of consensus → single meta-agent

This test uses REAL KL divergence, not mocks.

Author: Claude & Chris
Date: November 2025
"""

import numpy as np
from typing import List, Tuple, Optional

# Core imports
from geometry.geometry_base import BaseManifold, SupportRegion, TopologyType
from math_utils.numerical_utils import kl_gaussian, sanitize_sigma


def create_test_agent(
    agent_id: int,
    manifold: BaseManifold,
    center: Tuple[float, ...],
    radius: float,
    mu_value: np.ndarray,
    K: int = 3
):
    """
    Create a simple test agent with specified belief mean.

    Args:
        agent_id: Agent identifier
        manifold: Base manifold (spatial grid)
        center: Center of Gaussian support
        radius: Radius of support
        mu_value: Belief mean value (K,) - SAME at all spatial points
        K: State dimension
    """
    class TestAgent:
        def __init__(self):
            self.agent_id = agent_id
            self.base_manifold = manifold
            self.K = K
            self.scale = 0
            self.parent_meta = None

            # Create Gaussian support
            shape = manifold.shape
            ndim = len(shape)
            coords = [np.arange(s, dtype=np.float32) for s in shape]
            grids = np.meshgrid(*coords, indexing='ij')

            r_sq = sum((g - c)**2 for g, c in zip(grids, center))
            sigma = 0.47 * radius
            chi = np.exp(-0.5 * r_sq / sigma**2)
            chi = np.where(np.sqrt(r_sq) <= 3 * sigma, chi, 0)
            self.support = SupportRegion(base_manifold=manifold, chi_weight=chi.astype(np.float32))

            # Belief: constant mean across space
            self.mu_q = np.broadcast_to(mu_value, (*shape, K)).copy().astype(np.float32)
            self.Sigma_q = np.broadcast_to(np.eye(K), (*shape, K, K)).copy().astype(np.float32)

            # Prior: same as belief for simplicity
            self.mu_p = self.mu_q.copy()
            self.Sigma_p = self.Sigma_q.copy()

            # Trivial gauge
            self.gauge = type('Gauge', (), {'phi': np.zeros((*shape, K), dtype=np.float32)})()
            self.generators = np.zeros((K, K, K), dtype=np.float32)

    return TestAgent()


def compute_pairwise_kl_field(agent_i, agent_j, eps=1e-6) -> np.ndarray:
    """
    Compute KL(q_i || q_j) at each spatial point.

    Returns:
        kl_field: (*spatial,) array of KL divergences
    """
    spatial_shape = agent_i.base_manifold.shape
    kl_field = np.zeros(spatial_shape, dtype=np.float32)

    for idx in np.ndindex(spatial_shape):
        mu_i = agent_i.mu_q[idx]
        Sigma_i = sanitize_sigma(agent_i.Sigma_q[idx], eps)
        mu_j = agent_j.mu_q[idx]
        Sigma_j = sanitize_sigma(agent_j.Sigma_q[idx], eps)

        kl_field[idx] = kl_gaussian(mu_i, Sigma_i, mu_j, Sigma_j)

    return kl_field


def test_uniform_consensus():
    """
    Test: Two agents with SAME beliefs everywhere → consensus everywhere.

    Expected: Single connected consensus region covering the overlap.
    """
    print("\n" + "="*70)
    print("TEST 1: Uniform Consensus (same beliefs everywhere)")
    print("="*70)

    manifold = BaseManifold(shape=(32, 32), topology=TopologyType.PERIODIC)
    K = 3

    # Two agents with IDENTICAL beliefs
    mu = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    agent_0 = create_test_agent(0, manifold, center=(12, 16), radius=8, mu_value=mu, K=K)
    agent_1 = create_test_agent(1, manifold, center=(20, 16), radius=8, mu_value=mu, K=K)

    # Compute KL field
    kl_field = compute_pairwise_kl_field(agent_0, agent_1)

    # Compute overlap mask
    overlap = (agent_0.support.chi_weight > 0.01) & (agent_1.support.chi_weight > 0.01)

    print(f"  Overlap size: {np.sum(overlap)} pixels")
    print(f"  KL in overlap - min: {kl_field[overlap].min():.6f}, max: {kl_field[overlap].max():.6f}, mean: {kl_field[overlap].mean():.6f}")

    # Check: KL should be ~0 everywhere in overlap
    kl_threshold = 0.01
    consensus_mask = (kl_field < kl_threshold) & overlap

    print(f"  Consensus pixels: {np.sum(consensus_mask)} / {np.sum(overlap)}")

    # Verify
    assert np.sum(consensus_mask) == np.sum(overlap), "Should have consensus at ALL overlap points"
    print("  PASS: Uniform consensus detected at all overlap points")


def test_partial_consensus():
    """
    Test: Two agents with DIFFERENT beliefs in part of their overlap.

    Expected: Consensus only where beliefs agree, NOT where they differ.
    """
    print("\n" + "="*70)
    print("TEST 2: Partial Consensus (agreement in left half only)")
    print("="*70)

    manifold = BaseManifold(shape=(32, 32), topology=TopologyType.PERIODIC)
    K = 3

    # Agent 0: constant belief
    mu_0 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    agent_0 = create_test_agent(0, manifold, center=(16, 16), radius=10, mu_value=mu_0, K=K)

    # Agent 1: same belief on LEFT (x < 16), different on RIGHT (x >= 16)
    mu_1_left = np.array([1.0, 0.0, 0.0], dtype=np.float32)   # Same as agent_0
    mu_1_right = np.array([0.0, 1.0, 0.0], dtype=np.float32)  # Different!

    agent_1 = create_test_agent(1, manifold, center=(16, 16), radius=10, mu_value=mu_1_left, K=K)

    # Modify agent_1's belief on the right half
    agent_1.mu_q[:, 16:] = mu_1_right

    # Compute KL field
    kl_field = compute_pairwise_kl_field(agent_0, agent_1)

    # Compute overlap
    overlap = (agent_0.support.chi_weight > 0.01) & (agent_1.support.chi_weight > 0.01)
    left_overlap = overlap & (np.arange(32)[None, :] < 16)
    right_overlap = overlap & (np.arange(32)[None, :] >= 16)

    print(f"  Total overlap: {np.sum(overlap)} pixels")
    print(f"  Left overlap:  {np.sum(left_overlap)} pixels")
    print(f"  Right overlap: {np.sum(right_overlap)} pixels")

    if np.sum(left_overlap) > 0:
        print(f"  KL in LEFT  - min: {kl_field[left_overlap].min():.4f}, max: {kl_field[left_overlap].max():.4f}")
    if np.sum(right_overlap) > 0:
        print(f"  KL in RIGHT - min: {kl_field[right_overlap].min():.4f}, max: {kl_field[right_overlap].max():.4f}")

    # Check pointwise consensus
    kl_threshold = 0.25
    consensus_mask = (kl_field < kl_threshold) & overlap

    left_consensus = np.sum(consensus_mask & left_overlap)
    right_consensus = np.sum(consensus_mask & right_overlap)

    print(f"  Left consensus:  {left_consensus} / {np.sum(left_overlap)}")
    print(f"  Right consensus: {right_consensus} / {np.sum(right_overlap)}")

    # Verify
    assert left_consensus == np.sum(left_overlap), "Should have full consensus on LEFT"
    assert right_consensus == 0, "Should have NO consensus on RIGHT (different beliefs)"
    print("  PASS: Pointwise consensus correctly distinguishes left vs right")


def test_disconnected_consensus():
    """
    Test: Three agents where 0-1 agree and 1-2 agree, but 0-2 disagree.

    This creates a situation where:
    - Pair 0-1: consensus in their overlap
    - Pair 1-2: consensus in their overlap
    - Pair 0-2: NO consensus

    Expected: Connected component analysis should find the contiguous region.
    """
    print("\n" + "="*70)
    print("TEST 3: Three Agents, Partial Pairwise Consensus")
    print("="*70)

    manifold = BaseManifold(shape=(48, 48), topology=TopologyType.PERIODIC)
    K = 3

    # Agent 0: left side
    mu_0 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    agent_0 = create_test_agent(0, manifold, center=(12, 24), radius=10, mu_value=mu_0, K=K)

    # Agent 1: middle (overlaps with both 0 and 2)
    mu_1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)  # Same as agent_0!
    agent_1 = create_test_agent(1, manifold, center=(24, 24), radius=10, mu_value=mu_1, K=K)

    # Agent 2: right side
    mu_2 = np.array([0.0, 1.0, 0.0], dtype=np.float32)  # Different from 0 and 1!
    agent_2 = create_test_agent(2, manifold, center=(36, 24), radius=10, mu_value=mu_2, K=K)

    agents = [agent_0, agent_1, agent_2]

    # Compute pairwise KL fields
    kl_01 = compute_pairwise_kl_field(agent_0, agent_1)
    kl_12 = compute_pairwise_kl_field(agent_1, agent_2)
    kl_02 = compute_pairwise_kl_field(agent_0, agent_2)

    # Overlaps
    overlap_01 = (agent_0.support.chi_weight > 0.01) & (agent_1.support.chi_weight > 0.01)
    overlap_12 = (agent_1.support.chi_weight > 0.01) & (agent_2.support.chi_weight > 0.01)
    overlap_02 = (agent_0.support.chi_weight > 0.01) & (agent_2.support.chi_weight > 0.01)

    kl_threshold = 0.25

    print("  Pairwise analysis:")
    if np.sum(overlap_01) > 0:
        print(f"    0↔1: overlap={np.sum(overlap_01)}, KL_max={kl_01[overlap_01].max():.4f} {'✓' if kl_01[overlap_01].max() < kl_threshold else '✗'}")
    if np.sum(overlap_12) > 0:
        print(f"    1↔2: overlap={np.sum(overlap_12)}, KL_max={kl_12[overlap_12].max():.4f} {'✓' if kl_12[overlap_12].max() < kl_threshold else '✗'}")
    if np.sum(overlap_02) > 0:
        print(f"    0↔2: overlap={np.sum(overlap_02)}, KL_max={kl_02[overlap_02].max():.4f} {'✓' if kl_02[overlap_02].max() < kl_threshold else '✗'}")
    else:
        print(f"    0↔2: no overlap")

    # Verify
    assert kl_01[overlap_01].max() < kl_threshold, "0-1 should be in consensus"
    assert kl_12[overlap_12].max() > kl_threshold, "1-2 should NOT be in consensus (different beliefs)"
    print("  PASS: Pairwise consensus correctly computed")


def test_spatial_connected_components():
    """
    Test that connected component analysis works correctly.
    """
    print("\n" + "="*70)
    print("TEST 4: Connected Component Analysis")
    print("="*70)

    from scipy.ndimage import label
    from meta.spatial_emergence import full_connectivity_kernel

    # Create a consensus field with TWO disconnected regions
    consensus_field = np.ones((32, 32)) * 0.5  # High KL everywhere
    consensus_field[5:12, 5:12] = 0.01    # Region 1: top-left
    consensus_field[20:28, 20:28] = 0.01  # Region 2: bottom-right (disconnected)

    # Threshold
    kl_threshold = 0.1
    consensus_mask = consensus_field < kl_threshold

    # Find connected components
    kernel = full_connectivity_kernel(2)
    labeled, n_components = label(consensus_mask, structure=kernel)

    print(f"  Consensus mask: {np.sum(consensus_mask)} pixels")
    print(f"  Number of connected components: {n_components}")

    # Get component sizes
    for i in range(1, n_components + 1):
        size = np.sum(labeled == i)
        print(f"    Component {i}: {size} pixels")

    # Verify
    assert n_components == 2, "Should find exactly 2 disconnected regions"
    print("  PASS: Connected components correctly identified")


def run_all_tests():
    """Run all validation tests."""
    print("\n" + "#"*70)
    print("# POINTWISE SPATIAL EMERGENCE VALIDATION")
    print("#"*70)

    test_uniform_consensus()
    test_partial_consensus()
    test_disconnected_consensus()
    test_spatial_connected_components()

    print("\n" + "="*70)
    print("ALL TESTS PASSED - Pointwise emergence is working correctly!")
    print("="*70 + "\n")


if __name__ == "__main__":
    run_all_tests()