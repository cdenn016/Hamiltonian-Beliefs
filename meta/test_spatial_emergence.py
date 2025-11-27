# -*- coding: utf-8 -*-
"""
Created on Tue Nov 25 17:16:41 2025

@author: chris and christine
"""

#!/usr/bin/env python3
"""
Tests for Spatial Emergence: Meta-Agent Formation in Arbitrary Dimensions
==========================================================================

Tests the spatial emergence detection system for:
- 1D manifolds (intervals, rings)
- 2D manifolds (grids, toruses)
- 3D manifolds (volumes)
- Arbitrary nD manifolds

Key properties verified:
1. Connected consensus regions form single meta-agents
2. Disconnected regions form separate meta-agents
3. Smooth boundary transitions (differentiable support)
4. Scale-free hierarchy (works at any level)
5. Coherence-weighted support computation

Author: Claude & Chris
Date: November 2025
"""

import numpy as np
import pytest
from typing import List, Tuple

# Import spatial emergence components
from meta.spatial_emergence import (
    SpatialConsensusRegion,
    SpatialEmergenceDetector,
    full_connectivity_kernel,
    axis_connectivity_kernel,
    compute_cluster_consensus_field,
    find_consensus_components,
    smooth_consensus_boundary,
    create_smooth_consensus_support,
    summarize_emergence_detection
)

from geometry.geometry_base import BaseManifold, SupportRegion, TopologyType


# =============================================================================
# Test Fixtures
# =============================================================================

class MockAgent:
    """Mock agent for testing without full agent infrastructure."""

    def __init__(self,
                 agent_id: int,
                 base_manifold: BaseManifold,
                 center: Tuple[float, ...] = None,
                 radius: float = 5.0,
                 K: int = 3):
        self.agent_id = agent_id
        self.base_manifold = base_manifold
        self.K = K

        # Create support
        if base_manifold.is_point:
            chi = np.array(1.0, dtype=np.float32)
        else:
            chi = self._create_gaussian_chi(center, radius)

        self.support = SupportRegion(base_manifold=base_manifold, chi_weight=chi)

        # Create mock belief/prior fields
        spatial_shape = base_manifold.shape if not base_manifold.is_point else ()
        self.mu_q = np.random.randn(*spatial_shape, K).astype(np.float32)
        self.Sigma_q = np.eye(K, dtype=np.float32)
        if not base_manifold.is_point:
            self.Sigma_q = np.broadcast_to(self.Sigma_q, (*spatial_shape, K, K)).copy()

        self.mu_p = self.mu_q.copy()
        self.Sigma_p = self.Sigma_q.copy()

        # Mock gauge
        self.gauge = MockGauge(base_manifold, K)
        self.generators = np.zeros((K, K, K), dtype=np.float32)  # Trivial generators

    def _create_gaussian_chi(self, center, radius):
        """Create Gaussian support centered at given location."""
        shape = self.base_manifold.shape
        ndim = len(shape)

        if center is None:
            center = tuple(s // 2 for s in shape)

        coords = [np.arange(s, dtype=np.float32) for s in shape]
        grids = np.meshgrid(*coords, indexing='ij')

        r_sq = sum((g - c)**2 for g, c in zip(grids, center))
        sigma = 0.47 * radius

        chi = np.exp(-0.5 * r_sq / sigma**2)
        chi = np.where(np.sqrt(r_sq) <= 3 * sigma, chi, 0)

        return chi.astype(np.float32)


class MockGauge:
    """Mock gauge field."""
    def __init__(self, base_manifold, K):
        if base_manifold.is_point:
            self.phi = np.zeros(K, dtype=np.float32)
        else:
            self.phi = np.zeros((*base_manifold.shape, K), dtype=np.float32)


class MockConsensusDetector:
    """Mock consensus detector for testing."""

    def __init__(self, kl_threshold: float = 0.1):
        self.belief_threshold = kl_threshold
        self.model_threshold = kl_threshold

    def find_consensus_clusters(self, system) -> List[List[int]]:
        """Return predefined clusters for testing."""
        n = len(system.agents)
        if n <= 2:
            return [[i for i in range(n)]]
        # Group agents by proximity of their agent_ids
        return [[0, 1], [2, 3]] if n >= 4 else [[i for i in range(n)]]

    def _get_transport(self, agent_i, agent_j):
        """Return identity transport (no gauge rotation)."""
        K = agent_i.K
        if agent_i.base_manifold.is_point:
            return np.eye(K, dtype=np.float32)
        else:
            shape = agent_i.base_manifold.shape
            return np.broadcast_to(np.eye(K), (*shape, K, K)).copy().astype(np.float32)

    def check_belief_consensus_spatial(self, agent_i, agent_j, omega_ij=None):
        """Return mock consensus based on support overlap."""
        if agent_i.base_manifold.is_point:
            return np.array(True), np.array(0.01)

        # Consensus where supports overlap
        chi_overlap = agent_i.support.chi_weight * agent_j.support.chi_weight
        consensus_mask = chi_overlap > 0.1
        kl_field = np.where(consensus_mask, 0.01, 1.0).astype(np.float32)

        return consensus_mask, kl_field

    def check_model_consensus_spatial(self, agent_i, agent_j, omega_ij=None):
        """Same as belief consensus for mock."""
        return self.check_belief_consensus_spatial(agent_i, agent_j, omega_ij)

    def check_full_consensus(self, agent_i, agent_j):
        """Return mock consensus state."""
        class MockState:
            belief_divergence = 0.01
            model_divergence = 0.01
            is_epistemically_dead = True
        return MockState()


# =============================================================================
# Tests: Connectivity Kernels
# =============================================================================

class TestConnectivityKernels:
    """Test dimension-agnostic connectivity kernels."""

    def test_full_connectivity_1d(self):
        """1D full connectivity: [1, 1, 1]."""
        kernel = full_connectivity_kernel(1)
        assert kernel.shape == (3,)
        assert np.all(kernel == 1)

    def test_full_connectivity_2d(self):
        """2D full connectivity: 3x3 of ones (8-connectivity)."""
        kernel = full_connectivity_kernel(2)
        assert kernel.shape == (3, 3)
        assert np.sum(kernel) == 9  # All ones

    def test_full_connectivity_3d(self):
        """3D full connectivity: 3x3x3 cube (26-connectivity)."""
        kernel = full_connectivity_kernel(3)
        assert kernel.shape == (3, 3, 3)
        assert np.sum(kernel) == 27

    def test_full_connectivity_nd(self):
        """nD full connectivity: 3^n hypercube."""
        for ndim in range(1, 5):
            kernel = full_connectivity_kernel(ndim)
            assert kernel.shape == (3,) * ndim
            assert np.sum(kernel) == 3**ndim

    def test_axis_connectivity_2d(self):
        """2D axis connectivity: cross pattern (4-connectivity)."""
        kernel = axis_connectivity_kernel(2)
        assert kernel.shape == (3, 3)
        # Cross pattern: center + 4 neighbors
        assert kernel[1, 1] == 1  # Center
        assert kernel[0, 1] == 1  # Top
        assert kernel[2, 1] == 1  # Bottom
        assert kernel[1, 0] == 1  # Left
        assert kernel[1, 2] == 1  # Right
        # Corners should be 0
        assert kernel[0, 0] == 0
        assert kernel[0, 2] == 0
        assert kernel[2, 0] == 0
        assert kernel[2, 2] == 0


# =============================================================================
# Tests: Smooth Boundary Processing
# =============================================================================

class TestSmoothBoundary:
    """Test smooth boundary processing for smooth sections."""

    def test_smooth_preserves_shape(self):
        """Smoothing should preserve array shape."""
        mask = np.zeros((32, 32), dtype=bool)
        mask[10:20, 10:20] = True

        smoothed = smooth_consensus_boundary(mask, sigma=1.5)
        assert smoothed.shape == mask.shape

    def test_smooth_range(self):
        """Smoothed values should be in [0, 1]."""
        mask = np.zeros((32, 32), dtype=bool)
        mask[10:20, 10:20] = True

        smoothed = smooth_consensus_boundary(mask, sigma=1.5)
        assert np.all(smoothed >= 0)
        assert np.all(smoothed <= 1)

    def test_smooth_interior_preserved(self):
        """Interior of large regions should remain ~1."""
        mask = np.zeros((64, 64), dtype=bool)
        mask[20:44, 20:44] = True  # Large 24x24 region

        smoothed = smooth_consensus_boundary(mask, sigma=1.5)
        # Deep interior should be ~1
        assert smoothed[32, 32] > 0.95

    def test_smooth_exterior_preserved(self):
        """Exterior far from region should remain ~0."""
        mask = np.zeros((64, 64), dtype=bool)
        mask[30:34, 30:34] = True  # Small region

        smoothed = smooth_consensus_boundary(mask, sigma=1.5)
        # Far from region should be ~0
        assert smoothed[5, 5] < 0.05

    def test_smooth_gradient_at_boundary(self):
        """Boundary should have smooth gradient, not step."""
        mask = np.zeros((64, 64), dtype=bool)
        mask[20:44, 20:44] = True

        smoothed = smooth_consensus_boundary(mask, sigma=2.0)

        # Check that values transition smoothly across boundary
        boundary_line = smoothed[32, 15:50]  # Horizontal line across boundary

        # Should not have any jumps > 0.3 between adjacent pixels
        diffs = np.abs(np.diff(boundary_line))
        assert np.all(diffs < 0.3), "Boundary should be smooth, not step"

    def test_0d_passthrough(self):
        """0D should pass through unchanged."""
        mask = np.array(True)
        smoothed = smooth_consensus_boundary(mask, sigma=1.5)
        assert smoothed.ndim == 0
        assert float(smoothed) == 1.0


# =============================================================================
# Tests: Connected Component Analysis
# =============================================================================

class TestConnectedComponents:
    """Test connected component finding."""

    def test_single_region(self):
        """Single connected region should produce one component."""
        consensus_field = np.ones((32, 32)) * 0.5  # Above threshold everywhere
        consensus_field[10:22, 10:22] = 0.01  # Low KL in consensus region
        active_mask = np.ones((32, 32), dtype=bool)

        components = find_consensus_components(
            consensus_field, active_mask,
            kl_threshold=0.1,
            min_region_size=4
        )

        assert len(components) == 1
        assert components[0]['size'] == 12 * 12

    def test_two_disconnected_regions(self):
        """Two disconnected regions should produce two components."""
        consensus_field = np.ones((32, 32)) * 0.5
        consensus_field[5:10, 5:10] = 0.01    # Region 1
        consensus_field[20:25, 20:25] = 0.01  # Region 2 (disconnected)
        active_mask = np.ones((32, 32), dtype=bool)

        components = find_consensus_components(
            consensus_field, active_mask,
            kl_threshold=0.1,
            min_region_size=4
        )

        assert len(components) == 2

    def test_diagonal_connectivity(self):
        """Full connectivity should connect diagonal neighbors."""
        consensus_field = np.ones((32, 32)) * 0.5
        # Two squares connected only diagonally
        consensus_field[10:15, 10:15] = 0.01
        consensus_field[15:20, 15:20] = 0.01  # Touches corner
        active_mask = np.ones((32, 32), dtype=bool)

        # With full connectivity: should be ONE component
        components_full = find_consensus_components(
            consensus_field, active_mask,
            kl_threshold=0.1, connectivity='full'
        )
        assert len(components_full) == 1

        # With axis connectivity: should be TWO components
        components_axis = find_consensus_components(
            consensus_field, active_mask,
            kl_threshold=0.1, connectivity='axis'
        )
        assert len(components_axis) == 2

    def test_min_size_filtering(self):
        """Small regions below min_size should be filtered."""
        consensus_field = np.ones((32, 32)) * 0.5
        consensus_field[10:20, 10:20] = 0.01  # 100 pixels
        consensus_field[25:27, 25:27] = 0.01  # 4 pixels
        active_mask = np.ones((32, 32), dtype=bool)

        # With min_size=10: only large region
        components = find_consensus_components(
            consensus_field, active_mask,
            kl_threshold=0.1, min_region_size=10
        )
        assert len(components) == 1
        assert components[0]['size'] == 100

    def test_3d_components(self):
        """Should work with 3D volumes."""
        consensus_field = np.ones((16, 16, 16)) * 0.5
        consensus_field[5:10, 5:10, 5:10] = 0.01  # 3D cube region
        active_mask = np.ones((16, 16, 16), dtype=bool)

        components = find_consensus_components(
            consensus_field, active_mask,
            kl_threshold=0.1, min_region_size=4
        )

        assert len(components) == 1
        assert components[0]['size'] == 5 * 5 * 5


# =============================================================================
# Tests: Spatial Emergence Detector
# =============================================================================

class TestSpatialEmergenceDetector:
    """Test the full spatial emergence detection pipeline."""

    def test_1d_emergence(self):
        """Test emergence on 1D manifold."""
        manifold = BaseManifold(shape=(64,), topology=TopologyType.PERIODIC)

        # Create two overlapping agents
        agents = [
            MockAgent(0, manifold, center=(20,), radius=10),
            MockAgent(1, manifold, center=(25,), radius=10)
        ]

        detector = MockConsensusDetector(kl_threshold=0.1)
        spatial_detector = SpatialEmergenceDetector(
            consensus_detector=detector,
            kl_threshold=0.1,
            min_region_size=4,
            smooth_sigma=1.5
        )

        class System:
            def __init__(self, agents):
                self.agents = agents
                self.n_agents = len(agents)

        regions = spatial_detector.detect_emergence_regions(System(agents))

        # Should detect at least one region
        assert len(regions) >= 1
        # Region should have 1D shape
        assert regions[0].component_mask.ndim == 1

    def test_2d_emergence(self):
        """Test emergence on 2D manifold."""
        manifold = BaseManifold(shape=(32, 32), topology=TopologyType.PERIODIC)

        # Create overlapping agents
        agents = [
            MockAgent(0, manifold, center=(12, 12), radius=8),
            MockAgent(1, manifold, center=(16, 16), radius=8)
        ]

        detector = MockConsensusDetector(kl_threshold=0.1)
        spatial_detector = SpatialEmergenceDetector(
            consensus_detector=detector,
            kl_threshold=0.1,
            min_region_size=4,
            smooth_sigma=1.5
        )

        class System:
            def __init__(self, agents):
                self.agents = agents
                self.n_agents = len(agents)

        regions = spatial_detector.detect_emergence_regions(System(agents))

        assert len(regions) >= 1
        assert regions[0].component_mask.ndim == 2

    def test_3d_emergence(self):
        """Test emergence on 3D manifold."""
        manifold = BaseManifold(shape=(16, 16, 16), topology=TopologyType.PERIODIC)

        agents = [
            MockAgent(0, manifold, center=(6, 6, 6), radius=5),
            MockAgent(1, manifold, center=(8, 8, 8), radius=5)
        ]

        detector = MockConsensusDetector(kl_threshold=0.1)
        spatial_detector = SpatialEmergenceDetector(
            consensus_detector=detector,
            kl_threshold=0.1,
            min_region_size=4,
            smooth_sigma=1.0
        )

        class System:
            def __init__(self, agents):
                self.agents = agents
                self.n_agents = len(agents)

        regions = spatial_detector.detect_emergence_regions(System(agents))

        assert len(regions) >= 1
        assert regions[0].component_mask.ndim == 3

    def test_meta_agent_support_creation(self):
        """Test that meta-agent support is correctly computed from region."""
        manifold = BaseManifold(shape=(32, 32), topology=TopologyType.PERIODIC)

        agents = [
            MockAgent(0, manifold, center=(16, 16), radius=8),
            MockAgent(1, manifold, center=(16, 16), radius=8)  # Same center
        ]

        detector = MockConsensusDetector(kl_threshold=0.1)
        spatial_detector = SpatialEmergenceDetector(
            consensus_detector=detector,
            kl_threshold=0.1,
            smooth_sigma=1.5
        )

        class System:
            def __init__(self, agents):
                self.agents = agents
                self.n_agents = len(agents)

        regions = spatial_detector.detect_emergence_regions(System(agents))

        if regions:
            support = spatial_detector.create_meta_agent_support(
                region=regions[0],
                agents=agents,
                weight_by_coherence=True
            )

            # Support should be valid
            assert support.chi_weight.shape == (32, 32)
            assert np.all(support.chi_weight >= 0)
            assert np.all(support.chi_weight <= 1)
            # Should have some support
            assert np.max(support.chi_weight) > 0


# =============================================================================
# Tests: SpatialConsensusRegion Properties
# =============================================================================

class TestSpatialConsensusRegion:
    """Test SpatialConsensusRegion dataclass."""

    def test_volume_computation(self):
        """Volume should count True pixels in mask."""
        manifold = BaseManifold(shape=(32, 32))

        mask = np.zeros((32, 32), dtype=bool)
        mask[10:20, 10:20] = True  # 100 pixels

        region = SpatialConsensusRegion(
            cluster_indices=[0, 1],
            component_id=0,
            component_mask=mask,
            consensus_field=np.ones((32, 32)) * 0.01,
            smoothed_support=mask.astype(np.float32),
            base_manifold=manifold
        )

        assert region.volume == 100

    def test_volume_fraction(self):
        """Volume fraction should be volume / total_points."""
        manifold = BaseManifold(shape=(32, 32))  # 1024 total

        mask = np.zeros((32, 32), dtype=bool)
        mask[10:20, 10:20] = True  # 100 pixels

        region = SpatialConsensusRegion(
            cluster_indices=[0, 1],
            component_id=0,
            component_mask=mask,
            consensus_field=np.ones((32, 32)) * 0.01,
            smoothed_support=mask.astype(np.float32),
            base_manifold=manifold
        )

        assert abs(region.volume_fraction - 100/1024) < 1e-6

    def test_centroid_computation(self):
        """Centroid should be center of mass."""
        manifold = BaseManifold(shape=(32, 32))

        mask = np.zeros((32, 32), dtype=bool)
        mask[10:20, 10:20] = True  # Center should be (14.5, 14.5)

        region = SpatialConsensusRegion(
            cluster_indices=[0, 1],
            component_id=0,
            component_mask=mask,
            consensus_field=np.ones((32, 32)) * 0.01,
            smoothed_support=mask.astype(np.float32),
            base_manifold=manifold
        )

        cx, cy = region.centroid
        assert abs(cx - 14.5) < 0.1
        assert abs(cy - 14.5) < 0.1

    def test_bounding_box(self):
        """Bounding box should enclose the region."""
        manifold = BaseManifold(shape=(32, 32))

        mask = np.zeros((32, 32), dtype=bool)
        mask[5:15, 10:25] = True

        region = SpatialConsensusRegion(
            cluster_indices=[0, 1],
            component_id=0,
            component_mask=mask,
            consensus_field=np.ones((32, 32)) * 0.01,
            smoothed_support=mask.astype(np.float32),
            base_manifold=manifold
        )

        bbox = region.bounding_box
        assert bbox[0] == (5, 14)   # y range
        assert bbox[1] == (10, 24)  # x range


# =============================================================================
# Tests: Summary Function
# =============================================================================

class TestSummaryFunction:
    """Test the summarize_emergence_detection utility."""

    def test_empty_regions(self):
        """Should handle empty region list."""
        summary = summarize_emergence_detection([])
        assert summary['n_regions'] == 0
        assert summary['total_volume'] == 0

    def test_single_region(self):
        """Should summarize single region correctly."""
        manifold = BaseManifold(shape=(32, 32))
        mask = np.zeros((32, 32), dtype=bool)
        mask[10:20, 10:20] = True

        region = SpatialConsensusRegion(
            cluster_indices=[0, 1],
            component_id=0,
            component_mask=mask,
            consensus_field=np.ones((32, 32)) * 0.01,
            smoothed_support=mask.astype(np.float32),
            base_manifold=manifold
        )

        summary = summarize_emergence_detection([region])

        assert summary['n_regions'] == 1
        assert summary['total_volume'] == 100
        assert summary['clusters'] == [[0, 1]]


# =============================================================================
# Integration Test
# =============================================================================

class TestIntegration:
    """Integration tests combining multiple components."""

    def test_full_pipeline_2d(self):
        """Test complete pipeline from agents to regions in 2D."""
        manifold = BaseManifold(shape=(64, 64), topology=TopologyType.PERIODIC)

        # Create 4 agents in two pairs
        agents = [
            MockAgent(0, manifold, center=(16, 16), radius=10),
            MockAgent(1, manifold, center=(20, 20), radius=10),  # Overlaps with 0
            MockAgent(2, manifold, center=(48, 48), radius=10),
            MockAgent(3, manifold, center=(52, 52), radius=10),  # Overlaps with 2
        ]

        # Mock detector that returns two clusters
        class TwoClusterDetector(MockConsensusDetector):
            def find_consensus_clusters(self, system):
                return [[0, 1], [2, 3]]

        detector = TwoClusterDetector(kl_threshold=0.1)
        spatial_detector = SpatialEmergenceDetector(
            consensus_detector=detector,
            kl_threshold=0.1,
            min_region_size=4,
            smooth_sigma=1.5
        )

        class System:
            def __init__(self, agents):
                self.agents = agents
                self.n_agents = len(agents)

        regions = spatial_detector.detect_emergence_regions(System(agents))

        # Should get two separate regions (from two clusters)
        assert len(regions) >= 2

        # Each region should have different centroid
        centroids = [r.centroid for r in regions]
        dist = np.sqrt(sum((c1 - c2)**2 for c1, c2 in zip(centroids[0], centroids[1])))
        assert dist > 20, "Regions should be spatially separated"


# =============================================================================
# Run tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])