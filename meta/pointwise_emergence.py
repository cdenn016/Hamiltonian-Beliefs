#!/usr/bin/env python3
"""
Pointwise Emergence: Meta-Agent Formation in Connected Consensus Regions

For spatial manifolds, meta-agents emerge locally in regions where constituents
reach consensus, rather than requiring global agreement across the entire grid.

Key Idea:
- Compute spatial consensus mask: consensus_mask(x,y) = (KL(x,y) < threshold)
- Find connected components: each connected region becomes ONE meta-agent
- Meta-agent support: χ_M(x,y) > 0 only in consensus region

This prevents proliferation: adjacent points c and c+dc with consensus form
the SAME meta-agent, not different ones.

Author: Claude & Chris
Date: November 2025
"""

import numpy as np
from typing import List, Tuple, Optional
from scipy.ndimage import label as find_connected_components
from geometry.geometry_base import SupportRegion

def compute_spatial_consensus_mask(
    consensus_detector,
    agent_i,
    agent_j,
    check_models: bool = True
) -> np.ndarray:
    """
    Compute spatial mask showing where two agents reach consensus pointwise.

    Args:
        consensus_detector: ConsensusDetector instance
        agent_i, agent_j: Agents to compare
        check_models: If True, require both belief and model consensus

    Returns:
        consensus_mask: (*spatial,) boolean array, True where agents agree
    """
    # Compute transport once
    omega_ij = consensus_detector._get_transport(agent_i, agent_j)

    # Belief consensus pointwise
    belief_mask, belief_kl = consensus_detector.check_belief_consensus_spatial(
        agent_i, agent_j, omega_ij=omega_ij
    )

    if not check_models:
        return belief_mask

    # Model consensus pointwise
    model_mask, model_kl = consensus_detector.check_model_consensus_spatial(
        agent_i, agent_j, omega_ij=omega_ij
    )

    # Require BOTH belief and model consensus (epistemic death)
    full_consensus = belief_mask & model_mask

    return full_consensus


def find_consensus_regions(
    consensus_mask: np.ndarray,
    min_region_size: int = 4
) -> Tuple[np.ndarray, int, List[Tuple[int, np.ndarray]]]:
    """
    Identify connected regions in consensus mask.

    Uses 8-connectivity for 2D grids (neighbors include diagonals).

    Args:
        consensus_mask: (*spatial,) boolean array
        min_region_size: Minimum number of points to form a region

    Returns:
        labeled_array: (*spatial,) int array, each region has unique label
        n_regions: Number of distinct regions found
        regions: List of (region_id, region_mask) for regions >= min_size
    """
    # Find connected components
    # For 2D: structure=None uses 8-connectivity (includes diagonals)
    # For 1D: structure=None uses 2-connectivity (left-right neighbors)
    labeled, n_regions = find_connected_components(consensus_mask)

    # Filter by size
    valid_regions = []
    for region_id in range(1, n_regions + 1):  # Regions labeled 1, 2, 3, ...
        region_mask = (labeled == region_id)
        region_size = np.sum(region_mask)

        if region_size >= min_region_size:
            valid_regions.append((region_id, region_mask))

    return labeled, n_regions, valid_regions


def check_consensus_volume(
    region_mask: np.ndarray,
    total_volume: int,
    min_volume_fraction: float
) -> bool:
    """
    Check if consensus region covers sufficient fraction of manifold.

    Args:
        region_mask: Boolean mask for this region
        total_volume: Total number of spatial points
        min_volume_fraction: Minimum fraction required (e.g., 0.3 = 30%)

    Returns:
        True if region is large enough
    """
    region_volume = np.sum(region_mask)
    volume_fraction = region_volume / total_volume
    return volume_fraction >= min_volume_fraction


def compute_regional_weights(
    constituents: List,
    region_mask: np.ndarray,
    coherence_scores: np.ndarray
) -> np.ndarray:
    """
    Compute spatially-varying weights for meta-agent formation in a region.

    Within the consensus region, weights are:
        w_i(x) = χ_i(x) · C̄_i   (presence × coherence)

    Outside the region:
        w_i(x) = 0   (meta-agent has no support there)

    Args:
        constituents: List of constituent agents
        region_mask: (*spatial,) boolean, True in consensus region
        coherence_scores: (n_constituents,) float array

    Returns:
        weights: (*spatial, n_constituents) float array
    """
    # Get spatial shape from first agent
    spatial_shape = constituents[0].base_manifold.shape
    n_constituents = len(constituents)

    # Initialize weights
    weights = np.zeros((*spatial_shape, n_constituents))

    # Compute weights only in consensus region
    for i, agent in enumerate(constituents):
        # Weight by presence × coherence
        chi_i = agent.support.chi_weight  # (*spatial,)
        w_i = chi_i * coherence_scores[i]

        # Zero out weights outside consensus region
        weights[..., i] = np.where(region_mask, w_i, 0.0)

    return weights


def create_regional_support(
    base_manifold,
    region_mask: np.ndarray,
    constituents: List,
    coherence_scores: np.ndarray
) -> 'SupportRegion':
    """
    Create support region for meta-agent covering only consensus region.

    χ_M(x) = Σᵢ w_i(x) · χ_i(x)   where w_i(x) = 0 outside region

    Args:
        base_manifold: Base manifold
        region_mask: (*spatial,) boolean, consensus region
        constituents: List of constituent agents
        coherence_scores: (n_constituents,) coherence values

    Returns:
        support: SupportRegion with non-zero weight only in consensus region
    """
    

    spatial_shape = base_manifold.shape

    # Compute weighted sum of constituent supports within region
    chi_M = np.zeros(spatial_shape)
    total_weight = np.zeros(spatial_shape)

    for i, agent in enumerate(constituents):
        chi_i = agent.support.chi_weight
        C_i = coherence_scores[i]

        # Weight by presence × coherence, but only in region
        w_i = chi_i * C_i
        w_i = np.where(region_mask, w_i, 0.0)

        chi_M += w_i * chi_i
        total_weight += w_i

    # Normalize
    chi_M = np.where(total_weight > 1e-12, chi_M / total_weight, 0.0)

    # Ensure exactly zero outside region (for numerical cleanliness)
    chi_M = np.where(region_mask, chi_M, 0.0)

    # Create support
    support = SupportRegion(
        base_manifold=base_manifold,
        chi_soft=chi_M,
        chi_weight=chi_M,
        chi_hard=region_mask.astype(float),  # Hard boundary at region edge
        pattern="regional_consensus"
    )

    return support


def should_form_regional_meta_agent(
    region_mask: np.ndarray,
    total_volume: int,
    min_volume_fraction: float,
    min_region_size: int
) -> Tuple[bool, str]:
    """
    Determine if a consensus region should form a meta-agent.

    Criteria:
    1. Region size >= min_region_size (avoid tiny isolated patches)
    2. Volume fraction >= min_volume_fraction (significant coverage)

    Args:
        region_mask: Boolean mask for region
        total_volume: Total spatial volume
        min_volume_fraction: Minimum volume fraction (e.g., 0.3)
        min_region_size: Minimum number of points (e.g., 4)

    Returns:
        (should_form, reason): Boolean and explanation string
    """
    region_size = np.sum(region_mask)
    volume_fraction = region_size / total_volume

    # Check size
    if region_size < min_region_size:
        return False, f"Region too small ({region_size} < {min_region_size})"

    # Check volume
    if volume_fraction < min_volume_fraction:
        return False, f"Volume too small ({volume_fraction:.1%} < {min_volume_fraction:.1%})"

    return True, f"Valid region ({region_size} points, {volume_fraction:.1%} coverage)"


def visualize_consensus_regions(
    consensus_mask: np.ndarray,
    labeled_array: np.ndarray,
    valid_regions: List[Tuple[int, np.ndarray]],
    save_path: Optional[str] = None
):
    """
    Visualize consensus regions (debug utility).

    Shows:
    - Full consensus mask
    - Individual labeled regions
    - Valid regions that will form meta-agents

    Args:
        consensus_mask: Full consensus boolean mask
        labeled_array: Region labels from find_connected_components
        valid_regions: List of (region_id, mask) for valid regions
        save_path: Where to save plot (optional)
    """
    import matplotlib.pyplot as plt

    n_valid = len(valid_regions)

    fig, axes = plt.subplots(1, 2 + n_valid, figsize=(4 * (2 + n_valid), 4))

    # Full consensus mask
    axes[0].imshow(consensus_mask, cmap='Greys', origin='lower')
    axes[0].set_title('Full Consensus Mask')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('y')

    # All labeled regions
    axes[1].imshow(labeled_array, cmap='tab20', origin='lower')
    axes[1].set_title(f'All Regions (n={np.max(labeled_array)})')
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('y')

    # Individual valid regions
    for idx, (region_id, region_mask) in enumerate(valid_regions):
        axes[2 + idx].imshow(region_mask, cmap='Greys', origin='lower')
        region_size = np.sum(region_mask)
        axes[2 + idx].set_title(f'Region {region_id} ({region_size} points)')
        axes[2 + idx].set_xlabel('x')
        axes[2 + idx].set_ylabel('y')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()