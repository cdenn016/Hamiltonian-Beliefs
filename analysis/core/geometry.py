"""
Geometry Utilities for Analysis
================================

Helper functions for extracting spatial information from systems.
"""

import numpy as np
from typing import Tuple


def get_spatial_shape_from_system(system) -> Tuple:
    """
    Infer spatial shape from first agent's support or base_manifold.

    Returns:
        Tuple of spatial dimensions (e.g., () for 0D, (N,) for 1D, (N, M) for 2D)
    """
    a0 = system.agents[0]

    if hasattr(a0, "support") and a0.support is not None:
        shape = getattr(a0.support, "base_shape", None)
        if shape is None:
            shape = getattr(a0.support, "mask", np.array([])).shape
        return tuple(shape)

    if hasattr(a0, "base_manifold"):
        return tuple(a0.base_manifold.shape)

    if hasattr(a0, "support_shape"):
        return tuple(a0.support_shape)

    return ()


def pick_reference_agent(system) -> int:
    """
    Pick a reference agent i that actually has neighbors.

    Used for spatial plots where we need an agent with overlap relationships.

    Returns:
        Index of agent with neighbors, or 0 if none found
    """
    for i in range(system.n_agents):
        neighbors = system.get_neighbors(i)
        if neighbors:
            return i
    return 0


def get_ndim_from_shape(shape: Tuple) -> int:
    """
    Get number of spatial dimensions from shape.

    Args:
        shape: Spatial shape tuple

    Returns:
        0 for point manifold, 1 for 1D, 2 for 2D
    """
    if not shape or shape == ():
        return 0
    elif len(shape) == 1:
        return 1
    else:
        return 2