"""
Core Analysis Utilities
========================

Data loading, preprocessing, and geometry helpers.
"""

from .loaders import (
    load_history,
    load_system,
    get_mu_tracker,
    filter_history_steps,
    filter_mu_tracker,
    normalize_history,
    DEFAULT_SKIP_STEPS,
)

from .geometry import (
    get_spatial_shape_from_system,
    pick_reference_agent,
    get_ndim_from_shape,
)

from .power_law_attention import (
    extract_attention_weights,
    analyze_attention_distribution,
    scan_temperature_dependence,
    compute_rank_frequency,
    fit_power_law,
    fit_exponential,
    compare_power_law_vs_exponential,
    plot_attention_distribution,
    plot_rank_frequency,
    PowerLawFitResult,
    ExponentialFitResult,
    AttentionDistributionAnalysis,
)

__all__ = [
    # Loaders
    'load_history',
    'load_system',
    'get_mu_tracker',
    'filter_history_steps',
    'filter_mu_tracker',
    'normalize_history',
    'DEFAULT_SKIP_STEPS',
    # Geometry
    'get_spatial_shape_from_system',
    'pick_reference_agent',
    'get_ndim_from_shape',
    # Power-law attention analysis
    'extract_attention_weights',
    'analyze_attention_distribution',
    'scan_temperature_dependence',
    'compute_rank_frequency',
    'fit_power_law',
    'fit_exponential',
    'compare_power_law_vs_exponential',
    'plot_attention_distribution',
    'plot_rank_frequency',
    'PowerLawFitResult',
    'ExponentialFitResult',
    'AttentionDistributionAnalysis',
]