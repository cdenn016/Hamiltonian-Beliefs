# -*- coding: utf-8 -*-
"""
Created on Thu Nov 27 09:42:52 2025

@author: chris and christine
"""

"""
Power-Law Analysis of Attention Patterns β_ij
==============================================

Analyzes the statistical properties of the REAL attention weights:

    β_ij = softmax(-KL(q_i || Ω_ij[q_j]) / κ)

Key questions:
1. What is the distribution of β_ij values? (rank-frequency, histogram)
2. Does β_ij show power-law or exponential decay with some variable?
3. How does the distribution change with κ (temperature)?
4. Are there regime transitions where scaling behavior changes?

This module uses compute_softmax_weights() from softmax_grads.py -
the ACTUAL KL-based formula, NOT a fake exponential approximation.

Author: Chris & Claude
Date: November 2025
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Literal
from dataclasses import dataclass
from scipy import stats
from scipy.optimize import minimize_scalar
import warnings


@dataclass
class PowerLawFitResult:
    """Results from power-law fitting."""
    exponent: float           # α in p(x) ∝ x^(-α)
    x_min: float             # Lower cutoff
    log_likelihood: float    # Log-likelihood of fit
    ks_statistic: float      # Kolmogorov-Smirnov statistic
    ks_pvalue: float         # p-value for power-law hypothesis
    n_tail: int              # Number of data points in tail


@dataclass
class ExponentialFitResult:
    """Results from exponential fitting."""
    rate: float              # λ in p(x) ∝ exp(-λx)
    log_likelihood: float    # Log-likelihood
    ks_statistic: float      # KS statistic
    ks_pvalue: float         # p-value


@dataclass
class AttentionDistributionAnalysis:
    """Complete analysis of β_ij distribution."""
    beta_values: np.ndarray          # All β_ij values (flattened)
    n_agents: int
    n_pairs: int
    kappa: float                     # Temperature used

    # Basic statistics
    mean: float
    std: float
    median: float
    max_val: float
    min_val: float

    # Distribution fits
    power_law_fit: Optional[PowerLawFitResult] = None
    exponential_fit: Optional[ExponentialFitResult] = None

    # Model comparison
    preferred_model: Optional[str] = None  # 'power_law' or 'exponential'
    likelihood_ratio: Optional[float] = None


def extract_attention_weights(
    system,
    mode: Literal['belief', 'prior'] = 'belief',
    kappa: Optional[float] = None
) -> Tuple[np.ndarray, Dict[Tuple[int, int], np.ndarray]]:
    """
    Extract all β_ij attention weights from a MultiAgentSystem.

    Uses the REAL softmax-over-KL formula from softmax_grads.py.

    Args:
        system: MultiAgentSystem instance
        mode: 'belief' for β weights, 'prior' for γ weights
        kappa: Temperature (uses config default if None)

    Returns:
        all_weights: Flattened array of all β_ij values
        weight_dict: Dict mapping (i,j) → β_ij field
    """
    from gradients.softmax_grads import compute_softmax_weights

    if kappa is None:
        kappa = system.config.kappa_beta if mode == 'belief' else system.config.kappa_gamma

    weight_dict = {}
    all_weights = []

    for i in range(system.n_agents):
        # Get β_ij for all neighbors of agent i
        beta_fields = compute_softmax_weights(system, i, mode, kappa)

        for j, beta_ij in beta_fields.items():
            weight_dict[(i, j)] = beta_ij
            # Flatten and collect all values
            all_weights.extend(beta_ij.flatten())

    return np.array(all_weights), weight_dict


def compute_basic_statistics(beta_values: np.ndarray) -> Dict[str, float]:
    """Compute basic statistics of β_ij distribution."""
    # Filter out zeros for meaningful statistics
    nonzero = beta_values[beta_values > 1e-10]

    return {
        'mean': float(np.mean(beta_values)),
        'std': float(np.std(beta_values)),
        'median': float(np.median(beta_values)),
        'max': float(np.max(beta_values)),
        'min': float(np.min(beta_values)),
        'n_total': len(beta_values),
        'n_nonzero': len(nonzero),
        'fraction_nonzero': len(nonzero) / len(beta_values) if len(beta_values) > 0 else 0,
        'entropy': float(stats.entropy(beta_values[beta_values > 0])) if np.any(beta_values > 0) else 0,
    }


def fit_power_law(
    data: np.ndarray,
    x_min: Optional[float] = None,
    x_min_method: str = 'clauset'
) -> PowerLawFitResult:
    """
    Fit power-law distribution to data using MLE.

    For continuous data: p(x) ∝ x^(-α) for x ≥ x_min
    MLE estimator: α = 1 + n / Σ ln(x_i / x_min)

    Args:
        data: Array of positive values
        x_min: Lower cutoff (estimated if None)
        x_min_method: 'clauset' for KS-minimization, 'percentile' for 10th percentile

    Returns:
        PowerLawFitResult with fitted parameters
    """
    data = np.asarray(data)
    data = data[data > 0]  # Remove zeros

    if len(data) < 10:
        return PowerLawFitResult(
            exponent=np.nan, x_min=np.nan, log_likelihood=np.nan,
            ks_statistic=np.nan, ks_pvalue=np.nan, n_tail=0
        )

    # Estimate x_min if not provided
    if x_min is None:
        if x_min_method == 'clauset':
            x_min = _estimate_xmin_clauset(data)
        else:
            x_min = np.percentile(data, 10)

    # Filter data above x_min
    tail_data = data[data >= x_min]
    n = len(tail_data)

    if n < 5:
        return PowerLawFitResult(
            exponent=np.nan, x_min=x_min, log_likelihood=np.nan,
            ks_statistic=np.nan, ks_pvalue=np.nan, n_tail=n
        )

    # MLE for power-law exponent
    # α = 1 + n / Σ ln(x_i / x_min)
    alpha = 1 + n / np.sum(np.log(tail_data / x_min))

    # Log-likelihood
    log_likelihood = n * np.log(alpha - 1) - n * np.log(x_min) - alpha * np.sum(np.log(tail_data / x_min))

    # KS test
    theoretical_cdf = lambda x: 1 - (x_min / x) ** (alpha - 1)
    empirical_cdf = np.arange(1, n + 1) / n
    sorted_data = np.sort(tail_data)

    ks_stat = np.max(np.abs(empirical_cdf - theoretical_cdf(sorted_data)))

    # p-value via bootstrap (simplified)
    ks_pvalue = _bootstrap_ks_pvalue(tail_data, alpha, x_min, n_bootstrap=100)

    return PowerLawFitResult(
        exponent=alpha,
        x_min=x_min,
        log_likelihood=log_likelihood,
        ks_statistic=ks_stat,
        ks_pvalue=ks_pvalue,
        n_tail=n
    )


def fit_exponential(data: np.ndarray, x_min: Optional[float] = None) -> ExponentialFitResult:
    """
    Fit exponential distribution to data using MLE.

    p(x) ∝ exp(-λx) for x ≥ x_min
    MLE: λ = 1 / (mean(x) - x_min)

    Args:
        data: Array of positive values
        x_min: Lower cutoff (uses min(data) if None)

    Returns:
        ExponentialFitResult with fitted parameters
    """
    data = np.asarray(data)
    data = data[data > 0]

    if len(data) < 10:
        return ExponentialFitResult(
            rate=np.nan, log_likelihood=np.nan,
            ks_statistic=np.nan, ks_pvalue=np.nan
        )

    if x_min is None:
        x_min = np.min(data)

    tail_data = data[data >= x_min]
    n = len(tail_data)

    if n < 5:
        return ExponentialFitResult(
            rate=np.nan, log_likelihood=np.nan,
            ks_statistic=np.nan, ks_pvalue=np.nan
        )

    # MLE for exponential rate
    rate = 1.0 / (np.mean(tail_data) - x_min)

    # Log-likelihood
    log_likelihood = n * np.log(rate) - rate * np.sum(tail_data - x_min)

    # KS test
    theoretical_cdf = lambda x: 1 - np.exp(-rate * (x - x_min))
    empirical_cdf = np.arange(1, n + 1) / n
    sorted_data = np.sort(tail_data)

    ks_stat = np.max(np.abs(empirical_cdf - theoretical_cdf(sorted_data)))

    # Use scipy for p-value
    _, ks_pvalue = stats.kstest(tail_data - x_min, 'expon', args=(0, 1/rate))

    return ExponentialFitResult(
        rate=rate,
        log_likelihood=log_likelihood,
        ks_statistic=ks_stat,
        ks_pvalue=ks_pvalue
    )


def compare_power_law_vs_exponential(
    data: np.ndarray,
    x_min: Optional[float] = None
) -> Tuple[str, float, PowerLawFitResult, ExponentialFitResult]:
    """
    Compare power-law vs exponential fits using likelihood ratio.

    Uses Vuong's test for non-nested model comparison.

    Args:
        data: Array of positive values
        x_min: Shared lower cutoff for both fits

    Returns:
        preferred: 'power_law' or 'exponential'
        likelihood_ratio: Log-likelihood ratio (positive favors power-law)
        pl_fit: Power-law fit result
        exp_fit: Exponential fit result
    """
    if x_min is None:
        x_min = np.percentile(data[data > 0], 10)

    pl_fit = fit_power_law(data, x_min=x_min)
    exp_fit = fit_exponential(data, x_min=x_min)

    if np.isnan(pl_fit.log_likelihood) or np.isnan(exp_fit.log_likelihood):
        return 'inconclusive', 0.0, pl_fit, exp_fit

    # Likelihood ratio
    lr = pl_fit.log_likelihood - exp_fit.log_likelihood

    preferred = 'power_law' if lr > 0 else 'exponential'

    return preferred, lr, pl_fit, exp_fit


def analyze_attention_distribution(
    system,
    mode: Literal['belief', 'prior'] = 'belief',
    kappa: Optional[float] = None,
    fit_distributions: bool = True
) -> AttentionDistributionAnalysis:
    """
    Complete analysis of β_ij attention weight distribution.

    Args:
        system: MultiAgentSystem instance
        mode: 'belief' or 'prior'
        kappa: Temperature parameter
        fit_distributions: Whether to fit power-law/exponential

    Returns:
        AttentionDistributionAnalysis with full results
    """
    if kappa is None:
        kappa = system.config.kappa_beta if mode == 'belief' else system.config.kappa_gamma

    # Extract weights
    beta_values, _ = extract_attention_weights(system, mode, kappa)

    # Basic stats
    basic_stats = compute_basic_statistics(beta_values)

    # Initialize result
    result = AttentionDistributionAnalysis(
        beta_values=beta_values,
        n_agents=system.n_agents,
        n_pairs=len(beta_values),
        kappa=kappa,
        mean=basic_stats['mean'],
        std=basic_stats['std'],
        median=basic_stats['median'],
        max_val=basic_stats['max'],
        min_val=basic_stats['min'],
    )

    # Fit distributions if requested
    if fit_distributions and len(beta_values) > 20:
        nonzero = beta_values[beta_values > 1e-10]
        if len(nonzero) > 20:
            preferred, lr, pl_fit, exp_fit = compare_power_law_vs_exponential(nonzero)
            result.power_law_fit = pl_fit
            result.exponential_fit = exp_fit
            result.preferred_model = preferred
            result.likelihood_ratio = lr

    return result


def scan_temperature_dependence(
    system,
    kappa_values: np.ndarray,
    mode: Literal['belief', 'prior'] = 'belief'
) -> List[AttentionDistributionAnalysis]:
    """
    Analyze how β_ij distribution changes with temperature κ.

    Args:
        system: MultiAgentSystem
        kappa_values: Array of κ values to scan
        mode: 'belief' or 'prior'

    Returns:
        List of AttentionDistributionAnalysis for each κ
    """
    results = []
    for kappa in kappa_values:
        analysis = analyze_attention_distribution(system, mode, kappa)
        results.append(analysis)
    return results


def compute_rank_frequency(beta_values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute rank-frequency distribution (Zipf plot data).

    Args:
        beta_values: Array of attention weights

    Returns:
        ranks: 1, 2, 3, ... N
        sorted_values: β values sorted descending
    """
    sorted_vals = np.sort(beta_values)[::-1]  # Descending
    ranks = np.arange(1, len(sorted_vals) + 1)
    return ranks, sorted_vals


# =============================================================================
# Helper Functions
# =============================================================================

def _estimate_xmin_clauset(data: np.ndarray, n_candidates: int = 50) -> float:
    """
    Estimate optimal x_min using Clauset et al. method.

    Minimizes KS statistic between data and fitted power-law.
    """
    sorted_data = np.sort(data)
    n = len(sorted_data)

    # Candidate x_min values
    candidates = sorted_data[::max(1, n // n_candidates)]

    best_xmin = candidates[0]
    best_ks = np.inf

    for xmin in candidates[:-5]:  # Need at least 5 points in tail
        tail = sorted_data[sorted_data >= xmin]
        if len(tail) < 5:
            continue

        # Fit power-law
        alpha = 1 + len(tail) / np.sum(np.log(tail / xmin))

        # KS statistic
        theoretical_cdf = 1 - (xmin / tail) ** (alpha - 1)
        empirical_cdf = np.arange(1, len(tail) + 1) / len(tail)

        ks = np.max(np.abs(np.sort(theoretical_cdf) - empirical_cdf))

        if ks < best_ks:
            best_ks = ks
            best_xmin = xmin

    return best_xmin


def _bootstrap_ks_pvalue(
    data: np.ndarray,
    alpha: float,
    x_min: float,
    n_bootstrap: int = 100
) -> float:
    """
    Estimate p-value for power-law fit via bootstrap.
    """
    n = len(data)

    # Observed KS statistic
    theoretical_cdf = lambda x: 1 - (x_min / x) ** (alpha - 1)
    empirical_cdf = np.arange(1, n + 1) / n
    sorted_data = np.sort(data)
    ks_observed = np.max(np.abs(empirical_cdf - theoretical_cdf(sorted_data)))

    # Bootstrap
    n_larger = 0
    for _ in range(n_bootstrap):
        # Generate synthetic power-law data
        u = np.random.random(n)
        synthetic = x_min * (1 - u) ** (-1 / (alpha - 1))

        # Fit and compute KS
        alpha_syn = 1 + n / np.sum(np.log(synthetic / x_min))
        theoretical_syn = lambda x, a=alpha_syn: 1 - (x_min / x) ** (a - 1)
        sorted_syn = np.sort(synthetic)
        ks_syn = np.max(np.abs(empirical_cdf - theoretical_syn(sorted_syn)))

        if ks_syn >= ks_observed:
            n_larger += 1

    return n_larger / n_bootstrap


# =============================================================================
# Visualization (to be used with matplotlib)
# =============================================================================

def plot_attention_distribution(
    analysis: AttentionDistributionAnalysis,
    ax=None,
    log_scale: bool = True
):
    """
    Plot histogram and fitted distributions.

    Args:
        analysis: AttentionDistributionAnalysis result
        ax: Matplotlib axis (creates new figure if None)
        log_scale: Use log-log scale
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    beta = analysis.beta_values[analysis.beta_values > 1e-10]

    # Histogram
    if log_scale:
        bins = np.logspace(np.log10(beta.min()), np.log10(beta.max()), 50)
        ax.hist(beta, bins=bins, density=True, alpha=0.7, label='Data')
        ax.set_xscale('log')
        ax.set_yscale('log')
    else:
        ax.hist(beta, bins=50, density=True, alpha=0.7, label='Data')

    # Plot fits if available
    x_plot = np.linspace(beta.min(), beta.max(), 200)

    if analysis.power_law_fit is not None and not np.isnan(analysis.power_law_fit.exponent):
        pl = analysis.power_law_fit
        # Normalize power-law PDF
        norm = (pl.exponent - 1) / pl.x_min
        y_pl = norm * (x_plot / pl.x_min) ** (-pl.exponent)
        y_pl[x_plot < pl.x_min] = 0
        ax.plot(x_plot, y_pl, 'r-', lw=2,
                label=f'Power-law (α={pl.exponent:.2f})')

    if analysis.exponential_fit is not None and not np.isnan(analysis.exponential_fit.rate):
        exp = analysis.exponential_fit
        y_exp = exp.rate * np.exp(-exp.rate * (x_plot - beta.min()))
        ax.plot(x_plot, y_exp, 'g--', lw=2,
                label=f'Exponential (λ={exp.rate:.2f})')

    ax.set_xlabel('β_ij')
    ax.set_ylabel('Density')
    ax.set_title(f'Attention Distribution (κ={analysis.kappa:.2f})')
    ax.legend()

    return ax


def plot_rank_frequency(
    analysis: AttentionDistributionAnalysis,
    ax=None
):
    """
    Plot Zipf-style rank-frequency distribution.

    Args:
        analysis: AttentionDistributionAnalysis result
        ax: Matplotlib axis
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    ranks, values = compute_rank_frequency(analysis.beta_values)

    ax.loglog(ranks, values, 'b.', alpha=0.5, markersize=2)

    # Fit line for reference
    if analysis.power_law_fit is not None and not np.isnan(analysis.power_law_fit.exponent):
        # For Zipf: rank^(-1/(α-1))
        zipf_exp = 1 / (analysis.power_law_fit.exponent - 1)
        y_fit = values[0] * (ranks / ranks[0]) ** (-zipf_exp)
        ax.loglog(ranks, y_fit, 'r-', lw=2, alpha=0.7,
                  label=f'Zipf (exp={zipf_exp:.2f})')

    ax.set_xlabel('Rank')
    ax.set_ylabel('β_ij')
    ax.set_title('Rank-Frequency Plot')
    ax.legend()
    ax.grid(True, alpha=0.3)

    return ax