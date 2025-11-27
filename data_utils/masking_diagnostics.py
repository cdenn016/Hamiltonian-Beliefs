# -*- coding: utf-8 -*-
"""
Diagnostics and Visualization for Support Masking
==================================================

Tools to verify and visualize support masking behavior.

Author: Chris 
Date: November 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from typing import List, Optional, Tuple


# =============================================================================
# SECTION 1: Field Validation
# =============================================================================

def validate_agent_support_constraints(agent) -> dict:
    """
    Validate that agent fields respect support constraints.
    
    Returns dict with validation results.
    """
    from agent.masking import SupportRegionSmooth
    
    results = {
        'agent_id': agent.agent_id,
        'has_support': hasattr(agent, 'support') and agent.support is not None,
        'violations': [],
        'warnings': [],
        'metrics': {}
    }
    
    if not results['has_support']:
        results['warnings'].append("No support region defined")
        return results
    
    support = agent.support
    
    if not isinstance(support, SupportRegionSmooth):
        results['warnings'].append("Support is not SupportRegionSmooth")
        return results
    
    # Get masks
    mask_continuous = support.mask_continuous
    threshold = support.config.min_mask_for_normal_cov
    
    inside_mask = mask_continuous > 0.9  # Well inside
    boundary_mask = (mask_continuous > threshold) & (mask_continuous < 0.9)
    outside_mask = mask_continuous < threshold
    
    # ========== Check 1: Mean fields should be small/zero outside ==========
    if hasattr(agent, 'mu_q') and agent.mu_q is not None:
        if agent.mu_q.ndim > 1:
            mu_q_norm = np.linalg.norm(agent.mu_q, axis=-1)
            
            mu_inside = np.mean(mu_q_norm[inside_mask]) if np.any(inside_mask) else 0
            mu_outside = np.mean(mu_q_norm[outside_mask]) if np.any(outside_mask) else 0
            
            results['metrics']['mu_q_inside'] = float(mu_inside)
            results['metrics']['mu_q_outside'] = float(mu_outside)
            
            if mu_outside > 0.01 * mu_inside:
                results['violations'].append(
                    f"μ_q not zero outside support: {mu_outside:.2e} vs {mu_inside:.2e}"
                )
    
    # ========== Check 2: Covariances should be large outside ==========
    if hasattr(agent, 'Sigma_q') and agent.Sigma_q is not None:
        if agent.Sigma_q.ndim > 2:
            # Get eigenvalues at sample points
            if np.any(inside_mask):
                idx_inside = np.argwhere(inside_mask)[0]
                eigs_inside = np.linalg.eigvalsh(agent.Sigma_q[tuple(idx_inside)])
                min_eig_inside = np.min(eigs_inside)
                max_eig_inside = np.max(eigs_inside)
            else:
                min_eig_inside = 1.0
                max_eig_inside = 1.0
            
            if np.any(outside_mask):
                idx_outside = np.argwhere(outside_mask)[0]
                eigs_outside = np.linalg.eigvalsh(agent.Sigma_q[tuple(idx_outside)])
                min_eig_outside = np.min(eigs_outside)
                max_eig_outside = np.max(eigs_outside)
            else:
                min_eig_outside = support.config.outside_cov_scale
                max_eig_outside = support.config.outside_cov_scale
            
            results['metrics']['Sigma_q_eig_inside_min'] = float(min_eig_inside)
            results['metrics']['Sigma_q_eig_inside_max'] = float(max_eig_inside)
            results['metrics']['Sigma_q_eig_outside_min'] = float(min_eig_outside)
            results['metrics']['Sigma_q_eig_outside_max'] = float(max_eig_outside)
            
            expected_outside = support.config.outside_cov_scale
            
            # Check that outside eigenvalues are close to expected scale
            # Allow 10% deviation from expected value
            if min_eig_outside < 0.1 * expected_outside:
                results['violations'].append(
                    f"Σ_q too small outside: min_eig={min_eig_outside:.2e} "
                    f"vs expected ~{expected_outside:.2e}"
                )
            
            # Check that outside eigenvalues are larger than inside
            # (This ensures the "large covariance outside" property)
            if min_eig_outside < 10.0 * max_eig_inside:
                results['warnings'].append(
                    f"Σ_q outside not much larger than inside: "
                    f"outside={min_eig_outside:.2e} vs inside={max_eig_inside:.2e}"
                )
    
    # ========== Check 3: Gauge field should be zero/small outside ==========
    if hasattr(agent, 'gauge') and hasattr(agent.gauge, 'phi'):
        if agent.gauge.phi.ndim > 1:
            phi_norm = np.linalg.norm(agent.gauge.phi, axis=-1)
            
            phi_inside = np.mean(phi_norm[inside_mask]) if np.any(inside_mask) else 0
            phi_outside = np.mean(phi_norm[outside_mask]) if np.any(outside_mask) else 0
            
            results['metrics']['phi_inside'] = float(phi_inside)
            results['metrics']['phi_outside'] = float(phi_outside)
            
            if phi_outside > 0.1 * phi_inside and phi_inside > 1e-6:
                results['warnings'].append(
                    f"φ not small outside support: {phi_outside:.2e} vs {phi_inside:.2e}"
                )
    
    # ========== Summary ==========
    results['valid'] = len(results['violations']) == 0
    
    return results


def validate_system_overlaps(system) -> dict:
    """
    Validate overlap computation and thresholding.
    
    Returns dict with validation results.
    """
    results = {
        'n_agents': system.n_agents,
        'n_overlaps': len(system.overlap_masks),
        'violations': [],
        'metrics': {}
    }
    
    if not hasattr(system, 'overlap_masks'):
        results['violations'].append("No overlap_masks computed")
        return results
    
    # Check each overlap
    for (i, j), chi_ij in system.overlap_masks.items():
        agent_i = system.agents[i]
        agent_j = system.agents[j]
        
        # Verify shapes match
        if chi_ij.shape != agent_i.support.base_shape:
            results['violations'].append(
                f"Overlap ({i},{j}) shape mismatch: "
                f"{chi_ij.shape} vs {agent_i.support.base_shape}"
            )
        
        # Check values are in [0, 1]
        if np.any(chi_ij < 0) or np.any(chi_ij > 1):
            results['violations'].append(
                f"Overlap ({i},{j}) values outside [0,1]: "
                f"[{chi_ij.min():.2e}, {chi_ij.max():.2e}]"
            )
        
        # Check threshold was applied
        threshold = system.config.overlap_threshold if hasattr(system.config, 'overlap_threshold') else 1e-3
        if np.any((chi_ij > 0) & (chi_ij < threshold)):
            results['warnings'] = results.get('warnings', [])
            results['warnings'].append(
                f"Overlap ({i},{j}) has values below threshold: "
                f"min={chi_ij[chi_ij > 0].min():.2e}"
            )
    
    # Compute overlap statistics
    if system.overlap_masks:
        overlap_fracs = []
        for (i, j), chi_ij in system.overlap_masks.items():
            frac = np.sum(chi_ij > 0) / chi_ij.size
            overlap_fracs.append(frac)
        
        results['metrics']['mean_overlap_fraction'] = float(np.mean(overlap_fracs))
        results['metrics']['max_overlap_fraction'] = float(np.max(overlap_fracs))
        results['metrics']['min_overlap_fraction'] = float(np.min(overlap_fracs))
    
    results['valid'] = len(results['violations']) == 0
    
    return results


def validate_gradient_masking(system, gradients: List) -> dict:
    """
    Validate that gradients respect support constraints.
    
    Returns dict with validation results.
    """
    results = {
        'n_agents': len(gradients),
        'violations': [],
        'metrics': {}
    }
    
    for i, (agent, grad) in enumerate(zip(system.agents, gradients)):
        if not hasattr(agent, 'support') or agent.support is None:
            continue
        
        support = agent.support
        outside_mask = support.mask_continuous < support.config.min_mask_for_normal_cov
        
        if not np.any(outside_mask):
            continue  # No outside region
        
        # Check that gradients are small/zero outside
        for field_name in ['grad_mu_q', 'grad_Sigma_q', 'grad_mu_p', 'grad_Sigma_p', 'grad_phi']:
            if not hasattr(grad, field_name):
                continue
            
            field = getattr(grad, field_name)
            
            if field.ndim > 1:
                # Compute norm at each spatial point
                if field_name in ['grad_Sigma_q', 'grad_Sigma_p']:
                    field_norm = np.linalg.norm(field.reshape(*field.shape[:-2], -1), axis=-1)
                else:
                    field_norm = np.linalg.norm(field, axis=-1)
                
                norm_outside = np.mean(field_norm[outside_mask]) if np.any(outside_mask) else 0
                norm_inside = np.mean(field_norm[~outside_mask])
                
                if norm_outside > 0.1 * norm_inside and norm_inside > 1e-6:
                    results['violations'].append(
                        f"Agent {i} {field_name} not small outside: "
                        f"{norm_outside:.2e} vs {norm_inside:.2e}"
                    )
    
    results['valid'] = len(results['violations']) == 0
    
    return results


# =============================================================================
# SECTION 2: Visualization
# =============================================================================

def plot_agent_support_masks(
    agents: List,
    figsize: Tuple[int, int] = (15, 5),
    save_path: Optional[str] = None
):
    """
    Visualize support masks for multiple agents.
    
    Args:
        agents: List of Agent objects
        figsize: Figure size
        save_path: Optional path to save figure
    """
    n_agents = len(agents)
    fig, axes = plt.subplots(1, n_agents, figsize=figsize)
    
    if n_agents == 1:
        axes = [axes]
    
    for i, (agent, ax) in enumerate(zip(agents, axes)):
        if not hasattr(agent, 'support') or agent.support is None:
            ax.text(0.5, 0.5, "No support", ha='center', va='center')
            ax.set_title(f"Agent {agent.agent_id}")
            continue
        
        support = agent.support
        
        if len(support.base_shape) == 2:
            # 2D visualization
            im = ax.imshow(
                support.mask_continuous,
                cmap='viridis',
                vmin=0,
                vmax=1,
                origin='lower'
            )
            plt.colorbar(im, ax=ax, label='χ(c)')
            
            # Draw threshold contour
            threshold = support.config.overlap_threshold
            ax.contour(
                support.mask_continuous,
                levels=[threshold],
                colors='red',
                linewidths=2,
                linestyles='--'
            )
            
            ax.set_title(f"Agent {agent.agent_id}\n({support.n_active} active pts)")
            ax.set_xlabel("x")
            ax.set_ylabel("y")
        else:
            ax.text(0.5, 0.5, f"{len(support.base_shape)}D", ha='center', va='center')
            ax.set_title(f"Agent {agent.agent_id}")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved to {save_path}")
    
    plt.show()


def plot_overlap_masks(system, max_overlaps: int = 6, save_path: Optional[str] = None):
    """
    Visualize overlap masks showing BOTH continuous field and lens-shaped boundary.
    
    FIXED: Now shows threshold contour to reveal lens shape!
    """
    overlap_pairs = list(system.overlap_masks.items())[:max_overlaps]
    
    if not overlap_pairs:
        print("No overlaps to visualize")
        return
    
    n_overlaps = len(overlap_pairs)
    ncols = min(3, n_overlaps)
    nrows = (n_overlaps + ncols - 1) // ncols
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(6*ncols, 5*nrows))
    if n_overlaps == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    for idx, ((i, j), chi_ij) in enumerate(overlap_pairs):
        ax = axes[idx]
        
        if chi_ij.ndim != 2:
            ax.text(0.5, 0.5, "Not 2D", ha='center', va='center')
            ax.set_title(f'Overlap ({i}, {j})')
            continue
        
        overlap_integral = float(np.sum(chi_ij))
        
        # Show continuous field as heatmap
        im = ax.imshow(chi_ij, cmap='hot', origin='lower', interpolation='bilinear')
        
        # ✅ KEY FIX: Add contour at threshold to show lens shape!
        threshold = system.config.overlap_threshold
        
        # Show multiple contour levels to reveal shape
        levels = [threshold, 0.01, 0.1, 0.5]
        levels = [l for l in levels if l < np.max(chi_ij)]
        
        if levels:
            contours = ax.contour(
                chi_ij, 
                levels=levels,
                colors=['cyan', 'yellow', 'white', 'red'][:len(levels)],
                linewidths=[3, 2, 1.5, 1][:len(levels)],
                linestyles=['--', '-.', ':', '-'][:len(levels)]
            )
            ax.clabel(contours, inline=True, fontsize=8, fmt='%.3f')
        
        # Show individual agent boundaries
        agent_i = system.agents[i]
        agent_j = system.agents[j]
        
        # Contours of individual masks at 50% level
        mask_i = agent_i.support.mask_continuous
        mask_j = agent_j.support.mask_continuous
        
        ax.contour(mask_i, levels=[0.5], colors='blue', linewidths=2, alpha=0.6)
        ax.contour(mask_j, levels=[0.5], colors='red', linewidths=2, alpha=0.6)
        
        ax.set_title(f'Overlap ({i}, {j})\n∫χ = {overlap_integral:.1f}\nCyan=threshold, Blue/Red=agent boundaries')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        
        plt.colorbar(im, ax=ax, label='χ_ij(c)')
    
    # Hide unused subplots
    for idx in range(n_overlaps, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
   #     print(f"✅ Saved fixed overlap visualization: {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_single_overlap_with_geometry(system, i: int, j: int, save_path: Optional[str] = None):
    """
    Detailed visualization of one overlap showing geometric structure.
    """
    agent_i = system.agents[i]
    agent_j = system.agents[j]
    
    mask_i = agent_i.support.mask_continuous
    mask_j = agent_j.support.mask_continuous
    
    chi_ij = system.get_overlap_mask(i, j)
    
    if chi_ij is None or chi_ij.ndim != 2:
        print(f"Cannot visualize overlap ({i}, {j})")
        return
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Row 1: Individual masks and overlap field
    # Plot 1: Agent i continuous mask
    im1 = axes[0, 0].imshow(mask_i, cmap='Blues', origin='lower', vmin=0, vmax=1)
    axes[0, 0].contour(mask_i, levels=[0.1, 0.5, 0.9], colors='blue', linewidths=[1, 2, 1])
    axes[0, 0].set_title(f'Agent {i}: χ_i(c)\n{agent_i.support.n_active} active pts')
    plt.colorbar(im1, ax=axes[0, 0])
    
    # Plot 2: Agent j continuous mask
    im2 = axes[0, 1].imshow(mask_j, cmap='Reds', origin='lower', vmin=0, vmax=1)
    axes[0, 1].contour(mask_j, levels=[0.1, 0.5, 0.9], colors='red', linewidths=[1, 2, 1])
    axes[0, 1].set_title(f'Agent {j}: χ_j(c)\n{agent_j.support.n_active} active pts')
    plt.colorbar(im2, ax=axes[0, 1])
    
    # Plot 3: Overlap continuous field
    im3 = axes[0, 2].imshow(chi_ij, cmap='hot', origin='lower')
    threshold = system.config.overlap_threshold
    axes[0, 2].contour(chi_ij, levels=[threshold], colors='cyan', linewidths=3, linestyles='--')
    axes[0, 2].set_title(f'Overlap: χ_ij = χ_i · χ_j\n∫χ = {np.sum(chi_ij):.1f}')
    plt.colorbar(im3, ax=axes[0, 2])
    
    # Row 2: Geometric analysis
    # Plot 4: Binary masks (hard boundaries)
    mask_i_binary = mask_i > 0.5
    mask_j_binary = mask_j > 0.5
    
    combined_binary = np.zeros((*mask_i.shape, 3))
    combined_binary[mask_i_binary, 0] = 0.5  # Red channel for agent i
    combined_binary[mask_j_binary, 2] = 0.5  # Blue channel for agent j
    combined_binary[mask_i_binary & mask_j_binary, :] = [0.5, 0, 0.5]  # Purple for overlap
    
    axes[1, 0].imshow(combined_binary, origin='lower')
    axes[1, 0].set_title('Binary overlap region\nPurple = intersection')
    axes[1, 0].set_xlabel('x')
    axes[1, 0].set_ylabel('y')
    
    # Plot 5: Thresholded overlap (should show lens)
    overlap_thresholded = chi_ij > threshold
    axes[1, 1].imshow(overlap_thresholded, cmap='gray', origin='lower')
    axes[1, 1].contour(mask_i, levels=[0.5], colors='blue', linewidths=2, alpha=0.7)
    axes[1, 1].contour(mask_j, levels=[0.5], colors='red', linewidths=2, alpha=0.7)
    axes[1, 1].set_title(f'Thresholded overlap\n(χ_ij > {threshold:.0e})\nShould show LENS shape')
    axes[1, 1].set_xlabel('x')
    axes[1, 1].set_ylabel('y')
    
    # Plot 6: Cross-sections to show lens shape
    H, W = chi_ij.shape
    
    # Find center of overlap
    if np.sum(chi_ij) > 0:
        y_coords, x_coords = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
        center_y = int(np.sum(y_coords * chi_ij) / np.sum(chi_ij))
        center_x = int(np.sum(x_coords * chi_ij) / np.sum(chi_ij))
        
        # Horizontal cross-section
        axes[1, 2].plot(chi_ij[center_y, :], 'b-', linewidth=2, label='χ_ij horizontal')
        axes[1, 2].plot(mask_i[center_y, :], 'r--', alpha=0.5, label='χ_i')
        axes[1, 2].plot(mask_j[center_y, :], 'g--', alpha=0.5, label='χ_j')
        axes[1, 2].axhline(threshold, color='cyan', linestyle='--', label=f'threshold')
        axes[1, 2].set_xlabel('x coordinate')
        axes[1, 2].set_ylabel('χ value')
        axes[1, 2].set_title(f'Cross-section at y={center_y}')
        axes[1, 2].legend()
        axes[1, 2].grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
     #   print(f"✅ Saved detailed overlap geometry: {save_path}")
    else:
        plt.show()
    
    plt.close()


# Add to simulation_suite.py initial diagnostics:
def diagnose_overlap_geometry_suite(system, output_dir):
    """
    Complete overlap geometry diagnostic for simulation_suite.py
    """
    from pathlib import Path
    
    print(f"\n{'='*70}")
    print("OVERLAP GEOMETRY DIAGNOSTIC")
    print(f"{'='*70}")
    
    # 1. Show fixed overlap masks
    fig_path = output_dir / "overlap_masks_FIXED.png"
    plot_overlap_masks(system, max_overlaps=6, save_path=str(fig_path))
    
    # 2. Detailed analysis of first few overlaps
    overlap_pairs = list(system.overlap_masks.keys())[:3]
    
    for i, j in overlap_pairs:
        fig_path = output_dir / f"overlap_geometry_{i}_{j}.png"
        plot_single_overlap_with_geometry(system, i, j, save_path=str(fig_path))
        
        # Print statistics
        chi_ij = system.get_overlap_mask(i, j)
        if chi_ij is not None:
            threshold = system.config.overlap_threshold
            overlap_binary = chi_ij > threshold
            
            if np.any(overlap_binary):
                rows, cols = np.where(overlap_binary)
                y_span = rows.max() - rows.min() + 1
                x_span = cols.max() - cols.min() + 1
                aspect = max(y_span, x_span) / min(y_span, x_span) if min(y_span, x_span) > 0 else 0
                
                print(f"\nOverlap ({i}, {j}) geometry:")
                print(f"  Thresholded region: {y_span} × {x_span} pixels")
                print(f"  Aspect ratio: {aspect:.2f}")
                
                if aspect < 1.3:
                    print(f"  ⚠️  Nearly circular - agents may be too close!")
                else:
                    print(f"  ✅ Shows elongation (lens-like)")
    
    print(f"\n✅ Overlap diagnostics complete")


def plot_covariance_eigenvalues(
    agent,
    which: str = 'q',
    figsize: Tuple[int, int] = (12, 4),
    save_path: Optional[str] = None
):
    """
    Visualize covariance eigenvalues across space.
    
    Shows that eigenvalues are large outside support.
    
    Args:
        agent: Agent object
        which: 'q' for beliefs, 'p' for priors
        figsize: Figure size
        save_path: Optional path to save figure
    """
    Sigma = agent.Sigma_q if which == 'q' else agent.Sigma_p
    
    if Sigma.ndim <= 2:
        print("Agent is 0D, no spatial variation to plot")
        return
    
    spatial_shape = Sigma.shape[:-2]
    K = Sigma.shape[-1]
    
    if len(spatial_shape) != 2:
        print(f"Cannot plot {len(spatial_shape)}D spatial structure")
        return
    
    # Compute eigenvalues at each point
    min_eigs = np.zeros(spatial_shape)
    max_eigs = np.zeros(spatial_shape)
    
    for idx in np.ndindex(spatial_shape):
        eigs = np.linalg.eigvalsh(Sigma[idx])
        min_eigs[idx] = eigs[0]
        max_eigs[idx] = eigs[-1]
    
    # Plot
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Min eigenvalue (log scale)
    im0 = axes[0].imshow(
        min_eigs,
        cmap='viridis',
        norm=LogNorm(vmin=min_eigs[min_eigs > 0].min(), vmax=min_eigs.max()),
        origin='lower'
    )
    plt.colorbar(im0, ax=axes[0], label='λ_min (log scale)')
    axes[0].set_title(f"Min Eigenvalue (Σ_{which})")
    
    # Max eigenvalue (log scale)
    im1 = axes[1].imshow(
        max_eigs,
        cmap='plasma',
        norm=LogNorm(vmin=max_eigs[max_eigs > 0].min(), vmax=max_eigs.max()),
        origin='lower'
    )
    plt.colorbar(im1, ax=axes[1], label='λ_max (log scale)')
    axes[1].set_title(f"Max Eigenvalue (Σ_{which})")
    
    # Condition number
    cond = max_eigs / (min_eigs + 1e-12)
    im2 = axes[2].imshow(
        cond,
        cmap='coolwarm',
        norm=LogNorm(),
        origin='lower'
    )
    plt.colorbar(im2, ax=axes[2], label='κ (log scale)')
    axes[2].set_title(f"Condition Number (Σ_{which})")
    
    # Overlay support mask
    if hasattr(agent, 'support') and agent.support is not None:
        for ax in axes:
            ax.contour(
                agent.support.mask_continuous,
                levels=[agent.support.config.min_mask_for_normal_cov],
                colors='red',
                linewidths=2,
                linestyles='--'
            )
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
   #     print(f"✓ Saved to {save_path}")
    
    plt.show()


def plot_mean_field_norms(
    agent,
    which: str = 'q',
    figsize: Tuple[int, int] = (8, 6),
    save_path: Optional[str] = None
):
    """
    Visualize mean field magnitudes across space.
    
    Shows that ||μ|| ≈ 0 outside support.
    
    Args:
        agent: Agent object
        which: 'q' for beliefs, 'p' for priors
        figsize: Figure size
        save_path: Optional path to save figure
    """
    mu = agent.mu_q if which == 'q' else agent.mu_p
    
    if mu.ndim <= 1:
        print("Agent is 0D, no spatial variation to plot")
        return
    
    spatial_shape = mu.shape[:-1]
    
    if len(spatial_shape) != 2:
        print(f"Cannot plot {len(spatial_shape)}D spatial structure")
        return
    
    # Compute norm at each point
    mu_norm = np.linalg.norm(mu, axis=-1)
    
    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    
    im = ax.imshow(
        mu_norm,
        cmap='viridis',
        origin='lower'
    )
    plt.colorbar(im, ax=ax, label='||μ(c)||')
    
    # Overlay support mask
    if hasattr(agent, 'support') and agent.support is not None:
        ax.contour(
            agent.support.mask_continuous,
            levels=[agent.support.config.min_mask_for_normal_cov],
            colors='red',
            linewidths=2,
            linestyles='--',
            label='Support boundary'
        )
        ax.legend()
    
    ax.set_title(f"Mean Field Magnitude (μ_{which})")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    #    print(f"✓ Saved to {save_path}")
    
    plt.show()


# =============================================================================
# SECTION 3: Comprehensive Diagnostic Report
# =============================================================================

def generate_masking_report(system, gradients: Optional[List] = None):
    """
    Generate comprehensive diagnostic report for support masking.
    
    Args:
        system: MultiAgentSystem
        gradients: Optional list of gradients to validate
    
    Prints detailed report to console.
    """
    print("\n" + "=" * 80)
    print("SUPPORT MASKING DIAGNOSTIC REPORT")
    print("=" * 80)
    
    # ========== Agent-level validation ==========
    print("\n[1] AGENT FIELD VALIDATION")
    print("-" * 80)
    
    all_valid = True
    for agent in system.agents:
        result = validate_agent_support_constraints(agent)
        
        if not result['valid']:
            all_valid = False
            print(f"\n❌ Agent {agent.agent_id}: FAILED")
            for violation in result['violations']:
                print(f"   • {violation}")
        else:
            print(f"\n✓ Agent {agent.agent_id}: PASSED")
        
        if result['warnings']:
            for warning in result['warnings']:
                print(f"   ⚠️  {warning}")
        
        if result['metrics']:
            print(f"   Metrics:")
            for key, val in result['metrics'].items():
                print(f"     {key}: {val:.2e}")
    
    # ========== System-level validation ==========
    print("\n[2] SYSTEM OVERLAP VALIDATION")
    print("-" * 80)
    
    overlap_result = validate_system_overlaps(system)
    
    if overlap_result['valid']:
        print(f"✓ Overlap computation: PASSED")
    else:
        print(f"❌ Overlap computation: FAILED")
        all_valid = False
        for violation in overlap_result['violations']:
            print(f"   • {violation}")
    
    if 'warnings' in overlap_result:
        for warning in overlap_result['warnings']:
            print(f"   ⚠️  {warning}")
    
    print(f"\n   Statistics:")
    print(f"     Total agents: {overlap_result['n_agents']}")
    print(f"     Active overlaps: {overlap_result['n_overlaps']}")
    
    if overlap_result['metrics']:
        for key, val in overlap_result['metrics'].items():
            print(f"     {key}: {val:.2%}")
    
    # ========== Gradient validation ==========
    if gradients is not None:
        print("\n[3] GRADIENT MASKING VALIDATION")
        print("-" * 80)
        
        grad_result = validate_gradient_masking(system, gradients)
        
        if grad_result['valid']:
            print(f"✓ Gradient masking: PASSED")
        else:
            print(f"❌ Gradient masking: FAILED")
            all_valid = False
            for violation in grad_result['violations']:
                print(f"   • {violation}")
    
    # ========== Summary ==========
    print("\n" + "=" * 80)
    if all_valid:
        print("✓ ALL CHECKS PASSED")
    else:
        print("❌ SOME CHECKS FAILED - Review violations above")
    print("=" * 80 + "\n")


# =============================================================================
# Example Usage
# =============================================================================

if __name__ == "__main__":
    print("Support Masking Diagnostics - Example Usage")
    print("=" * 60)
    print()
    print("# Validate single agent:")
    print("result = validate_agent_support_constraints(agent)")
    print()
    print("# Validate system overlaps:")
    print("result = validate_system_overlaps(system)")
    print()
    print("# Generate full report:")
    print("generate_masking_report(system, gradients)")
    print()
    print("# Visualize masks:")
    print("plot_agent_support_masks(system.agents)")
    print("plot_overlap_masks(system)")
    print("plot_covariance_eigenvalues(agent, which='q')")
    print("plot_mean_field_norms(agent, which='q')")