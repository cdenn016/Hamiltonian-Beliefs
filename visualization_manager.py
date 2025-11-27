"""
Unified Visualization Manager
==============================

Coordinates all visualization modules for comprehensive analysis:
- Meta-agent visualization (meta/visualization.py)
- Energy landscapes (meta/energy_visualization.py)
- Spatial field imaging (meta/agent_field_visualizer.py)
- Live monitoring (meta/live_monitor.py)
- Pullback geometry (geometry/pullback_metrics.py)
- Analysis plots (analysis/plots/*)

This manager provides a unified interface for creating comprehensive
analysis reports that integrate geometry, dynamics, and emergence.
"""

import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, List
import matplotlib.pyplot as plt


class VisualizationManager:
    """
    Unified manager for all visualization modules.

    Coordinates:
    - Meta-scale visualization
    - Energy landscape analysis
    - Spatial field visualization
    - Pullback geometry visualization
    - Standard analysis plots
    """

    def __init__(self,
                 output_dir: Path,
                 dpi: int = 150,
                 style: str = 'seaborn-v0_8-darkgrid'):
        """
        Initialize visualization manager.

        Args:
            output_dir: Directory for saving all visualizations
            dpi: Resolution for saved figures
            style: Matplotlib style to use
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dpi = dpi

        # Set matplotlib style if available
        try:
            plt.style.use(style)
        except:
            # Fallback to default if style not available
            pass

        # Lazy-load visualization modules (import only when needed)
        self._meta_viz = None
        self._energy_viz = None
        self._field_viz = None
        self._live_monitor = None
        self._pullback_viz = None

        print(f"VisualizationManager initialized")
        print(f"  Output directory: {self.output_dir}")
        print(f"  DPI: {self.dpi}")

    def create_full_report(self,
                          system,
                          history=None,
                          run_config: Optional[Dict] = None,
                          include_meta: bool = True,
                          include_energy: bool = True,
                          include_fields: bool = True,
                          include_pullback: bool = True,
                          include_analysis: bool = True):
        """
        Create comprehensive visualization report.

        Args:
            system: MultiAgentSystem or MultiScaleSystem
            history: Training history (dict or TrainingHistory object)
            run_config: Optional configuration dict
            include_meta: Generate meta-agent visualization
            include_energy: Generate energy landscape plots
            include_fields: Generate spatial field visualizations
            include_pullback: Generate pullback geometry visualization
            include_analysis: Generate standard analysis plots

        Returns:
            Dictionary of generated file paths
        """
        print("\n" + "="*70)
        print("GENERATING COMPREHENSIVE VISUALIZATION REPORT")
        print("="*70)

        generated_files = {}

        # Meta-agent visualization (for hierarchical systems)
        if include_meta and self._is_hierarchical(system):
            print("\n📊 Generating meta-agent visualization...")
            try:
                meta_files = self.visualize_meta_agents(system, history)
                generated_files['meta'] = meta_files
                print(f"  ✓ Generated {len(meta_files)} meta-agent plots")
            except Exception as e:
                print(f"  ⚠️  Meta visualization failed: {e}")

        # Energy landscape visualization
        if include_energy and history is not None:
            print("\n⚡ Generating energy landscape visualization...")
            try:
                energy_files = self.visualize_energy_landscape(history, system)
                generated_files['energy'] = energy_files
                print(f"  ✓ Generated {len(energy_files)} energy plots")
            except Exception as e:
                print(f"  ⚠️  Energy visualization failed: {e}")

        # Spatial field visualization
        if include_fields:
            print("\n🗺️  Generating spatial field visualization...")
            try:
                field_files = self.visualize_fields(system, history)
                generated_files['fields'] = field_files
                print(f"  ✓ Generated {len(field_files)} field plots")
            except Exception as e:
                print(f"  ⚠️  Field visualization failed: {e}")

        # Pullback geometry visualization
        if include_pullback and self._has_pullback_geometry(system):
            print("\n🌌 Generating pullback geometry visualization...")
            try:
                pullback_files = self.visualize_pullback_geometry(system, history)
                generated_files['pullback'] = pullback_files
                print(f"  ✓ Generated {len(pullback_files)} pullback geometry plots")
            except Exception as e:
                print(f"  ⚠️  Pullback visualization failed: {e}")

        # Standard analysis plots
        if include_analysis and history is not None:
            print("\n📈 Generating standard analysis plots...")
            try:
                analysis_files = self.create_analysis_plots(history, system)
                generated_files['analysis'] = analysis_files
                print(f"  ✓ Generated {len(analysis_files)} analysis plots")
            except Exception as e:
                print(f"  ⚠️  Analysis plots failed: {e}")

        print("\n" + "="*70)
        print(f"✓ Report generation complete")
        print(f"  Total files: {sum(len(v) if isinstance(v, list) else 1 for v in generated_files.values())}")
        print(f"  Output: {self.output_dir}")
        print("="*70 + "\n")

        return generated_files

    def visualize_meta_agents(self, system, history=None) -> List[Path]:
        """
        Generate meta-agent visualization for hierarchical systems.

        Uses meta/visualization.py module.
        """
        if self._meta_viz is None:
            from meta import visualization as meta_viz
            self._meta_viz = meta_viz

        output_files = []
        meta_dir = self.output_dir / "meta_agents"
        meta_dir.mkdir(exist_ok=True)

        # Generate meta-agent hierarchy visualization
        if hasattr(self._meta_viz, 'visualize_meta_agent_hierarchy'):
            fig_path = meta_dir / "hierarchy.png"
            self._meta_viz.visualize_meta_agent_hierarchy(
                system,
                save_path=fig_path,
                dpi=self.dpi
            )
            output_files.append(fig_path)

        # Generate consensus visualization
        if hasattr(self._meta_viz, 'visualize_consensus_detection'):
            fig_path = meta_dir / "consensus.png"
            self._meta_viz.visualize_consensus_detection(
                system,
                save_path=fig_path,
                dpi=self.dpi
            )
            output_files.append(fig_path)

        return output_files

    def visualize_energy_landscape(self, history, system=None) -> List[Path]:
        """
        Generate energy landscape visualization.

        Uses meta/energy_visualization.py module.
        """
        if self._energy_viz is None:
            from meta import energy_visualization as energy_viz
            self._energy_viz = energy_viz

        output_files = []
        energy_dir = self.output_dir / "energy"
        energy_dir.mkdir(exist_ok=True)

        # Generate energy evolution plot
        if hasattr(self._energy_viz, 'plot_energy_evolution'):
            fig_path = energy_dir / "evolution.png"
            self._energy_viz.plot_energy_evolution(
                history,
                save_path=fig_path,
                dpi=self.dpi
            )
            output_files.append(fig_path)

        return output_files

    def visualize_fields(self, system, history=None) -> List[Path]:
        """
        Generate spatial field visualizations (mu, Sigma, phi).

        Uses meta/agent_field_visualizer.py module.
        """
        if self._field_viz is None:
            from meta import agent_field_visualizer as field_viz
            self._field_viz = field_viz

        output_files = []
        field_dir = self.output_dir / "fields"
        field_dir.mkdir(exist_ok=True)

        # Generate field visualizations based on dimensionality
        if hasattr(self._field_viz, 'visualize_all_fields'):
            try:
                paths = self._field_viz.visualize_all_fields(
                    system,
                    output_dir=field_dir,
                    dpi=self.dpi
                )
                output_files.extend(paths)
            except Exception as e:
                print(f"    Field visualization error: {e}")

        return output_files

    def visualize_pullback_geometry(self, system, history=None) -> List[Path]:
        """
        Generate pullback geometry visualizations (emergent spacetime).

        Uses geometry/pullback_metrics.py module.
        """
        if self._pullback_viz is None:
            try:
                from geometry import pullback_metrics
                self._pullback_viz = pullback_metrics
            except ImportError:
                print("    ⚠️  Pullback geometry module not available")
                return []

        output_files = []
        pullback_dir = self.output_dir / "pullback_geometry"
        pullback_dir.mkdir(exist_ok=True)

        # Generate pullback metric visualization
        if hasattr(self._pullback_viz, 'visualize_pullback_metrics'):
            try:
                paths = self._pullback_viz.visualize_pullback_metrics(
                    system,
                    output_dir=pullback_dir,
                    dpi=self.dpi
                )
                output_files.extend(paths)
            except Exception as e:
                print(f"    Pullback visualization error: {e}")

        return output_files

    def create_analysis_plots(self, history, system=None) -> List[Path]:
        """
        Generate standard analysis plots.

        Uses analysis/plots/* modules.
        """
        from analysis.core import normalize_history

        output_files = []
        analysis_dir = self.output_dir / "analysis"
        analysis_dir.mkdir(exist_ok=True)

        # Normalize history for plotting
        hist_dict = normalize_history(history)
        if hist_dict is None:
            return output_files

        # Energy components (if available)
        try:
            from analysis.plots import energy
            fig_path = analysis_dir / "energy_components.png"
            energy.plot_energy_components(hist_dict, fig_path)
            output_files.append(fig_path)
        except Exception as e:
            print(f"    Energy plot error: {e}")

        # Mu tracking (if available)
        try:
            from analysis.plots import mu_tracking
            from analysis.core import get_mu_tracker

            mu_tracker = get_mu_tracker(history)
            if mu_tracker is not None:
                fig_path = analysis_dir / "mu_tracking.png"
                mu_tracking.plot_mu_summary(mu_tracker, fig_path)
                output_files.append(fig_path)
        except Exception as e:
            print(f"    Mu tracking plot error: {e}")

        return output_files

    def _is_hierarchical(self, system) -> bool:
        """Check if system is hierarchical (MultiScaleSystem)."""
        return hasattr(system, 'agents') and isinstance(system.agents, dict)

    def _has_pullback_geometry(self, system) -> bool:
        """Check if system has pullback geometry tracking."""
        return hasattr(system, 'geometry_tracker') or hasattr(system, 'pullback_metrics')

    def close(self):
        """Clean up resources."""
        plt.close('all')


def create_visualization_report(system,
                                history=None,
                                output_dir: Path = Path("visualization_report"),
                                **kwargs):
    """
    Convenience function to create a full visualization report.

    Args:
        system: MultiAgentSystem or MultiScaleSystem
        history: Training history
        output_dir: Output directory
        **kwargs: Additional arguments for VisualizationManager.create_full_report()

    Returns:
        Dictionary of generated file paths
    """
    manager = VisualizationManager(output_dir)
    try:
        return manager.create_full_report(system, history, **kwargs)
    finally:
        manager.close()