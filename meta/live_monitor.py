# -*- coding: utf-8 -*-
"""
Created on Wed Nov 19 21:07:39 2025

@author: chris and christine
"""

"""
Live Monitoring Dashboard for Meta-Agent Evolution

Provides real-time visualization and monitoring of hierarchical evolution
using matplotlib with live updates.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.gridspec import GridSpec
from typing import Dict, List, Optional, Callable
from collections import deque
import time

from meta.emergence import MultiScaleSystem
from meta.participatory_diagnostics import ParticipatoryDiagnostics


class LiveMonitor:
    """
    Real-time monitoring dashboard for meta-agent evolution.

    Displays:
    - Scale occupancy over time
    - Energy landscape
    - Condensation events
    - System health indicators
    """

    def __init__(self,
                 system: MultiScaleSystem,
                 diagnostics: Optional[ParticipatoryDiagnostics] = None,
                 history_length: int = 100,
                 update_interval: int = 100):  # milliseconds
        """
        Initialize live monitor.

        Args:
            system: MultiScaleSystem to monitor
            diagnostics: Optional diagnostics tracker
            history_length: Number of timesteps to display
            update_interval: Update interval in milliseconds
        """
        self.system = system
        self.diagnostics = diagnostics
        self.history_length = history_length
        self.update_interval = update_interval

        # History buffers
        self.time_history = deque(maxlen=history_length)
        self.scale_counts = {i: deque(maxlen=history_length) for i in range(10)}
        self.energy_history = deque(maxlen=history_length)
        self.condensation_times = []
        self.condensation_scales = []

        # Animation state
        self.animation: Optional[FuncAnimation] = None
        self.paused = False

        # Setup figure
        self._setup_figure()

    def _setup_figure(self):
        """Setup matplotlib figure and subplots."""
        self.fig = plt.figure(figsize=(16, 10))
        self.fig.suptitle('Live Meta-Agent Evolution Monitor', fontsize=16, fontweight='bold')

        # Create grid layout
        gs = GridSpec(3, 3, figure=self.fig, hspace=0.3, wspace=0.3)

        # Main plots
        self.ax_occupancy = self.fig.add_subplot(gs[0, :])  # Top: scale occupancy
        self.ax_energy = self.fig.add_subplot(gs[1, :2])     # Middle-left: energy
        self.ax_status = self.fig.add_subplot(gs[1, 2])      # Middle-right: status
        self.ax_condensation = self.fig.add_subplot(gs[2, :2])  # Bottom-left: condensations
        self.ax_hierarchy = self.fig.add_subplot(gs[2, 2])   # Bottom-right: hierarchy stats

        # Initialize plots
        self._init_plots()

    def _init_plots(self):
        """Initialize all plot elements."""
        # Occupancy plot
        self.ax_occupancy.set_title('Agent Population by Scale', fontweight='bold')
        self.ax_occupancy.set_xlabel('Time')
        self.ax_occupancy.set_ylabel('Active Agents')
        self.ax_occupancy.grid(alpha=0.3)
        self.occupancy_lines = {}

        # Energy plot
        self.ax_energy.set_title('Total System Energy', fontweight='bold')
        self.ax_energy.set_xlabel('Time')
        self.ax_energy.set_ylabel('Energy')
        self.ax_energy.grid(alpha=0.3)
        self.energy_line, = self.ax_energy.plot([], [], 'b-', linewidth=2)

        # Status indicators
        self.ax_status.set_title('System Status', fontweight='bold')
        self.ax_status.axis('off')
        self.status_text = self.ax_status.text(0.1, 0.5, '', fontsize=10,
                                              verticalalignment='center',
                                              family='monospace')

        # Condensation plot
        self.ax_condensation.set_title('Condensation Events', fontweight='bold')
        self.ax_condensation.set_xlabel('Time')
        self.ax_condensation.set_ylabel('Target Scale')
        self.ax_condensation.grid(alpha=0.3)
        self.condensation_scatter = self.ax_condensation.scatter([], [], s=100, alpha=0.6)

        # Hierarchy stats
        self.ax_hierarchy.set_title('Hierarchy Stats', fontweight='bold')
        self.ax_hierarchy.axis('off')
        self.hierarchy_text = self.ax_hierarchy.text(0.1, 0.5, '', fontsize=10,
                                                    verticalalignment='center',
                                                    family='monospace')

    def capture_state(self):
        """Capture current system state."""
        current_time = self.system.current_time
        self.time_history.append(current_time)

        # Count agents by scale
        for scale, agents in self.system.agents.items():
            n_active = sum(1 for a in agents if a.is_active)
            self.scale_counts[scale].append(n_active)

        # Pad scales that don't exist
        for scale in self.scale_counts.keys():
            if scale not in self.system.agents:
                self.scale_counts[scale].append(0)

        # Capture energy (if diagnostics available)
        if self.diagnostics and self.diagnostics.energy_snapshots:
            latest_snap = self.diagnostics.energy_snapshots[-1]
            total_energy = sum(s['E_total'] for s in latest_snap['by_scale'].values())
            self.energy_history.append(total_energy)
        else:
            self.energy_history.append(0)

        # Track condensation events
        for event in self.system.condensation_events:
            if event['time'] not in [t for t in self.condensation_times]:
                self.condensation_times.append(event['time'])
                self.condensation_scales.append(event['target_scale'])

    def update(self, frame):
        """Update all plots (called by animation)."""
        if self.paused:
            return

        # Capture current state
        self.capture_state()

        if not self.time_history:
            return

        times = list(self.time_history)

        # Update occupancy plot
        for scale, counts in self.scale_counts.items():
            if any(counts):  # Only plot scales that exist
                if scale not in self.occupancy_lines:
                    self.occupancy_lines[scale], = self.ax_occupancy.plot(
                        [], [], 'o-', label=f'ζ={scale}', linewidth=2
                    )

                self.occupancy_lines[scale].set_data(times, list(counts))

        self.ax_occupancy.relim()
        self.ax_occupancy.autoscale_view()
        self.ax_occupancy.legend(loc='upper left')

        # Update energy plot
        if self.energy_history:
            self.energy_line.set_data(times, list(self.energy_history))
            self.ax_energy.relim()
            self.ax_energy.autoscale_view()

        # Update status text
        status_str = self._format_status()
        self.status_text.set_text(status_str)

        # Update condensation scatter
        if self.condensation_times:
            self.condensation_scatter.set_offsets(
                np.c_[self.condensation_times, self.condensation_scales]
            )
            self.ax_condensation.relim()
            self.ax_condensation.autoscale_view()

        # Update hierarchy stats
        hierarchy_str = self._format_hierarchy_stats()
        self.hierarchy_text.set_text(hierarchy_str)

    def _format_status(self) -> str:
        """Format status indicators as string."""
        total_agents = sum(len(agents) for agents in self.system.agents.values())
        total_active = sum(
            len([a for a in agents if a.is_active])
            for agents in self.system.agents.values()
        )
        total_meta = sum(
            len([a for a in agents if a.is_meta])
            for agents in self.system.agents.values()
        )

        status = f"""
Time:         {self.system.current_time}
Total Agents: {total_agents}
Active:       {total_active}
Meta-agents:  {total_meta}
Condensations:{len(self.system.condensation_events)}
Max Scale:    {max(self.system.agents.keys()) if self.system.agents else 0}
        """
        return status.strip()

    def _format_hierarchy_stats(self) -> str:
        """Format hierarchy statistics as string."""
        if not self.system.condensation_events:
            return "No condensations yet"

        # Compute statistics
        coherences = [e['coherence']['belief'] for e in self.system.condensation_events]
        cluster_sizes = [e['n_constituents'] for e in self.system.condensation_events]

        stats = f"""
Condensations: {len(self.system.condensation_events)}

Avg Coherence: {np.mean(coherences):.3f}
Min Coherence: {np.min(coherences):.3f}
Max Coherence: {np.max(coherences):.3f}

Avg Cluster:   {np.mean(cluster_sizes):.1f}
Min Cluster:   {np.min(cluster_sizes)}
Max Cluster:   {np.max(cluster_sizes)}
        """
        return stats.strip()

    def start(self, update_callback: Optional[Callable] = None):
        """
        Start live monitoring.

        Args:
            update_callback: Optional callback to run before each update
                            Should return True to continue, False to stop
        """
        def animate(frame):
            # Run user callback
            if update_callback:
                should_continue = update_callback()
                if not should_continue:
                    self.stop()
                    return

            # Update plots
            self.update(frame)

        self.animation = FuncAnimation(
            self.fig,
            animate,
            interval=self.update_interval,
            blit=False
        )

        plt.show()

    def stop(self):
        """Stop monitoring."""
        if self.animation:
            self.animation.event_source.stop()

    def pause(self):
        """Pause/resume monitoring."""
        self.paused = not self.paused

    def save_snapshot(self, filepath: str):
        """Save current visualization to file."""
        self.fig.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"Saved snapshot to {filepath}")


class StepwiseMonitor:
    """
    Stepwise monitor for manual control (non-animated).

    Useful for integrating into existing evolution loops.
    """

    def __init__(self,
                 system: MultiScaleSystem,
                 diagnostics: Optional[ParticipatoryDiagnostics] = None,
                 update_every: int = 10):
        """
        Initialize stepwise monitor.

        Args:
            system: MultiScaleSystem to monitor
            diagnostics: Optional diagnostics tracker
            update_every: Update visualization every N steps
        """
        self.system = system
        self.diagnostics = diagnostics
        self.update_every = update_every
        self.step_count = 0

        self.live_monitor = LiveMonitor(system, diagnostics)

    def step(self):
        """
        Call this after each evolution step.

        Updates visualization every update_every steps.
        """
        self.step_count += 1

        if self.step_count % self.update_every == 0:
            self.live_monitor.capture_state()
            self.live_monitor.update(self.step_count)
            plt.pause(0.01)  # Force redraw

    def show(self):
        """Display the monitor (call once before evolution loop)."""
        plt.ion()  # Interactive mode
        self.live_monitor.fig.show()

    def close(self):
        """Close the monitor."""
        plt.close(self.live_monitor.fig)

    def save(self, filepath: str):
        """Save current state."""
        self.live_monitor.save_snapshot(filepath)


# ============================================================================
# Convenience Functions
# ============================================================================

def monitor_evolution(system: MultiScaleSystem,
                     evolution_step_fn: Callable,
                     n_steps: int,
                     diagnostics: Optional[ParticipatoryDiagnostics] = None,
                     update_every: int = 10):
    """
    Monitor evolution with stepwise updates.

    Args:
        system: MultiScaleSystem to evolve
        evolution_step_fn: Function that performs one evolution step
                          Should accept no arguments
        n_steps: Number of steps to run
        diagnostics: Optional diagnostics tracker
        update_every: Update visualization every N steps

    Example:
        >>> from meta.emergence import MultiScaleSystem
        >>> from meta.hierarchical_evolution import HierarchicalEvolutionEngine
        >>>
        >>> system = MultiScaleSystem(agents={...})
        >>> engine = HierarchicalEvolutionEngine(system)
        >>>
        >>> monitor_evolution(
        ...     system,
        ...     engine.step,
        ...     n_steps=100,
        ...     update_every=5
        ... )
    """
    monitor = StepwiseMonitor(system, diagnostics, update_every)
    monitor.show()

    try:
        print(f"Running evolution for {n_steps} steps...")
        print("Close the plot window to stop early.\n")

        for i in range(n_steps):
            evolution_step_fn()

            if diagnostics:
                diagnostics.capture_snapshot()

            monitor.step()

            # Check if window was closed
            if not plt.fignum_exists(monitor.live_monitor.fig.number):
                print("\nMonitor closed. Stopping evolution.")
                break

        print("\nEvolution complete!")

    except KeyboardInterrupt:
        print("\nInterrupted by user.")

    finally:
        # Save final state
        monitor.save('./final_state.png')
        print("Final state saved to ./final_state.png")

    # Keep window open
    plt.ioff()
    plt.show()


if __name__ == '__main__':
    """Demo of live monitoring."""
    print("Live Monitor Demo")
    print("This requires a running evolution loop.")
    print("See examples/meta_agent_analysis_demo.py for complete example.")