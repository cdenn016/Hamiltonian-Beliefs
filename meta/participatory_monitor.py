"""
Participatory "It from Bit" Universe Monitor

Validates that the participatory dynamics are wired up correctly:
1. Scale-0 agents condense into meta-agents properly
2. Priors evolve due to meta-agent activity above
3. System is in non-equilibrium dynamical regime
4. Level cap prevents runaway emergence
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict
import warnings

from meta.emergence import MultiScaleSystem, HierarchicalAgent
from meta.consensus import ConsensusDetector
from math_utils.numerical_utils import kl_gaussian


@dataclass
class ParticipatorySample:
    """Single timestep snapshot of participatory dynamics"""
    step: int
    num_agents_per_scale: Dict[int, int]
    num_meta_agents_formed: int
    prior_change_magnitudes: Dict[int, List[float]]  # scale -> list of KL divergences
    energy_gradient_norms: Dict[int, List[float]]  # scale -> list of gradient magnitudes
    consensus_clusters: List[List[int]]  # clusters ready to condense
    parent_child_links: int  # number of active parent->child connections

    # Non-equilibrium indicators
    energy_flux: float  # rate of energy change
    information_flux: float  # rate of information accumulation
    gradient_variance: float  # variance of gradients across scales


@dataclass
class ParticipatoryStatistics:
    """Aggregated statistics over monitoring period"""
    samples: List[ParticipatorySample] = field(default_factory=list)

    # Condensation tracking
    total_meta_agents_formed: int = 0
    condensation_events: List[Tuple[int, int, int]] = field(default_factory=list)  # (step, scale, num_agents)

    # Prior evolution tracking
    prior_evolution_detected: Dict[int, bool] = field(default_factory=dict)  # scale -> bool
    avg_prior_change_per_scale: Dict[int, float] = field(default_factory=dict)

    # Non-equilibrium indicators
    is_non_equilibrium: bool = False
    equilibrium_score: float = 0.0  # 0=equilibrium, 1=far-from-equilibrium

    # Level cap tracking
    max_scale_reached: int = 0
    level_cap_hit: bool = False


class ParticipatoryMonitor:
    """
    Monitors and validates participatory "it from bit" dynamics

    Tracks:
    - Agent condensation from scale-0 upward
    - Prior evolution due to top-down meta-agent influence
    - Non-equilibrium dynamical regime
    - Level emergence cap enforcement
    """

    def __init__(
        self,
        system: MultiScaleSystem,
        consensus_detector: Optional[ConsensusDetector] = None,
        check_interval: int = 10,
        prior_change_threshold: float = 1e-4,  # KL divergence threshold
        non_eq_threshold: float = 1e-3,  # gradient norm threshold
    ):
        self.system = system
        self.consensus_detector = consensus_detector
        self.check_interval = check_interval
        self.prior_change_threshold = prior_change_threshold
        self.non_eq_threshold = non_eq_threshold

        self.stats = ParticipatoryStatistics()
        self._last_check_step = 0

        # Cache previous priors for change detection
        self._previous_priors: Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray]] = {}
        self._previous_energy: Optional[float] = None
        self._previous_info: Optional[float] = None

    def take_snapshot(self, step: int, force: bool = False) -> Optional[ParticipatorySample]:
        """
        Take a snapshot of participatory dynamics

        Args:
            step: Current training step
            force: Force snapshot even if not at check_interval

        Returns:
            ParticipatorySample if snapshot taken, None otherwise
        """
        if not force and (step - self._last_check_step) < self.check_interval:
            return None

        self._last_check_step = step

        # Count agents per scale
        num_agents_per_scale = {}
        for scale, agents in self.system.agents.items():
            num_agents_per_scale[scale] = len(agents)

        # Detect consensus clusters ready to condense
        consensus_clusters = []
        if self.consensus_detector is not None:
            # Note: ConsensusDetector.find_consensus_clusters() expects a MultiAgentSystem,
            # but we have a MultiScaleSystem. For now, skip consensus detection in monitor.
            # Consensus detection should be done externally in the evolution loop.
            pass

        # Track prior changes
        prior_change_magnitudes = defaultdict(list)
        for scale, agents in self.system.agents.items():
            for i, agent in enumerate(agents):
                key = (scale, i)
                current_prior = (agent.mu_p.copy(), agent.L_p.copy())

                if key in self._previous_priors:
                    prev_mu_p, prev_L_p = self._previous_priors[key]
                    # Compute KL divergence between old and new prior
                    kl_change = kl_gaussian(
                        prev_mu_p, prev_L_p @ prev_L_p.T,
                        agent.mu_p, agent.L_p @ agent.L_p.T
                    )
                    prior_change_magnitudes[scale].append(kl_change)

                self._previous_priors[key] = current_prior

        # Track gradient norms (proxy for non-equilibrium)
        energy_gradient_norms = defaultdict(list)
        for scale, agents in self.system.agents.items():
            for agent in agents:
                # Compute approximate gradient magnitude
                # Use difference between belief and prior as proxy
                grad_mu = np.linalg.norm(agent.mu_q - agent.mu_p)
                energy_gradient_norms[scale].append(grad_mu)

        # Count parent-child links
        parent_child_links = 0
        for scale, agents in self.system.agents.items():
            for agent in agents:
                if hasattr(agent, 'parent_meta') and agent.parent_meta is not None:
                    parent_child_links += 1

        # Compute non-equilibrium indicators
        total_energy = self._compute_total_energy()
        total_info = self._compute_total_information()

        energy_flux = 0.0
        if self._previous_energy is not None:
            energy_flux = abs(total_energy - self._previous_energy)
        self._previous_energy = total_energy

        information_flux = 0.0
        if self._previous_info is not None:
            information_flux = abs(total_info - self._previous_info)
        self._previous_info = total_info

        # Compute gradient variance across scales
        all_gradients = []
        for grad_list in energy_gradient_norms.values():
            all_gradients.extend(grad_list)
        gradient_variance = np.var(all_gradients) if all_gradients else 0.0

        # Count newly formed meta-agents
        num_meta_agents_formed = sum(
            1 for scale in range(1, max(self.system.agents.keys()) + 1) if scale in self.system.agents
            for agent in self.system.agents[scale]
            if hasattr(agent, 'is_meta') and agent.is_meta
        )

        sample = ParticipatorySample(
            step=step,
            num_agents_per_scale=num_agents_per_scale,
            num_meta_agents_formed=num_meta_agents_formed,
            prior_change_magnitudes=dict(prior_change_magnitudes),
            energy_gradient_norms=dict(energy_gradient_norms),
            consensus_clusters=consensus_clusters,
            parent_child_links=parent_child_links,
            energy_flux=energy_flux,
            information_flux=information_flux,
            gradient_variance=gradient_variance,
        )

        self.stats.samples.append(sample)
        return sample

    def analyze_condensation(self) -> Dict:
        """
        Analyze whether scale-0 agents are properly condensing into meta-agents

        Returns:
            Dictionary with condensation analysis results
        """
        if not self.stats.samples:
            return {"status": "no_data"}

        # Track meta-agent formation over time
        meta_agent_counts = [s.num_meta_agents_formed for s in self.stats.samples]

        # Check if meta-agents are being formed
        condensation_occurring = any(count > 0 for count in meta_agent_counts)

        # Track consensus clusters vs actual condensation
        consensus_opportunities = sum(len(s.consensus_clusters) for s in self.stats.samples)

        # Check if scale-0 is producing scale-1
        scale_0_to_1_flow = False
        if len(self.stats.samples) > 1:
            for i in range(1, len(self.stats.samples)):
                prev = self.stats.samples[i-1]
                curr = self.stats.samples[i]
                if 1 in curr.num_agents_per_scale and curr.num_agents_per_scale[1] > prev.num_agents_per_scale.get(1, 0):
                    scale_0_to_1_flow = True
                    break

        return {
            "status": "ok" if condensation_occurring else "no_condensation",
            "condensation_occurring": condensation_occurring,
            "total_meta_agents": meta_agent_counts[-1] if meta_agent_counts else 0,
            "meta_agent_trajectory": meta_agent_counts,
            "consensus_opportunities": consensus_opportunities,
            "scale_0_to_1_flow": scale_0_to_1_flow,
        }

    def analyze_prior_evolution(self) -> Dict:
        """
        Analyze whether priors are evolving due to meta-agent activity

        Returns:
            Dictionary with prior evolution analysis results
        """
        if not self.stats.samples:
            return {"status": "no_data"}

        # Check if priors are changing at each scale
        prior_evolution = {}
        avg_changes = {}

        for sample in self.stats.samples:
            for scale, changes in sample.prior_change_magnitudes.items():
                if scale not in prior_evolution:
                    prior_evolution[scale] = []
                prior_evolution[scale].extend(changes)

        for scale, changes in prior_evolution.items():
            if changes:
                avg_change = np.mean(changes)
                avg_changes[scale] = avg_change

                # Priors are evolving if average change exceeds threshold
                if avg_change > self.prior_change_threshold:
                    self.stats.prior_evolution_detected[scale] = True
                    self.stats.avg_prior_change_per_scale[scale] = avg_change

        # Check for top-down influence
        top_down_influence = False
        if len(self.stats.samples) > 0:
            latest = self.stats.samples[-1]
            # If we have parent-child links and priors are changing, top-down is working
            if latest.parent_child_links > 0 and any(self.stats.prior_evolution_detected.values()):
                top_down_influence = True

        return {
            "status": "ok" if top_down_influence else "no_evolution",
            "prior_evolution_detected": dict(self.stats.prior_evolution_detected),
            "avg_prior_changes": avg_changes,
            "top_down_influence": top_down_influence,
            "parent_child_links": latest.parent_child_links if self.stats.samples else 0,
        }

    def analyze_non_equilibrium(self) -> Dict:
        """
        Analyze whether system is in non-equilibrium dynamical regime

        Returns:
            Dictionary with non-equilibrium analysis results
        """
        if len(self.stats.samples) < 2:
            return {"status": "insufficient_data"}

        # Compute indicators of non-equilibrium
        energy_fluxes = [s.energy_flux for s in self.stats.samples[1:]]  # Skip first (no previous)
        info_fluxes = [s.information_flux for s in self.stats.samples[1:]]
        gradient_variances = [s.gradient_variance for s in self.stats.samples]

        # Check for significant fluxes
        avg_energy_flux = np.mean(energy_fluxes) if energy_fluxes else 0.0
        avg_info_flux = np.mean(info_fluxes) if info_fluxes else 0.0
        avg_gradient_var = np.mean(gradient_variances) if gradient_variances else 0.0

        # Non-equilibrium if fluxes exceed threshold
        is_non_eq = (
            avg_energy_flux > self.non_eq_threshold or
            avg_info_flux > self.non_eq_threshold or
            avg_gradient_var > self.non_eq_threshold
        )

        # Compute equilibrium score (0=equilibrium, 1=far-from-equilibrium)
        # Normalize by thresholds and clip to [0, 1]
        equilibrium_score = min(1.0, max(
            avg_energy_flux / self.non_eq_threshold,
            avg_info_flux / self.non_eq_threshold,
            avg_gradient_var / self.non_eq_threshold
        ))

        self.stats.is_non_equilibrium = is_non_eq
        self.stats.equilibrium_score = equilibrium_score

        return {
            "status": "ok",
            "is_non_equilibrium": is_non_eq,
            "equilibrium_score": equilibrium_score,
            "avg_energy_flux": avg_energy_flux,
            "avg_information_flux": avg_info_flux,
            "avg_gradient_variance": avg_gradient_var,
            "trajectory": {
                "energy_flux": energy_fluxes,
                "info_flux": info_fluxes,
                "gradient_variance": gradient_variances,
            }
        }

    def check_level_cap(self, max_levels: Optional[int] = None) -> Dict:
        """
        Check level cap enforcement

        Args:
            max_levels: Maximum allowed levels (if None, use system's max_emergence_levels)

        Returns:
            Dictionary with level cap status
        """
        if max_levels is None:
            max_levels = getattr(self.system, 'max_emergence_levels', None)

        if max_levels is None:
            return {
                "status": "no_cap_set",
                "max_scale_reached": max(self.system.agents.keys()) if self.system.agents else 0,
                "warning": "No level cap configured - runaway emergence possible!"
            }

        max_scale = max(self.system.agents.keys()) if self.system.agents else 0
        self.stats.max_scale_reached = max_scale
        self.stats.level_cap_hit = (max_scale >= max_levels)

        return {
            "status": "ok",
            "max_levels_allowed": max_levels,
            "max_scale_reached": max_scale,
            "level_cap_hit": self.stats.level_cap_hit,
            "levels_remaining": max(0, max_levels - max_scale),
        }

    def validate_participatory_dynamics(self, max_levels: Optional[int] = None) -> Dict:
        """
        Complete validation of participatory "it from bit" universe

        Args:
            max_levels: Maximum allowed emergence levels

        Returns:
            Dictionary with comprehensive validation results
        """
        condensation = self.analyze_condensation()
        prior_evolution = self.analyze_prior_evolution()
        non_equilibrium = self.analyze_non_equilibrium()
        level_cap = self.check_level_cap(max_levels)

        # Overall health check
        all_ok = (
            condensation.get("condensation_occurring", False) and
            prior_evolution.get("top_down_influence", False) and
            non_equilibrium.get("is_non_equilibrium", False) and
            not level_cap.get("level_cap_hit", False)
        )

        return {
            "overall_status": "healthy" if all_ok else "issues_detected",
            "condensation": condensation,
            "prior_evolution": prior_evolution,
            "non_equilibrium": non_equilibrium,
            "level_cap": level_cap,
            "samples_collected": len(self.stats.samples),
        }

    def print_summary(self, max_levels: Optional[int] = None):
        """Print a human-readable summary of participatory dynamics"""
        print("\n" + "="*70)
        print("PARTICIPATORY 'IT FROM BIT' UNIVERSE MONITOR")
        print("="*70)

        validation = self.validate_participatory_dynamics(max_levels)

        print(f"\nOverall Status: {validation['overall_status'].upper()}")
        print(f"Samples Collected: {validation['samples_collected']}")

        print("\n" + "-"*70)
        print("1. AGENT CONDENSATION (Scale-0 → Meta-Agents)")
        print("-"*70)
        cond = validation['condensation']
        print(f"   Status: {'✓ OK' if cond['condensation_occurring'] else '✗ NO CONDENSATION'}")
        print(f"   Total Meta-Agents: {cond['total_meta_agents']}")
        print(f"   Scale-0 → Scale-1 Flow: {'✓ Yes' if cond.get('scale_0_to_1_flow') else '✗ No'}")
        print(f"   Consensus Opportunities: {cond['consensus_opportunities']}")

        print("\n" + "-"*70)
        print("2. PRIOR EVOLUTION (Top-Down Meta-Agent Influence)")
        print("-"*70)
        prior = validation['prior_evolution']
        print(f"   Status: {'✓ OK' if prior['top_down_influence'] else '✗ NO EVOLUTION'}")
        print(f"   Parent-Child Links: {prior['parent_child_links']}")
        if prior['avg_prior_changes']:
            print(f"   Average Prior Changes by Scale:")
            for scale, change in sorted(prior['avg_prior_changes'].items()):
                indicator = "✓" if change > self.prior_change_threshold else "✗"
                print(f"      Scale {scale}: {change:.6f} {indicator}")

        print("\n" + "-"*70)
        print("3. NON-EQUILIBRIUM DYNAMICS")
        print("-"*70)
        neq = validation['non_equilibrium']
        if neq['status'] != 'insufficient_data':
            print(f"   Status: {'✓ NON-EQUILIBRIUM' if neq['is_non_equilibrium'] else '✗ EQUILIBRIUM'}")
            print(f"   Equilibrium Score: {neq['equilibrium_score']:.4f} (0=eq, 1=far-from-eq)")
            print(f"   Avg Energy Flux: {neq['avg_energy_flux']:.6f}")
            print(f"   Avg Information Flux: {neq['avg_information_flux']:.6f}")
            print(f"   Avg Gradient Variance: {neq['avg_gradient_variance']:.6f}")
        else:
            print(f"   Status: INSUFFICIENT DATA")

        print("\n" + "-"*70)
        print("4. EMERGENCE LEVEL CAP")
        print("-"*70)
        cap = validation['level_cap']
        if cap['status'] == 'no_cap_set':
            print(f"   Status: ⚠ WARNING - NO CAP SET")
            print(f"   Max Scale Reached: {cap['max_scale_reached']}")
            print(f"   Warning: {cap['warning']}")
        else:
            print(f"   Status: {'✓ OK' if not cap['level_cap_hit'] else '⚠ CAP HIT'}")
            print(f"   Max Levels Allowed: {cap['max_levels_allowed']}")
            print(f"   Max Scale Reached: {cap['max_scale_reached']}")
            print(f"   Levels Remaining: {cap['levels_remaining']}")

        print("\n" + "="*70)

    def _compute_total_energy(self) -> float:
        """Compute total system energy (proxy)"""
        total = 0.0
        for scale, agents in self.system.agents.items():
            for agent in agents:
                # Simple proxy: sum of KL divergences between belief and prior
                kl = kl_gaussian(
                    agent.mu_q, agent.L_q @ agent.L_q.T,
                    agent.mu_p, agent.L_p @ agent.L_p.T
                )
                total += kl
        return total

    def _compute_total_information(self) -> float:
        """Compute total information (proxy via entropy)"""
        total = 0.0
        for scale, agents in self.system.agents.items():
            for agent in agents:
                # Information ~ -log(det(Σ))
                logdet = 2.0 * np.sum(np.log(np.diag(agent.L_q)))
                total += -logdet
        return total