#!/usr/bin/env python3
"""
Click this file to run and generate phase space plots.
Output will be in: ./orbit_analysis/
"""

import numpy as np
from pathlib import Path
from agent.system import MultiAgentSystem
from agent.agents import Agent
from agent.hamiltonian_trainer import HamiltonianTrainer
from config import SystemConfig, AgentConfig, TrainingConfig

# === CONFIGURATION (edit these if you want) ===
N_AGENTS = 2
N_STEPS = 100
K_LATENT = 3
FRICTION = 0.0  # 0 = conservative, >0 = damped
OUTPUT_DIR = Path('./orbit_analysis')

# === RUN SIMULATION ===
print("Setting up simulation...")

system_config = SystemConfig(
    lambda_self=1.0,
    lambda_belief_align=1.0,
    lambda_prior_align=0.0
)

agent_config = AgentConfig(
    spatial_shape=(),  # 0D particles
    K=K_LATENT,
    mu_scale=0.5,
    sigma_scale=0.3
)

rng = np.random.default_rng(42)
agents = [Agent(i, agent_config, rng=rng) for i in range(N_AGENTS)]
system = MultiAgentSystem(agents, system_config)

config = TrainingConfig(
    n_steps=N_STEPS,
    log_every=N_STEPS // 5,
    save_history=True
)

trainer = HamiltonianTrainer(
    system, config,
    friction=FRICTION,
    track_phase_space=True,
    phase_space_output_dir=OUTPUT_DIR
)

print(f"\nRunning {N_STEPS} steps...")
history = trainer.train(dt=0.01)

print(f"\n=== DONE ===")
print(f"Phase space plots saved to: {OUTPUT_DIR.absolute()}")
print(f"\nFiles:")
for f in OUTPUT_DIR.iterdir():
    print(f"  - {f.name}")
