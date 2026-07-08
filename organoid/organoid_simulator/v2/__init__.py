"""v2 -- biophysical Brian2 rebuild of the organoid simulator.

A drop-in ``cl.sim.SimulatorDataSource`` (like v1's ``LIFDataSource``) whose internals
are a biophysically grounded, delay-coupled, criticality-tunable spiking network:
AdEx conductance neurons (§M1), conductance synapses with per-edge conduction delays
(§M2), Ornstein-Uhlenbeck background bombardment (§M3), voltage/reward plasticity
(§M4, from v2.2), and a point-source electrode forward model (§M5).

See ``organoid_simulator/v2`` module docstrings and the repo plan for the full spec.
"""
from .brian_source import BrianOrganoidDataSource, make_brian_source

__all__ = ["BrianOrganoidDataSource", "make_brian_source"]
