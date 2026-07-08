"""LIF-based dynamical simulator backend for the CL SDK.

A drop-in replacement for the SDK's ``RandomDataSource`` that reacts to
stimulation, for sandboxing closed-loop stimulus->response hypotheses without
hardware. See :mod:`organoid_simulator.lif_data_source`.
"""
from .lif_data_source import (
    LIFDataSource,
    make_lif_source,
    grid_connectivity,
    small_world_connectivity,
)

__all__ = [
    "LIFDataSource",
    "make_lif_source",
    "grid_connectivity",
    "small_world_connectivity",
]
