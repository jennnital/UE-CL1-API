"""
Spatial geometry + distance-dependent connectivity for the v2 substrate.

This is the module that turns "a bag of neurons" into a *spatially embedded,
delay-coupled* network -- the structural precondition for the v2 upgrades that
v1 could not express:

* Neurons live at 2-D positions on a sheet that spans the 8x8 MEA (§Architecture).
* Electrodes sit at the 64 grid points and *sample / stimulate local populations*
  rather than owning one neuron each (the key departure noted in the plan).
* Synapses carry **per-edge conduction delays** derived from inter-soma distance
  and a finite conduction velocity (§M2) -- distance becomes latency, which is what
  lets the reservoir "convert time into space".

Two geometry modes, both produced here so the rest of v2 is mode-agnostic:

* **parity** (`n_neurons == 64`): exactly one neuron per electrode, placed on the
  electrode. Electrode ``i`` reads neuron ``i`` 1:1 -- the fast drop-in mode that
  reproduces v1's "1 neuron = 1 channel" picture but with real dynamics underneath.
* **sheet** (`n_neurons` large, e.g. 1000): neurons scattered uniformly on the
  sheet, 80% excitatory / 20% inhibitory; electrodes forward-model a weighted sum
  of nearby neurons (see :mod:`organoid_simulator.v2.recording`).

Connectivity itself is distance-dependent (Gaussian falloff, à la cortical
cultures): nearby neurons are likely to connect with short delay, far ones rarely
and with long delay. A v1-style ``(64, 64)`` matrix may also be passed in as a seed
(interpreted as excitatory edges) so the existing
:func:`grid_connectivity` / :func:`small_world_connectivity` graphs keep working.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# --- SDK-fixed layout constants (mirror v1's lif_data_source) ----------------
CHANNEL_COUNT = 64
GROUNDED_CHANNELS = (0, 7, 56, 63)
REFERENCE_CHANNELS = (4,)
NON_SPIKING = frozenset(GROUNDED_CHANNELS + REFERENCE_CHANNELS)

# --- Physical scale of the sheet ---------------------------------------------
# MEA pitch: distance between adjacent electrodes. 200 um is a typical CL1-scale
# electrode spacing; it also sets the absolute distances that feed the conduction
# delays, so it is a genuine model parameter, not cosmetic.
ELECTRODE_PITCH_UM = 200.0
GRID_SIDE = 8


def electrode_positions(pitch_um: float = ELECTRODE_PITCH_UM) -> np.ndarray:
    """(64, 2) electrode coordinates in micrometres.

    Channel ``ch`` sits at grid ``(row, col) = (ch % 8, ch // 8)`` -- identical to
    v1's ``_grid_pos`` so any channel-indexed structure carries over unchanged.
    """
    pos = np.zeros((CHANNEL_COUNT, 2), dtype=np.float64)
    for ch in range(CHANNEL_COUNT):
        row, col = ch % GRID_SIDE, ch // GRID_SIDE
        pos[ch] = (col * pitch_um, row * pitch_um)
    return pos


@dataclass
class Geometry:
    """Neuron/electrode spatial layout shared across the whole v2 network."""

    neuron_pos: np.ndarray        # (N, 2) micrometres
    is_exc: np.ndarray            # (N,) bool -- True for excitatory (RS) neurons
    electrode_pos: np.ndarray     # (64, 2) micrometres
    parity: bool                  # True in the N==64 1:1 mode
    pitch_um: float

    @property
    def n_neurons(self) -> int:
        return self.neuron_pos.shape[0]

    @property
    def n_exc(self) -> int:
        return int(self.is_exc.sum())

    @property
    def n_inh(self) -> int:
        return int((~self.is_exc).sum())


def build_geometry(
    n_neurons: int = 64,
    exc_fraction: float = 0.8,
    pitch_um: float = ELECTRODE_PITCH_UM,
    seed: int = 0,
) -> Geometry:
    """Lay out ``n_neurons`` on the sheet spanning the 8x8 electrode grid.

    ``n_neurons == 64`` selects **parity mode**: one neuron per electrode, sitting
    exactly on it. To keep parity mode a faithful, debuggable analogue of v1 (where
    every live channel spikes), *all* parity neurons are excitatory and the
    grounded/reference channels are still allowed to hold a neuron -- the recording
    layer, not the geometry, enforces their silence. Any other ``n_neurons`` selects
    **sheet mode**: uniform-random positions with an 80/20 E/I split.
    """
    rng = np.random.default_rng(seed)
    elec = electrode_positions(pitch_um)

    if n_neurons == CHANNEL_COUNT:
        # Parity: neuron i lives on electrode i. All excitatory (see docstring).
        return Geometry(
            neuron_pos=elec.copy(),
            is_exc=np.ones(CHANNEL_COUNT, dtype=bool),
            electrode_pos=elec,
            parity=True,
            pitch_um=pitch_um,
        )

    # Sheet mode: scatter neurons across the electrode-spanned area, with a small
    # margin so edge neurons still have neighbours on all sides.
    span = (GRID_SIDE - 1) * pitch_um
    margin = 0.5 * pitch_um
    lo, hi = -margin, span + margin
    pos = rng.uniform(lo, hi, size=(n_neurons, 2))

    # 80/20 E/I. Dale's principle is enforced downstream: a neuron is purely
    # excitatory or purely inhibitory, and that identity picks g_e vs g_i on targets.
    n_exc = int(round(exc_fraction * n_neurons))
    is_exc = np.zeros(n_neurons, dtype=bool)
    is_exc[rng.permutation(n_neurons)[:n_exc]] = True

    return Geometry(
        neuron_pos=pos,
        is_exc=is_exc,
        electrode_pos=elec,
        parity=False,
        pitch_um=pitch_um,
    )


@dataclass
class Connectivity:
    """Directed, weighted, delayed edges, split by presynaptic transmitter type.

    Excitatory and inhibitory edges are stored separately because Brian2 routes them
    to different post-synaptic conductances (``g_e`` vs ``g_i``) -- see
    :func:`organoid_simulator.v2.network.build_network`.
    Indices are into ``Geometry.neuron_pos`` rows.
    """

    exc_pre: np.ndarray   # (Ee,) presynaptic neuron index (excitatory source)
    exc_post: np.ndarray  # (Ee,) postsynaptic neuron index
    exc_w: np.ndarray     # (Ee,) peak conductance, nS
    exc_delay: np.ndarray # (Ee,) conduction delay, ms
    inh_pre: np.ndarray
    inh_post: np.ndarray
    inh_w: np.ndarray
    inh_delay: np.ndarray

    @property
    def n_exc_syn(self) -> int:
        return self.exc_pre.shape[0]

    @property
    def n_inh_syn(self) -> int:
        return self.inh_pre.shape[0]


def _delays_from_distance(
    dist_um: np.ndarray,
    velocity_um_per_ms: float,
    min_delay_ms: float,
    jitter_ms: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Conduction delay (ms) for each edge: distance / velocity + floor + jitter.

    Velocity ~0.2-0.5 m/s = 200-500 um/ms for unmyelinated culture axons (§M2). A
    small floor keeps even coincident somata from having a zero-length delay buffer,
    and jitter breaks the perfectly-degenerate delays that a pure distance rule gives.
    """
    d = dist_um / velocity_um_per_ms + min_delay_ms
    if jitter_ms > 0:
        d = d + rng.normal(0.0, jitter_ms, size=d.shape)
    return np.clip(d, min_delay_ms, None)


def build_connectivity(
    geometry: Geometry,
    *,
    conn_lambda_um: float = 300.0,
    p_connect: float = 0.15,
    w_exc_nS: float = 1.2,
    w_inh_nS: float = 6.0,
    weight_sigma: float = 0.4,
    velocity_um_per_ms: float = 300.0,
    min_delay_ms: float = 0.5,
    delay_jitter_ms: float = 0.2,
    max_delay_ms: float = 12.0,
    seed_matrix: np.ndarray | None = None,
    seed: int = 0,
) -> Connectivity:
    """Distance-dependent recurrent connectivity with per-edge conduction delays.

    Connection probability falls off as a Gaussian of inter-soma distance,
    ``P(connect) = p_connect * exp(-(d / conn_lambda_um)^2)`` -- the standard
    cortical-culture rule (nearby > far). Inhibitory synapses are made stronger per
    edge than excitatory ones (``w_inh_nS > w_exc_nS``), the usual asymmetry that
    lets a 20% inhibitory minority balance an 80% excitatory majority. Peak
    conductances are lognormal (a few strong pathways, many weak) and delays come
    from :func:`_delays_from_distance`.

    If ``seed_matrix`` (a v1-style ``(64, 64)`` graph, ``W[post, pre]``) is given, its
    nonzero entries are used as the edge set instead of the probabilistic rule
    (weights scaled to ``w_exc_nS``); this requires parity geometry and reproduces a
    specific v1 structural graph with real delays layered on. Delays are capped at
    ``max_delay_ms`` to bound Brian2's per-synapse delay buffers (§M2 roadblock).
    """
    rng = np.random.default_rng(seed)
    pos = geometry.neuron_pos
    is_exc = geometry.is_exc
    n = geometry.n_neurons

    if seed_matrix is not None:
        return _connectivity_from_matrix(
            geometry, seed_matrix, w_exc_nS, w_inh_nS, weight_sigma,
            velocity_um_per_ms, min_delay_ms, delay_jitter_ms, max_delay_ms, rng,
        )

    # Pairwise distances (N, N). N<=~1000 so the dense matrix is cheap (<8 MB).
    diff = pos[:, None, :] - pos[None, :, :]
    dist = np.sqrt((diff ** 2).sum(-1))

    prob = p_connect * np.exp(-((dist / conn_lambda_um) ** 2))
    np.fill_diagonal(prob, 0.0)                      # no autapses
    mask = rng.random((n, n)) < prob                 # mask[pre, post] -> edge pre->post

    pre_idx, post_idx = np.nonzero(mask)             # both (E,)
    edge_dist = dist[pre_idx, post_idx]
    edge_delay = _delays_from_distance(
        edge_dist, velocity_um_per_ms, min_delay_ms, delay_jitter_ms, rng
    )
    edge_delay = np.clip(edge_delay, min_delay_ms, max_delay_ms)

    src_is_exc = is_exc[pre_idx]
    # Lognormal peak conductance, mean 1, scaled by the E/I base weights.
    lognorm = rng.lognormal(mean=0.0, sigma=weight_sigma, size=pre_idx.shape)
    w = np.where(src_is_exc, w_exc_nS, w_inh_nS) * lognorm

    e = src_is_exc
    i = ~src_is_exc
    return Connectivity(
        exc_pre=pre_idx[e].astype(np.int32),
        exc_post=post_idx[e].astype(np.int32),
        exc_w=w[e].astype(np.float64),
        exc_delay=edge_delay[e].astype(np.float64),
        inh_pre=pre_idx[i].astype(np.int32),
        inh_post=post_idx[i].astype(np.int32),
        inh_w=w[i].astype(np.float64),
        inh_delay=edge_delay[i].astype(np.float64),
    )


def _connectivity_from_matrix(
    geometry, seed_matrix, w_exc_nS, w_inh_nS, weight_sigma,
    velocity_um_per_ms, min_delay_ms, delay_jitter_ms, max_delay_ms, rng,
) -> Connectivity:
    """Build edges from a v1 ``(64, 64)`` ``W[post, pre]`` matrix (parity mode only)."""
    if not geometry.parity:
        raise ValueError("seed_matrix requires parity (N=64) geometry")
    W = np.asarray(seed_matrix, dtype=np.float64)
    if W.shape != (CHANNEL_COUNT, CHANNEL_COUNT):
        raise ValueError(f"seed_matrix must be ({CHANNEL_COUNT}, {CHANNEL_COUNT}), got {W.shape}")

    post_idx, pre_idx = np.nonzero(W)                # W[post, pre] -> edge pre->post
    pos = geometry.neuron_pos
    edge_dist = np.sqrt(((pos[pre_idx] - pos[post_idx]) ** 2).sum(-1))
    edge_delay = _delays_from_distance(
        edge_dist, velocity_um_per_ms, min_delay_ms, delay_jitter_ms, rng
    )
    edge_delay = np.clip(edge_delay, min_delay_ms, max_delay_ms)

    is_exc = geometry.is_exc
    src_is_exc = is_exc[pre_idx]
    # Preserve the matrix's relative weights, rescaled to the nS base weights.
    base = np.where(src_is_exc, w_exc_nS, w_inh_nS)
    w = base * W[post_idx, pre_idx]

    e = src_is_exc
    i = ~src_is_exc
    return Connectivity(
        exc_pre=pre_idx[e].astype(np.int32),
        exc_post=post_idx[e].astype(np.int32),
        exc_w=w[e].astype(np.float64),
        exc_delay=edge_delay[e].astype(np.float64),
        inh_pre=pre_idx[i].astype(np.int32),
        inh_post=post_idx[i].astype(np.int32),
        inh_w=w[i].astype(np.float64),
        inh_delay=edge_delay[i].astype(np.float64),
    )
