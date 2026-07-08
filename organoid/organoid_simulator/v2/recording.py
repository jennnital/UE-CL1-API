"""
Electrode forward model (§M5): neurons -> 64 extracellular electrode frames.

A real MEA electrode records a small, noisy echo of the transmembrane currents of
*nearby* neurons -- not any single neuron's intracellular voltage. This module turns
the network's spikes into that signal with a point-source approximation:

* Each neuron ``n`` couples to each electrode ``e`` by a **footprint weight**
  ``w[e, n]`` that falls off with soma-electrode distance (``V_e ∝ Σ_n I_n /(4πσ r)``,
  Ness 2013/2015). Grounded/reference electrodes have all-zero footprints and never
  see anything, matching the hardware contract.
* When a neuron fires, a canonical biphasic extracellular action-potential waveform
  is stamped onto every electrode, scaled by that neuron's footprint on the
  electrode. Overlapping spikes sum linearly (superposition).
* The summed microvolt trace is quantised to ``int16`` in raw sample units -- the
  exact ``frames`` contract v1 produced, but now physically motivated rather than a
  hand-stamped snippet per channel.

This keeps the same 25 kHz, 64-channel, ``int16`` frame contract as v1's
``LIFDataSource`` (constants imported from it) so the SDK loop and viewers are
unchanged. In **parity mode** (N=64) each neuron's footprint peaks on its own
electrode, recovering v1's 1:1 picture; in **sheet mode** each electrode genuinely
mixes a local population.
"""
from __future__ import annotations

import numpy as np

from ..lif_data_source import (
    CHANNEL_COUNT, NON_SPIKING,
    UV_PER_SAMPLE_UNIT, SPIKE_SAMPLES_BEFORE, SPIKE_SAMPLES_TOTAL,
    INT16_MAX, INT16_MIN,
)
from .connectivity import Geometry


def build_footprints(
    geometry: Geometry,
    *,
    sigma_um: float = 120.0,
    r_floor_um: float = 20.0,
    cutoff: float = 0.02,
) -> np.ndarray:
    """(64, N) electrode<-neuron footprint weights.

    Weight ``w[e, n] = 1 / (1 + (r_en / sigma_um)^2)`` (a soft point-source / Lorentzian
    falloff; ``r_floor_um`` prevents a singularity when a soma sits on an electrode).
    Weights below ``cutoff`` of the per-electrode max are zeroed to keep each electrode
    a *local* sampler and the stamping loop sparse. Grounded/reference electrodes are
    zeroed entirely so they stay silent (SDK contract).
    """
    elec = geometry.electrode_pos            # (64, 2)
    npos = geometry.neuron_pos               # (N, 2)
    diff = elec[:, None, :] - npos[None, :, :]
    r = np.sqrt((diff ** 2).sum(-1))         # (64, N) micrometres
    r = np.maximum(r, r_floor_um)
    w = 1.0 / (1.0 + (r / sigma_um) ** 2)

    # Locality: drop the long tail so an electrode reads only its neighbourhood.
    peak = w.max(axis=1, keepdims=True)
    w[w < cutoff * peak] = 0.0

    # Grounded/reference electrodes record nothing.
    w[list(NON_SPIKING), :] = 0.0
    return w


def spike_template(amplitude_uV: float = 80.0) -> np.ndarray:
    """Canonical biphasic extracellular AP: sharp negative trough then slower positive
    rebound, mean-centred and scaled so the trough is ``-amplitude_uV`` (same shape v1
    used, so downstream spike analyses see a familiar waveform)."""
    idx = np.arange(SPIKE_SAMPLES_TOTAL, dtype=np.float64)
    trough = -np.exp(-((idx - SPIKE_SAMPLES_BEFORE) / 4.0) ** 2)
    rebound = 0.4 * np.exp(-((idx - (SPIKE_SAMPLES_BEFORE + 15)) / 8.0) ** 2)
    t = trough + rebound
    t -= t.mean()
    t /= -t.min()
    t *= amplitude_uV
    return np.ascontiguousarray(t, dtype=np.float64)


class ExtracellularRecorder:
    """Renders neuron spikes into int16 electrode frames (see module docstring).

    Stateful across ``render`` calls: a spike near the end of one chunk stamps a
    waveform that spills into the next, so pending contributions are held in a dict
    keyed by absolute frame timestamp (same technique as v1's ``_pending_waveform``).
    """

    def __init__(
        self,
        geometry: Geometry,
        *,
        rng: np.random.Generator,
        sigma_um: float = 120.0,
        spike_amplitude_uV: float = 80.0,
        noise_floor_uV: float = 6.0,
        home_footprint_min: float = 0.15,
    ):
        self._geo = geometry
        self._rng = rng
        self._footprints = build_footprints(geometry, sigma_um=sigma_um)  # (64, N)
        self._template = spike_template(spike_amplitude_uV)
        self._noise_uV = float(noise_floor_uV)
        # A neuron reports a *detected* spike on its dominant electrode (argmax
        # footprint) provided that footprint is strong enough and the electrode is
        # spiking-eligible. This is the neuron->channel map for DataSourceSpike.
        home = self._footprints.argmax(axis=0)                     # (N,)
        home_w = self._footprints[home, np.arange(geometry.n_neurons)]
        self._home_elec = home
        self._home_ok = home_w >= home_footprint_min * self._footprints.max()
        # Absolute-ts -> (64,) uV waveform contributions not yet emitted.
        self._pending: dict[int, np.ndarray] = {}

    @property
    def footprints(self) -> np.ndarray:
        return self._footprints

    def reset(self) -> None:
        self._pending.clear()

    def render(self, from_ts: int, frame_count: int, spikes):
        """Build ``(frame_count, 64)`` int16 frames and per-electrode spike events.

        Args:
            from_ts: absolute frame timestamp of the first frame in this chunk.
            frame_count: number of frames to render.
            spikes: iterable of ``(abs_ts, neuron_idx)`` for spikes in this chunk.
        Returns ``(frames, spike_events)`` where ``spike_events`` is a list of
        ``(abs_ts, channel, waveform_uV)`` tuples for eligible detections.
        """
        # Drop any pending contribution that can no longer be emitted (reads are
        # monotonic, so anything before the read head is unreachable).
        for ts in [t for t in self._pending if t < from_ts]:
            del self._pending[ts]

        frames_uV = self._rng.normal(0.0, self._noise_uV, size=(frame_count, CHANNEL_COUNT))
        spike_events = []

        for abs_ts, n in spikes:
            n = int(n)
            fp = self._footprints[:, n]                    # (64,) electrode weights
            self._stamp(int(abs_ts), fp)
            if self._home_ok[n]:
                spike_events.append((int(abs_ts), int(self._home_elec[n]), self._template.copy()))

        # Fold this chunk's pending contributions into the frame window.
        to_ts = from_ts + frame_count
        for ts in [t for t in self._pending if from_ts <= t < to_ts]:
            frames_uV[ts - from_ts] += self._pending.pop(ts)

        frames = np.clip(
            np.round(frames_uV / UV_PER_SAMPLE_UNIT), INT16_MIN, INT16_MAX
        ).astype(np.int16)
        return np.ascontiguousarray(frames), spike_events

    def _stamp(self, ts: int, footprint: np.ndarray) -> None:
        """Add ``footprint``-scaled spike waveform to frames [ts, ts+75)."""
        active = np.nonzero(footprint)[0]
        if active.size == 0:
            return
        fp = footprint[active]
        for k in range(SPIKE_SAMPLES_TOTAL):
            fts = ts + k
            vec = self._pending.get(fts)
            if vec is None:
                vec = np.zeros(CHANNEL_COUNT, dtype=np.float64)
                self._pending[fts] = vec
            vec[active] += self._template[k] * fp
