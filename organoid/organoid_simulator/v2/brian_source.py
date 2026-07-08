"""
``BrianOrganoidDataSource`` -- the v2 substrate as a drop-in CL SDK data source.

This mirrors v1's ``LIFDataSource`` contract exactly (``metadata`` / ``open`` /
``read`` / ``on_stims`` / ``apply_reward``) so the existing SDK loop, ``live_server``,
and demos run against it unchanged -- but the internals are the biophysical Brian2
network from :mod:`organoid_simulator.v2.network`, recorded through the forward model
in :mod:`organoid_simulator.v2.recording`.

Register it exactly like v1::

    import cl
    from organoid_simulator.v2 import make_brian_source

    cl.sim.set_simulator_data_source(
        "organoid_simulator.v2.brian_source:make_brian_source",
        config={"n_neurons": 64, "random_seed": 42},
    )
    with cl.open() as neurons:
        for tick in neurons.loop(ticks_per_second=50):
            neurons.stim(cl.ChannelSet(27), cl.StimDesign(400, -2.0, 400, 2.0))
            spikes = tick.analysis.spikes    # evoked by the stim above

Clock reconciliation (the plan's noted roadblock): Brian2 runs on a continuous
``defaultclock`` at ``dt_ms``; the SDK reads on a 25 kHz frame clock. With the default
``dt_ms == 0.04`` one Brian step is exactly one frame, so a spike at Brian time ``t``
maps to frame ``round(t / frame_dt)``. ``read`` advances the network by exactly
``frame_count`` frames per call and consumes only the spikes recorded since the last
call, keeping the two clocks locked.

Performance note: this runs Brian2 in **runtime (Cython)** mode so it can be driven
chunk-by-chunk. Expect a few-× real time on CPU for ``N=64`` and slower for
``N≈1000``; ``supports_accelerated=True`` because throughput is compute-bound, not
wall-clock-bound. Standalone mode would be faster but cannot be stepped, so it is
deliberately not used (see plan §Performance).
"""
from __future__ import annotations

import os

import numpy as np

from brian2 import network_operation, ms, pA, nA, nS, defaultclock, second

from cl.sim import (
    DataSourceBatch,
    DataSourceSpike,
    SimulatorDataSource,
    SimulatorDataSourceMetadata,
)

from ..lif_data_source import (
    CHANNEL_COUNT, FRAMES_PER_SECOND, US_PER_FRAME, UV_PER_SAMPLE_UNIT, NON_SPIKING,
)
from .connectivity import build_geometry, build_connectivity
from .recording import build_footprints
from .network import build_network, SynParams, BackgroundParams
from .recording import ExtracellularRecorder


# Mode-aware defaults. Parity (N=64) is the fast drop-in: sparse connectivity and a
# Fellous-2003 resting regime (~5 Hz, reactive) -- good for closed-loop sandboxing.
# Sheet (N large) is the realism/reservoir mode: denser recurrence and a background
# working point tuned to sit at criticality (branching ratio ≈ 1, avalanche α ≈ 1.5 at
# N≈1000; see criticality_check.py). Both are just starting points -- any field can be
# overridden per-source, and criticality_check.py --sweep re-tunes for other N.
PARITY_PRESET = dict(
    p_connect=0.15, conn_lambda_um=300.0, w_exc_nS=1.2, w_inh_nS=6.0,
    exc_scale=1.0, inh_scale=1.0,
    background=dict(ge0=12.0, gi0=10.0, sigma_e=6.0, sigma_i=6.0),
)
SHEET_PRESET = dict(
    p_connect=0.4, conn_lambda_um=450.0, w_exc_nS=1.5, w_inh_nS=6.0,
    exc_scale=2.0, inh_scale=2.0,
    background=dict(ge0=10.0, gi0=5.0, sigma_e=1.5, sigma_i=1.5),
)


class BrianOrganoidDataSource(SimulatorDataSource):
    """Biophysical Brian2 organoid substrate wrapped as a CL SDK simulator source."""

    def __init__(
        self,
        *,
        n_neurons: int = 64,
        random_seed: int | None = None,
        dt_ms: float = US_PER_FRAME / 1000.0,   # 0.04 ms == one 25 kHz frame
        # geometry / connectivity. The tuning-sensitive fields (conn density, weights,
        # E/I scales, background) default to None -> the parity/sheet preset above,
        # chosen by n_neurons. Pass any of them to override the preset.
        exc_fraction: float = 0.8,
        conn_lambda_um: float | None = None,
        p_connect: float | None = None,
        w_exc_nS: float | None = None,
        w_inh_nS: float | None = None,
        velocity_um_per_ms: float = 300.0,
        min_delay_ms: float = 0.5,
        max_delay_ms: float = 12.0,
        # E/I tuning (criticality knobs, §M3)
        exc_scale: float | None = None,
        inh_scale: float | None = None,
        # Scale on AdEx spike-frequency adaptation (<1 weakens it). Strong adaptation
        # makes evoked propagation all-or-nothing then refractory; lowering it gives
        # consistent, repeatable propagation (used by the viewer and trial sweeps).
        adaptation_scale: float = 1.0,
        # OU background bombardment (§M3). Dict of any of ge0/gi0/sigma_e/sigma_i (nS);
        # None -> preset. (e.g. all-zero sigma for a silent culture that only reacts to stim.)
        background: dict | None = None,
        # recording forward model (§M5)
        footprint_sigma_um: float = 120.0,
        spike_amplitude_uV: float = 80.0,
        noise_floor_uV: float = 6.0,
        # stimulation. A *focal* stim footprint (tighter than the recording sigma)
        # so a pulse drives mainly the neurons directly under the electrode; activity
        # then reaches downstream neurons through synapses + conduction delays, making
        # the causal delay pathway observable rather than swamped by direct co-drive.
        stim_sigma_um: float = 70.0,
        stim_gain_nA_per_uA: float = 400.0,
        # plasticity (§M4, active from v2.2)
        plasticity: bool = False,
        # Optional overrides for the plasticity rule (PlasticityParams fields, e.g.
        # {"A_ltp": 0.006, "lr_nS": 0.4}). By default the LTP/LTD voltage thresholds are
        # auto-calibrated to sit just above the network's resting potential (so only
        # genuine, event-driven depolarisation -- not resting fluctuation -- drives
        # plasticity); pass theta_ltp/theta_ltd here to override that.
        plasticity_params: dict | None = None,
    ):
        # __init__ runs in BOTH parent and simulator subprocess -- store plain config
        # only; all Brian2 objects are created in open().
        self._n_neurons = int(n_neurons)
        self._random_seed = random_seed
        self._dt_ms = float(dt_ms)
        self._frame_dt_s = US_PER_FRAME / 1_000_000.0     # seconds per frame
        self._exc_fraction = float(exc_fraction)

        preset = PARITY_PRESET if self._n_neurons == CHANNEL_COUNT else SHEET_PRESET
        def _d(val, key):  # user override else preset
            return preset[key] if val is None else val

        self._conn = dict(
            conn_lambda_um=_d(conn_lambda_um, "conn_lambda_um"),
            p_connect=_d(p_connect, "p_connect"),
            w_exc_nS=_d(w_exc_nS, "w_exc_nS"),
            w_inh_nS=_d(w_inh_nS, "w_inh_nS"),
            velocity_um_per_ms=velocity_um_per_ms,
            min_delay_ms=min_delay_ms, max_delay_ms=max_delay_ms,
        )
        self._exc_scale = float(_d(exc_scale, "exc_scale"))
        self._inh_scale = float(_d(inh_scale, "inh_scale"))
        self._adaptation_scale = float(adaptation_scale)
        self._background = dict(background) if background else dict(preset["background"])
        self._footprint_sigma_um = float(footprint_sigma_um)
        self._spike_amplitude_uV = float(spike_amplitude_uV)
        self._noise_floor_uV = float(noise_floor_uV)
        self._stim_sigma_um = float(stim_sigma_um)
        self._stim_gain = float(stim_gain_nA_per_uA)
        self._plastic = bool(plasticity)
        self._plasticity_params = dict(plasticity_params) if plasticity_params else {}

        self._metadata = SimulatorDataSourceMetadata(
            channel_count=CHANNEL_COUNT,
            frames_per_second=FRAMES_PER_SECOND,
            uV_per_sample_unit=UV_PER_SAMPLE_UNIT,
            start_timestamp=0,
            duration_frames=None,
            seekable=False,             # state depends on the full stim/spike history
            supports_accelerated=True,  # compute-bound, no wall-clock dependency
        )

        # Simulation state (populated in open()).
        self._bundle = None
        self._recorder: ExtracellularRecorder | None = None
        self._stim_footprints: np.ndarray | None = None   # (64, N) normalised stim weights
        self._pending_stim: dict[int, np.ndarray] = {}     # frame ts -> (N,) injected current, amp-scaled
        self._spikes_seen = 0
        self._reward_M = 0.0
        self._stim_active = False

    @property
    def metadata(self) -> SimulatorDataSourceMetadata:
        return self._metadata

    # -- lifecycle -----------------------------------------------------------

    def open(self) -> None:
        seed = self._random_seed
        if seed is None:
            env = os.getenv("CL_SDK_RANDOM_SEED")
            seed = int(env) if env else int(np.random.SeedSequence(None).generate_state(1)[0])
        seed = int(seed)
        rng = np.random.default_rng(seed)

        geo = build_geometry(
            n_neurons=self._n_neurons, exc_fraction=self._exc_fraction, seed=seed
        )
        conn = build_connectivity(geo, seed=seed, **self._conn)
        bg = BackgroundParams(**self._background) if self._background else None
        plasticity = self._resolve_plasticity(bg) if self._plastic else None
        self._bundle = build_network(
            geo, conn,
            dt_ms=self._dt_ms, background=bg,
            exc_scale=self._exc_scale, inh_scale=self._inh_scale,
            adaptation_scale=self._adaptation_scale,
            plastic=self._plastic, plasticity=plasticity, seed=seed,
        )
        self._recorder = ExtracellularRecorder(
            geo, rng=rng,
            sigma_um=self._footprint_sigma_um,
            spike_amplitude_uV=self._spike_amplitude_uV,
            noise_floor_uV=self._noise_floor_uV,
        )
        # Focal stim footprints: a tighter Gaussian than the recording footprint AND a
        # high locality cutoff, so a pulse drives essentially only the neuron(s)
        # directly under the electrode (a near-delta in parity mode). This keeps direct
        # electrical co-drive from swamping the synaptic + conduction-delay pathway --
        # so downstream firing is genuinely caused, and arrives after the edge delay.
        fp = build_footprints(geo, sigma_um=self._stim_sigma_um, cutoff=0.4)   # (64, N)
        peak = fp.max(axis=1, keepdims=True)
        peak[peak == 0] = 1.0
        self._stim_footprints = fp / peak

        # Base (unscaled) excitatory weights, kept so set_coupling_gain can set an
        # absolute coupling level (w = base * gain) rather than drift multiplicatively.
        self._exc_base_nS = conn.exc_w.copy() if conn.n_exc_syn else np.zeros(0)
        self._coupling_gain = self._exc_scale

        self._pending_stim.clear()
        self._spikes_seen = 0
        self._reward_M = 0.0
        self._stim_active = False

        self._install_stim_operation()
        print(f"Brian organoid v2 source: seed={seed}, N={geo.n_neurons} "
              f"(E={geo.n_exc}/I={geo.n_inh}), exc_syn={conn.n_exc_syn}, inh_syn={conn.n_inh_syn}")

    def _resolve_plasticity(self, background) -> "PlasticityParams":
        """Build the PlasticityParams, auto-calibrating the LTP/LTD voltage thresholds
        to just above the analytic resting potential so only genuine, event-driven
        depolarisation drives learning (selectivity), unless the user overrode them.

        Resting Vm is the RS conductance-clamped steady state
        ``(gL*EL + ge0*Ee + gi0*Ei)/(gL+ge0+gi0)`` -- an analytic estimate that tracks
        the background, so plasticity behaves consistently across configs.
        """
        from .network import PlasticityParams, RS_EXC, SynParams
        bg = background or BackgroundParams()
        syn = SynParams()
        p = RS_EXC
        v_rest = ((p.gL * p.EL + bg.ge0 * syn.Ee + bg.gi0 * syn.Ei)
                  / (p.gL + bg.ge0 + bg.gi0))
        theta = v_rest + 4.0     # a few mV above rest: above typical OU fluctuation
        params = dict(theta_ltp=theta, theta_ltd=theta)
        params.update(self._plasticity_params)   # explicit user overrides win
        return PlasticityParams(**params)

    def close(self) -> None:
        self._bundle = None
        self._recorder = None
        self._pending_stim.clear()

    # -- stim handling -------------------------------------------------------

    def on_stims(self, stims) -> None:
        """Expand each committed stim pulse into per-frame injected current on the
        neurons under the stimulated electrode, keyed by absolute frame timestamp.

        Only the cathodic (negative-current, by CL convention) phase excites, matching
        v1 and the biophysics: the charge-balanced anodic recharge does not undo an
        all-or-none threshold crossing during the cathodic phase.
        """
        assert self._stim_footprints is not None, "on_stims before open()"
        for stim in stims:
            weights = self._stim_footprints[stim.channel]        # (N,)
            frame_offset = 0
            for duration_us, current_uA in zip(stim.phase_durations_us, stim.phase_currents_uA):
                n_frames = max(1, round(duration_us / US_PER_FRAME))
                cathodic = max(0.0, -float(current_uA))          # only cathodic excites
                if cathodic > 0.0:
                    drive = (self._stim_gain * cathodic) * weights   # nA per neuron
                    for k in range(n_frames):
                        ts = int(stim.timestamp) + frame_offset + k
                        vec = self._pending_stim.get(ts)
                        if vec is None:
                            vec = np.zeros(self._bundle.geometry.n_neurons)
                            self._pending_stim[ts] = vec
                        vec += drive
                frame_offset += n_frames

    def _install_stim_operation(self) -> None:
        """Add a frame-clock network_operation that writes I_stim from the pending
        schedule, so stim->spike latency is preserved to the frame (not quantised to a
        whole read chunk)."""
        G = self._bundle.neurons
        pending = self._pending_stim
        frame_dt_s = self._frame_dt_s
        state = {"active": False}

        @network_operation(dt=self._dt_ms * ms, when="start")
        def _apply_stim():
            ts = int(round(float(defaultclock.t / second) / frame_dt_s))
            vec = pending.pop(ts, None)
            if vec is not None:
                G.I_stim = vec * nA
                state["active"] = True
            elif state["active"]:
                G.I_stim = 0 * pA
                state["active"] = False

        self._bundle.net.add(_apply_stim)

    # -- read loop -----------------------------------------------------------

    def read(self, from_timestamp: int, frame_count: int) -> DataSourceBatch:
        assert self._bundle is not None and self._recorder is not None, "read() before open()"

        # Drop stims that can no longer be applied (monotonic reads).
        for ts in [t for t in self._pending_stim if t < from_timestamp]:
            del self._pending_stim[ts]

        # Advance the network by exactly this many frames.
        self._bundle.net.run(frame_count * self._frame_dt_s * second)

        # Pull spikes recorded since the last read; map Brian time -> frame ts.
        mon = self._bundle.spikemon
        total = mon.num_spikes
        new_i = np.asarray(mon.i[self._spikes_seen:total])
        new_t = np.asarray(mon.t[self._spikes_seen:total] / second)
        self._spikes_seen = total

        to_ts = from_timestamp + frame_count
        spikes_in = []
        for n, t in zip(new_i, new_t):
            ts = int(round(t / self._frame_dt_s))
            if ts < from_timestamp:
                ts = from_timestamp
            elif ts >= to_ts:
                ts = to_ts - 1
            spikes_in.append((ts, int(n)))

        frames, events = self._recorder.render(from_timestamp, frame_count, spikes_in)

        spikes = [
            DataSourceSpike(timestamp=ts, channel=ch, samples=wf.astype(np.float32),
                            channel_mean_sample=0.0)
            for ts, ch, wf in events
        ]
        return DataSourceBatch(frames=frames, spikes=spikes)

    # -- reward / plasticity -------------------------------------------------

    def apply_reward(self, reward: float) -> None:
        """Consolidate the accumulated eligibility trace into the excitatory weights
        (§M4 three-factor rule): ``dw = lr * reward * e``, then reset ``e``.

        This is the third (neuromodulatory) factor: the Clopath rule has been writing
        *what propagated with what* into each synapse's eligibility trace ``e``; the
        global reward decides whether to strengthen (r>0) or weaken (r<0) those
        pathways. Call between closed-loop episodes (DishBrain-style feedback). A
        no-op unless ``plasticity=True``.
        """
        self._reward_M = float(reward)
        b = self._bundle
        if b is None or not b.plastic or b.syn_e is None:
            return
        p = b.plasticity
        w = np.asarray(b.syn_e.w_syn[:] / nS)
        elig = np.asarray(b.syn_e.elig[:])
        # Per-synapse ceiling RELATIVE to the current operating weight (structural
        # base x coupling gain), so a strong-coupling regime isn't crushed to a small
        # absolute cap. A synapse may grow up to `w_growth_factor`x its structural
        # strength; `w_max_nS` stays only as a final hard safety limit.
        ceiling = np.minimum(self._exc_base_nS * self._coupling_gain * p.w_growth_factor,
                             p.w_max_nS)
        new_w = np.clip(w + p.lr_nS * float(reward) * elig, 0.0, ceiling)
        b.syn_e.w_syn = new_w * nS
        b.syn_e.elig = 0.0   # each reward consolidates activity since the previous one

    # -- viewer-compat helpers (used by live_server's structural overlay) ----

    @property
    def connectivity_matrix(self) -> np.ndarray:
        """(64, 64) channel-level structural graph, aggregated from the synapses via
        each neuron's home electrode. ``W[post_ch, pre_ch]`` = summed excitatory weight.
        Lets v1's ``live_server`` draw a structural overlay against the v2 substrate."""
        assert self._bundle is not None, "connectivity_matrix before open()"
        W = np.zeros((CHANNEL_COUNT, CHANNEL_COUNT))
        syn = self._bundle.syn_e
        if syn is None:
            return W
        home = self._recorder._home_elec
        pre = np.asarray(syn.i[:])
        post = np.asarray(syn.j[:])
        w = np.asarray(syn.w_syn[:] / nS)
        for a, b, ww in zip(pre, post, w):
            W[home[b], home[a]] += ww
        return W

    def set_coupling_gain(self, gain: float) -> None:
        """Live-set the excitatory coupling level (maps v1's ``coupling_gain`` control
        to the v2 substrate). Sets an *absolute* scale ``w = base_weight * gain`` from
        the stored base weights, so repeated calls don't drift. With plasticity on this
        overwrites any learned weights (as v1 did)."""
        if self._bundle is None or self._bundle.syn_e is None:
            return
        target = max(0.0, float(gain))
        self._bundle.syn_e.w_syn = (self._exc_base_nS * target) * nS
        self._coupling_gain = target


def make_brian_source(**kwargs) -> BrianOrganoidDataSource:
    """Factory for ``cl.sim.set_simulator_data_source``. All kwargs forward to
    :class:`BrianOrganoidDataSource` and must be JSON-serialisable."""
    return BrianOrganoidDataSource(**kwargs)
