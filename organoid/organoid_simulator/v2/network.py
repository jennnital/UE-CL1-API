"""
Brian2 network builder for the v2 substrate.

This is where the biophysics lives. :func:`build_network` assembles, from a
:class:`~organoid_simulator.v2.connectivity.Geometry` and
:class:`~organoid_simulator.v2.connectivity.Connectivity`, a runtime-mode Brian2
``Network`` implementing:

* **§M1 AdEx neurons** (Brette & Gerstner 2005), conductance-based, in two classes:
  regular-spiking (RS) excitatory and fast-spiking (FS) inhibitory. The AdEx
  ``exp((v - VT)/DeltaT)`` spike-initiation term and the adaptation current ``w``
  give the bursting/adaptation richness LIF lacks.
* **§M2 conductance synapses with per-edge conduction delays.** Excitatory edges
  add to ``g_e`` (reversal ``Ee``), inhibitory to ``g_i`` (reversal ``Ei``); each
  edge fires its target after its own ``delay`` -- distance becomes latency.
* **§M3 Ornstein-Uhlenbeck conductance bombardment.** ``g_e``/``g_i`` are OU
  processes (mean-reverting to ``ge0``/``gi0`` with white-noise drive), so every
  neuron sits in a fluctuating high-conductance state -- real membrane noise, not a
  cosmetic display shimmer. Synaptic events ride on top of the same conductance.

Plasticity (§M4) is layered on the excitatory synapses when ``plastic=True`` (added
in the v2.2 stage); v2.0/v2.1 run a static substrate.

Everything runs in Brian2 **runtime mode with the Cython target** so the network can
be driven chunk-by-chunk from :class:`~organoid_simulator.v2.brian_source`. Units are
carried explicitly (Brian2 quantities), never bare floats.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

import brian2 as b2
from brian2 import (
    NeuronGroup, Synapses, SpikeMonitor, Network, defaultclock,
    ms, mV, nS, pF, pA, nA, amp, second, prefs,
)

from .connectivity import Geometry, Connectivity

# Cython codegen (verified available on this machine). Runtime mode -- NOT
# standalone -- because standalone cannot be driven chunk-by-chunk (§cl-sdk wrapping).
prefs.codegen.target = "cython"

# The shared background/synapse constants live in the group-specific namespace (see
# build_network). Brian2 warns whenever a caller happens to have a Python local of
# the same name (e.g. `ge0`) in scope at run() time; the group namespace always wins,
# so the warning is noise -- silence just that class.
b2.BrianLogger.suppress_name("resolution_conflict")


# --------------------------------------------------------------------------- #
# AdEx parameter sets. Two cell classes give the network its E/I dynamics.
# Values seed from Brette & Gerstner (2005) / Naud et al. (2008): RS is the
# canonical adapting pyramidal cell; FS is a fast, weakly-adapting interneuron.
# --------------------------------------------------------------------------- #
@dataclass
class CellParams:
    C: float          # pF   membrane capacitance
    gL: float         # nS   leak conductance
    EL: float         # mV   leak reversal / resting potential
    VT: float         # mV   spike-initiation threshold (soft)
    DeltaT: float     # mV   spike sharpness
    tau_w: float      # ms   adaptation time constant
    a: float          # nS   subthreshold adaptation coupling
    b: float          # pA   spike-triggered adaptation increment
    Vr: float         # mV   reset potential
    refractory: float # ms   absolute refractory period


RS_EXC = CellParams(C=281, gL=30, EL=-70.6, VT=-50.4, DeltaT=2.0,
                    tau_w=144, a=4.0, b=80.5, Vr=-70.6, refractory=2.0)
# FS interneuron: lower capacitance, sharper, negligible adaptation, higher reset ->
# fast, sustained firing characteristic of inhibitory basket cells.
FS_INH = CellParams(C=200, gL=20, EL=-70.0, VT=-50.0, DeltaT=1.5,
                    tau_w=30, a=0.0, b=10.0, Vr=-58.0, refractory=1.0)


@dataclass
class SynParams:
    Ee: float = 0.0       # mV   excitatory (AMPA) reversal
    Ei: float = -80.0     # mV   inhibitory (GABA) reversal
    tau_e: float = 5.0    # ms   excitatory conductance decay
    tau_i: float = 10.0   # ms   inhibitory conductance decay


@dataclass
class PlasticityParams:
    """Clopath voltage-gated STDP + three-factor reward + homeostasis (§M4).

    LTP is gated by the *postsynaptic depolarisation* left by arriving/propagating
    activity (traces ``u_p``/``u_m``), so learning is mechanistically tied to
    propagation -- the correlation v1's coincidence STDP faked. Each voltage/spike
    event writes not to the weight but to a slow **eligibility trace** ``e`` per
    synapse; the weight only moves when ``apply_reward(r)`` consolidates it
    (``dw = lr * r * e``), the third (neuromodulatory) factor. A slow homeostatic
    voltage average scales LTD so over-active neurons depress more, keeping weights
    bounded without v1's crude hard clip (a hard ``w_max`` remains only as a safety).
    """
    tau_ltp: float = 7.0        # ms   fast depolarisation trace (potentiation)
    tau_ltd: float = 12.0       # ms   slow depolarisation trace (depression)
    tau_homeo: float = 1000.0   # ms   homeostatic voltage average
    tau_x: float = 15.0         # ms   presynaptic trace
    tau_elig: float = 1000.0    # ms   eligibility-trace decay (bridges to reward)
    # Clopath trace threshold θ- sits just above rest (not at spike threshold): the
    # low-pass depolarisation traces u_p/u_m only clear it when the neuron has actually
    # been depolarised by activity, which is the whole point (LTP tied to propagation).
    theta_ltp: float = -64.0    # mV   θ- for the fast (LTP) trace u_p
    theta_ltd: float = -64.0    # mV   θ- for the slow (LTD) trace u_m
    A_ltp: float = 0.006        # potentiation rate (per mV*trace)
    A_ltd: float = 0.004        # depression rate (per mV)
    u_ref_mV: float = 6.0       # mV   homeostatic target (mean depol above θ-)
    lr_nS: float = 0.4          # weight change per unit reward per unit eligibility
    w_growth_factor: float = 3.0  # a synapse may grow up to this x its structural weight
    w_max_nS: float = 60.0      # final hard safety clip (rarely binds; see apply_reward)


@dataclass
class BackgroundParams:
    """Ornstein-Uhlenbeck conductance bombardment (§M3, Destexhe 2001)."""
    # Defaults tuned (parity mode) to a Fellous-2003-like resting regime:
    # ~4-5 Hz spontaneous firing, Vm mean ~-58 mV with ~5 mV fluctuations.
    ge0: float = 12.0     # nS   mean background excitatory conductance
    gi0: float = 10.0     # nS   mean background inhibitory conductance
    sigma_e: float = 6.0  # nS   excitatory conductance fluctuation size
    sigma_i: float = 6.0  # nS   inhibitory conductance fluctuation size


@dataclass
class NetworkBundle:
    """Handles the SDK wrapper needs to drive and read the network."""
    net: Network
    neurons: NeuronGroup
    syn_e: Synapses | None
    syn_i: Synapses | None
    spikemon: SpikeMonitor
    geometry: Geometry
    dt_ms: float
    plastic: bool
    # Plasticity params (present when plastic=True) -- apply_reward uses lr/w_max.
    plasticity: "PlasticityParams | None" = None
    # Namespace object carrying the global neuromodulator M for the three-factor
    # reward rule (set by apply_reward). Present even when plastic=False (unused).
    reward_ctrl: dict = field(default_factory=dict)


# AdEx membrane + adaptation, conductance synapses, and OU background all in one
# equation block. Per-neuron constants (C, gL, ... Vcut) let RS and FS coexist in a
# single NeuronGroup. xi_e/xi_i are independent white-noise terms (Euler-Maruyama).
_NEURON_EQS = """
dv/dt = ( gL*(EL - v)
          + gL*DeltaT*exp((v - VT)/DeltaT)
          - w_adapt
          + g_e*(Ee - v) + g_i*(Ei - v)
          + I_stim ) / C                                          : volt (unless refractory)
dw_adapt/dt = ( a*(v - EL) - w_adapt ) / tau_w                    : amp

dg_e/dt = (ge0 - g_e)/tau_e + sigma_e*sqrt(2/tau_e)*xi_e          : siemens
dg_i/dt = (gi0 - g_i)/tau_i + sigma_i*sqrt(2/tau_i)*xi_i          : siemens

I_stim : amp                                                      : (constant over dt, set externally)

C      : farad    (constant)
gL     : siemens  (constant)
EL     : volt     (constant)
VT     : volt     (constant)
DeltaT : volt     (constant)
tau_w  : second   (constant)
a      : siemens  (constant)
b      : amp      (constant)
Vr     : volt     (constant)
Vcut   : volt     (constant)
refractory_time : second (constant)
"""

# The extra "(constant over dt...)" annotation on I_stim above is a comment, not
# valid Brian2 flags -- strip it to a clean parameter declaration.
_NEURON_EQS = _NEURON_EQS.replace(
    "I_stim : amp                                                      : (constant over dt, set externally)",
    "I_stim : amp",
)

# Clopath low-pass voltage traces, added to the neuron block only when plasticity is
# on. u_p (fast) gates LTP, u_m (slow) gates LTD, u_homeo (very slow) drives the
# homeostatic scaling of LTD. All track v with different time constants.
_PLASTIC_TRACE_EQS = """
du_p/dt     = (v - u_p)/tau_ltp        : volt
du_m/dt     = (v - u_m)/tau_ltd        : volt
du_homeo/dt = (v - u_homeo)/tau_homeo  : volt
"""


def build_network(
    geometry: Geometry,
    connectivity: Connectivity,
    *,
    dt_ms: float = 0.04,
    syn: SynParams | None = None,
    background: BackgroundParams | None = None,
    exc_scale: float = 1.0,
    inh_scale: float = 1.0,
    adaptation_scale: float = 1.0,
    plastic: bool = False,
    plasticity: PlasticityParams | None = None,
    seed: int = 0,
) -> NetworkBundle:
    """Assemble the runtime-mode Brian2 network.

    Args:
        geometry / connectivity: spatial layout and edges from
            :mod:`organoid_simulator.v2.connectivity`.
        dt_ms: integration timestep. Default 0.04 ms == one 25 kHz frame, so one
            Brian step maps to exactly one recorded frame (clean clock reconciliation).
            The AdEx ``exp`` term stays stable at this dt for the parameter sets here;
            drop to 0.02 ms if a hotter regime rings.
        syn / background: synaptic reversals/time-constants and OU background params.
        exc_scale / inh_scale: global multipliers on all excitatory / inhibitory
            synaptic weights -- the primary knobs for tuning the E/I balance toward
            criticality (§M3) without regenerating connectivity.
        plastic: enable §M4 plasticity on excitatory synapses (v2.2). v2.0/2.1: False.
        seed: seeds Brian2's RNG so OU noise + any random init are reproducible.
    """
    syn = syn or SynParams()
    background = background or BackgroundParams()
    if plastic and plasticity is None:
        plasticity = PlasticityParams()

    b2.seed(seed)
    defaultclock.dt = dt_ms * ms

    n = geometry.n_neurons
    eqs = _NEURON_EQS + (_PLASTIC_TRACE_EQS if plastic else "")
    G = NeuronGroup(
        n, eqs,
        threshold="v > Vcut",
        reset="v = Vr; w_adapt += b",
        refractory="refractory_time",   # per-neuron refractory (set below)
        method="euler",
        name="neurons",
    )

    # Per-neuron refractory needs to be a parameter; Brian2 accepts a variable name
    # in `refractory=`. Add it and the shared background/synapse constants via the
    # run namespace. (Declared here rather than in the eqs string to keep that block
    # focused on dynamics.)
    G.namespace.update({
        "Ee": syn.Ee * mV, "Ei": syn.Ei * mV,
        "tau_e": syn.tau_e * ms, "tau_i": syn.tau_i * ms,
        "ge0": background.ge0 * nS, "gi0": background.gi0 * nS,
        "sigma_e": background.sigma_e * nS, "sigma_i": background.sigma_i * nS,
    })
    if plastic:
        G.namespace.update({
            "tau_ltp": plasticity.tau_ltp * ms, "tau_ltd": plasticity.tau_ltd * ms,
            "tau_homeo": plasticity.tau_homeo * ms,
        })

    _assign_cell_params(G, geometry, adaptation_scale)

    # Initial state: rest, adaptation zero, conductances at their OU means.
    G.v = G.EL
    G.w_adapt = 0 * pA
    G.g_e = background.ge0 * nS
    G.g_i = background.gi0 * nS
    G.I_stim = 0 * pA
    if plastic:
        G.u_p = G.EL
        G.u_m = G.EL
        G.u_homeo = G.EL

    objects = [G]

    syn_e = (_build_exc_synapses(G, connectivity, exc_scale, plastic, plasticity)
             if connectivity.n_exc_syn else None)
    syn_i = _build_inh_synapses(G, connectivity, inh_scale) if connectivity.n_inh_syn else None
    if syn_e is not None:
        objects.append(syn_e)
    if syn_i is not None:
        objects.append(syn_i)

    spikemon = SpikeMonitor(G, name="spikes")
    objects.append(spikemon)

    net = Network(*objects)
    return NetworkBundle(
        net=net, neurons=G, syn_e=syn_e, syn_i=syn_i, spikemon=spikemon,
        geometry=geometry, dt_ms=dt_ms, plastic=plastic,
        plasticity=plasticity if plastic else None,
    )


def _assign_cell_params(G: NeuronGroup, geometry: Geometry,
                        adaptation_scale: float = 1.0) -> None:
    """Write the RS/FS parameter sets into the group's per-neuron constants.

    ``adaptation_scale`` (<1) scales down the AdEx adaptation (``a``, ``b``). Strong
    adaptation makes the network a relaxation oscillator that fires an all-or-nothing
    grid-wide burst and then goes refractory -- so evoked propagation alternates
    strong/silent. Reducing it yields consistent, graded, repeatable propagation, which
    is what the interactive viewer and repeated-trial classification want.
    """
    is_exc = geometry.is_exc
    # `refractory_time` is declared in the equation block and referenced by the
    # group's refractory= argument, so Brian2 resolves it as a per-neuron variable.
    for mask, p in ((is_exc, RS_EXC), (~is_exc, FS_INH)):
        idx = np.nonzero(mask)[0]
        if idx.size == 0:
            continue
        G.C[idx] = p.C * pF
        G.gL[idx] = p.gL * nS
        G.EL[idx] = p.EL * mV
        G.VT[idx] = p.VT * mV
        G.DeltaT[idx] = p.DeltaT * mV
        G.tau_w[idx] = p.tau_w * ms
        G.a[idx] = p.a * adaptation_scale * nS
        G.b[idx] = p.b * adaptation_scale * pA
        G.Vr[idx] = p.Vr * mV
        # Detect the spike well above VT, where the exp term has run away.
        G.Vcut[idx] = (p.VT + 5 * p.DeltaT) * mV
        G.refractory_time[idx] = p.refractory * ms


def _build_exc_synapses(G, conn: Connectivity, scale: float, plastic: bool,
                        p: "PlasticityParams | None" = None) -> Synapses:
    """Excitatory (AMPA) synapses: on presynaptic spike, bump target ``g_e`` after
    the edge's conduction delay.

    Static in v2.0/2.1 (``plastic=False``). When ``plastic=True`` the synapse also
    carries the §M4 rule (Clopath voltage-gated STDP writing to a reward-gated
    eligibility trace with homeostatic LTD scaling):

    * ``x_bar`` -- presynaptic trace (event-driven), decays with ``tau_x``.
    * ``e``     -- eligibility trace (clock-driven), decays with ``tau_elig``.
    * on **post** spike: LTP into ``e``, ∝ presynaptic trace × how far the fast
      depolarisation trace ``u_p`` sits above ``theta_ltp`` -- i.e. potentiation is
      *driven by the postsynaptic depolarisation that propagating activity produced*.
    * on **pre** spike: LTD out of ``e``, ∝ how far the slow trace ``u_m`` sits above
      ``theta_ltd``, scaled by a homeostatic factor ``(u_homeo above theta_ltd / u_ref)``
      so chronically depolarised (over-active) neurons depress more -> self-stabilising.

    The weight itself is not touched here; ``apply_reward`` consolidates ``e`` into
    ``w_syn`` (the third, neuromodulatory factor). ``clip(...,0,inf)`` one-sided gates
    keep LTP/LTD from flipping sign, and voltages are divided by ``mV`` to keep the
    dimensionless trace arithmetic unit-clean.
    """
    if not plastic:
        S = Synapses(
            G, G,
            model="w_syn : siemens",
            on_pre="g_e_post += w_syn",
            method="euler",
            name="exc_synapses",
        )
        S.connect(i=conn.exc_pre, j=conn.exc_post)
        S.w_syn = conn.exc_w * nS * scale
        S.delay = conn.exc_delay * ms
        return S

    p = p or PlasticityParams()
    # NB: `e` is reserved (Euler's constant) in Brian2 -- the eligibility trace is `elig`.
    model = """
    w_syn : siemens
    delig/dt = -elig/tau_elig      : 1  (clock-driven)
    dx_bar/dt = -x_bar/tau_x       : 1  (event-driven)
    """
    on_pre = """
    g_e_post += w_syn
    x_bar += 1
    elig = elig - A_ltd * clip((u_m_post - theta_ltd)/mV, 0, inf) * clip((u_homeo_post - theta_ltd)/mV, 0, inf) / u_ref
    """
    on_post = """
    elig = elig + A_ltp * x_bar * clip((u_p_post - theta_ltp)/mV, 0, inf)
    """
    S = Synapses(
        G, G, model=model, on_pre=on_pre, on_post=on_post,
        method="euler", name="exc_synapses",
        namespace={
            "tau_elig": p.tau_elig * ms, "tau_x": p.tau_x * ms,
            "theta_ltp": p.theta_ltp * mV, "theta_ltd": p.theta_ltd * mV,
            "A_ltp": p.A_ltp, "A_ltd": p.A_ltd, "u_ref": p.u_ref_mV,
        },
    )
    S.connect(i=conn.exc_pre, j=conn.exc_post)
    S.w_syn = conn.exc_w * nS * scale
    S.delay = conn.exc_delay * ms
    S.elig = 0
    S.x_bar = 0
    return S


def _build_inh_synapses(G, conn: Connectivity, scale: float) -> Synapses:
    """Inhibitory (GABA) synapses: bump target ``g_i`` after the conduction delay."""
    S = Synapses(
        G, G,
        model="w_syn : siemens",
        on_pre="g_i_post += w_syn",
        method="euler",
        name="inh_synapses",
    )
    S.connect(i=conn.inh_pre, j=conn.inh_post)
    S.w_syn = conn.inh_w * nS * scale
    S.delay = conn.inh_delay * ms
    return S
