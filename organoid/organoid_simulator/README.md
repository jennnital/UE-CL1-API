# `organoid_simulator` — a reactive LIF simulator for the CL1 SDK

A drop-in [`cl.sim.SimulatorDataSource`](https://docs.corticallabs.com/) that, unlike the SDK's
built-in `RandomDataSource`, **reacts to stimulation**. Stimulating a channel injects current
into a model neuron, which can spike, and (with coupling) drive its neighbours — so you can
sandbox closed-loop stimulus→response hypotheses, plasticity, and read-out design on a laptop
before spending money on real wetware.

This document explains what the model is, every knob it exposes, how connectivity and plasticity
work, and how to use the tools. All code snippets are runnable from the repo root.

---

## 1. Why this exists

The SDK ships a simulator, but its `RandomDataSource` generates Poisson noise and **ignores
stimulation entirely** (`on_stims()` is never overridden) — stimulating a channel has zero effect
on what comes back. That's fine for building an API against, useless for modelling dynamics.

`LIFDataSource` implements the same `SimulatorDataSource` contract but runs an actual dynamical
system, so the loop you already use is unchanged:

```python
import cl
from organoid_simulator import make_lif_source

cl.sim.set_simulator_data_source(
    "organoid_simulator.lif_data_source:make_lif_source",
    config={"random_seed": 42, "coupling_gain_mV": 6.0},   # JSON-serialisable kwargs
)

with cl.open() as neurons:
    for tick in neurons.loop(ticks_per_second=50):
        neurons.stim(cl.ChannelSet(10), cl.StimDesign(200, -1.5, 200, 1.5))
        spikes = tick.analysis.spikes      # now genuinely reacts to the stim above
```

---

## 2. How it plugs into the SDK

The SDK's producer subprocess calls three methods on the data source:

| Method | When | What we do |
|---|---|---|
| `metadata` | once, before `open()` | declare 64 ch, 25 kHz, `seekable=False` (state depends on stim history) |
| `on_stims(stims)` | each tick, **before** `read()` | expand each stim pulse into per-frame injected current, keyed by exact timestamp |
| `read(from_ts, n)` | each tick | integrate the model `n` frames, return `DataSourceBatch(frames, spikes)` |

Because `on_stims` runs before `read` for the same window, a stim can drive a spike in the very
batch it lands in, and stim→spike latency is preserved to the frame (40 µs), not quantised to a tick.

You don't have to go through the SDK at all — for experiments it's often cleaner to drive the object
directly (this is exactly what the demos do):

```python
from organoid_simulator import LIFDataSource
from cl.sim import DataSourceStim

src = LIFDataSource(random_seed=0, coupling_gain_mV=6.0)
src.open()
# stimulate channel 27 with a cathodic-first biphasic pulse at frame 100
src.on_stims([DataSourceStim(timestamp=100, channel=27,
                             phase_durations_us=(400, 400), phase_currents_uA=(-2.8, 2.8))])
batch = src.read(0, 1000)                 # advance 1000 frames (40 ms)
print(len(batch.spikes), "spikes;", batch.frames.shape, "int16 frames")
```

---

## 3. The neuron model

Each of the 64 channels is a **leaky integrate-and-fire** neuron. Per frame (dt = 40 µs):

```
V += (dt/τ_mem) · (V_rest − V) + drive              # membrane update, all terms in mV
if V ≥ V_thresh and not refractory:  emit spike; V ← V_reset; start refractory
```

`drive` (mV) is the sum of:

- **Background** `N(0, background_drive_std_mV)` per frame — a stochastic membrane drive that
  produces spontaneous firing. Higher → livelier baseline; 0 → silent.
- **Stimulation** `stim_gain_mV_per_uA · max(0, −I_ext)` — see below.
- **Coupling** `Σ_j W[i,j] · fired_j(t−1)` — drive from presynaptic partners that fired last frame.

Three modelling choices worth understanding (they are deliberate, and documented in the source):

1. **The recorded frames are extracellular, not membrane voltage.** A real MEA records a small
   (tens-of-µV) noisy echo, not the ~100 mV intracellular swing. So `V` never touches `frames`;
   frames are a Gaussian noise floor + a canonical extracellular spike waveform stamped in when a
   neuron fires. (Writing `V` into an int16 frame would overflow ±32767 by ~15× and be the wrong
   physical quantity.)
2. **Only the cathodic (depolarising) phase of a stim excites.** CL pulses are charge-balanced, so a
   purely linear integrator would net ~zero drive and never fire. Physically the leading cathodic
   phase drives the spike and threshold makes it irreversible; we model that with
   `max(0, −I_ext)` (negative current = cathodic = excitatory, by CL convention).
3. **A finite-range clamp** keeps `V` in [−150, +100] mV — far outside physiology, so it never
   distorts real dynamics; it only stops a runaway coupling "seizure" from producing NaNs.

**Performance:** the threshold-and-reset nonlinearity forces a sequential per-frame Python loop
(vectorised only across the 64 channels), so accelerated mode runs ~5–50× real time — ample for
closed-loop iteration, not a GPU kernel.

---

## 4. Connectivity — weighted, small-world, pluggable

`W[i, j]` is the weight of the directed edge from presynaptic `j` to postsynaptic `i`; the coupling
drive to `i` is `Σ_j W[i,j]·fired_j`. Grounded/reference channels (0, 4, 7, 56, 63) and self-edges
are always zeroed. The **actual coupling = this graph × `coupling_gain_mV`**, so `coupling_gain_mV`
is a single global knob and the matrix sets the *shape*.

Two generators ship in the module (both return a 64×64 matrix):

```python
from organoid_simulator import grid_connectivity, small_world_connectivity
```

| Graph | Clustering | Avg path | Character |
|---|---|---|---|
| `grid_connectivity()` | 0.00 | 5.1 | regular 4-neighbour lattice, all weights 1 |
| `small_world_connectivity()` | 0.46 | 2.2 | **default** — weighted, high clustering + short paths |

The default `small_world_connectivity()` is a spatially-embedded Watts–Strogatz graph: each neuron
receives from its `k` spatially-nearest neighbours (dense local structure → high clustering), a
fraction `rewire_p` of those edges are rewired to random long-range sources (→ short paths), edge
weights are lognormal (a few strong pathways), and the whole matrix is scaled so the mean total
incoming weight matches the grid's — keeping `coupling_gain_mV` comparable across graphs. Signals
therefore diffuse locally but occasionally jump — closer to a real culture than uniform diffusion.

**`LIFDataSource` uses a fresh small-world graph by default.** Set `coupling_gain_mV > 0` to
activate coupling (it is 0 by default, i.e. independent neurons).

### Plug in your own graph

```python
import numpy as np
from organoid_simulator import LIFDataSource, small_world_connectivity, grid_connectivity

# (a) tune the built-in small-world generator
LIFDataSource(coupling_gain_mV=6.0, small_world_k=6, small_world_rewire_p=0.3,
              small_world_weight_sigma=0.8, connectivity_seed=7)

# (b) supply any 64×64 weight matrix you like
my_W = small_world_connectivity(k=10, rewire_p=0.05, seed=1)
my_W[20, 43] = 5.0                                  # hand-craft a strong 43→20 edge
LIFDataSource(coupling_gain_mV=6.0, connectivity=my_W)

# (c) use a functional-connectivity matrix you measured, or the plain grid
LIFDataSource(coupling_gain_mV=6.0, connectivity=grid_connectivity())
```

Inspect the resolved graph with `src.connectivity_matrix` (structure) and `src.coupling_weights`
(= structure × gain, or the learned matrix if plasticity is on). Change the global strength live
with `src.set_coupling_gain(new_gain)`.

> Note: a custom matrix is passed as a NumPy array, so it only works via direct instantiation, not
> through `set_simulator_data_source(config=...)` (config must be JSON-serialisable). The small-world
> *parameters* (seed/k/p) do serialise, so the SDK path can still pick a specific generated graph.

---

## 5. Plasticity — reward-modulated learning (opt-in)

Off by default (the substrate is fixed). Turn on with `plasticity=True`. It's a **three-factor
rule**: an eligibility trace records *what fired together*, and a later `apply_reward(r)` call
consolidates it (`W += lr · r · eligibility`, then clip to `[0, weight_max_mV]`). Call
`apply_reward` between closed-loop episodes; it resets the trace so each reward consolidates only
recent activity.

```python
src = LIFDataSource(coupling_gain_mV=2.0, plasticity=True, plasticity_lr=0.1, weight_max_mV=14.0)
src.open()
# ... run some episodes, driving activity you want to reinforce ...
src.apply_reward(+1.0)     # potentiate recently co-active pairs (−1.0 to depress)
W = src.coupling_weights   # inspect what was learned
```

### Two eligibility rules

- **Default (1-frame rule):** pairs a spike only with the *immediately preceding* frame (40 µs). It
  reinforces near-simultaneous co-firing — i.e. **spatial** coupling. It cannot bind events separated
  by milliseconds.
- **Trace STDP (`stdp=True`):** each neuron keeps decaying pre/post traces (`stdp_tau_pre_ms`,
  `stdp_tau_post_ms`), so a firing neuron credits synapses from partners active over the last ~τ —
  a real **millisecond** pairing window, and directional (causal pre→post potentiation, acausal
  depression scaled by `stdp_ltd_ratio`). This is what lets the network learn temporal sequences.

```python
# learn a temporal association A→B across a 4 ms gap, then recall B from A alone:
src = LIFDataSource(coupling_gain_mV=0.0, plasticity=True, stdp=True,
                    plasticity_lr=0.15, weight_max_mV=25.0, stdp_tau_pre_ms=20.0)
```

See `spatiotemporal_demo.py` for the full before/after story. The **why** — including the diagnostic
that showed the 1-frame rule can't cross ms gaps — is in [§8 Findings](#8-findings--honest-limits).

---

## 6. Parameter reference

`LIFDataSource(...)` — all optional:

| Param | Default | Meaning |
|---|---|---|
| `random_seed` | `None` | RNG seed (falls back to `CL_SDK_RANDOM_SEED`, then random) |
| `tau_mem_ms` | 20 | membrane time constant |
| `v_rest_mV` / `v_thresh_mV` / `v_reset_mV` | −70 / −50 / −75 | resting / threshold / reset potential |
| `refractory_ms` | 2 | absolute refractory period |
| `background_drive_std_mV` | 0.60 | std of stochastic membrane drive (spontaneous rate) |
| `stim_gain_mV_per_uA` | 5.0 | depolarisation per µA of cathodic stim current |
| `spike_amplitude_uV` | 80 | scale of the stamped extracellular spike waveform |
| `noise_floor_uV` | 6.0 | std of recording-noise floor in the frames |
| `coupling_gain_mV` | **0.0** | global coupling strength (0 = independent neurons) |
| `stim_artifact` / `stim_artifact_gain_uV_per_uA` | False / 4000 | cosmetic amplifier-saturation bump on stimulated channels |
| `plasticity` | False | enable reward-modulated learning |
| `plasticity_lr` / `eligibility_tau_ms` / `weight_max_mV` | 0.02 / 200 / 6.0 | learning rate / trace decay / weight clip |
| `stdp` | False | use the trace-based STDP rule (ms window) instead of the 1-frame rule |
| `stdp_tau_pre_ms` / `stdp_tau_post_ms` / `stdp_ltd_ratio` | 20 / 20 / 0.5 | STDP trace time constants / depression strength |
| `connectivity` | `None` | custom 64×64 weight matrix; `None` → generated small-world |
| `small_world_k` / `small_world_rewire_p` / `small_world_weight_sigma` / `connectivity_seed` | 8 / 0.15 / 0.5 / 0 | small-world generator params |

Key methods: `open()` / `close()` (lifecycle), `read(from_ts, n)`, `on_stims(stims)`,
`apply_reward(r)`, `set_coupling_gain(g)`; properties `coupling_weights`, `connectivity_matrix`.

---

## 7. The tools

All are self-contained; run from the repo root.

| File | What | Run |
|---|---|---|
| `smoke_test.py` | validates the in-SDK-loop path (baseline / evoked latency / throughput) | `CL_SDK_ACCELERATED_TIME=1 PYTHONPATH=. python organoid_simulator/smoke_test.py` |
| `demonstrations.py` | figures for (A) classification, (B) functional connectivity, (C) reinforcement | `PYTHONPATH=. python organoid_simulator/demonstrations.py` → `demo_output/*.png` |
| `spatiotemporal_demo.py` | sequence learning & recall with trace STDP vs the 1-frame rule | `uv run organoid_simulator/spatiotemporal_demo.py` → `demo_output/D_spatiotemporal.png` |
| `live_server.py` | **interactive** 8×8 grid; click to stimulate, watch it propagate (small-world) | `uv run organoid_simulator/live_server.py` → http://127.0.0.1:8008 |
| `classify_server.py` | **interactive** classification: encoder → organoid → linear decoder, live | `uv run organoid_simulator/classify_server.py` → http://127.0.0.1:8011 |

The two servers stream to the browser over Server-Sent Events and take clicks via POST — pure
standard library, no web dependencies. `live_server.py` uses the small-world default (realistic
propagation); `classify_server.py` is pinned to the grid on purpose (it teaches the *spatial* band
code, which small-world shortcuts would scramble).

---

## 8. Findings & honest limits

This simulator was built incrementally, and a few results are worth carrying forward:

- **Classification here is spatial.** With distinct electrodes per class and a spike-*count*
  read-out, classification is essentially "which region lit up". Crank the coupling slider in
  `classify_server.py` and watch confidence blur as activity spreads across bands.
- **The count read-out is blind to timing.** On a task where classes differ *only* in temporal order
  (A→B vs B→A, identical electrodes and counts), a count read-out scores 50% (chance) while a
  time-binned read-out scores 100% — the information is in the response, the read-out discards it.
- **The 1-frame plasticity rule is spatial; trace STDP fixes the window.** Training an A→B
  association: the 1-frame rule leaves the weight at 0 for any gap >40 µs; trace STDP grows it across
  0.04–16 ms and directionally (A→B, not B→A), enabling sequence recall (`spatiotemporal_demo.py`).
- **Still missing for full spatio-temporal computation:** a time-resolved read-out (cheap) and
  heterogeneous synaptic *delays* / diverse time constants (a "richer reservoir"). Without delays, a
  rate read-out neuron can't be made order-selective regardless of weights.
- **Simplifications:** no true amplifier blanking (the stim artifact is cosmetic); the charge→voltage
  map is linear (not a strength-duration curve); reward-modulated learning uses global credit
  assignment, so it learns but plateaus and can erode with over-training.

These are limits of a deliberately minimal model, not bugs — they're the honest boundary of what a
homogeneous LIF network with a count read-out can do.
