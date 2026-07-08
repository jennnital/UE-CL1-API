# `organoid_simulator.v2` — a biophysical Brian2 organoid substrate for the CL1 SDK

A drop-in [`cl.sim.SimulatorDataSource`](https://docs.corticallabs.com/) — exactly like v1's
`LIFDataSource`, so the SDK loop, `live_server`, and demos run against it unchanged — but whose
internals are a **biophysically grounded, delay-coupled, criticality-tuned spiking network in
which learning is mechanistically tied to propagating activity**. This is the v2 rebuild
motivated by five structural limits diagnosed in v1's leaky-integrate-and-fire backbone.

```python
import cl
from organoid_simulator.v2 import make_brian_source

cl.sim.set_simulator_data_source(
    "organoid_simulator.v2.brian_source:make_brian_source",
    config={"n_neurons": 64, "random_seed": 42},   # JSON-serialisable kwargs
)
with cl.open() as neurons:
    for tick in neurons.loop(ticks_per_second=50):
        neurons.stim(cl.ChannelSet(27), cl.StimDesign(600, -3.0, 600, 3.0))
        spikes = tick.analysis.spikes   # evoked by the stim above
```

---

## What v2 fixes (vs v1)

| v1 limitation | v2 mechanism |
|---|---|
| No real plasticity/learning (1-frame coincidence STDP) | **Clopath voltage-gated STDP + reward-modulated three-factor rule** (§M4) |
| Surface-level background noise (display shimmer) | **Ornstein–Uhlenbeck conductance bombardment** + E/I criticality (§M3) |
| No causal pathways (instantaneous coupling) | **Conductance synapses with per-edge conduction delays** (§M2) |
| Propagation not correlated with LTP | LTP is *driven by* the post-synaptic depolarisation from propagating activity (§M4) |
| Classification = spatial + argmax (time-blind) | Substrate is an **LSM "liquid"**; a trained readout on time-resolved features (§M5, v2.3) |

## Architecture

```
organoid_simulator/v2/
  connectivity.py   spatial geometry + distance-dependent connectivity & conduction delays (§M2)
  network.py        Brian2 build: AdEx E/I neurons (§M1), conductance synapses, OU background (§M3), plasticity (§M4)
  recording.py      electrode forward model: neurons -> 64 int16 frames (§M5)
  brian_source.py   BrianOrganoidDataSource(SimulatorDataSource): open/read/on_stims/apply_reward
  readout.py        (v2.3) LSM logistic readout on time-resolved reservoir features
  smoke_test_v2.py  demos_v2.py  criticality_check.py    verification
```

**Population model.** `N` AdEx neurons (80% RS excitatory / 20% FS inhibitory) on a 2-D sheet
spanning the 8×8 MEA. Electrodes *sample and stimulate local populations* through distance-weighted
footprints — the biologically correct picture. Two modes, chosen by `n_neurons`:

* **parity** (`n_neurons=64`): one neuron per electrode, 1:1 — the fast drop-in, Fellous-2003
  resting regime (~5 Hz, reactive). Good for closed-loop sandboxing. ~0.5× real time.
* **sheet** (`n_neurons≈1000`): scattered neurons, denser recurrence, background tuned to sit at
  **criticality**. The realism / reservoir-computing mode. ~0.2× real time.

Mode-appropriate defaults (connectivity density, E/I scales, OU background) are picked
automatically; every field can be overridden per-source (see `PARITY_PRESET` / `SHEET_PRESET` in
`brian_source.py`).

## The model in one screen

* **§M1 Neurons** — adaptive-exponential integrate-and-fire (Brette & Gerstner 2005),
  conductance-based, RS + FS classes. `dt = 0.04 ms` (one 25 kHz frame → clean clock mapping).
* **§M2 Synapses** — `g_e`/`g_i` conductance events with per-edge delays `d = distance / velocity`
  (≈0.3 m/s). Distance becomes latency — the basis of reservoir computation.
* **§M3 Background** — OU excitatory/inhibitory conductance bombardment sets a fluctuating
  high-conductance state; recurrent E/I is tuned near criticality.
* **§M4 Plasticity** (`plasticity=True`) — Clopath voltage traces gate LTP by post-synaptic
  depolarisation; each event writes to a decaying **eligibility trace**; `apply_reward(r)`
  consolidates it into weights (`dw = lr·r·e`). Homeostatic LTD scaling keeps weights bounded.
* **§M5 Recording** — each electrode frame is the distance-weighted (point-source `~1/r`) sum of
  nearby neurons' spike waveforms, quantised to `int16` — the same `frames` contract as v1.

## Verification (all reproduced on this machine)

```bash
CL_SDK_ACCELERATED_TIME=1 python -m organoid_simulator.v2.smoke_test_v2   # §1-2,6 unit/reactivity/delay/throughput
python -m organoid_simulator.v2.criticality_check --n 1000 --duration 15  # §3 criticality (--sweep to retune)
python -m organoid_simulator.v2.demos_v2                                  # §4 propagation↔LTP + reward-gated recall
python -m organoid_simulator.v2.readout                                   # §5 A→B vs B→A order task
```

Representative results:

* **Reactivity + delay** — stim → evoked spikes within 1 tick; a monosynaptic EPSP arrives at
  **2.68 ms** across an edge whose set conduction delay is **2.65 ms** (distance → latency).
* **Criticality** (N=1000) — branching ratio **σ = 1.00**, avalanche exponent **α = 1.52**
  (Beggs & Plenz targets: σ≈1, α≈1.5).
* **Propagation-driven LTP** — causal (pre→post) pairing potentiates **~11×** more than
  anti-causal; reward-gated consolidation **doubles** the postsynaptic EPSP (recall).
* **Order task** — LSM readout **100%** (echo-only control, stim channels masked: **97.5%**),
  while the time-blind count readout that v1 used collapses to **~53%** (chance).

## Performance & caveats

Brian2 runs in **runtime (Cython)** mode so the network can be stepped chunk-by-chunk from the SDK
loop; standalone mode is faster but not chunk-drivable. Expect ~0.5× real time at `N=64` and ~0.2×
at `N≈1000` on CPU — fine for offline experiments and accelerated-time runs, slower than v1's pure
numpy. Criticality (a narrow E/I band) and plasticity stability are genuine tuning, not turnkey;
`criticality_check.py --sweep` re-locates the critical background working point for other `N`.
`SpikeMonitor` retains all spikes for the session, so extremely long live runs grow memory —
reopen the source to reset.
