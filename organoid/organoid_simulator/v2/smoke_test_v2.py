"""
Smoke test for the v2 Brian2 organoid data source.

Runs the checks the plan's Verification §1-2 + §6 call for, through the *real* CL SDK
loop (so it exercises the actual plug-in path, not the class in isolation):

  1. baseline    -- no stim: frames are well-formed int16, grounded/reference channels
                    stay silent, other channels fire at a low non-zero (Fellous-like) rate.
  2. evoked      -- a stim on one electrode produces spikes shortly after (reactivity).
  3. delay       -- (network-level) a monosynaptic EPSP arrives after the edge's set
                    conduction delay, proving causal delayed pathways (§M2).
  4. throughput  -- measured accelerated-mode frames/sec on this machine (§6, honest
                    reporting of the Brian2-runtime bottleneck).

Run (accelerated time makes it finish in a reasonable wall time)::

    CL_SDK_ACCELERATED_TIME=1 python -m organoid_simulator.v2.smoke_test_v2

Plain script style, matching v1's ``smoke_test.py`` (no pytest in this repo).
"""
from __future__ import annotations

import os
import time

# Allow running as a plain file as well as via `-m`: put the repo root on sys.path so
# the `organoid_simulator.*` imports below resolve.
if __name__ == "__main__" and __package__ in (None, ""):
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

os.environ.setdefault("CL_SDK_ACCELERATED_TIME", "1")

import numpy as np  # noqa: E402

import cl  # noqa: E402  (import after setting accelerated-time env)

from organoid_simulator.lif_data_source import NON_SPIKING, FRAMES_PER_SECOND  # noqa: E402

TICKS_PER_SECOND = 50
FACTORY = "organoid_simulator.v2.brian_source:make_brian_source"


def _register(**config) -> None:
    cl.sim.set_simulator_data_source(FACTORY, config=config)


def _warm_cython_cache() -> None:
    """Build + briefly run the network once outside the SDK loop.

    Brian2 compiles a fresh Cython module the first time it sees a given set of
    equations; on a cold cache that compile can exceed the SDK producer's first-tick
    timeout and abort the very first ``cl.open()``. Warming the cache here makes the
    subsequent SDK-driven runs start instantly.
    """
    from brian2 import ms
    from organoid_simulator.v2.connectivity import build_geometry, build_connectivity
    from organoid_simulator.v2.network import build_network
    geo = build_geometry(64, seed=0)
    conn = build_connectivity(geo, seed=0)
    build_network(geo, conn, seed=0).net.run(20 * ms)


def test_baseline() -> None:
    print("\n=== 1. baseline (no stimulation) ===")
    _register(n_neurons=64, random_seed=1)

    spike_channels: set[int] = set()
    total_spikes = 0
    seconds = 3.0

    with cl.open() as neurons:
        for tick in neurons.loop(ticks_per_second=TICKS_PER_SECOND,
                                 stop_after_seconds=seconds, ignore_jitter=True):
            for s in tick.analysis.spikes:
                spike_channels.add(int(s.channel))
                total_spikes += 1

    silent_violations = spike_channels & NON_SPIKING
    rate = total_spikes / seconds

    assert not silent_violations, f"grounded/reference channels spiked: {silent_violations}"
    assert total_spikes > 0, "no spontaneous activity -- background too weak?"
    print(f"  spikes: {total_spikes} over {seconds:.0f}s  (~{rate:.0f}/s across the array)")
    print(f"  active channels: {len(spike_channels)}/59 spiking-eligible")
    print(f"  grounded/reference silent: OK ({sorted(NON_SPIKING)})")
    print("  PASS")


def test_evoked() -> None:
    print("\n=== 2. evoked response (stim -> spikes) ===")
    stim_channel = 27
    stim_at_tick = 15   # after a short warm-up
    gaps: list[int] = []

    for seed in (1, 2, 3):
        # Quiet the OU background so any spike near the stim site is stim-caused.
        _register(n_neurons=64, random_seed=seed,
                  background=dict(ge0=8, gi0=10, sigma_e=1.0, sigma_i=1.0))
        stim_iter = evoked_iter = None

        with cl.open() as neurons:
            for i, tick in enumerate(neurons.loop(ticks_per_second=TICKS_PER_SECOND,
                                                  stop_after_seconds=1.0, ignore_jitter=True)):
                if i == stim_at_tick and stim_iter is None:
                    neurons.stim(cl.ChannelSet(stim_channel),
                                 cl.StimDesign(600, -3.0, 600, 3.0))
                    stim_iter = i
                if stim_iter is not None and evoked_iter is None:
                    if tick.analysis.spikes:
                        evoked_iter = i

        assert evoked_iter is not None, f"seed {seed}: stim produced no evoked spikes"
        gaps.append(evoked_iter - stim_iter)
        print(f"  seed {seed}: evoked activity within {evoked_iter - stim_iter} tick(s) of the stim")

    assert all(0 <= g <= 3 for g in gaps), f"evoked response too slow: {gaps} ticks"
    print("  PASS")


def test_delay() -> None:
    """Network-level proof of §M2: a monosynaptic EPSP arrives after the edge delay."""
    print("\n=== 3. conduction delay (distance -> latency) ===")
    from brian2 import ms, mV, nA, StateMonitor
    from organoid_simulator.v2.connectivity import build_geometry, build_connectivity
    from organoid_simulator.v2 import network as netmod
    from organoid_simulator.v2.network import BackgroundParams

    geo = build_geometry(64, seed=0)
    conn = build_connectivity(geo, seed=0)
    k = int(np.argmax(conn.exc_delay))
    pre, post, d = int(conn.exc_pre[k]), int(conn.exc_post[k]), float(conn.exc_delay[k])

    bg = BackgroundParams(ge0=6, gi0=8, sigma_e=0.0, sigma_i=0.0)   # silent: isolate EPSP
    bundle = netmod.build_network(geo, conn, background=bg, exc_scale=12.0, seed=1)
    G = bundle.neurons
    sm = StateMonitor(G, "v", record=[post], dt=0.04 * ms)
    bundle.net.add(sm)

    bundle.net.run(150 * ms)          # settle
    G.I_stim[pre] = 5 * nA
    bundle.net.run(3 * ms)
    G.I_stim[pre] = 0 * nA
    bundle.net.run(25 * ms)

    si = np.asarray(bundle.spikemon.i); st = np.asarray(bundle.spikemon.t / ms)
    pre_t = st[si == pre][0]
    t = np.array(sm.t / ms); v = np.array(sm.v[0] / mV)
    dv = np.gradient(v, t)
    win = t > pre_t
    onset = t[win][np.argmax(dv[win] > 2.0)]
    latency = onset - pre_t
    print(f"  edge {pre}->{post}: set delay {d:.2f} ms, measured EPSP latency {latency:.2f} ms")
    assert abs(latency - d) < 0.5, f"conduction latency {latency:.2f} != set delay {d:.2f}"
    print("  PASS")


def test_throughput() -> None:
    print("\n=== 4. throughput (accelerated mode) ===")
    _register(n_neurons=64, random_seed=1)
    seconds = 3.0
    t0 = time.perf_counter()
    with cl.open() as neurons:
        for _ in neurons.loop(ticks_per_second=TICKS_PER_SECOND,
                              stop_after_seconds=seconds, ignore_jitter=True):
            pass
    wall = time.perf_counter() - t0
    sim_frames = seconds * FRAMES_PER_SECOND
    print(f"  simulated {seconds:.0f}s in {wall:.2f}s wall  "
          f"({sim_frames / wall:,.0f} frames/s, ~{seconds / wall:.2f}x real time)")
    print("  (Brian2 runtime/Cython is the bottleneck; N=64 parity keeps it usable.)")
    print("  PASS")


if __name__ == "__main__":
    _warm_cython_cache()
    test_baseline()
    test_evoked()
    test_delay()
    test_throughput()
    print("\nAll v2 smoke checks passed.")
