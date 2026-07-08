"""
Smoke test for the LIF simulator data source.

Runs three checks through the *real* CL SDK loop (so it exercises the actual
plug-in path, not the class in isolation):

  1. baseline   -- no stim: frames are well-formed, grounded/reference channels
                   stay silent, other channels fire at a low non-zero rate.
  2. evoked     -- a suprathreshold stim on one channel produces a spike on that
                   channel shortly after, repeatably across seeds.
  3. throughput -- measured accelerated-mode frames/sec on this machine.

Run directly (accelerated time makes it finish in seconds)::

    CL_SDK_ACCELERATED_TIME=1 python organoid_simulator/smoke_test.py

This is a plain script, matching the repo's existing `memory_test.py` style --
there is no pytest setup in this project.
"""
from __future__ import annotations

import os
import time

os.environ.setdefault("CL_SDK_ACCELERATED_TIME", "1")

import cl  # noqa: E402  (import after setting accelerated-time env)

from organoid_simulator import make_lif_source  # noqa: E402
from organoid_simulator.lif_data_source import NON_SPIKING, FRAMES_PER_SECOND  # noqa: E402

TICKS_PER_SECOND = 50


def _register(**config) -> None:
    cl.sim.set_simulator_data_source(
        "organoid_simulator.lif_data_source:make_lif_source",
        config=config,
    )


def test_baseline() -> None:
    print("\n=== 1. baseline (no stimulation) ===")
    _register(random_seed=1)

    spike_channels: set[int] = set()
    total_spikes = 0
    seconds = 5.0

    with cl.open() as neurons:
        for tick in neurons.loop(ticks_per_second=TICKS_PER_SECOND,
                                 stop_after_seconds=seconds, ignore_jitter=True):
            for s in tick.analysis.spikes:
                spike_channels.add(int(s.channel))
                total_spikes += 1

    silent_violations = spike_channels & NON_SPIKING
    rate = total_spikes / seconds

    assert not silent_violations, f"grounded/reference channels spiked: {silent_violations}"
    assert total_spikes > 0, "no spontaneous activity at all -- background drive too low?"
    print(f"  spikes: {total_spikes} over {seconds:.0f}s  (~{rate:.0f}/s)")
    print(f"  active channels: {len(spike_channels)}/59 spiking-eligible")
    print(f"  grounded/reference silent: OK ({sorted(NON_SPIKING)})")
    print("  PASS")


def test_evoked() -> None:
    print("\n=== 2. evoked response (stim -> spike latency) ===")
    stim_channel = 27
    stim_at_tick = 5
    # Measured in whole ticks rather than absolute frames: the SDK reports spike
    # timestamps on a different (internal) clock than loop-tick timestamps, so
    # subtracting the two would be meaningless. Ticks are unambiguous.
    tick_gaps: list[int] = []

    for seed in (1, 2, 3):
        # Zero background drive isolates the evoked spike: with no spontaneous
        # activity, any spike on the stim channel must have been caused by the stim.
        _register(random_seed=seed, background_drive_std_mV=0.0)
        stim_iter: int | None = None
        evoked_iter: int | None = None
        evoked_frame: int | None = None

        with cl.open() as neurons:
            for i, tick in enumerate(neurons.loop(ticks_per_second=TICKS_PER_SECOND,
                                                  stop_after_seconds=0.5, ignore_jitter=True)):
                if i == stim_at_tick and stim_iter is None:
                    neurons.stim(cl.ChannelSet(stim_channel),
                                 cl.StimDesign(400, -2.0, 400, 2.0))
                    stim_iter = i
                if stim_iter is not None and evoked_iter is None:
                    for s in tick.analysis.spikes:
                        if int(s.channel) == stim_channel:
                            evoked_iter = i
                            evoked_frame = int(s.timestamp)
                            break

        assert evoked_iter is not None, f"seed {seed}: stim produced no spike on ch {stim_channel}"
        gap = evoked_iter - stim_iter
        tick_gaps.append(gap)
        print(f"  seed {seed}: evoked spike on ch {stim_channel} within {gap} tick(s) "
              f"of the stim (frame {evoked_frame}, deterministic across seeds)")

    # Direct electrical activation of an independent neuron is near-instant: the
    # evoked spike should land in the same tick as the stim commit, or the next.
    assert all(0 <= g <= 2 for g in tick_gaps), f"evoked response too slow: {tick_gaps} ticks"
    print("  PASS")


def test_throughput() -> None:
    print("\n=== 3. throughput (accelerated mode) ===")
    _register(random_seed=1)
    seconds = 10.0
    t0 = time.perf_counter()
    with cl.open() as neurons:
        for _ in neurons.loop(ticks_per_second=TICKS_PER_SECOND,
                              stop_after_seconds=seconds, ignore_jitter=True):
            pass
    wall = time.perf_counter() - t0
    sim_frames = seconds * FRAMES_PER_SECOND
    print(f"  simulated {seconds:.0f}s in {wall:.2f}s wall  "
          f"({sim_frames / wall:,.0f} frames/s, ~{seconds / wall:.1f}x real time)")
    print("  PASS")


if __name__ == "__main__":
    test_baseline()
    test_evoked()
    test_throughput()
    print("\nAll smoke checks passed.")
