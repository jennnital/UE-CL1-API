"""
Spatio-temporal learning demo for the LIF organoid simulator.

Motivation: the original plasticity rule paired a spike only with the
*immediately preceding* frame (a 40 us window), so it could reinforce
simultaneous / spatially-coupled co-firing but never bind events separated by
milliseconds. That makes classification lean entirely on spatial proximity and
fail on signals whose information is in *timing*.

This demo shows the fix: the trace-based STDP rule (LIFDataSource(..., stdp=True))
gives the plasticity a real millisecond pairing window, so the network can learn
a temporal sequence A->B and later *recall* B when shown A alone -- and the
association is directional (order-sensitive), which spatial coincidence is not.

Three panels (saved to organoid_simulator/demo_output/D_spatiotemporal.png):
  1. Pairing window  -- learned W[B<-A] vs the A->B training gap, for the old
                        1-frame rule vs trace STDP.
  2. Recall behaviour -- stimulate A alone; count B's evoked spikes before vs
                        after training, old rule vs STDP.
  3. Directionality   -- after training A->B with STDP, probe A-alone (recalls B)
                        vs B-alone (does NOT recall A).

Run:  uv run organoid_simulator/spatiotemporal_demo.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from cl.sim import DataSourceStim
from organoid_simulator.lif_data_source import LIFDataSource

OUT = Path(__file__).parent / "demo_output"
OUT.mkdir(exist_ok=True)

A, B = 20, 43          # far-apart electrodes: no initial coupling either way


def burst(ch, start, n=8, period=25, amp=2.8, pw=400):
    return [DataSourceStim(timestamp=start + p * period, channel=ch,
            phase_durations_us=(pw, pw), phase_currents_uA=(-amp, amp)) for p in range(n)]


def _make_source(stdp: bool, lr: float = 0.03, weight_max: float = 14.0) -> LIFDataSource:
    # coupling_gain=0 -> the ONLY route from A to B is the *learned* weight, so
    # any recall is unambiguously due to plasticity, not built-in coupling.
    s = LIFDataSource(random_seed=1, coupling_gain_mV=0.0, plasticity=True, stdp=stdp,
                      plasticity_lr=lr, weight_max_mV=weight_max, background_drive_std_mV=0.05,
                      stdp_tau_pre_ms=20.0)
    s.open()
    return s


def _run(s, ts, stims, window, block=50):
    if stims:
        s.on_stims(stims)
    times, t, end = [], ts, ts + window
    while t < end:
        for sp in s.read(t, min(block, end - t)).spikes:
            times.append((sp.timestamp - ts, sp.channel))
        t += block
    return times, ts + window


def _train_AB(s, ts, lag, trials=40):
    for _ in range(trials):
        _, ts = _run(s, ts, burst(A, ts + 5) + burst(B, ts + 5 + lag), max(1600, lag + 800))
        s.apply_reward(+1.0)
    return ts


def _probe_count(s, ts, drive_ch, target_ch, n_pulses=12):
    """Stimulate `drive_ch` alone; count spikes on `target_ch` (recall)."""
    times, ts = _run(s, ts, burst(drive_ch, ts + 5, n=n_pulses), 1600)
    n = sum(1 for (_, ch) in times if ch == target_ch)
    return n, ts


def panel1_pairing_window(ax):
    lags = [1, 25, 50, 100, 200, 400, 800]
    w_stdp, w_old = [], []
    for stdp, out in ((True, w_stdp), (False, w_old)):
        for lag in lags:
            s = _make_source(stdp)
            _train_AB(s, 0, lag, trials=30)
            out.append(s.coupling_weights[B, A])
    ms = [l * 0.04 for l in lags]
    ax.plot(ms, w_stdp, "o-", color="#2a9d8f", label="trace STDP (τ=20 ms)")
    ax.plot(ms, w_old,  "s--", color="#999999", label="original 1-frame rule")
    ax.set_xscale("log")
    ax.set_xlabel("A→B training gap (ms)"); ax.set_ylabel("learned W[B←A] (mV)")
    ax.set_title("1. Temporal pairing window"); ax.legend()


def panel2_recall(ax):
    labels, before, after = [], [], []
    for stdp, name in ((False, "1-frame rule"), (True, "trace STDP")):
        s = _make_source(stdp, lr=0.15, weight_max=25.0)
        pre, ts = _probe_count(s, 0, drive_ch=A, target_ch=B)     # A-alone before training
        ts = _train_AB(s, ts, lag=100, trials=80)                 # learn A->B (4 ms gap)
        post, ts = _probe_count(s, ts, drive_ch=A, target_ch=B)   # A-alone after training
        labels.append(name); before.append(pre); after.append(post)
    x = np.arange(len(labels)); w = 0.35
    ax.bar(x - w/2, before, w, label="before training", color="#555b6e")
    ax.bar(x + w/2, after,  w, label="after training",  color="#2a9d8f")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("B spikes evoked by stimulating A alone")
    ax.set_title("2. Recall: does A alone now evoke B?"); ax.legend()
    return after


def panel3_directionality(ax):
    s = _make_source(stdp=True, lr=0.15, weight_max=25.0)
    ts = _train_AB(s, 0, lag=100, trials=80)          # train A->B only
    ab, ts = _probe_count(s, ts, drive_ch=A, target_ch=B)   # A should recall B
    ba, ts = _probe_count(s, ts, drive_ch=B, target_ch=A)   # B should NOT recall A
    ax.bar(["stim A → count B", "stim B → count A"], [ab, ba],
           color=["#2a9d8f", "#e76f51"])
    ax.set_ylabel("evoked spikes on the other electrode")
    ax.set_title("3. Directionality (order-sensitive)")
    return ab, ba


def main():
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
    panel1_pairing_window(ax[0])
    recall = panel2_recall(ax[1])
    ab, ba = panel3_directionality(ax[2])
    fig.suptitle("Spatio-temporal learning: trace STDP lets the network learn a sequence A→B "
                 "and recall it — the 1-frame rule cannot", fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "D_spatiotemporal.png", dpi=110)
    plt.close(fig)

    print(f"[recall] B-spikes from A-alone after training:  1-frame rule={recall[0]}, trace STDP={recall[1]}")
    print(f"[directionality] after training A→B (STDP):  stim A → {ab} B-spikes,  stim B → {ba} A-spikes")
    print(f"figure written to {OUT / 'D_spatiotemporal.png'}")


if __name__ == "__main__":
    main()
