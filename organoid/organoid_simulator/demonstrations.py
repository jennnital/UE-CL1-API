"""
Capability demonstrations for the LIF organoid simulator.

Three self-contained demos, each saving a figure to ``organoid_simulator/demo_output/``:

  A) classification   -- distinct stimulation patterns evoke distinct, linearly
                          decodable spatial spike responses (reservoir readout).
  B) connectivity      -- Spearman functional connectivity (the repo's own
                          metrics.graph_calculations) recovers the ground-truth
                          coupling structure when coupling is enabled.
  C) reinforcement     -- a 2-choice sensorimotor task learned online via
                          reward-modulated plasticity, with DishBrain / free-energy
                          style structured-vs-chaotic feedback; a plasticity-OFF run
                          is the honest control (stays at chance).

These drive the ``LIFDataSource`` object directly rather than through ``cl.open()``:
the demos need trial resets, weight read-out and reward signalling that the stock
SDK loop API does not expose, and driving the source object exercises the exact
same dynamics the SDK's producer would. The full in-SDK-loop path is validated
separately by ``organoid_simulator/smoke_test.py``.

Run:  PYTHONPATH=. python organoid_simulator/demonstrations.py
"""
from __future__ import annotations

import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from cl.sim import DataSourceStim

from organoid_simulator.lif_data_source import (
    LIFDataSource, CHANNEL_COUNT, NON_SPIKING, grid_connectivity,
)

try:
    from metrics.graph_calculations import construct_functional_connectivity
except ImportError:
    # metrics/graph_calculations.py has a dead top-level `from gnm import metrics`
    # that fails on the installed gnm version (the name isn't exported, and nothing
    # in the file uses it). Stub the attribute so we can use the repo's real,
    # unrelated FC code without editing that file. See writeup for the suggested fix.
    import types
    import gnm
    if not hasattr(gnm, "metrics"):
        gnm.metrics = types.ModuleType("gnm.metrics")
    from metrics.graph_calculations import construct_functional_connectivity

OUT_DIR = Path(__file__).parent / "demo_output"
OUT_DIR.mkdir(exist_ok=True)


# --------------------------------------------------------------------------- #
# Shared helpers for driving the source directly.
# --------------------------------------------------------------------------- #

def ch(row: int, col: int) -> int:
    """8x8 MEA channel index (channel = row + 8*col, matching the notebooks)."""
    return row + 8 * col


def grid_image(vec: np.ndarray) -> np.ndarray:
    """Reshape a 64-vector into an 8x8 image (img[row, col] = channel row+8*col)."""
    img = np.full((8, 8), np.nan)
    for j in range(CHANNEL_COUNT):
        img[j % 8, j // 8] = vec[j]
    return img


def biphasic(channel: int, timestamp: int, amp_uA: float = 2.5,
             pw_us: int = 400) -> DataSourceStim:
    """Cathodic-first charge-balanced biphasic pulse (excitatory by CL convention)."""
    return DataSourceStim(
        timestamp          = int(timestamp),
        channel            = int(channel),
        phase_durations_us = (pw_us, pw_us),
        phase_currents_uA  = (-abs(amp_uA), abs(amp_uA)),
    )


def burst(channels, start_ts: int, n_pulses: int = 6, period_frames: int = 60,
          amp_uA: float = 2.5, pw_us: int = 400) -> list[DataSourceStim]:
    """A train of biphasic pulses on each of `channels`."""
    stims = []
    for p in range(n_pulses):
        t = start_ts + p * period_frames
        for c in channels:
            stims.append(biphasic(c, t, amp_uA=amp_uA, pw_us=pw_us))
    return stims


def run_window(source: LIFDataSource, start_ts: int, n_frames: int,
               stims: list[DataSourceStim] | None = None, block: int = 200):
    """Advance the source by `n_frames`, delivering `stims`. Returns (frames, spikes)."""
    if stims:
        source.on_stims(stims)   # stored by absolute timestamp; safe to deliver up front
    parts, spikes = [], []
    ts, end = start_ts, start_ts + n_frames
    while ts < end:
        n = min(block, end - ts)
        batch = source.read(ts, n)
        parts.append(batch.frames)
        spikes.extend(batch.spikes)
        ts += n
    return np.vstack(parts), spikes


def spike_counts(spikes) -> np.ndarray:
    """Per-channel spike counts as a 64-vector."""
    counts = np.zeros(CHANNEL_COUNT)
    for s in spikes:
        counts[s.channel] += 1
    return counts


# --------------------------------------------------------------------------- #
# A) Classification
# --------------------------------------------------------------------------- #

def demo_classification(seed: int = 0):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import confusion_matrix, accuracy_score
    from sklearn.decomposition import PCA

    # Three classes, each a disjoint 2x2 block of electrodes on the array.
    class_channels = {
        0: [ch(1, 1), ch(2, 1), ch(1, 2), ch(2, 2)],
        1: [ch(3, 3), ch(4, 3), ch(3, 4), ch(4, 4)],
        2: [ch(5, 5), ch(6, 5), ch(5, 6), ch(6, 6)],
    }
    trials_per_class = 40
    settle, readout = 800, 1200   # frames (~32 ms settle, ~48 ms readout window)

    # Coupling ON so evoked activity spreads -> distributed spatial fingerprint.
    source = LIFDataSource(random_seed=seed, coupling_gain_mV=2.5,
                           background_drive_std_mV=0.5, connectivity=grid_connectivity())
    source.open()

    X, y, ts = [], [], 0
    rng = np.random.default_rng(seed)
    order = [c for c in class_channels for _ in range(trials_per_class)]
    rng.shuffle(order)
    for cls in order:
        run_window(source, ts, settle); ts += settle                       # inter-trial settle
        stims = burst(class_channels[cls], ts + 5, n_pulses=6, period_frames=60)
        _, spikes = run_window(source, ts, readout, stims=stims); ts += readout
        X.append(spike_counts(spikes)); y.append(cls)
    X, y = np.array(X), np.array(y)

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=seed, stratify=y)
    clf = LogisticRegression(max_iter=2000, C=1.0).fit(Xtr, ytr)
    acc = accuracy_score(yte, clf.predict(Xte))
    cm = confusion_matrix(yte, clf.predict(Xte))
    print(f"[A] classification test accuracy: {acc*100:.1f}%  (3 classes, chance 33%)")

    # ---- figure ----
    fig, ax = plt.subplots(1, 3, figsize=(13, 4))
    mean_resp = np.stack([X[y == c].mean(0) for c in class_channels])
    im0 = ax[0].imshow(grid_image(mean_resp.sum(0)), cmap="magma")
    ax[0].set_title("Mean evoked response (all classes)\n8x8 MEA")
    fig.colorbar(im0, ax=ax[0], fraction=0.046, label="mean spikes/trial")

    pca = PCA(n_components=2).fit(X)
    Z = pca.transform(X)
    for c, col in zip(class_channels, ["#e6194B", "#3cb44b", "#4363d8"]):
        ax[1].scatter(Z[y == c, 0], Z[y == c, 1], c=col, label=f"class {c}", alpha=0.7)
    ax[1].set_title("Response feature space (PCA)")
    ax[1].set_xlabel("PC1"); ax[1].set_ylabel("PC2"); ax[1].legend()

    im2 = ax[2].imshow(cm, cmap="Blues")
    ax[2].set_title(f"Confusion matrix\ntest accuracy {acc*100:.0f}%")
    ax[2].set_xlabel("predicted"); ax[2].set_ylabel("true")
    ax[2].set_xticks(range(3)); ax[2].set_yticks(range(3))
    for i in range(3):
        for j in range(3):
            ax[2].text(j, i, cm[i, j], ha="center",
                       va="center", color="black")
    fig.colorbar(im2, ax=ax[2], fraction=0.046)
    fig.suptitle("A) Stimulus classification from evoked spatial spike patterns", fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "A_classification.png", dpi=110)
    plt.close(fig)
    return acc


# --------------------------------------------------------------------------- #
# B) Functional connectivity
# --------------------------------------------------------------------------- #

def _fc_binned_activity(coupling_gain: float, seed: int,
                        total_frames: int = 250_000, bin_frames: int = 50):
    """Run spontaneous activity and return binned spike counts as a single
    (n_bins, 64) array. Feeding binned spike counts (rather than raw voltage) to
    the repo's Spearman FC is the standard way to read out connectivity: co-firing
    coupled neighbours produce correlated counts, whereas raw voltage is dominated
    by independent per-channel recording noise."""
    source = LIFDataSource(random_seed=seed, coupling_gain_mV=coupling_gain,
                           background_drive_std_mV=0.8, noise_floor_uV=3.0,
                           connectivity=grid_connectivity())
    source.open()
    n_bins = total_frames // bin_frames
    counts = np.zeros((n_bins, CHANNEL_COUNT))
    ts, block = 0, 2000
    while ts < total_frames:
        batch = source.read(ts, min(block, total_frames - ts))
        for sp in batch.spikes:
            counts[min(sp.timestamp // bin_frames, n_bins - 1), sp.channel] += 1
        ts += block
    return source, counts


def demo_connectivity(seed: int = 0):
    src_off, counts_off = _fc_binned_activity(0.0, seed)   # independent neurons
    src_on,  counts_on  = _fc_binned_activity(7.0, seed)   # 4-neighbour coupling

    fc_off = construct_functional_connectivity([counts_off])   # repo's own Spearman FC
    fc_on  = construct_functional_connectivity([counts_on])
    ground_truth = (src_on.coupling_weights > 0).astype(float)   # the coupling we built in

    # How well does recovered FC match the ground-truth coupling mask?
    mask = ~np.eye(CHANNEL_COUNT, dtype=bool)
    gt = ground_truth[mask]

    def _match(fc):
        v = np.abs(fc[mask])
        return 0.0 if v.std() == 0 else float(np.corrcoef(v, gt)[0, 1])

    corr_on, corr_off = _match(fc_on), _match(fc_off)
    print(f"[B] |FC| vs ground-truth-coupling correlation:  coupling ON={corr_on:.2f}  OFF={corr_off:.2f}")

    fig, ax = plt.subplots(1, 3, figsize=(14, 4.3))
    for a, mat, title in zip(
        ax,
        [fc_off, fc_on, ground_truth],
        [f"FC, coupling OFF\n(match r={corr_off:.2f})",
         f"FC, coupling ON\n(match r={corr_on:.2f})",
         "Ground-truth coupling\n(4-neighbour grid)"],
    ):
        vmax = np.abs(mat).max() or 1.0
        im = a.imshow(mat, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        a.set_title(title); a.set_xlabel("channel"); a.set_ylabel("channel")
        fig.colorbar(im, ax=a, fraction=0.046)
    fig.suptitle("B) Spearman functional connectivity recovers the built-in coupling structure",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "B_connectivity.png", dpi=110)
    plt.close(fig)
    return corr_on, corr_off


# --------------------------------------------------------------------------- #
# C) Reinforcement via structured/chaotic (free-energy style) feedback
# --------------------------------------------------------------------------- #

def _run_rl(plasticity: bool, seed: int, n_episodes: int = 360):
    """2-choice sensorimotor task. Two spatially-separated modules; for each the
    sensory electrode's neighbours act as LEFT/RIGHT motor units. The agent must
    map stimulus A->RIGHT and stimulus B->LEFT. Correct actions earn structured
    (predictable) feedback; errors earn chaotic (random) feedback -- the
    free-energy contingency. Reward-modulated plasticity consolidates the
    sensory->motor coupling that produced the action."""
    # Module A around sensory ch(2,2); Module B around sensory ch(5,5).
    modules = {
        "A": dict(sensory=ch(2, 2), left=ch(2, 1), right=ch(2, 3), target="R"),
        "B": dict(sensory=ch(5, 5), left=ch(5, 4), right=ch(5, 6), target="L"),
    }
    reward_channels = [ch(1, 1), ch(6, 6)]   # "structured feedback" electrodes

    source = LIFDataSource(random_seed=seed, coupling_gain_mV=6.0, plasticity=plasticity,
                           plasticity_lr=0.13, weight_max_mV=14.0,
                           background_drive_std_mV=0.18, connectivity=grid_connectivity())
    source.open()
    W_before = source.coupling_weights

    rng = np.random.default_rng(seed + 1)
    settle, sense_win, fb_win = 400, 1400, 600
    correct_hist, surprise_hist = [], []
    ts = 0

    for _ in range(n_episodes):
        mod = modules[rng.choice(["A", "B"])]

        run_window(source, ts, settle); ts += settle                       # settle
        # sensory stimulation (a tight burst so coupling to the motor units accumulates)
        stims = burst([mod["sensory"]], ts + 5, n_pulses=16, period_frames=20, amp_uA=2.8)
        _, spikes = run_window(source, ts, sense_win, stims=stims); ts += sense_win

        counts = spike_counts(spikes)
        action = "R" if counts[mod["right"]] > counts[mod["left"]] else \
                 ("L" if counts[mod["left"]] > counts[mod["right"]] else
                  rng.choice(["L", "R"]))          # break ties randomly (exploration)
        correct = (action == mod["target"])
        correct_hist.append(int(correct))

        # Reward-modulated consolidation of the sensory->motor coupling just used.
        source.apply_reward(+1.0 if correct else -1.0)

        # Deliver the feedback the agent "experiences": structured if correct
        # (predictable burst on the reward electrodes), chaotic if wrong (random
        # channels at random times). Surprise = unpredictability of that input.
        if correct:
            fb = burst(reward_channels, ts + 5, n_pulses=4, period_frames=80, amp_uA=2.0)
            surprise = 0.0
        else:
            rand_ch = rng.choice([j for j in range(CHANNEL_COUNT) if j not in NON_SPIKING],
                                 size=6, replace=False)
            fb = [biphasic(int(c), ts + int(rng.integers(0, fb_win - 100)), amp_uA=2.0)
                  for c in rand_ch]
            surprise = 1.0
        run_window(source, ts, fb_win, stims=fb); ts += fb_win
        source._elig[...] = 0.0   # feedback epoch should not itself drive learning
        surprise_hist.append(surprise)

    return {
        "correct": np.array(correct_hist),
        "surprise": np.array(surprise_hist),
        "W_before": W_before,
        "W_after": source.coupling_weights,
        "modules": modules,
    }


def _smooth(x, w=25):
    if len(x) < w:
        return x.astype(float)
    return np.convolve(x, np.ones(w) / w, mode="valid")


def _mean_sem_curve(runs, key, w=25):
    """Smoothed mean and standard error across seeds for a per-episode series."""
    curves = np.stack([_smooth(r[key], w) for r in runs])
    mean = curves.mean(0)
    sem = curves.std(0) / np.sqrt(curves.shape[0])
    return mean, sem


def demo_reinforcement(seeds=(0, 1, 2, 3, 4)):
    on_runs  = [_run_rl(plasticity=True,  seed=s) for s in seeds]
    off_runs = [_run_rl(plasticity=False, seed=s) for s in seeds]

    acc_on  = np.mean([r["correct"][-50:].mean() for r in on_runs])
    acc_off = np.mean([r["correct"][-50:].mean() for r in off_runs])
    print(f"[C] final accuracy (last 50 ep, mean of {len(seeds)} seeds):  "
          f"plasticity ON={acc_on*100:.0f}%  OFF={acc_off*100:.0f}%  (chance 50%)")

    fig, ax = plt.subplots(1, 3, figsize=(14, 4.3))

    for runs, colour, lbl in [(on_runs, "#2a9d8f", "plasticity ON"),
                              (off_runs, "#999999", "plasticity OFF (control)")]:
        m, e = _mean_sem_curve(runs, "correct")
        x = np.arange(len(m))
        ax[0].plot(x, m, label=lbl, color=colour)
        ax[0].fill_between(x, m - e, m + e, color=colour, alpha=0.25)
    ax[0].axhline(0.5, ls="--", c="k", lw=0.8, label="chance")
    ax[0].set_ylim(0, 1); ax[0].set_xlabel("episode"); ax[0].set_ylabel("accuracy (25-ep MA)")
    ax[0].set_title(f"Task learning curve (mean ± SEM, {len(seeds)} seeds)")
    ax[0].legend(loc="lower right")

    for runs, colour, lbl in [(on_runs, "#e76f51", "plasticity ON"),
                              (off_runs, "#999999", "plasticity OFF")]:
        m, e = _mean_sem_curve(runs, "surprise")
        x = np.arange(len(m))
        ax[1].plot(x, m, label=lbl, color=colour)
        ax[1].fill_between(x, m - e, m + e, color=colour, alpha=0.25)
    ax[1].set_ylim(0, 1); ax[1].set_xlabel("episode")
    ax[1].set_ylabel("surprise = chaotic-feedback fraction")
    ax[1].set_title("Free-energy proxy (minimised as it learns)"); ax[1].legend()

    dW = on_runs[0]["W_after"] - on_runs[0]["W_before"]
    vmax = np.abs(dW).max() or 1.0
    im = ax[2].imshow(dW, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax[2].set_title("Learned coupling change  W_after - W_before")
    ax[2].set_xlabel("presynaptic ch"); ax[2].set_ylabel("postsynaptic ch")
    fig.colorbar(im, ax=ax[2], fraction=0.046, label="Δ weight (mV)")

    fig.suptitle("C) Reinforcement via structured/chaotic feedback (reward-modulated plasticity)",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "C_reinforcement.png", dpi=110)
    plt.close(fig)
    return acc_on, acc_off


if __name__ == "__main__":
    os.environ.setdefault("PYTHONWARNINGS", "ignore")
    print("Running LIF simulator capability demonstrations...\n")
    demo_classification()
    demo_connectivity()
    demo_reinforcement()
    print(f"\nFigures written to {OUT_DIR}/")
