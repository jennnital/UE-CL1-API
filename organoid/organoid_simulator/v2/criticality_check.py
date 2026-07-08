"""
Criticality diagnostic + E/I tuning for the v2 substrate (§M3, Verification §3).

Dissociated cultures on MEAs self-organise near a critical point, where neuronal
*avalanches* -- cascades of activity bounded by quiescence -- have a power-law size
distribution (exponent ≈ -1.5) and a branching ratio σ ≈ 1 (Beggs & Plenz 2003;
Pasquale et al. 2008). Criticality maximises dynamic range (Shew et al. 2009), which
is exactly what makes the network a useful reservoir. This module measures both
statistics and provides a small E/I sweep to locate the near-critical band, which is
narrow (between silence and runaway/seizure), so tuning is expected.

Two entry points:

* :func:`measure_criticality` -- given population spike times, return branching ratio,
  avalanche-size power-law exponent, and firing rate.
* ``python -m organoid_simulator.v2.criticality_check [--sweep]`` -- build a network,
  run it, and either report the single-point diagnostic or sweep ``exc_scale`` to find
  the E/I balance nearest σ ≈ 1.
"""
from __future__ import annotations

# Allow running as a plain file as well as via `-m` (see demos_v2.py for why).
if __name__ == "__main__" and __package__ in (None, ""):
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
    __package__ = "organoid_simulator.v2"

import argparse
from dataclasses import dataclass

import numpy as np

from brian2 import ms, second

from .connectivity import build_geometry, build_connectivity
from .network import build_network, BackgroundParams


@dataclass
class CriticalityResult:
    branching_ratio: float     # σ; ≈1 at criticality
    avalanche_exponent: float  # α in P(S) ~ S^-α; ≈1.5 at criticality
    mean_rate_hz: float        # per-neuron firing rate
    n_avalanches: int


def _avalanches(bin_counts: np.ndarray) -> list[np.ndarray]:
    """Split binned population activity into avalanches: maximal runs of non-empty
    bins bounded by empty bins (Beggs & Plenz 2003)."""
    active = bin_counts > 0
    avalanches = []
    start = None
    for i, a in enumerate(active):
        if a and start is None:
            start = i
        elif not a and start is not None:
            avalanches.append(bin_counts[start:i])
            start = None
    if start is not None:
        avalanches.append(bin_counts[start:])
    return avalanches


def _branching_ratio(avalanches: list[np.ndarray]) -> float:
    """Branching ratio σ as the avalanche descendants/ancestors ratio (Beggs & Plenz
    2003): σ = Σ events in non-first bins / Σ events in non-last bins, over all
    avalanches. σ<1 subcritical, σ>1 supercritical, σ≈1 critical.

    Preferred here over the regression-slope estimator because AdEx spike-frequency
    adaptation suppresses the bin after a burst, which biases a raw A(t+1)-on-A(t)
    regression negative -- this ratio measures within-avalanche propagation instead.
    """
    ancestors = descendants = 0.0
    n = 0
    for a in avalanches:
        if a.size >= 2:
            ancestors += a[:-1].sum()
            descendants += a[1:].sum()
            n += 1
    if n < 5 or ancestors == 0:
        return float("nan")
    return float(descendants / ancestors)


def _powerlaw_exponent(sizes: np.ndarray, s_min: int = 2) -> float:
    """MLE exponent α for a discrete power law P(S) ~ S^-α (Clauset et al. 2009),
    fit to avalanche sizes >= s_min."""
    s = sizes[sizes >= s_min].astype(float)
    if s.size < 20:
        return float("nan")
    return float(1.0 + s.size / np.sum(np.log(s / (s_min - 0.5))))


def measure_criticality(
    spike_times_s: np.ndarray,
    n_neurons: int,
    duration_s: float,
    bin_ms: float | None = None,
) -> CriticalityResult:
    """Compute criticality statistics from a flat array of spike times (seconds).

    ``bin_ms`` defaults to the mean inter-spike interval of the whole population, the
    conventional avalanche bin width; it makes the avalanche definition self-scaling
    to the network's activity level.
    """
    spike_times_s = np.sort(np.asarray(spike_times_s))
    n_spikes = spike_times_s.size
    mean_rate = n_spikes / n_neurons / duration_s if duration_s > 0 else 0.0

    if n_spikes < 50:
        return CriticalityResult(float("nan"), float("nan"), mean_rate, 0)

    if bin_ms is None:
        mean_isi_s = duration_s / n_spikes
        bin_ms = max(0.5, mean_isi_s * 1000.0)

    n_bins = int(np.ceil(duration_s * 1000.0 / bin_ms))
    edges = np.arange(n_bins + 1) * (bin_ms / 1000.0)
    counts, _ = np.histogram(spike_times_s, bins=edges)

    avs = _avalanches(counts)
    br = _branching_ratio(avs)
    sizes = np.array([a.sum() for a in avs]) if avs else np.array([])
    alpha = _powerlaw_exponent(sizes)
    return CriticalityResult(br, alpha, mean_rate, len(avs))


def _run_and_collect(exc_scale, inh_scale, n_neurons, duration_s, seed, background=None):
    """Build a network at the sheet-mode (criticality) preset and return all spike
    times (s) from a no-stim run."""
    from .brian_source import SHEET_PRESET
    geo = build_geometry(n_neurons, seed=seed)
    conn = build_connectivity(
        geo, seed=seed,
        p_connect=SHEET_PRESET["p_connect"],
        conn_lambda_um=SHEET_PRESET["conn_lambda_um"],
        w_exc_nS=SHEET_PRESET["w_exc_nS"],
        w_inh_nS=SHEET_PRESET["w_inh_nS"],
    )
    bg = BackgroundParams(**(background if background else SHEET_PRESET["background"]))
    bundle = build_network(geo, conn, exc_scale=exc_scale, inh_scale=inh_scale,
                           background=bg, seed=seed)
    bundle.net.run(duration_s * second)
    return np.asarray(bundle.spikemon.t / second), geo.n_neurons


def sweep_background(ge0_values, n_neurons=1000, duration_s=15.0, seed=1):
    """Sweep the background excitatory working point ``ge0`` (nS) -- the primary
    criticality knob (it sets how close resting Vm sits to threshold, hence whether a
    synaptic EPSP propagates) -- and report the stats, flagging the point nearest σ=1.
    """
    from .brian_source import SHEET_PRESET
    base = SHEET_PRESET["background"]
    print(f"Background (excitability) sweep (N={n_neurons}, {duration_s:.0f}s each):")
    print(f"  {'ge0(nS)':>8} {'rate(Hz)':>9} {'branching σ':>12} {'exponent α':>11} {'#aval':>7}")
    results = []
    for ge0 in ge0_values:
        bg = dict(base, ge0=float(ge0))
        st, n = _run_and_collect(SHEET_PRESET["exc_scale"], SHEET_PRESET["inh_scale"],
                                 n_neurons, duration_s, seed, background=bg)
        r = measure_criticality(st, n, duration_s, bin_ms=3.0)
        results.append((ge0, r))
        print(f"  {ge0:8.2f} {r.mean_rate_hz:9.2f} {r.branching_ratio:12.3f} "
              f"{r.avalanche_exponent:11.3f} {r.n_avalanches:7d}")
    valid = [(g, r) for g, r in results if np.isfinite(r.branching_ratio)]
    if valid:
        best = min(valid, key=lambda x: abs(x[1].branching_ratio - 1.0))
        print(f"\n  nearest criticality: ge0={best[0]:.2f} nS "
              f"(σ={best[1].branching_ratio:.3f}, α={best[1].avalanche_exponent:.3f})")
    return results


def main():
    ap = argparse.ArgumentParser(description="v2 criticality diagnostic")
    ap.add_argument("--sweep", action="store_true", help="sweep exc_scale for the critical band")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--duration", type=float, default=15.0)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    from .brian_source import SHEET_PRESET
    if args.sweep:
        # Sweep the background working point (excitability), the primary criticality
        # knob, at the preset E/I scales.
        sweep_background(np.arange(8.0, 13.01, 1.0), n_neurons=args.n,
                         duration_s=args.duration, seed=args.seed)
    else:
        st, n = _run_and_collect(SHEET_PRESET["exc_scale"], SHEET_PRESET["inh_scale"],
                                 args.n, args.duration, args.seed)
        r = measure_criticality(st, n, args.duration, bin_ms=3.0)
        print(f"N={n}, {args.duration:.0f}s no-stim run:")
        print(f"  per-neuron rate : {r.mean_rate_hz:.2f} Hz")
        print(f"  branching ratio : {r.branching_ratio:.3f}   (criticality ≈ 1)")
        print(f"  avalanche α     : {r.avalanche_exponent:.3f}   (criticality ≈ 1.5)")
        print(f"  # avalanches    : {r.n_avalanches}")


if __name__ == "__main__":
    main()
