# Unreal-CL1-API

**A closed-loop UDP interface between Cortical Labs' CL1 biocomputer and Unreal Engine** —
receive living-neuron spikes as gameplay events, send electrical stimulation back as feedback.

This is the reference implementation of the pipeline described in:

> **Assembloid Agency: Unreal Engine API for brain-on-a-chip platforms**
> Jenn Leung, Chloe Loewith — *NeurIPS 2025 Creative AI Track*
> 📄 Paper / forum: https://openreview.net/forum?id=BroaBkQAGa
> 📥 PDF: https://openreview.net/pdf?id=BroaBkQAGa

The two UDP streams mirror Figure 1 of the paper:

```
   neural spikes  →  game state     (CL1  → Unreal,  port 12345)
   game rewards   →  neural stim     (Unreal → CL1,   port 12346)
```

The spike direction is **byte-compatible with Cortical Labs' own CL-01 / CL-01A
examples**, so the Unreal listener also works against their stock notebooks and against
the NEST simulator.

---

## Contents

| Path | Runs where | Role |
|------|-----------|------|
| `assembloid_cl_bridge.py` | on the **CL1** (or anywhere with `--simulate`) | runs the CL-API loop, streams spikes out, applies incoming stim |
| `AssembloidAgency/` | in **Unreal Engine** (plugin) | receives spikes as events, sends stim from gameplay |
| `PROTOCOL.md` | — | the shared UDP wire-format contract |

---

## Part 1 — Python bridge (CL1 side)

Built directly on the CL-API (`cl.open()`, `neurons.loop()`, `neurons.stim()`,
`cl.StimDesign`, `cl.BurstDesign`, `cl.ChannelSet`). Standard library only.

```bash
# On the CL1 device, streaming to the PC running Unreal at 192.168.1.50
python assembloid_cl_bridge.py --unreal-ip 192.168.1.50

# Develop the Unreal side with no hardware (synthetic Poisson spikes):
python assembloid_cl_bridge.py --simulate --unreal-ip 127.0.0.1
```

Key options (`--help` for all): `--unreal-ip/--unreal-port` (spike destination),
`--stim-listen-port` (incoming stim), `--tick-rate` (default 25000, the CL max),
`--strict-timestamps` (one packet per spike), plus the safety limits below.

**Architecture.** All hardware access is single-threaded: the CL-API loop both sends
spikes and is the only caller of `neurons.stim()`. A background thread reads the stim
socket, validates each command, and enqueues it; the loop drains that queue every tick.

**Safety envelope.** Out-of-range stim commands are logged and dropped — never silently
altered. The defaults are deliberately conservative; set them to match your own lab/IRB
protocol (`--max-amplitude-ua`, `--max-pulse-us`, `--max-pulses`, `--max-freq-hz`,
`--max-channel`). This is the practical counterpart to the paper's note (§5) that
overstimulation affects substrate longevity.

---

## Part 2 — Unreal Engine plugin

Depends on getnamo's **UDP-Unreal** plugin (it embeds `FUDPNative` directly).

### Install

1. Install **UDP-Unreal** into your project's `Plugins/` folder
   (https://github.com/getnamo/UDP-Unreal). Its module is `UDPWrapper`.
2. Copy the `AssembloidAgency/` folder into your project's `Plugins/` folder.
3. Enable both plugins. A C++ project is required (the plugin links `UDPWrapper`);
   if yours is Blueprint-only, add any C++ class once to convert it to mixed.
4. Rebuild.

> The `.uplugin` lists a dependency named `UDPWrapper`; if your installed UDP-Unreal
> `.uplugin` uses a different name, match it there.

### Use (Blueprint or C++)

Add an **Assembloid Agency** component to any Actor (e.g. a `BP_NeuronBridge` placed in
your level). Set `CL Device IP` to the CL1's address; leave ports at 12345 / 12346 to
match the bridge defaults. With `Auto Connect On Begin Play` ticked it connects on play.

**Receiving spikes** — bind the `On Spikes Received` event. Each fires with an
`FCLSpikeEvent` (`Timestamp`, `Channels`, `SourceIP`) on the game thread. You can also
poll `Get Spike Rate Hz(Channel)` and `Get Total Spike Count(Channel)`.

**Sending stimulation:**

```cpp
// Encode target proximity as stim frequency (paper §3.4): near = 100 Hz burst.
Bridge->SendStimulus(
    /*Channels*/    {27, 28, 35, 36},
    /*FrequencyHz*/ 100,
    /*PulseWidthUs*/180,
    /*AmplitudeUA*/ 1.5f,
    /*DurationMs*/  200);   // → 20 pulses at 100 Hz

Bridge->SendSinglePulse({27}, 180, 1.5f);   // one charge-balanced biphasic pulse
```

`SendStimulus` maps to a `cl.StimDesign` (charge-balanced biphasic) plus a
`cl.BurstDesign` on the bridge side. `DurationMs` becomes a pulse count as
`round(DurationMs/1000 × FrequencyHz)`.

### Wiring it into the paper's templates

- **3D navigation / FPS (§3.4):** map `GetSpikeRateHz(channel)` to pawn axis input;
  call `SendStimulus` with proximity-encoded frequency for sensory feedback.
- **Learning Agents:** read spike rates in your agent's observation step and issue
  `SendStimulus` as the reward/feedback action, keeping UE `Learning Agents` as the RL
  harness around this transport layer.
- **NEST instead of CL1:** point the bridge's sockets at a NEST UDP script using the same
  `PROTOCOL.md` formats; the Unreal side is unchanged.

---

## Quick local test (no hardware)

```bash
# Terminal 1 — simulated bridge
python assembloid_cl_bridge.py --simulate --unreal-ip 127.0.0.1

# Terminal 2 — Play in Editor with the component pointed at 127.0.0.1.
# You should see On Spikes Received firing; SendStimulus calls print on the bridge.
```

---

## Repository layout

```
PROTOCOL.md                       wire-format contract (both directions)
assembloid_cl_bridge.py           CL1-side bridge (+ simulator)
AssembloidAgency/                 Unreal Engine plugin
  AssembloidAgency.uplugin
  Source/AssembloidAgency/
    AssembloidAgency.Build.cs
    Public/AssembloidAgencyTypes.h
    Public/AssembloidAgencyComponent.h
    Private/AssembloidAgency.cpp
    Private/AssembloidAgencyComponent.cpp
```

---

## Acknowledgements & related work

- Cortical Labs **CL-API** documentation: https://github.com/Cortical-Labs/cl-api-doc
- getnamo **UDP-Unreal**: https://github.com/getnamo/UDP-Unreal
- Leung, Loewith & Frisch (2025), *Organoid Array Computing: The Design Space of
  Organoid Intelligence*, Antikythera Digital Journal.

## Citing

If you use this code, please cite the paper:

```bibtex
@inproceedings{leung2025assembloid,
  title     = {Assembloid Agency: Unreal Engine API for brain-on-a-chip platforms},
  author    = {Leung, Jenn and Loewith, Chloe},
  booktitle = {The Thirty-ninth Annual Conference on Neural Information
               Processing Systems, Creative AI Track},
  year      = {2025},
  url       = {https://openreview.net/forum?id=BroaBkQAGa}
}
```

## License

Code released under the **MIT License** (see `LICENSE`), compatible with the MIT-licensed
UDP-Unreal dependency. The accompanying paper is licensed CC BY 4.0.
