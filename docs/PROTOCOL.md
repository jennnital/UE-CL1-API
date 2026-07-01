# Unreal-CL1-API — UDP Wire Protocol

Two one-directional UDP streams form the closed loop between the CL1 and Unreal:

```
   spikes   CL1 ──:12345──▶ Unreal      (official "spike firehose", CL-01A)
   stim     Unreal ──:12346──▶ CL1      (Assembloid Agency STIM format)
```

Both directions use **little-endian** byte order.

References:
* Hogan et al., *CL API: Real-Time Closed-Loop Interactions with Biological
  Neural Networks*, arXiv:2602.11632.
* Cortical Labs *CL-01 / CL-01A. Detecting and Reacting to Spikes (UDP Spike
  Receiver)*: https://github.com/Cortical-Labs/cl-api-doc

---

## 1. Spike packet  (CL1 → Unreal)  — official CL format

This is the Cortical Labs "spike firehose" format defined in CL-01A, unchanged:

| Offset | Size | Type        | Field       | Notes                                          |
|-------:|-----:|-------------|-------------|------------------------------------------------|
| 0      | 8    | uint64 LE   | `timestamp` | CL frame index (25 kHz frames, 40 µs each)     |
| 8      | 1×N  | uint8 each  | `channels`  | one byte per channel that spiked (CL1: 0–63)   |

* Minimum valid size: **9 bytes** (timestamp + ≥1 channel).
* The CL-01A receiver parses it as `struct.unpack('<Q', data[0:8])` followed by
  one channel per remaining byte — this protocol is byte-identical, so a stock
  CL-01A receiver and the Unreal plugin consume the same packets.
* `assembloid_cl_bridge.py` groups all channels seen in a loop tick under that
  tick's first-spike timestamp; `--strict-timestamps` instead emits one packet
  per spike with exact per-spike timestamps.

`struct` reference: `'<Q' + 'B' * N`.

---

## 2. Stim command packet  (Unreal → CL1)

Cortical Labs notes UDP stimulation is a planned lower-layer feature ("registers
the remote host as a target for a 'spike firehose', and then opens a socket to
listen to reply packets which it then translates into stimulation api calls"),
but has not yet frozen a wire format. Unreal-CL1-API defines one; the bridge maps
it onto the official `StimDesign` / `BurstDesign` / `ChannelSet` calls.

> If your own CL1 code expects a different blob layout, use the plugin's
> `SendRawBytes()` passthrough instead.

Fixed 16-byte header, then the channel list:

| Offset | Size | Type      | Field            | Notes                                              |
|-------:|-----:|-----------|------------------|----------------------------------------------------|
| 0      | 1    | uint8     | `magic[0]`       | `'A'` (0x41)                                       |
| 1      | 1    | uint8     | `magic[1]`       | `'A'` (0x41)                                       |
| 2      | 1    | uint8     | `version`        | `1`                                                |
| 3      | 1    | uint8     | `msg_type`       | `1` = STIM                                          |
| 4      | 1    | uint8     | `flags`          | bit0 = charge-balanced biphasic (1 = on, default)  |
| 5      | 1    | uint8     | `num_channels`   | length of the channel list that follows            |
| 6      | 2    | uint16 LE | `pulse_width_us` | per-phase width (µs)                               |
| 8      | 4    | float32 LE| `amplitude_uA`   | magnitude (µA); phase 1 negative, phase 2 positive |
| 12     | 2    | uint16 LE | `num_pulses`     | ≥1; >1 means a burst                              |
| 14     | 2    | uint16 LE | `freq_hz`        | burst rate, used when `num_pulses > 1`            |
| 16     | 1×N  | uint8 each| `channels`       | electrodes to stimulate (0–63)                     |

`struct` reference for the header: `'<2sBBBBHfHH'`.

### STIMPLAN packet (Unreal → CL1)

A `msg_type` of `5` carries an atomic stimulation plan with multiple groups.
The `num_channels` field becomes `num_groups`; each group appends:

| Offset | Size | Type      | Field            | Notes                          |
|-------:|-----:|-----------|------------------|--------------------------------|
| 0      | 1    | uint8     | `group_channels` | number of channels in group    |
| 1      | 1    | uint8     | `num_phases`     | number of phase pairs in design|
| 2      | 2    | uint16 LE | `lead_time_us`   | desired lead time before stim  |
| 4      | 6×P  | phase data | `phase_width_us`, `phase_amplitude_uA` | P phase entries, each `uint16 + float32` |
| 4+6P   | 2    | uint16 LE | `num_pulses`     | ≥1; >1 means a burst           |
| 6+6P   | 2    | uint16 LE | `freq_hz`        | burst rate, used when `num_pulses > 1` |
| 8+6P   | 1×M  | uint8 each| `channels`       | electrodes in this group       |

The packet is then repeated for each group. The bridge parses all groups,
validates them against the safety envelope, and applies them atomically with
`interrupt_then_stim` if the flag is set.

Currently the plugin sends zero interrupt channels in the STIMPLAN header.
If `flags` bit1 is set, the bridge uses `interrupt_then_stim` semantics for
all groups.

### Mapping to the CL-API (bridge side, per the whitepaper §6.1)

```python
design = cl.StimDesign(pulse_width_us, -amplitude_uA,   # phase 1 (cathodic)
                       pulse_width_us, +amplitude_uA)    # phase 2 (anodic, balanced)
chans  = cl.ChannelSet(*channels)
if num_pulses > 1 and freq_hz > 0:
    neurons.stim(chans, design, cl.BurstDesign(num_pulses, freq_hz))
else:
    neurons.stim(chans, design)
```

The Unreal `SendStimulus(... DurationMs)` helper converts a duration to a pulse
count before sending: `num_pulses = max(1, round(DurationMs/1000 * FreqHz))`.

---

## 3. Safety envelope

The bridge **rejects** stim commands outside a configurable envelope (defaults
are deliberately conservative — set them to match your own lab/IRB protocol):

| Limit                | Default       |
|----------------------|---------------|
| `|amplitude_uA|`     | ≤ 10.0 µA     |
| `pulse_width_us`     | 10 – 1000 µs  |
| `num_pulses`         | ≤ 100         |
| `freq_hz`            | 1 – 200 Hz    |
| allowed channels     | 0 – 63        |

Rejected commands are logged and dropped, never silently altered.
