# Assembloid Agency — UDP Wire Protocol v1

Two one-directional UDP streams form the closed loop between Cortical Labs' **CL1**
and **Unreal Engine**, mirroring Figure 1 of *Assembloid Agency*:

```
   CL1  ──(spike firehose, port 12345)──▶  Unreal Engine
   CL1  ◀──(stim commands,  port 12346)──  Unreal Engine
```

The two directions use **different ports** and **different packet formats**. Both use
**little-endian** byte order throughout.

---

## 1. Spike packet  (CL1 → Unreal)

This format is intentionally **byte-compatible with Cortical Labs' own examples**
(`CL-01. Detecting and Reacting to Spikes` and `CL-01A. UDP Spike Receiver`), so the
same Unreal listener also works against their stock notebooks.

| Offset | Size | Type        | Field       | Notes                                              |
|-------:|-----:|-------------|-------------|----------------------------------------------------|
| 0      | 8    | uint64 LE   | `timestamp` | CL sample index of the first spike (40 µs / frame) |
| 8      | 1×N  | uint8 each  | `channels`  | One byte per channel that spiked this packet       |

* Minimum valid size: **9 bytes** (timestamp + ≥1 channel).
* A packet groups all channels reported in a single bridge tick under one timestamp,
  matching the Cortical Labs `struct.pack('<QB', timestamp, channel)` + extra channel
  bytes convention. Run the bridge with `--strict-timestamps` to instead emit one
  packet per spike (exact per-spike timestamps, more packets).
* Default MEA channels are `0–59`.

`struct` reference: `'<Q' + 'B' * N`.

---

## 2. Stim command packet  (Unreal → CL1)

Cortical Labs has not yet frozen a stim wire format ("first-class support … at a lower
layer" is on their roadmap), so Assembloid Agency defines one. A 2-byte magic + version
lets the bridge validate and version the contract.

Fixed 16-byte header, then the channel list:

| Offset | Size | Type      | Field              | Notes                                             |
|-------:|-----:|-----------|--------------------|---------------------------------------------------|
| 0      | 1    | uint8     | `magic[0]`         | `'A'` (0x41)                                       |
| 1      | 1    | uint8     | `magic[1]`         | `'A'` (0x41)                                       |
| 2      | 1    | uint8     | `version`          | `1`                                               |
| 3      | 1    | uint8     | `msg_type`         | `1` = STIM                                         |
| 4      | 1    | uint8     | `flags`            | bit0 = charge-balanced biphasic (1 = on, default) |
| 5      | 1    | uint8     | `num_channels`     | length of the channel list that follows           |
| 6      | 2    | uint16 LE | `pulse_width_us`   | per-phase width (µs)                               |
| 8      | 4    | float32 LE| `amplitude_uA`     | magnitude (µA); phase 1 negative, phase 2 positive |
| 12     | 2    | uint16 LE | `num_pulses`       | ≥1; >1 means a burst                              |
| 14     | 2    | uint16 LE | `freq_hz`          | burst rate, used when `num_pulses > 1`            |
| 16     | 1×N  | uint8 each| `channels`         | electrodes to stimulate                           |

`struct` reference for the header: `'<2sBBBBHfHH'`.

### Mapping to the CL-API

The bridge translates a STIM packet into Cortical Labs calls:

```python
design = cl.StimDesign(pulse_width_us, -amplitude_uA,   # phase 1 (cathodic)
                       pulse_width_us, +amplitude_uA)    # phase 2 (anodic, charge-balanced)
chans  = cl.ChannelSet(*channels)

if num_pulses > 1 and freq_hz > 0:
    neurons.stim(chans, design, cl.BurstDesign(num_pulses, freq_hz))
else:
    neurons.stim(chans, design)
```

The Unreal `SendStimulus(... DurationMs)` helper converts a duration to a pulse count
before sending: `num_pulses = max(1, round(DurationMs/1000 * FreqHz))`.

---

## 3. Safety envelope

The bridge **clamps/rejects** stim commands outside a configurable envelope (defaults are
deliberately conservative — set them to match your own lab/IRB protocol). Defaults:

| Limit                | Default       |
|----------------------|---------------|
| `|amplitude_uA|`     | ≤ 10.0 µA     |
| `pulse_width_us`     | 10 – 1000 µs  |
| `num_pulses`         | ≤ 100         |
| `freq_hz`            | 1 – 200 Hz    |
| allowed channels     | 0 – 59        |

Rejected commands are logged and dropped, never silently altered.
