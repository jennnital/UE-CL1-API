# ue-cl1-api design notes: 

Compared against three sources:
- **Official receiver** — cl-api-doc, CL-01A Appendix A (the firehose wire format)
- **CL API paper §6** — arXiv:2602.11632 (the stimulation contract)
- **Assembloid Agency** — Leung & Loewith, NeurIPS 2025 Creative AI Track (the
  UE API surface this plugin targets)

## Summary

| Aspect | PROTOCOL.md v1 | Assessment |
|---|---|---|
| Spike packet (uint64 LE ts + uint8 channels) | ✓ | Correct, byte-compatible with the official receiver. Keep. |
| Timestamp = frame index @ 40 µs | ✓ | Correct — and *more* accurate than Assembloid §3.3's "ms" prose. Convert in UE, don't mislabel. |
| Versioned `'AA'` header (magic+version+msg_type+flags) | ✓✓ | Better than a fixed magic; enables clean, non-breaking extension. |
| Multi-channel stim in one packet | ✓ | Good. A >1-channel `ChannelSet` auto-syncs (§6.1.3). |
| Charge-balanced biphasic mapping | ✓ | Faithful to §6.1: `StimDesign(pw,-amp,pw,+amp)` cathodic-first. |
| Safety envelope + "reject, never silently alter" | ✓✓ | Directly implements Assembloid §5 ethical guardrails. |
| **No interrupt path** | ✗ | **Main gap.** Re-sending `stim()` *appends to the queue* (§6.1.1) → latency build-up under continuous control. |
| `freq_hz` as uint16 | ~ | Discards sub-Hz; the paper uses 37.9 Hz. float32 is free. |
| Symmetric-only pulse | ~ | `StimDesign` allows asymmetric phases; v1 cannot express them. |
| No `lead_time_us` | ~ | §6.1 default 80 µs; overriding enables precise inter-event timing. |
| Channel range 0–59 | ~ | CL1 reference is **64 channels** (examples use ch 60, 62). Set the ceiling to your MEA. |
| Simultaneous *different* designs | ✗ | Needs a `StimPlan` (§6.1.4, atomic, same-frame visibility). Future msg_type. |

## v2 additions implemented here (non-breaking — reuse the 16-byte header)

1. **`flags` bit1 = interrupt-then-stim.** Maps to `interrupt_then_stim(...)`
   (§6.1.1): atomically replaces ongoing activity at a stimulus boundary, so
   continuous proximity→frequency control changes the rate instead of stacking
   bursts. `SendStimulus(..., bInterruptFirst=true)` sets it.
2. **`msg_type` 2 = INTERRUPT.** Clean stop on the listed channels
   (`neurons.interrupt(...)`). `InterruptStim(channels)`.
3. **`msg_type` 4 = RECORD.** Triggers CL1-side `neurons.record()` /
   `recording.stop()` (§6.4) — the authoritative HDF5 capture (raw samples +
   spikes + stims + data streams on one time base), far richer than a UE log.
   `flags` bit0 = start/stop. `RecordSessionData(bStart)`.

v1 STIM packets parse unchanged; a v1-only bridge ignores the new types.

## Recommended but NOT yet wired (would change the header — propose for v3)

- `freq_hz` → float32 (preserve sub-Hz before the CL1's own 20 µs quantization).
- `lead_time_us` (u32) optional field, 0 = CL default 80 µs.
- Per-phase fields (`pw1,amp1,pw2,amp2`) for asymmetric biphasic.
- A `STIMPLAN` msg_type carrying multiple (channel-set, design, burst) groups,
  admitted atomically and made visible in the same frame (§6.1.4), with
  `TransactionRejected` surfaced back to UE via an ACK packet.

## Units & semantics to keep straight

- **Timestamp:** frame index, 40 µs/frame. seconds = frame / 25000. The plugin
  fills `FCl1Spike.TimeSeconds`; treat Assembloid's "ms" as informal.
- **Voltage:** the firehose carries **none**. Spike waveforms are float **µV**
  (not mV) in the CL API's `Spike.samples`, fetched lazily (§6.1.5). Add an
  explicit packet field if a template needs amplitude.
- **Reward:** in the DishBrain paradigm reward *is* stimulation, so
  `SendRewardSignal` is a thin wrapper over stim; the reward↔stim mapping is an
  experiment choice, not a wire-level fact.

## Closed-loop timing reminder (Assembloid §5)

Feedback must land inside the biological window of plasticity. Keep the bridge
loop body minimal (no blocking I/O on the loop thread — sending is offloaded to
a background thread here) and prefer interrupt-then-stim for rate changes so the
stimulus stays causally tied to the spike that triggered it.
