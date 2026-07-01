#!/usr/bin/env python3
"""
bridge.py - Cortical Labs CL1 <-> Unreal Engine UDP bridge (Assembloid Agency).

Implements the closed loop described in:
  * Leung & Loewith, "Assembloid Agency: Unreal Engine API for brain-on-a-chip
    platforms", NeurIPS 2025 Creative AI Track  (the API surface + UDP design)
  * Hogan et al., "CL API: Real-Time Closed-Loop Interactions with Biological
    Neural Networks", arXiv:2602.11632  (the stimulation contract, Section 6)
  * cl-api-doc, CL-01A Appendix A  (the spike firehose wire format)

Wire protocol = "Assembloid Agency UDP Wire Protocol" (see PROTOCOL.md), with
the non-breaking v2 additions justified against Section 6 of the CL paper:
  * flags bit1  -> interrupt-then-stim  (Sec 6.1.1, avoids queue build-up)
  * msg_type 2  -> INTERRUPT            (Sec 6.1.1, clean stop)
  * msg_type 4  -> RECORD               (Sec 6.4, CL1-side HDF5 recording)
These reuse the v1 16-byte header exactly, so v1 STIM packets still parse.

------------------------------------------------------------------------------
SPIKE PACKET  CL1 -> UE  (unchanged; byte-compatible with the official receiver)
------------------------------------------------------------------------------
    byte 0..7   uint64 LE   timestamp (frame index, 40 us/frame)
    byte 8..N   uint8       one byte per channel that spiked this tick

------------------------------------------------------------------------------
STIM / CONTROL PACKET  UE -> CL1   (PROTOCOL.md header '<2sBBBBHfHH' = 16 bytes)
------------------------------------------------------------------------------
    0   2  2s    magic            b'AA'
    2   1  u8    version          1 or 2
    3   1  u8    msg_type         1=STIM, 2=INTERRUPT, 4=RECORD
    4   1  u8    flags            bit0 charge-balanced biphasic (default 1)
                                  bit1 interrupt-then-stim (v2)
                                  (RECORD: bit0 = 1 start / 0 stop)
    5   1  u8    num_channels     length of trailing channel list
    6   2  u16   pulse_width_us   per-phase width
    8   4  f32   amplitude_uA     magnitude; phase1 = -amp, phase2 = +amp
    12  2  u16   num_pulses       >=1; >1 => burst
    14  2  u16   freq_hz          burst rate (used if num_pulses > 1)
    16  1xN u8   channels         electrodes


When running bridge.py you can use these simulate modes without CL1 device:

# Random simulation with deterministic seed
python3 bridge.py --simulator --random-seed 42

# Replay a recording with accelerated time
python3 bridge.py --replay-path /path/to/recording.h5 --accelerated-time

# Custom random simulation parameters
python3 bridge.py --simulator --sample-mean 200 --spike-percentile 99.9
"""

import argparse
import os
import queue
import socket
import struct
import sys
import threading
import time

# --- Spike firehose (CL1 -> UE) ------------------------------------------- #
TS_STRUCT = struct.Struct("<Q")  # uint64 LE timestamp, matches the receiver

# --- Stim / control (UE -> CL1) ------------------------------------------- #
AA_MAGIC = b"AA"
AA_HEADER = struct.Struct("<2sBBBBHfHH")  # 16 bytes
assert AA_HEADER.size == 16

MSG_STIM = 1
MSG_INTERRUPT = 2
MSG_RECORD = 4
MSG_STIMPLAN = 5

FLAG_BIPHASIC = 0x01           # charge-balanced biphasic (phase1 -, phase2 +)
FLAG_INTERRUPT_THEN_STIM = 0x02  # v2: replace ongoing activity before stimulating


# --------------------------------------------------------------------------- #
# Safety envelope (Assembloid Agency Sec 5: overstimulation -> substrate
# longevity; ethical guardrails). Reject + drop, never silently alter.
# --------------------------------------------------------------------------- #
class SafetyEnvelope:
    def __init__(self, max_amp_uA=10.0, min_pw_us=10, max_pw_us=1000,
                 max_pulses=100, min_freq=1.0, max_freq=200.0,
                 min_channel=0, max_channel=63):
        self.max_amp_uA = max_amp_uA
        self.min_pw_us = min_pw_us
        self.max_pw_us = max_pw_us
        self.max_pulses = max_pulses
        self.min_freq = min_freq
        self.max_freq = max_freq
        self.min_channel = min_channel
        self.max_channel = max_channel

    def check(self, amp_uA, pw_us, num_pulses, freq_hz, channels):
        """Return (ok, reason). Bursts (num_pulses>1) are checked against freq."""
        amp_list = amp_uA if isinstance(amp_uA, (list, tuple)) else [amp_uA]
        pw_list = pw_us if isinstance(pw_us, (list, tuple)) else [pw_us]

        for amp in amp_list:
            if abs(amp) > self.max_amp_uA:
                return False, f"|amplitude|={abs(amp):.2f}uA > {self.max_amp_uA}uA"
        for pw in pw_list:
            if not (self.min_pw_us <= pw <= self.max_pw_us):
                return False, f"pulse_width={pw}us outside [{self.min_pw_us},{self.max_pw_us}]"
        if num_pulses > self.max_pulses:
            return False, f"num_pulses={num_pulses} > {self.max_pulses}"
        if num_pulses > 1 and not (self.min_freq <= freq_hz <= self.max_freq):
            return False, f"freq={freq_hz}Hz outside [{self.min_freq},{self.max_freq}]"
        for c in channels:
            if not (self.min_channel <= c <= self.max_channel):
                return False, f"channel {c} outside [{self.min_channel},{self.max_channel}]"
        return True, ""


# --------------------------------------------------------------------------- #
# Background sender: keeps sendto() off the real-time loop thread.
# --------------------------------------------------------------------------- #
class SpikeSender(threading.Thread):
    def __init__(self, dest_host, dest_port, maxsize=4096):
        super().__init__(daemon=True, name="SpikeSender")
        self.dest = (dest_host, dest_port)
        self.q = queue.Queue(maxsize=maxsize)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._stop = threading.Event()
        self.dropped = 0
        self.sent = 0

    def submit(self, packet):
        try:
            self.q.put_nowait(packet)
        except queue.Full:
            self.dropped += 1  # never block the neural loop

    def run(self):
        while not self._stop.is_set():
            try:
                packet = self.q.get(timeout=0.25)
            except queue.Empty:
                continue
            try:
                self.sock.sendto(packet, self.dest)
                self.sent += 1
            except OSError:
                pass

    def stop(self):
        self._stop.set()


# --------------------------------------------------------------------------- #
# Background receiver: parses UE -> CL1 control packets into structured cmds.
# --------------------------------------------------------------------------- #
class ControlReceiver(threading.Thread):
    def __init__(self, listen_host, listen_port, maxsize=1024):
        super().__init__(daemon=True, name="ControlReceiver")
        self.q = queue.Queue(maxsize=maxsize)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((listen_host, listen_port))
        self.sock.settimeout(0.25)
        self._stop = threading.Event()
        self.bad = 0

    def run(self):
        while not self._stop.is_set():
            try:
                data, _ = self.sock.recvfrom(2048)
            except socket.timeout:
                continue
            except OSError:
                continue
            cmd = self._parse(data)
            if cmd is None:
                self.bad += 1
                continue
            try:
                self.q.put_nowait(cmd)
            except queue.Full:
                self.bad += 1

    @staticmethod
    def _parse(data):
        if len(data) < AA_HEADER.size:
            return None
        magic, ver, mtype, flags, nch, pw, amp, npulses, freq = \
            AA_HEADER.unpack_from(data, 0)
        if magic != AA_MAGIC:
            return None

        if mtype == MSG_STIMPLAN:
            num_groups = nch
            offset = AA_HEADER.size
            if len(data) < offset + 1:
                return None
            interrupt_nch = data[offset]
            offset += 1
            interrupt_channels = list(data[offset:offset + interrupt_nch])
            if len(interrupt_channels) != interrupt_nch:
                return None
            offset += interrupt_nch

            groups = []
            for _ in range(num_groups):
                if len(data) < offset + 4:
                    return None
                group_nch = data[offset]
                num_phases = data[offset + 1]
                lead_time_us = struct.unpack_from("<H", data, offset + 2)[0]
                offset += 4

                phases = []
                for _ in range(num_phases):
                    if len(data) < offset + 6:
                        return None
                    phase_pw = struct.unpack_from("<H", data, offset)[0]
                    phase_amp = struct.unpack_from("<f", data, offset + 2)[0]
                    phases.append((phase_pw, phase_amp))
                    offset += 6

                if len(data) < offset + 4:
                    return None
                group_npulses = struct.unpack_from("<H", data, offset)[0]
                group_freq = struct.unpack_from("<H", data, offset + 2)[0]
                offset += 4

                channels = list(data[offset:offset + group_nch])
                if len(channels) != group_nch:
                    return None
                offset += group_nch

                groups.append(dict(channels=channels,
                                   phases=phases,
                                   num_pulses=group_npulses,
                                   freq_hz=float(group_freq),
                                   lead_time_us=lead_time_us))

            if offset != len(data):
                return None

            return dict(version=ver, msg_type=mtype, flags=flags,
                        interrupt_channels=interrupt_channels,
                        groups=groups)

        channels = list(data[AA_HEADER.size:AA_HEADER.size + nch])
        if len(channels) != nch:
            return None
        return dict(version=ver, msg_type=mtype, flags=flags,
                    pulse_width_us=pw, amplitude_uA=amp,
                    num_pulses=npulses, freq_hz=float(freq),
                    channels=channels)

    def drain(self, limit=16):
        out = []
        for _ in range(limit):
            try:
                out.append(self.q.get_nowait())
            except queue.Empty:
                break
        return out

    def stop(self):
        self._stop.set()


# --------------------------------------------------------------------------- #
# Spike packet builders (CL1 -> UE)
# --------------------------------------------------------------------------- #
def pack_per_tick(timestamp, channels):
    return TS_STRUCT.pack(timestamp) + bytes(bytearray(c & 0xFF for c in channels))


def pack_per_spike(timestamp, channel):
    return TS_STRUCT.pack(timestamp) + bytes((channel & 0xFF,))


# --------------------------------------------------------------------------- #
# Apply a parsed control command via the CL API (Section 6 mapping).
# --------------------------------------------------------------------------- #
def apply_command(cl, neurons, cmd, envelope, recording_state):
    mtype = cmd["msg_type"]

    if mtype == MSG_RECORD:
        # Sec 6.4: CL1-side HDF5 recording (raw samples + spikes + stims + streams).
        start = bool(cmd["flags"] & FLAG_BIPHASIC)  # bit0 reused as start/stop
        if start and recording_state.get("rec") is None:
            recording_state["rec"] = neurons.record()
            print("[bridge] recording started", file=sys.stderr)
        elif not start and recording_state.get("rec") is not None:
            rec = recording_state.pop("rec")
            rec.stop()
            try:
                print(f"[bridge] recording stopped: {rec.file['path']}", file=sys.stderr)
            except Exception:
                print("[bridge] recording stopped", file=sys.stderr)
        return

    if mtype == MSG_INTERRUPT:
        channels = cmd.get("channels", [])
        if not channels:
            return
        # Sec 6.1.1: clean stop of ongoing activity on these channels.
        try:
            neurons.interrupt(cl.ChannelSet(*channels))
        except Exception as e:
            print(f"[bridge] interrupt failed: {e}", file=sys.stderr)
        return

    if mtype == MSG_STIMPLAN:
        interrupt_channels = cmd.get("interrupt_channels", [])
        if interrupt_channels:
            try:
                neurons.interrupt(cl.ChannelSet(*interrupt_channels))
            except Exception as e:
                print(f"[bridge] interrupt failed: {e}", file=sys.stderr)
                return

        plan_groups = cmd.get("groups", [])
        if not plan_groups:
            return

        interrupt_first = bool(cmd["flags"] & FLAG_INTERRUPT_THEN_STIM)
        try:
            for group in plan_groups:
                channels = group["channels"]
                if not channels:
                    continue

                phases = group["phases"]
                lead_time = group.get("lead_time_us", 80)
                npulses = group["num_pulses"]
                freq = group["freq_hz"]

                amp_list = [amp for _, amp in phases]
                pw_list = [pw for pw, _ in phases]
                ok, reason = envelope.check(amp_list, pw_list, npulses, freq, channels)
                if not ok:
                    print(f"[bridge] REJECT stim plan ({reason})", file=sys.stderr)
                    return

                flat_phase_args = []
                for pw, amp in phases:
                    flat_phase_args.extend((pw, amp))

                design = cl.StimDesign(*flat_phase_args)
                cset = cl.ChannelSet(*channels)
                burst = cl.BurstDesign(npulses, freq) if npulses > 1 and freq > 0 else None

                if burst is not None:
                    if interrupt_first:
                        try:
                            neurons.interrupt_then_stim(cset, design, burst, lead_time)
                        except TypeError:
                            neurons.interrupt_then_stim(cset, design, burst)
                    else:
                        try:
                            neurons.stim(cset, design, burst, lead_time)
                        except TypeError:
                            neurons.stim(cset, design, burst)
                else:
                    if interrupt_first:
                        try:
                            neurons.interrupt_then_stim(cset, design, lead_time)
                        except TypeError:
                            neurons.interrupt_then_stim(cset, design)
                    else:
                        try:
                            neurons.stim(cset, design, lead_time)
                        except TypeError:
                            neurons.stim(cset, design)
        except cl.TransactionRejected:
            print("[bridge] stim TransactionRejected (admission/capacity)", file=sys.stderr)
        except Exception as e:
            print(f"[bridge] stim failed: {e}", file=sys.stderr)
        return

    if mtype != MSG_STIM:
        return

    amp = cmd["amplitude_uA"]
    pw = cmd["pulse_width_us"]
    npulses = cmd["num_pulses"]
    freq = cmd["freq_hz"]

    ok, reason = envelope.check(amp, pw, npulses, freq, channels)
    if not ok:
        print(f"[bridge] REJECT stim ({reason})", file=sys.stderr)  # log + drop
        return

    # Sec 6.1: cathodic-first charge-balanced biphasic design.
    design = cl.StimDesign(pw, -abs(amp), pw, +abs(amp))
    cset = cl.ChannelSet(*channels)
    interrupt_first = bool(cmd["flags"] & FLAG_INTERRUPT_THEN_STIM)

    try:
        if npulses > 1 and freq > 0:
            burst = cl.BurstDesign(npulses, freq)
            if interrupt_first:
                neurons.interrupt_then_stim(cset, design, burst)  # Sec 6.1.1
            else:
                neurons.stim(cset, design, burst)
        else:
            if interrupt_first:
                neurons.interrupt_then_stim(cset, design)
            else:
                neurons.stim(cset, design)
    except cl.TransactionRejected:
        # Sec 5.4: atomic admission failed (e.g. queue capacity). Keep loop alive.
        print("[bridge] stim TransactionRejected (admission/capacity)", file=sys.stderr)
    except Exception as e:
        print(f"[bridge] stim failed: {e}", file=sys.stderr)


# --------------------------------------------------------------------------- #
# Main run loop (real hardware / simulator path)
# --------------------------------------------------------------------------- #
def run_bridge(args):
    # Configure simulator via environment variables if requested
    if args.simulator or args.replay_path:
        if args.replay_path:
            os.environ["CL_SDK_REPLAY_PATH"] = args.replay_path
        if args.replay_start_offset is not None:
            os.environ["CL_SDK_REPLAY_START_OFFSET"] = str(args.replay_start_offset)
        if args.sample_mean is not None:
            os.environ["CL_SDK_SAMPLE_MEAN"] = str(args.sample_mean)
        if args.spike_percentile is not None:
            os.environ["CL_SDK_SPIKE_PERCENTILE"] = str(args.spike_percentile)
        if args.random_seed is not None:
            os.environ["CL_SDK_RANDOM_SEED"] = str(args.random_seed)
        if args.spike_visibility is not None:
            os.environ["CL_SDK_SPIKE_VISIBILITY"] = str(args.spike_visibility)
        if args.accelerated_time:
            os.environ["CL_SDK_ACCELERATED_TIME"] = "1"
        if args.disable_visualisation:
            os.environ["CL_SDK_VISUALISATION"] = "0"

    import cl  # lazy import so --selftest works without the SDK

    envelope = SafetyEnvelope(
        max_amp_uA=args.max_amp, min_pw_us=args.min_pw, max_pw_us=args.max_pw,
        max_pulses=args.max_pulses, min_freq=args.min_freq, max_freq=args.max_freq,
        min_channel=0, max_channel=args.max_channel)

    sender = SpikeSender(args.ue_host, args.ue_port)
    sender.start()

    receiver = None
    if args.control_listen_port:
        receiver = ControlReceiver(args.bind, args.control_listen_port)
        receiver.start()
        print(f"[bridge] listening for UE->CL1 control on "
              f"{args.bind}:{args.control_listen_port}", file=sys.stderr)

    print(f"[bridge] spikes -> {args.ue_host}:{args.ue_port} "
          f"(mode={args.mode}, ticks_per_second={args.rate}); "
          f"safety: |amp|<={args.max_amp}uA, ch 0-{args.max_channel}", file=sys.stderr)

    loop_kwargs = dict(ticks_per_second=args.rate)
    if args.ignore_jitter:
        loop_kwargs["ignore_jitter"] = True
    elif args.jitter_tolerance:
        loop_kwargs["jitter_tolerance_frames"] = args.jitter_tolerance

    recording_state = {}
    tick_count = 0
    with cl.open() as neurons:
        for tick in neurons.loop(**loop_kwargs):
            spikes = tick.analysis.spikes

            # ---- CL1 -> UE: emit spikes -------------------------------------
            if spikes:
                if args.mode == "per_spike":
                    for s in spikes:
                        sender.submit(pack_per_spike(s.timestamp, s.channel))
                else:
                    channels = [s.channel for s in spikes]
                    sender.submit(pack_per_tick(spikes[0].timestamp, channels))

            # ---- UE -> CL1: apply queued control commands -------------------
            if receiver is not None:
                for cmd in receiver.drain(limit=args.max_cmds_per_tick):
                    apply_command(cl, neurons, cmd, envelope, recording_state)

            tick_count += 1
            if args.heartbeat and tick_count % args.heartbeat == 0:
                print(f"[bridge] tick={tick_count} sent={sender.sent} "
                      f"dropped={sender.dropped}", file=sys.stderr)

    sender.stop()
    if receiver:
        receiver.stop()


# --------------------------------------------------------------------------- #
# Self-test: synthetic spikes, no SDK/hardware. Validate UE plugin first.
# --------------------------------------------------------------------------- #
def run_selftest(args):
    import random
    sender = SpikeSender(args.ue_host, args.ue_port)
    sender.start()
    print(f"[selftest] synthetic spikes -> {args.ue_host}:{args.ue_port} "
          f"(mode={args.mode}) - Ctrl-C to stop", file=sys.stderr)
    ts, frame_dt, n_channels = 0, 1.0 / args.rate, args.max_channel + 1
    try:
        while True:
            ts += 1
            firing = [c for c in range(n_channels)
                      if random.random() < args.selftest_rate / args.rate]
            if firing:
                if args.mode == "per_spike":
                    for c in firing:
                        sender.submit(pack_per_spike(ts, c))
                else:
                    sender.submit(pack_per_tick(ts, firing))
            time.sleep(frame_dt)
    except KeyboardInterrupt:
        pass
    finally:
        sender.stop()
        print(f"\n[selftest] sent={sender.sent} dropped={sender.dropped}",
              file=sys.stderr)


def build_parser():
    p = argparse.ArgumentParser(description="CL1 <-> Unreal Engine UDP bridge (Assembloid Agency)")
    p.add_argument("--ue-host", default="127.0.0.1")
    p.add_argument("--ue-port", type=int, default=12345)
    p.add_argument("--bind", default="0.0.0.0")
    p.add_argument("--control-listen-port", type=int, default=0,
                   help="UDP port for UE->CL1 stim/control (0 = disabled, e.g. 12346)")
    p.add_argument("--mode", choices=["per_tick", "per_spike"], default="per_tick")
    p.add_argument("--rate", type=int, default=1000, help="loop ticks/sec (ceiling 25000)")
    p.add_argument("--max-cmds-per-tick", type=int, default=16)
    p.add_argument("--ignore-jitter", action="store_true")
    p.add_argument("--jitter-tolerance", type=int, default=0)
    p.add_argument("--heartbeat", type=int, default=1000)
    # Safety envelope (Assembloid Sec 5 defaults; set to your IRB/lab protocol)
    p.add_argument("--max-amp", type=float, default=10.0)
    p.add_argument("--min-pw", type=int, default=10)
    p.add_argument("--max-pw", type=int, default=1000)
    p.add_argument("--max-pulses", type=int, default=100)
    p.add_argument("--min-freq", type=float, default=1.0)
    p.add_argument("--max-freq", type=float, default=200.0)
    p.add_argument("--max-channel", type=int, default=59,
                   help="highest allowed electrode (PROTOCOL.md=59; CL1 ref is 64ch => 63)")
    # Simulator options (CL SDK)
    p.add_argument("--simulator", action="store_true",
                   help="Enable CL SDK simulator (local development mode)")
    p.add_argument("--replay-path", type=str, default=None,
                   help="Path to recording file to replay in simulator")
    p.add_argument("--replay-start-offset", type=int, default=None,
                   help="Starting frame offset in replay recording (default: random)")
    p.add_argument("--sample-mean", type=int, default=None,
                   help="Mean samples value for random simulation (default: 170)")
    p.add_argument("--spike-percentile", type=float, default=None,
                   help="Percentile threshold for spike detection (default: 99.995)")
    p.add_argument("--random-seed", type=int, default=None,
                   help="Random seed for deterministic simulation")
    p.add_argument("--spike-visibility", type=int, default=None,
                   help="Amplify spike visibility in sample data (default: 0)")
    p.add_argument("--accelerated-time", action="store_true",
                   help="Enable accelerated (non-wall-clock) time in simulator")
    p.add_argument("--disable-visualisation", action="store_true",
                   help="Disable WebSocket visualization server in simulator")
    # Self-test
    p.add_argument("--selftest", action="store_true")
    p.add_argument("--selftest-rate", type=float, default=5.0)
    return p


def main():
    args = build_parser().parse_args()
    if args.selftest:
        run_selftest(args)
    else:
        run_bridge(args)


if __name__ == "__main__":
    main()
