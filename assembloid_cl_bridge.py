#!/usr/bin/env python3
"""
assembloid_cl_bridge.py
=======================

Closed-loop UDP bridge between Cortical Labs' CL1 and Unreal Engine, for the
Assembloid Agency project.

It does two things at once:

  1.  SPIKES  CL1 -> Unreal
      Runs the CL-API loop, and for every tick that contains spikes it sends a
      UDP packet to Unreal. The packet format is byte-compatible with Cortical
      Labs' own CL-01 / CL-01A examples (uint64 timestamp + one byte per channel).

  2.  STIM    Unreal -> CL1
      Listens on a second UDP port for Assembloid Agency STIM packets coming
      from Unreal, validates them against a safety envelope, and applies them
      to the culture via neurons.stim() from the loop thread.

All hardware access (loop + stim) happens on a single thread. The stim socket
is read on a background thread that only enqueues validated requests; the loop
thread drains that queue each tick and is the sole caller of neurons.stim().

See PROTOCOL.md for the exact wire formats.

Run on the CL1 device (where `import cl` works), or anywhere with `--simulate`
to develop the Unreal side without hardware.

    python assembloid_cl_bridge.py --unreal-ip 192.168.1.50
    python assembloid_cl_bridge.py --simulate            # no hardware needed
"""

import argparse
import logging
import queue
import socket
import struct
import sys
import threading
import time

# --------------------------------------------------------------------------- #
#  Wire formats (see PROTOCOL.md)
# --------------------------------------------------------------------------- #

SPIKE_TS_STRUCT = struct.Struct("<Q")          # uint64 timestamp, then 1 byte/channel

STIM_MAGIC = b"AA"
STIM_VERSION = 1
STIM_MSG = 1
# magic(2) version(1) msg_type(1) flags(1) num_channels(1)
# pulse_width_us(uint16) amplitude_uA(float32) num_pulses(uint16) freq_hz(uint16)
STIM_HEADER = struct.Struct("<2sBBBBHfHH")
STIM_HEADER_SIZE = STIM_HEADER.size           # 16 bytes
STIM_FLAG_BIPHASIC = 0x01

log = logging.getLogger("cl_bridge")


# --------------------------------------------------------------------------- #
#  Safety envelope
# --------------------------------------------------------------------------- #

class SafetyLimits:
    """Conservative defaults. Set these to match your own lab / IRB protocol."""

    def __init__(self, max_amplitude_ua=10.0, min_pulse_us=10, max_pulse_us=1000,
                 max_pulses=100, min_freq_hz=1, max_freq_hz=200,
                 min_channel=0, max_channel=59):
        self.max_amplitude_ua = max_amplitude_ua
        self.min_pulse_us = min_pulse_us
        self.max_pulse_us = max_pulse_us
        self.max_pulses = max_pulses
        self.min_freq_hz = min_freq_hz
        self.max_freq_hz = max_freq_hz
        self.min_channel = min_channel
        self.max_channel = max_channel

    def validate(self, cmd):
        """Return (ok: bool, reason: str). Commands are never silently altered."""
        if abs(cmd["amplitude_ua"]) > self.max_amplitude_ua:
            return False, f"amplitude {cmd['amplitude_ua']:.2f}µA exceeds {self.max_amplitude_ua}µA"
        if not (self.min_pulse_us <= cmd["pulse_width_us"] <= self.max_pulse_us):
            return False, f"pulse width {cmd['pulse_width_us']}µs out of range"
        if not (1 <= cmd["num_pulses"] <= self.max_pulses):
            return False, f"num_pulses {cmd['num_pulses']} out of range"
        if cmd["num_pulses"] > 1 and not (self.min_freq_hz <= cmd["freq_hz"] <= self.max_freq_hz):
            return False, f"freq {cmd['freq_hz']}Hz out of range"
        for ch in cmd["channels"]:
            if not (self.min_channel <= ch <= self.max_channel):
                return False, f"channel {ch} out of range"
        if not cmd["channels"]:
            return False, "no channels specified"
        return True, ""


# --------------------------------------------------------------------------- #
#  STIM packet parsing
# --------------------------------------------------------------------------- #

def parse_stim_packet(data):
    """Parse a STIM packet into a dict, or return None if malformed."""
    if len(data) < STIM_HEADER_SIZE:
        return None
    (magic, version, msg_type, flags, num_channels,
     pulse_width_us, amplitude_ua, num_pulses, freq_hz) = STIM_HEADER.unpack_from(data, 0)

    if magic != STIM_MAGIC or version != STIM_VERSION or msg_type != STIM_MSG:
        return None
    if len(data) < STIM_HEADER_SIZE + num_channels:
        return None

    channels = list(data[STIM_HEADER_SIZE:STIM_HEADER_SIZE + num_channels])
    return {
        "channels": channels,
        "pulse_width_us": pulse_width_us,
        "amplitude_ua": amplitude_ua,
        "num_pulses": num_pulses,
        "freq_hz": freq_hz,
        "biphasic": bool(flags & STIM_FLAG_BIPHASIC),
    }


# --------------------------------------------------------------------------- #
#  Stim receiver thread: socket -> validate -> queue
# --------------------------------------------------------------------------- #

class StimReceiver(threading.Thread):
    def __init__(self, listen_ip, listen_port, limits, out_queue, stop_event):
        super().__init__(name="StimReceiver", daemon=True)
        self.listen_ip = listen_ip
        self.listen_port = listen_port
        self.limits = limits
        self.out_queue = out_queue
        self.stop_event = stop_event
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((self.listen_ip, self.listen_port))
        self.sock.settimeout(0.25)

    def run(self):
        log.info("Listening for STIM packets on %s:%d", self.listen_ip, self.listen_port)
        while not self.stop_event.is_set():
            try:
                data, addr = self.sock.recvfrom(2048)
            except socket.timeout:
                continue
            except OSError:
                break

            cmd = parse_stim_packet(data)
            if cmd is None:
                log.warning("Dropped malformed STIM packet (%d bytes) from %s", len(data), addr)
                continue

            ok, reason = self.limits.validate(cmd)
            if not ok:
                log.warning("Rejected STIM from %s: %s", addr, reason)
                continue

            # Drop, don't block the loop, if the queue is backed up.
            try:
                self.out_queue.put_nowait(cmd)
            except queue.Full:
                log.warning("Stim queue full; dropping command")
        self.sock.close()
        log.info("StimReceiver stopped")


# --------------------------------------------------------------------------- #
#  Spike sender
# --------------------------------------------------------------------------- #

class SpikeSender:
    def __init__(self, unreal_ip, unreal_port):
        self.peer = (unreal_ip, unreal_port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sent = 0

    def send(self, timestamp, channels):
        # uint64 timestamp + one uint8 per channel  (CL-compatible)
        payload = bytearray(SPIKE_TS_STRUCT.pack(int(timestamp) & 0xFFFFFFFFFFFFFFFF))
        payload.extend(int(c) & 0xFF for c in channels)
        self.sock.sendto(payload, self.peer)
        self.sent += 1

    def close(self):
        self.sock.close()


# --------------------------------------------------------------------------- #
#  Stim application (loop thread only)
# --------------------------------------------------------------------------- #

def apply_stim(neurons, cl, cmd):
    """Translate a validated STIM command into CL-API calls. Loop thread only."""
    amp = float(cmd["amplitude_ua"])
    pw = int(cmd["pulse_width_us"])
    chans = cl.ChannelSet(*cmd["channels"])

    if cmd["biphasic"]:
        # Charge-balanced: cathodic phase then anodic phase, equal width.
        design = cl.StimDesign(pw, -amp, pw, +amp)
    else:
        design = cl.StimDesign(pw, amp)

    if cmd["num_pulses"] > 1 and cmd["freq_hz"] > 0:
        neurons.stim(chans, design, cl.BurstDesign(cmd["num_pulses"], cmd["freq_hz"]))
    else:
        neurons.stim(chans, design)


def drain_stim_queue(stim_queue, apply_fn):
    """Apply every pending stim command. Returns count applied."""
    n = 0
    while True:
        try:
            cmd = stim_queue.get_nowait()
        except queue.Empty:
            break
        try:
            apply_fn(cmd)
            n += 1
        except Exception as exc:  # noqa: BLE001 - never let one bad stim kill the loop
            log.error("Stim apply failed: %s", exc)
    return n


# --------------------------------------------------------------------------- #
#  Real CL1 run loop
# --------------------------------------------------------------------------- #

def run_hardware(args, sender, stim_queue, stop_event):
    import cl  # only imported on the device

    log.info("Opening CL1 connection ...")
    with cl.open() as neurons:
        log.info("Connected. Streaming spikes -> %s:%d at %d ticks/s",
                 args.unreal_ip, args.unreal_port, args.tick_rate)

        loop_kwargs = dict(ticks_per_second=args.tick_rate, ignore_jitter=True)
        if args.run_seconds > 0:
            loop_kwargs["stop_after_seconds"] = args.run_seconds

        apply_fn = lambda cmd: apply_stim(neurons, cl, cmd)  # noqa: E731

        last_report = time.monotonic()
        spikes_seen = 0
        stims_done = 0

        for tick in neurons.loop(**loop_kwargs):
            if stop_event.is_set():
                break

            # 1) outbound: spikes -> Unreal
            spikes = tick.analysis.spikes
            if spikes:
                spikes_seen += len(spikes)
                if args.strict_timestamps:
                    for s in spikes:
                        sender.send(s.timestamp, [s.channel])
                else:
                    sender.send(spikes[0].timestamp, [s.channel for s in spikes])

            # 2) inbound: stim commands -> culture
            stims_done += drain_stim_queue(stim_queue, apply_fn)

            now = time.monotonic()
            if now - last_report >= 1.0:
                log.info("spikes/s=%d  stims applied=%d  packets sent=%d",
                         spikes_seen, stims_done, sender.sent)
                spikes_seen = 0
                last_report = now


# --------------------------------------------------------------------------- #
#  Simulated run loop (no hardware)
# --------------------------------------------------------------------------- #

def run_simulated(args, sender, stim_queue, stop_event):
    import random

    log.info("SIMULATION mode: generating synthetic spikes, no hardware.")
    log.info("Streaming spikes -> %s:%d", args.unreal_ip, args.unreal_port)

    frames_per_tick = 25000 // max(1, args.sim_report_rate)
    timestamp = 0
    deadline = time.monotonic()
    interval = 1.0 / max(1, args.sim_report_rate)
    t_end = time.monotonic() + args.run_seconds if args.run_seconds > 0 else None

    def apply_fn(cmd):
        log.info("[SIM] would stim ch=%s  %dµs  %.2fµA  x%d @ %dHz",
                 cmd["channels"], cmd["pulse_width_us"], cmd["amplitude_ua"],
                 cmd["num_pulses"], cmd["freq_hz"])

    last_report = time.monotonic()
    spikes_seen = 0
    while not stop_event.is_set():
        if t_end and time.monotonic() >= t_end:
            break

        # Poisson-ish synthetic activity across a handful of channels
        n = random.choices([0, 1, 2, 3], weights=[55, 30, 10, 5])[0]
        if n:
            channels = random.sample(range(args.sim_min_ch, args.sim_max_ch + 1), n)
            sender.send(timestamp, channels)
            spikes_seen += n

        drain_stim_queue(stim_queue, apply_fn)

        timestamp += frames_per_tick
        now = time.monotonic()
        if now - last_report >= 1.0:
            log.info("[SIM] spikes/s=%d  packets sent=%d", spikes_seen, sender.sent)
            spikes_seen = 0
            last_report = now

        deadline += interval
        sleep = deadline - time.monotonic()
        if sleep > 0:
            time.sleep(sleep)
        else:
            deadline = time.monotonic()


# --------------------------------------------------------------------------- #
#  Entry point
# --------------------------------------------------------------------------- #

def build_parser():
    p = argparse.ArgumentParser(description="Assembloid Agency CL1 <-> Unreal UDP bridge")
    # Network
    p.add_argument("--unreal-ip", default="127.0.0.1",
                   help="IP of the machine running Unreal (spike destination)")
    p.add_argument("--unreal-port", type=int, default=12345,
                   help="Unreal UDP receive port for spikes")
    p.add_argument("--stim-listen-ip", default="0.0.0.0",
                   help="Local IP to listen on for stim commands (0.0.0.0 = all)")
    p.add_argument("--stim-listen-port", type=int, default=12346,
                   help="Local UDP port to receive stim commands from Unreal")
    # Loop
    p.add_argument("--tick-rate", type=int, default=25000,
                   help="CL-API loop ticks per second (max 25000)")
    p.add_argument("--run-seconds", type=float, default=0,
                   help="Stop after N seconds (0 = run until Ctrl-C)")
    p.add_argument("--strict-timestamps", action="store_true",
                   help="Send one packet per spike with exact timestamps")
    # Safety
    p.add_argument("--max-amplitude-ua", type=float, default=10.0)
    p.add_argument("--max-pulse-us", type=int, default=1000)
    p.add_argument("--max-pulses", type=int, default=100)
    p.add_argument("--max-freq-hz", type=int, default=200)
    p.add_argument("--max-channel", type=int, default=59)
    # Simulation
    p.add_argument("--simulate", action="store_true",
                   help="Generate synthetic spikes without CL hardware")
    p.add_argument("--sim-report-rate", type=int, default=200,
                   help="[sim] synthetic ticks per second")
    p.add_argument("--sim-min-ch", type=int, default=0)
    p.add_argument("--sim-max-ch", type=int, default=59)
    p.add_argument("--verbose", action="store_true")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )

    limits = SafetyLimits(
        max_amplitude_ua=args.max_amplitude_ua,
        max_pulse_us=args.max_pulse_us,
        max_pulses=args.max_pulses,
        max_freq_hz=args.max_freq_hz,
        max_channel=args.max_channel,
    )

    stop_event = threading.Event()
    stim_queue = queue.Queue(maxsize=256)
    sender = SpikeSender(args.unreal_ip, args.unreal_port)

    receiver = StimReceiver(args.stim_listen_ip, args.stim_listen_port,
                            limits, stim_queue, stop_event)
    receiver.start()

    try:
        if args.simulate:
            run_simulated(args, sender, stim_queue, stop_event)
        else:
            run_hardware(args, sender, stim_queue, stop_event)
    except KeyboardInterrupt:
        log.info("Interrupted by user")
    except ImportError:
        log.error("Could not `import cl` — run on the CL1 device, or use --simulate.")
        return 2
    finally:
        stop_event.set()
        receiver.join(timeout=1.0)
        sender.close()
        log.info("Bridge shut down. Total spike packets sent: %d", sender.sent)
    return 0


if __name__ == "__main__":
    sys.exit(main())
