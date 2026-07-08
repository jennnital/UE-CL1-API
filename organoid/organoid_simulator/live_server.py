"""
Real-time interactive visualiser for the LIF organoid simulator.

Serves an 8x8 electrode grid on localhost (much like the SDK's built-in
visualiser). Click any electrode to stimulate it; with coupling enabled you
watch the evoked activity propagate outward across the array in real time, then
fade. A slider changes the coupling strength live so you can see the regime move
from local flashes (weak coupling) to grid-wide waves (strong coupling).

Architecture (pure standard library, no extra deps):
  * A background thread steps the LIFDataSource in slow-motion and maintains a
    decaying per-channel "activity" array (phosphor-style persistence).
  * The browser opens a Server-Sent-Events stream (`GET /stream`) that pushes the
    64-value activity array at ~25 fps.
  * Clicks POST to `/stim`; the coupling slider POSTs to `/config`. Both are
    drained by the sim thread, so only that thread ever touches the simulator.

Run:  PYTHONPATH=. python organoid_simulator/live_server.py
      (optional: CL_LIVE_PORT=8008)
"""
from __future__ import annotations

import errno
import json
import os
import queue
import signal
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

# Allow running directly (`uv run organoid_simulator/live_server.py`) without
# needing PYTHONPATH=. — put the repo root on sys.path so the package imports.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from cl.sim import DataSourceStim
from organoid_simulator.lif_data_source import LIFDataSource, NON_SPIKING, CHANNEL_COUNT

# Tuned for a visible "click -> expanding wave that fades" in an excitable regime.
DEFAULT_COUPLING = 13.0
BACKGROUND_STD   = 0.10
CLICK_PULSES     = 60
CLICK_PERIOD     = 25
CLICK_AMP_UA     = 2.8
CHUNK_FRAMES     = 40      # sim frames advanced per step (~1.6 ms)
STEP_SLEEP_S     = 0.03    # wall time per step  -> ~0.05x real time (watchable)
DECAY            = 0.90    # activity persistence per step
STREAM_FPS       = 25
NOISE_STD        = 0.22    # mild, ever-present Gaussian shimmer (recording noise floor)
REWARD_EVERY     = 6       # consolidate Hebbian learning every N steps when plasticity is on
PLASTICITY_LR    = 0.02    # base rate; the UI slider scales the reward on top of this
WEIGHT_MAX       = 60.0    # high so apply_reward's clip never crushes gain*structure coupling


class LiveSim:
    """Owns the simulator and steps it on a background thread."""

    def __init__(self, coupling_gain: float = DEFAULT_COUPLING):
        # Built with plasticity ready but OFF (no rewards applied) until the UI
        # enables it. weight_max is high so apply_reward's clip cannot crush the
        # coupling weights (which are gain*structure and can exceed small caps).
        self.source = LIFDataSource(random_seed=0, coupling_gain_mV=coupling_gain,
                                    background_drive_std_mV=BACKGROUND_STD,
                                    plasticity=True, plasticity_lr=PLASTICITY_LR,
                                    weight_max_mV=WEIGHT_MAX)
        self.source.open()
        self.intensity = np.zeros(CHANNEL_COUNT)
        self.coupling  = coupling_gain
        self.noise_std = NOISE_STD
        self.plasticity_on   = False
        self.plasticity_rate = 1.0          # reward magnitude = "extent of plasticity"
        self._reward_ctr     = 0
        self._lock     = threading.Lock()
        self._clicks: queue.Queue[int]   = queue.Queue()
        self._config: queue.Queue[tuple] = queue.Queue()
        self._noise_rng = np.random.default_rng(1)
        self._dead_idx  = list(NON_SPIKING)
        self._ts       = 0
        self._running  = True

    # -- public API called from HTTP handler threads ---------------------
    def stimulate(self, channel: int) -> None:
        self._clicks.put(int(channel))

    def set_coupling(self, gain: float) -> None:
        self._config.put(("coupling", float(gain)))

    def set_plasticity(self, on: bool) -> None:
        self._config.put(("plast_on", bool(on)))

    def set_plasticity_rate(self, rate: float) -> None:
        self._config.put(("plast_rate", float(rate)))

    def set_noise(self, std: float) -> None:
        with self._lock:
            self.noise_std = max(0.0, float(std))

    def snapshot(self) -> list[float]:
        """Display values = decaying spike activity + a mild, ever-present Gaussian
        noise floor. The shimmer is display-only (folded in here, not driven through
        the membrane), so it stays consistently visible without perturbing the
        excitable click-wave dynamics."""
        with self._lock:
            disp = self.intensity + np.abs(self._noise_rng.normal(0.0, self.noise_std, CHANNEL_COUNT))
            disp[self._dead_idx] = 0.0
            return disp.round(3).tolist()

    # -- background stepping ---------------------------------------------
    def _click_burst(self, channel: int, start_ts: int) -> list[DataSourceStim]:
        return [
            DataSourceStim(
                timestamp          = start_ts + p * CLICK_PERIOD,
                channel            = channel,
                phase_durations_us = (400, 400),
                phase_currents_uA  = (-CLICK_AMP_UA, CLICK_AMP_UA),
            )
            for p in range(CLICK_PULSES)
        ]

    def run(self) -> None:
        while self._running:
            while not self._clicks.empty():
                ch = self._clicks.get()
                if ch not in NON_SPIKING and 0 <= ch < CHANNEL_COUNT:
                    self.source.on_stims(self._click_burst(ch, self._ts + 1))
            while not self._config.empty():
                kind, val = self._config.get()
                if kind == "coupling":
                    self.source.set_coupling_gain(val)   # note: resets learned weights
                    with self._lock:
                        self.coupling = val
                elif kind == "plast_on":
                    self.plasticity_on = bool(val)
                elif kind == "plast_rate":
                    self.plasticity_rate = float(val)

            batch = self.source.read(self._ts, CHUNK_FRAMES)
            with self._lock:
                self.intensity *= DECAY
                for sp in batch.spikes:
                    self.intensity[sp.channel] += 1.0
                np.clip(self.intensity, 0.0, 8.0, out=self.intensity)

            # Hebbian consolidation: periodically reward whatever fired together
            # so repeatedly-driven pathways strengthen ("fire together, wire together").
            if self.plasticity_on and self.plasticity_rate > 0:
                self._reward_ctr += 1
                if self._reward_ctr >= REWARD_EVERY:
                    self.source.apply_reward(self.plasticity_rate)
                    self._reward_ctr = 0

            self._ts += CHUNK_FRAMES
            time.sleep(STEP_SLEEP_S)

    def stop(self) -> None:
        self._running = False


SIM: LiveSim | None = None  # set in main()


PAGE = """<!DOCTYPE html><html><head><meta charset="utf-8">
<title>LIF Organoid — Live</title>
<style>
  :root { --bg:#0b0e14; --panel:#141922; --line:#2a3444; --txt:#cdd6e4; --accent:#4cc9f0; }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--txt);
         font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif; }
  .wrap { max-width:720px; margin:0 auto; padding:24px 16px 40px; }
  h1 { font-size:1.3rem; font-weight:600; margin:0 0 4px; }
  .sub { color:#8895ab; font-size:.9rem; margin:0 0 20px; }
  .grid { display:grid; grid-template-columns:repeat(8,1fr); gap:6px;
          background:var(--panel); padding:16px; border-radius:14px; border:1px solid var(--line); }
  .cell { position:relative; aspect-ratio:1/1; border-radius:9px; background:#000;
          cursor:pointer; transition:background .05s linear, box-shadow .05s linear;
          display:flex; align-items:center; justify-content:center;
          font-size:.62rem; color:#3a4658; user-select:none; }
  .cell:hover { outline:2px solid var(--accent); }
  .cell.dead { cursor:not-allowed; background:#0e1420; color:#33405a; }
  .cell.dead:hover { outline:none; }
  .controls { display:flex; align-items:center; gap:14px; margin-top:20px;
              background:var(--panel); padding:14px 18px; border-radius:12px; border:1px solid var(--line); }
  .controls label { font-size:.85rem; color:#8895ab; white-space:nowrap; }
  input[type=range] { flex:1; accent-color:var(--accent); }
  #gainval { font-variant-numeric:tabular-nums; min-width:3.2em; text-align:right; color:var(--txt); }
  .status { display:flex; align-items:center; gap:8px; font-size:.82rem; color:#8895ab; margin-top:14px; }
  .dot { width:9px; height:9px; border-radius:50%; background:#e5484d; }
  .dot.on { background:#30d158; }
  .legend { display:flex; align-items:center; gap:8px; margin-top:14px; font-size:.8rem; color:#8895ab; }
  .bar { height:10px; width:180px; border-radius:5px;
         background:linear-gradient(90deg,#000,#280b54,#9f2a63,#d44842,#f57d15,#fac127,#fcffa4); }
  .gridwrap { position:relative; }
  .net { position:absolute; inset:0; pointer-events:none; display:none; }
  .controls label { display:flex; align-items:center; gap:6px; cursor:pointer; }
  input[type=checkbox] { accent-color:var(--accent); width:15px; height:15px; margin:0; }
</style></head>
<body><div class="wrap">
  <h1>LIF Organoid — live 8×8 array</h1>
  <p class="sub">Click any electrode to stimulate it. With coupling on, watch the response propagate and fade.</p>
  <div class="gridwrap">
    <div class="grid" id="grid"></div>
    <svg class="net" id="net"></svg>
  </div>
  <div class="controls">
    <label style="cursor:default">coupling strength</label>
    <input type="range" id="gain" min="0" max="24" step="0.5" value="13">
    <span id="gainval">13.0 mV</span>
  </div>
  <div class="controls">
    <label style="cursor:default">background noise</label>
    <input type="range" id="noise" min="0" max="0.6" step="0.02" value="0.22">
    <span id="noiseval">0.22</span>
  </div>
  <div class="controls">
    <label><input type="checkbox" id="plast"> plasticity (Hebbian learning)</label>
    <input type="range" id="prate" min="0" max="3" step="0.1" value="1.0">
    <span id="prateval">extent 1.0</span>
  </div>
  <div class="controls">
    <label><input type="checkbox" id="structure"> show structural connectivity</label>
    <span style="color:#8895ab;font-size:.74rem">grey = local · cyan = long-range shortcut</span>
  </div>
  <div class="legend"><span>low</span><div class="bar"></div><span>high activity</span></div>
  <div class="status"><span class="dot" id="dot"></span><span id="stat">connecting…</span></div>
</div>
<script>
const DEAD = new Set([0,4,7,56,63]);   // grounded + reference channels
const cells = {};
const grid = document.getElementById('grid');
// visual (row, col) -> channel = row + 8*col  (matches the simulator layout)
for (let r=0; r<8; r++) for (let c=0; c<8; c++) {
  const chan = r + 8*c;
  const d = document.createElement('div');
  d.className = 'cell' + (DEAD.has(chan) ? ' dead' : '');
  d.textContent = chan;
  if (!DEAD.has(chan)) d.onclick = () => stim(chan);
  grid.appendChild(d);
  cells[chan] = d;
}

// inferno-ish colour ramp for activity in [0,1]
const STOPS = [[0,0,0],[40,11,84],[159,42,99],[212,72,66],[245,125,21],[250,193,39],[252,255,164]];
function heat(t){
  t = Math.max(0, Math.min(1, t));
  const x = t*(STOPS.length-1), i = Math.floor(x), f = x-i;
  const a = STOPS[i], b = STOPS[Math.min(i+1, STOPS.length-1)];
  const m = k => Math.round(a[k]+(b[k]-a[k])*f);
  return `rgb(${m(0)},${m(1)},${m(2)})`;
}

function stim(chan){
  cells[chan].animate([{transform:'scale(0.82)'},{transform:'scale(1)'}], {duration:160});
  fetch('/stim', {method:'POST', body: JSON.stringify({channel: chan})});
}

const gain = document.getElementById('gain'), gainval = document.getElementById('gainval');
gain.oninput = () => {
  gainval.textContent = parseFloat(gain.value).toFixed(1) + ' mV';
  fetch('/config', {method:'POST', body: JSON.stringify({coupling_gain: parseFloat(gain.value)})});
};

const noise = document.getElementById('noise'), noiseval = document.getElementById('noiseval');
noise.oninput = () => {
  noiseval.textContent = parseFloat(noise.value).toFixed(2);
  fetch('/config', {method:'POST', body: JSON.stringify({noise: parseFloat(noise.value)})});
};

// plasticity: checkbox toggles learning, slider sets its extent (reward magnitude)
const plast = document.getElementById('plast'), prate = document.getElementById('prate'),
      prateval = document.getElementById('prateval');
plast.onchange = () =>
  fetch('/config', {method:'POST', body: JSON.stringify({plasticity: plast.checked})});
prate.oninput = () => {
  prateval.textContent = 'extent ' + parseFloat(prate.value).toFixed(1);
  fetch('/config', {method:'POST', body: JSON.stringify({plasticity_rate: parseFloat(prate.value)})});
};

// structural connectivity overlay (the fixed, pre-determined graph)
const net = document.getElementById('net'), structure = document.getElementById('structure');
let netEdges = null;
const gridDist = (a, b) => Math.abs(a % 8 - b % 8) + Math.abs((a / 8 | 0) - (b / 8 | 0));
function drawNet() {
  if (!netEdges) return;
  const wrap = net.parentElement;
  net.setAttribute('width', wrap.clientWidth);
  net.setAttribute('height', wrap.clientHeight);
  const mx = netEdges.max_w || 1;
  let s = '';
  for (const [i, j, w] of netEdges.edges) {
    const a = cells[i], b = cells[j];
    const x1 = a.offsetLeft + a.offsetWidth/2, y1 = a.offsetTop + a.offsetHeight/2;
    const x2 = b.offsetLeft + b.offsetWidth/2, y2 = b.offsetTop + b.offsetHeight/2;
    const f = w / mx, shortcut = gridDist(i, j) > 2;
    const op = (0.06 + 0.55 * f).toFixed(2);
    const col = shortcut ? `rgba(76,201,240,${op})` : `rgba(185,195,215,${op})`;
    s += `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="${col}" stroke-width="${(0.5 + 1.8*f).toFixed(2)}"/>`;
  }
  net.innerHTML = s;
}
structure.onchange = async () => {
  if (structure.checked) {
    if (!netEdges) netEdges = await (await fetch('/connectivity')).json();
    drawNet(); net.style.display = 'block';
  } else {
    net.style.display = 'none';
  }
};
window.addEventListener('resize', () => { if (structure.checked) drawNet(); });

const dot = document.getElementById('dot'), stat = document.getElementById('stat');
const es = new EventSource('/stream');
es.onopen = () => { dot.classList.add('on'); stat.textContent = 'live — streaming activity'; };
es.onerror = () => { dot.classList.remove('on'); stat.textContent = 'disconnected — is the server running?'; };
es.onmessage = (e) => {
  const v = JSON.parse(e.data);
  for (let ch=0; ch<64; ch++){
    if (DEAD.has(ch)) continue;
    const t = Math.min(1, v[ch]/4);
    cells[ch].style.background = heat(t);
    cells[ch].style.boxShadow = t > 0.05 ? `0 0 ${Math.round(t*18)}px ${heat(t)}` : 'none';
  }
};
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass  # keep the console clean

    def _json_body(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        if not length:
            return {}
        try:
            return json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            return {}

    def do_GET(self):
        if self.path == "/" or self.path.startswith("/index"):
            body = PAGE.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/stream":
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.end_headers()
            try:
                while True:
                    intensity = SIM.snapshot()
                    self.wfile.write(f"data: {json.dumps(intensity)}\n\n".encode())
                    self.wfile.flush()
                    time.sleep(1.0 / STREAM_FPS)
            except (BrokenPipeError, ConnectionResetError):
                pass  # browser closed the stream
        elif self.path == "/connectivity":
            # The fixed, pre-determined structural graph (before gain / learning).
            C = SIM.source.connectivity_matrix
            edges, mx = [], 0.0
            for i in range(CHANNEL_COUNT):
                for j in range(i + 1, CHANNEL_COUNT):
                    w = max(float(C[i, j]), float(C[j, i]))
                    if w > 0:
                        edges.append([i, j, round(w, 3)])
                        mx = max(mx, w)
            body = json.dumps({"edges": edges, "max_w": round(mx, 3)}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_error(404)

    def do_POST(self):
        data = self._json_body()
        if self.path == "/stim" and "channel" in data:
            SIM.stimulate(int(data["channel"]))
            self._ok()
        elif self.path == "/config" and "coupling_gain" in data:
            SIM.set_coupling(float(data["coupling_gain"]))
            self._ok()
        elif self.path == "/config" and "noise" in data:
            SIM.set_noise(float(data["noise"]))
            self._ok()
        elif self.path == "/config" and "plasticity" in data:
            SIM.set_plasticity(bool(data["plasticity"]))
            self._ok()
        elif self.path == "/config" and "plasticity_rate" in data:
            SIM.set_plasticity_rate(float(data["plasticity_rate"]))
            self._ok()
        else:
            self.send_error(404)

    def _ok(self):
        self.send_response(204)
        self.send_header("Content-Length", "0")
        self.end_headers()


def _pids_on_port(port: int) -> list[int]:
    """PIDs listening on `port` (via lsof). Empty if lsof is unavailable."""
    try:
        out = subprocess.run(
            ["lsof", "-nP", f"-iTCP:{port}", "-sTCP:LISTEN", "-t"],
            capture_output=True, text=True, timeout=5,
        ).stdout
    except (FileNotFoundError, subprocess.SubprocessError):
        return []
    return [int(x) for x in out.split()]


def _cmdline(pid: int) -> str:
    try:
        return subprocess.run(["ps", "-p", str(pid), "-o", "command="],
                              capture_output=True, text=True, timeout=5).stdout.strip()
    except (FileNotFoundError, subprocess.SubprocessError):
        return ""


def _free_port_if_ours(port: int) -> bool:
    """Make `port` bindable if it's held by a *previous live_server.py* instance.

    Returns True if the port is free (or was freed by killing our own stale
    server), False if it's held by an unrelated process — which we refuse to kill.
    """
    pids = [p for p in _pids_on_port(port) if p != os.getpid()]
    if not pids:
        return True
    ok = True
    for pid in pids:
        cmd = _cmdline(pid)
        if "live_server.py" not in cmd:
            print(f"  Port {port} is held by an unrelated process (pid {pid}): {cmd[:80]}")
            ok = False
            continue
        print(f"  Port {port} held by a previous live_server (pid {pid}) — stopping it.")
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            continue
        for _ in range(20):                       # wait up to ~2s for a clean exit
            time.sleep(0.1)
            if pid not in _pids_on_port(port):
                break
        else:                                     # still alive → force it
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            time.sleep(0.3)
    return ok


def main():
    global SIM
    try:
        sys.stdout.reconfigure(line_buffering=True)   # flush status lines even when piped
    except Exception:
        pass
    host = "127.0.0.1"
    port = int(os.getenv("CL_LIVE_PORT", "8008"))
    SIM = LiveSim()
    threading.Thread(target=SIM.run, daemon=True).start()

    try:
        server = ThreadingHTTPServer((host, port), Handler)
    except OSError as e:
        if e.errno != errno.EADDRINUSE:
            raise
        if not _free_port_if_ours(port):
            print(f"\n  Cannot start: port {port} is in use by another program.")
            print(f"  Use a different port, e.g.  CL_LIVE_PORT=8020 uv run organoid_simulator/live_server.py\n")
            SIM.stop()
            return
        server = ThreadingHTTPServer((host, port), Handler)   # retry once, port now free

    url = f"http://127.0.0.1:{port}/"
    print(f"\n  LIF organoid live visualiser running at:  {url}")
    print("  Click electrodes to stimulate. Ctrl-C to stop.\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  Shutting down…")
    finally:
        SIM.stop()
        server.shutdown()


if __name__ == "__main__":
    main()
