"""
Real-time interactive visualiser for the v2 (Brian2/AdEx) organoid substrate.

Serves an 8x8 electrode grid on localhost. Click an electrode to stimulate it, or use
the **scenario buttons** to run canned demos -- e.g. *"Couple two channels"* fires two
well-separated electrodes and you watch the evoked activity propagate through the
delay-coupled network and meet in the middle: the headline v2 feature (conduction
delays turn coupling into a *travelling* wave, not an instantaneous flash). A coupling
slider moves the regime from local flashes (weak E-coupling) to grid-wide waves.

This is v1's ``live_server`` architecture pointed at ``BrianOrganoidDataSource`` (the
plan's "reuse the whole interactive viewer" step), plus scenario automation:

  * A background thread steps the Brian2 source in slow motion (its per-chunk compute
    cost naturally yields a watchable ~0.05x real-time pace) and maintains a decaying
    per-channel "activity" array (phosphor persistence).
  * The browser opens a Server-Sent-Events stream (`GET /stream`) pushing the 64-value
    activity array at ~25 fps.
  * Clicks POST to `/stim`, scenario buttons to `/scenario`, sliders to `/config`.
    All are drained by the sim thread, so only it ever touches the simulator.

Run:  uv run organoid_simulator/v2/live_server_v2.py     (optional: CL_LIVE_PORT=8009)
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

# Allow running directly without PYTHONPATH -- put the repo root on sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from cl.sim import DataSourceStim
from organoid_simulator.v2 import BrianOrganoidDataSource
from organoid_simulator.lif_data_source import NON_SPIKING, CHANNEL_COUNT

# --- excitable-regime defaults (tuned so a single stim erupts into a visible,
# travelling, then-fading wave; baseline is quiet until stimulated). ---
DEFAULT_COUPLING = 8.0      # excitatory weight scale (the coupling slider)
CHUNK_FRAMES     = 40       # sim frames advanced per step (~1.6 ms)
STEP_SLEEP_S     = 0.0      # Brian2's per-chunk compute already paces it (~0.05x RT)
DECAY            = 0.88     # activity persistence per step
STREAM_FPS       = 25
NOISE_STD        = 0.18     # display-only Gaussian shimmer (recording noise floor)
CLICK_PULSES     = 16
CLICK_PERIOD     = 20       # frames between pulses in a burst
CLICK_AMP_UA     = 2.8
REWARD_EVERY     = 6        # consolidate plasticity every N steps when on
PLASTICITY_RATE  = 1.0

# Scenario electrode choices (avoid grounded 0/7/56/63 and reference 4).
PAIR_A, PAIR_B = 10, 53     # opposite regions of the array
CENTER_CH      = 27


class LiveSim:
    """Owns the Brian2 simulator and steps it on a background thread."""

    def __init__(self, coupling: float = DEFAULT_COUPLING):
        self.source = BrianOrganoidDataSource(
            n_neurons=64, random_seed=0,
            p_connect=0.6, conn_lambda_um=350, w_exc_nS=1.5,
            exc_scale=coupling, inh_scale=1.0,
            # Weakened adaptation so a stim gives a consistent, far-reaching, repeatable
            # travelling wave instead of an all-or-nothing burst that then goes refractory.
            adaptation_scale=0.3,
            background=dict(ge0=10, gi0=5, sigma_e=1.5, sigma_i=1.5),
            plasticity=True,
        )
        self.source.open()
        self.intensity = np.zeros(CHANNEL_COUNT)
        self.coupling  = coupling
        self.noise_std = NOISE_STD
        self.plasticity_on   = False
        self.plasticity_rate = PLASTICITY_RATE
        self._reward_ctr = 0
        self._lock   = threading.Lock()
        self._stims: queue.Queue[tuple[int, int]] = queue.Queue()  # (channel, delay_frames)
        self._config: queue.Queue[tuple] = queue.Queue()
        self._noise_rng = np.random.default_rng(1)
        self._dead_idx  = list(NON_SPIKING)
        self._ts = 0
        self._running = True

    # -- public API called from HTTP handler threads ---------------------
    def stimulate(self, channel: int, delay_frames: int = 1) -> None:
        self._stims.put((int(channel), int(delay_frames)))

    def run_scenario(self, name: str) -> list[int]:
        """Queue a canned stimulation pattern. Returns the channels it will fire so the
        UI can highlight them."""
        if name == "couple_pair":
            # Fire two well-separated electrodes ~simultaneously; watch their waves
            # propagate through the coupled network and meet -> "see them couple".
            self.stimulate(PAIR_A, 1)
            self.stimulate(PAIR_B, 1)
            return [PAIR_A, PAIR_B]
        if name == "sequence":
            # Fire A, then B a beat later: a directional pre->post interaction.
            self.stimulate(PAIR_A, 1)
            self.stimulate(PAIR_B, 60)   # +~2.4 ms
            return [PAIR_A, PAIR_B]
        if name == "wave":
            self.stimulate(CENTER_CH, 1)
            return [CENTER_CH]
        if name == "reset":
            self._config.put(("reset", None))
            return []
        return []

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
        """Display values = decaying spike activity + a mild display-only shimmer."""
        with self._lock:
            disp = self.intensity + np.abs(self._noise_rng.normal(0.0, self.noise_std, CHANNEL_COUNT))
            disp[self._dead_idx] = 0.0
            return disp.round(3).tolist()

    # -- background stepping ---------------------------------------------
    def _burst(self, channel: int, start_ts: int) -> list[DataSourceStim]:
        return [
            DataSourceStim(
                timestamp=start_ts + p * CLICK_PERIOD, channel=channel,
                phase_durations_us=(400, 400),
                phase_currents_uA=(-CLICK_AMP_UA, CLICK_AMP_UA),
            )
            for p in range(CLICK_PULSES)
        ]

    def run(self) -> None:
        while self._running:
            while not self._stims.empty():
                ch, delay = self._stims.get()
                if ch not in NON_SPIKING and 0 <= ch < CHANNEL_COUNT:
                    self.source.on_stims(self._burst(ch, self._ts + max(1, delay)))
            while not self._config.empty():
                kind, val = self._config.get()
                if kind == "coupling":
                    self.source.set_coupling_gain(val)   # resets learned weights
                    with self._lock:
                        self.coupling = val
                elif kind == "plast_on":
                    self.plasticity_on = bool(val)
                elif kind == "plast_rate":
                    self.plasticity_rate = float(val)
                elif kind == "reset":
                    with self._lock:
                        self.intensity[:] = 0.0

            batch = self.source.read(self._ts, CHUNK_FRAMES)
            with self._lock:
                self.intensity *= DECAY
                for sp in batch.spikes:
                    self.intensity[sp.channel] += 1.0
                np.clip(self.intensity, 0.0, 8.0, out=self.intensity)

            if self.plasticity_on and self.plasticity_rate > 0:
                self._reward_ctr += 1
                if self._reward_ctr >= REWARD_EVERY:
                    self.source.apply_reward(self.plasticity_rate)
                    self._reward_ctr = 0

            self._ts += CHUNK_FRAMES
            if STEP_SLEEP_S:
                time.sleep(STEP_SLEEP_S)

    def stop(self) -> None:
        self._running = False


SIM: LiveSim | None = None  # set in main()


PAGE = """<!DOCTYPE html><html><head><meta charset="utf-8">
<title>v2 Brian Organoid — Live</title>
<style>
  :root { --bg:#0b0e14; --panel:#141922; --line:#2a3444; --txt:#cdd6e4; --accent:#4cc9f0; --warm:#f57d15; }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--txt);
         font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif; }
  .wrap { max-width:760px; margin:0 auto; padding:24px 16px 40px; }
  h1 { font-size:1.3rem; font-weight:600; margin:0 0 4px; }
  .sub { color:#8895ab; font-size:.9rem; margin:0 0 18px; }
  .grid { display:grid; grid-template-columns:repeat(8,1fr); gap:6px;
          background:var(--panel); padding:16px; border-radius:14px; border:1px solid var(--line); }
  .cell { position:relative; aspect-ratio:1/1; border-radius:9px; background:#000;
          cursor:pointer; transition:background .05s linear, box-shadow .05s linear;
          display:flex; align-items:center; justify-content:center;
          font-size:.62rem; color:#3a4658; user-select:none; }
  .cell:hover { outline:2px solid var(--accent); }
  .cell.dead { cursor:not-allowed; background:#0e1420; color:#33405a; }
  .cell.dead:hover { outline:none; }
  .cell.mark { outline:2px solid var(--warm); }
  .scenarios { display:flex; flex-wrap:wrap; gap:10px; margin-top:18px; }
  .btn { background:var(--panel); color:var(--txt); border:1px solid var(--line);
         border-radius:10px; padding:10px 14px; font-size:.86rem; cursor:pointer;
         transition:border-color .1s, background .1s; }
  .btn:hover { border-color:var(--accent); background:#1b2230; }
  .btn.primary { border-color:var(--accent); }
  .btn.ghost { color:#8895ab; }
  .controls { display:flex; align-items:center; gap:14px; margin-top:14px;
              background:var(--panel); padding:14px 18px; border-radius:12px; border:1px solid var(--line); }
  .controls label { font-size:.85rem; color:#8895ab; white-space:nowrap; }
  input[type=range] { flex:1; accent-color:var(--accent); }
  .val { font-variant-numeric:tabular-nums; min-width:3.4em; text-align:right; color:var(--txt); }
  .status { display:flex; align-items:center; gap:8px; font-size:.82rem; color:#8895ab; margin-top:14px; }
  .dot { width:9px; height:9px; border-radius:50%; background:#e5484d; }
  .dot.on { background:#30d158; }
  .legend { display:flex; align-items:center; gap:8px; margin-top:14px; font-size:.8rem; color:#8895ab; }
  .bar { height:10px; width:180px; border-radius:5px;
         background:linear-gradient(90deg,#000,#280b54,#9f2a63,#d44842,#f57d15,#fac127,#fcffa4); }
  .gridwrap { position:relative; }
  .net { position:absolute; inset:0; pointer-events:none; display:none; }
  .controls label.chk { display:flex; align-items:center; gap:6px; cursor:pointer; }
  input[type=checkbox] { accent-color:var(--accent); width:15px; height:15px; margin:0; }
</style></head>
<body><div class="wrap">
  <h1>v2 Brian organoid — live 8×8 array</h1>
  <p class="sub">AdEx spiking network with conduction-delayed conductance synapses. Click an electrode to
     stimulate it, or run a scenario below and watch activity <em>propagate</em> across the coupled network.</p>

  <div class="gridwrap">
    <div class="grid" id="grid"></div>
    <svg class="net" id="net"></svg>
  </div>

  <div class="scenarios">
    <button class="btn primary" onclick="scenario('couple_pair')">▶ Couple two channels</button>
    <button class="btn" onclick="scenario('sequence')">▶ Fire A then B (delayed)</button>
    <button class="btn" onclick="scenario('wave')">▶ Travelling wave (centre)</button>
    <button class="btn ghost" onclick="scenario('reset')">◼ Clear activity</button>
  </div>

  <div class="controls">
    <label style="cursor:default">coupling strength</label>
    <input type="range" id="gain" min="0" max="16" step="0.5" value="8">
    <span class="val" id="gainval">8.0×</span>
  </div>
  <div class="controls">
    <label style="cursor:default">background shimmer</label>
    <input type="range" id="noise" min="0" max="0.6" step="0.02" value="0.18">
    <span class="val" id="noiseval">0.18</span>
  </div>
  <div class="controls">
    <label class="chk"><input type="checkbox" id="plast"> plasticity (reward-gated)</label>
    <input type="range" id="prate" min="0" max="3" step="0.1" value="1.0">
    <span class="val" id="prateval">extent 1.0</span>
  </div>
  <div class="controls">
    <label class="chk"><input type="checkbox" id="structure"> show structural connectivity</label>
    <span style="color:#8895ab;font-size:.74rem">brighter = stronger synaptic pathway</span>
  </div>

  <div class="legend"><span>low</span><div class="bar"></div><span>high activity</span></div>
  <div class="status"><span class="dot" id="dot"></span><span id="stat">connecting…</span></div>
</div>
<script>
const DEAD = new Set([0,4,7,56,63]);
const cells = {};
const grid = document.getElementById('grid');
for (let r=0; r<8; r++) for (let c=0; c<8; c++) {
  const chan = r + 8*c;
  const d = document.createElement('div');
  d.className = 'cell' + (DEAD.has(chan) ? ' dead' : '');
  d.textContent = chan;
  if (!DEAD.has(chan)) d.onclick = () => stim(chan);
  grid.appendChild(d);
  cells[chan] = d;
}

const STOPS = [[0,0,0],[40,11,84],[159,42,99],[212,72,66],[245,125,21],[250,193,39],[252,255,164]];
function heat(t){
  t = Math.max(0, Math.min(1, t));
  const x = t*(STOPS.length-1), i = Math.floor(x), f = x-i;
  const a = STOPS[i], b = STOPS[Math.min(i+1, STOPS.length-1)];
  const m = k => Math.round(a[k]+(b[k]-a[k])*f);
  return `rgb(${m(0)},${m(1)},${m(2)})`;
}
function pulse(chan){
  cells[chan].animate([{transform:'scale(0.82)'},{transform:'scale(1)'}], {duration:160});
}
function stim(chan){ pulse(chan); fetch('/stim', {method:'POST', body: JSON.stringify({channel: chan})}); }

function mark(channels){
  Object.values(cells).forEach(c => c.classList.remove('mark'));
  channels.forEach(ch => { if(cells[ch]){ cells[ch].classList.add('mark'); pulse(ch); } });
  setTimeout(() => channels.forEach(ch => cells[ch] && cells[ch].classList.remove('mark')), 1500);
}
async function scenario(name){
  const r = await fetch('/scenario', {method:'POST', body: JSON.stringify({name})});
  const j = await r.json();
  if (j.channels && j.channels.length) mark(j.channels);
}

const gain = document.getElementById('gain'), gainval = document.getElementById('gainval');
gain.oninput = () => {
  gainval.textContent = parseFloat(gain.value).toFixed(1) + '×';
  fetch('/config', {method:'POST', body: JSON.stringify({coupling_gain: parseFloat(gain.value)})});
};
const noise = document.getElementById('noise'), noiseval = document.getElementById('noiseval');
noise.oninput = () => {
  noiseval.textContent = parseFloat(noise.value).toFixed(2);
  fetch('/config', {method:'POST', body: JSON.stringify({noise: parseFloat(noise.value)})});
};
const plast = document.getElementById('plast'), prate = document.getElementById('prate'),
      prateval = document.getElementById('prateval');
plast.onchange = () => fetch('/config', {method:'POST', body: JSON.stringify({plasticity: plast.checked})});
prate.oninput = () => {
  prateval.textContent = 'extent ' + parseFloat(prate.value).toFixed(1);
  fetch('/config', {method:'POST', body: JSON.stringify({plasticity_rate: parseFloat(prate.value)})});
};

const net = document.getElementById('net'), structure = document.getElementById('structure');
let netEdges = null;
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
    const f = w / mx, op = (0.06 + 0.55 * f).toFixed(2);
    s += `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="rgba(76,201,240,${op})" stroke-width="${(0.5 + 1.8*f).toFixed(2)}"/>`;
  }
  net.innerHTML = s;
}
structure.onchange = async () => {
  if (structure.checked) {
    if (!netEdges) netEdges = await (await fetch('/connectivity')).json();
    drawNet(); net.style.display = 'block';
  } else { net.style.display = 'none'; }
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
        pass

    def _json_body(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        if not length:
            return {}
        try:
            return json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            return {}

    def _send_json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

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
                    self.wfile.write(f"data: {json.dumps(SIM.snapshot())}\n\n".encode())
                    self.wfile.flush()
                    time.sleep(1.0 / STREAM_FPS)
            except (BrokenPipeError, ConnectionResetError):
                pass
        elif self.path == "/connectivity":
            C = SIM.source.connectivity_matrix
            edges, mx = [], 0.0
            for i in range(CHANNEL_COUNT):
                for j in range(i + 1, CHANNEL_COUNT):
                    w = max(float(C[i, j]), float(C[j, i]))
                    if w > 0:
                        edges.append([i, j, round(w, 3)])
                        mx = max(mx, w)
            self._send_json({"edges": edges, "max_w": round(mx, 3)})
        else:
            self.send_error(404)

    def do_POST(self):
        data = self._json_body()
        if self.path == "/stim" and "channel" in data:
            SIM.stimulate(int(data["channel"])); self._ok()
        elif self.path == "/scenario" and "name" in data:
            channels = SIM.run_scenario(str(data["name"]))
            self._send_json({"channels": channels})
        elif self.path == "/config" and "coupling_gain" in data:
            SIM.set_coupling(float(data["coupling_gain"])); self._ok()
        elif self.path == "/config" and "noise" in data:
            SIM.set_noise(float(data["noise"])); self._ok()
        elif self.path == "/config" and "plasticity" in data:
            SIM.set_plasticity(bool(data["plasticity"])); self._ok()
        elif self.path == "/config" and "plasticity_rate" in data:
            SIM.set_plasticity_rate(float(data["plasticity_rate"])); self._ok()
        else:
            self.send_error(404)

    def _ok(self):
        self.send_response(204)
        self.send_header("Content-Length", "0")
        self.end_headers()


def _pids_on_port(port: int) -> list[int]:
    try:
        out = subprocess.run(["lsof", "-nP", f"-iTCP:{port}", "-sTCP:LISTEN", "-t"],
                             capture_output=True, text=True, timeout=5).stdout
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
    pids = [p for p in _pids_on_port(port) if p != os.getpid()]
    if not pids:
        return True
    ok = True
    for pid in pids:
        cmd = _cmdline(pid)
        if "live_server_v2.py" not in cmd:
            print(f"  Port {port} is held by an unrelated process (pid {pid}): {cmd[:80]}")
            ok = False
            continue
        print(f"  Port {port} held by a previous live_server_v2 (pid {pid}) — stopping it.")
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            continue
        for _ in range(20):
            time.sleep(0.1)
            if pid not in _pids_on_port(port):
                break
        else:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            time.sleep(0.3)
    return ok


def _warm_cache() -> None:
    """Warm Brian2's Cython compile cache so the first stim doesn't stall the viewer."""
    from brian2 import ms
    from organoid_simulator.v2.connectivity import build_geometry, build_connectivity
    from organoid_simulator.v2.network import build_network
    g = build_geometry(64, seed=0); c = build_connectivity(g, seed=0)
    build_network(g, c, plastic=True, seed=0).net.run(20 * ms)


def main():
    global SIM
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass
    host = "127.0.0.1"
    port = int(os.getenv("CL_LIVE_PORT", "8009"))
    print("  Warming Brian2 Cython cache (first build compiles; ~10-20s)…")
    _warm_cache()
    SIM = LiveSim()
    threading.Thread(target=SIM.run, daemon=True).start()

    try:
        server = ThreadingHTTPServer((host, port), Handler)
    except OSError as e:
        if e.errno != errno.EADDRINUSE:
            raise
        if not _free_port_if_ours(port):
            print(f"\n  Cannot start: port {port} is in use by another program.")
            print(f"  Use a different port, e.g.  CL_LIVE_PORT=8021 uv run organoid_simulator/v2/live_server_v2.py\n")
            SIM.stop()
            return
        server = ThreadingHTTPServer((host, port), Handler)

    url = f"http://127.0.0.1:{port}/"
    print(f"\n  v2 Brian organoid live visualiser running at:  {url}")
    print("  Click electrodes or use the scenario buttons. Ctrl-C to stop.\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  Shutting down…")
    finally:
        SIM.stop()
        server.shutdown()


if __name__ == "__main__":
    main()
