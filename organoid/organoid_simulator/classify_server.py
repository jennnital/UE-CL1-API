"""
Real-time classification viewer for the LIF organoid simulator.

Watch the full encode -> organoid -> decode loop run live, trial by trial:

  * ENCODER  (models.encoders.SymbolToFixedChannelEncoder): each class is a
    symbol mapped to one fixed electrode. Presenting a class stimulates that
    electrode.
  * ORGANOID (LIFDataSource): the evoked activity spreads across the 8x8 array.
  * DECODER  (models.decoders.LinearArgmaxDecoder): weightlessly pools the 64
    channels into 4 class scores (adaptive-average pooling). Because channel =
    row + 8*col, the 4 pooling bins are the 4 vertical column-bands of the grid
    -- so the decoder is literally reading "which band lit up". argmax = class.

The viewer cycles through the classes automatically (or present them by hand),
showing the input electrode, the live response, the decoder's class scores, the
prediction vs truth, a running accuracy and a confusion matrix.

Turn the coupling slider up and watch accuracy fall as activity bleeds across
band boundaries -- an intuition for why a spatial code needs the response to
stay localised.

Run:  uv run organoid_simulator/classify_server.py   (optional CL_LIVE_PORT=8011)
"""
from __future__ import annotations

import json
import os
import queue
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from cl.sim import DataSourceStim
from utilities import Signal
from models.encoders import SymbolToFixedChannelEncoder
from models.decoders import LinearArgmaxDecoder
from organoid_simulator.lif_data_source import (
    LIFDataSource, NON_SPIKING, CHANNEL_COUNT, grid_connectivity,
)

N_CLASSES        = 4
DEFAULT_COUPLING = 2.5     # localised responses -> good separation. Raise to see it break.
BACKGROUND_STD   = 0.10
NOISE_STD        = 0.18    # display-only shimmer (recording noise floor)
CHUNK_FRAMES     = 40
STEP_SLEEP_S     = 0.02
DECAY            = 0.90

# Phase durations (in sim frames) of one trial.
SETTLE_FRAMES    = 400
READOUT_FRAMES   = 1400
HOLD_FRAMES      = 1200

# One representative electrode per class, each sitting in its own pooling band
# (band = channel // 16 = column-pair). All are live (non grounded/reference).
CLASS_ELECTRODES = [10, 26, 42, 50]   # bands 0, 1, 2, 3


def band_of(channel: int) -> int:
    return channel // (CHANNEL_COUNT // N_CLASSES)


class ClassifierSim:
    """Runs the encode->organoid->decode loop on a background thread."""

    def __init__(self, decoder_channels=[12, 28, 44, 60]):
        # Grid connectivity on purpose: this viewer teaches the *spatial* band
        # code, so activity must stay spatially local. (Small-world shortcuts would
        # scatter activity across bands and destroy the spatial readout -- see the
        # live propagation viewer for the small-world default instead.)
        self.source = LIFDataSource(random_seed=0, coupling_gain_mV=DEFAULT_COUPLING,
                                    background_drive_std_mV=BACKGROUND_STD,
                                    connectivity=grid_connectivity())
        self.source.open()

        # Encoder: symbols are the class ids; each is pinned to one band electrode.
        signal = Signal(pulse_width_us=400, amplitude=2.8, n_pulses=10, burst_rate_hz=200.0)
        self.encoder = SymbolToFixedChannelEncoder(
            max_channels=CHANNEL_COUNT,
            symbols=list(range(N_CLASSES)),
            min_channel_gap=0,
            signal=signal,
            channels_allowed=[int(c) for c in CLASS_ELECTRODES],
        )
        # class id -> electrode, and each class's true label is its electrode's band.
        self.class_channel = {c: int(self.encoder.symbol_to_channel_mapping[c])
                              for c in range(N_CLASSES)}
        self.true_class    = {c: band_of(self.class_channel[c]) for c in range(N_CLASSES)}
        self.signal        = signal

        self.decoder = LinearArgmaxDecoder(d_out=len(decoder_channels), n_classes=N_CLASSES)
        self.decoder_channels = decoder_channels

        self.intensity = np.zeros(CHANNEL_COUNT)
        self.coupling  = DEFAULT_COUPLING
        self.confusion = np.zeros((N_CLASSES, N_CLASSES), dtype=int)
        self.n_trials  = 0
        self.n_correct = 0

        # live trial state (published to the UI)
        self.phase        = "settle"
        self.cur_class    = None
        self.cur_channel  = None
        self.pred_class   = None
        self.scores       = [0.0] * N_CLASSES
        self.result_ready = False

        self.auto = True
        self._lock = threading.Lock()
        self._requests: queue.Queue[int] = queue.Queue()
        self._config: queue.Queue[tuple] = queue.Queue()
        self._noise_rng = np.random.default_rng(1)
        self._dead_idx  = list(NON_SPIKING)
        self._ts = 0
        self._counts = np.zeros(CHANNEL_COUNT)
        self._cycle = 0
        self._running = True

    # -- public API (HTTP threads) ---------------------------------------
    def present(self, class_id: int) -> None:
        self._requests.put(int(class_id))

    def set_coupling(self, gain: float) -> None:
        self._config.put(("coupling", float(gain)))

    def set_auto(self, on: bool) -> None:
        self._config.put(("auto", bool(on)))

    def snapshot(self) -> dict:
        with self._lock:
            disp = self.intensity + np.abs(self._noise_rng.normal(0.0, NOISE_STD, CHANNEL_COUNT))
            disp[self._dead_idx] = 0.0
            return {
                "intensity":     disp.round(3).tolist(),
                "phase":         self.phase,
                "input_class":   self.cur_class,
                "input_channel": self.cur_channel,
                "pred_class":    self.pred_class if self.result_ready else None,
                "true_class":    (self.true_class[self.cur_class] if self.cur_class is not None else None),
                "scores":        [round(s, 3) for s in self.scores] if self.result_ready else None,
                "result_ready":  self.result_ready,
                "accuracy":      (self.n_correct / self.n_trials) if self.n_trials else 0.0,
                "n_trials":      self.n_trials,
                "confusion":     self.confusion.tolist(),
                "coupling":      self.coupling,
                "auto":          self.auto,
                "class_channels": [self.class_channel[c] for c in range(N_CLASSES)],
            }

    # -- stimulation from the encoder's channel + signal -----------------
    def _encoded_burst(self, class_id: int, start_ts: int) -> list[DataSourceStim]:
        channel = self.class_channel[class_id]
        s = self.signal
        period = max(1, round(25_000 / s.burst_rate_hz))       # frames between pulses
        # cathodic-first biphasic (this model excites on the cathodic phase);
        # pulse width & count come from the encoder's Signal.
        return [
            DataSourceStim(
                timestamp          = start_ts + p * period,
                channel            = channel,
                phase_durations_us = (int(s.pulse_width_us), int(s.pulse_width_us)),
                phase_currents_uA  = (-abs(s.amplitude), abs(s.amplitude)),
            )
            for p in range(int(s.n_pulses))
        ]

    def _decode(self) -> None:
        counts = torch.tensor(self._counts[self.decoder_channels], dtype=torch.float32).unsqueeze(0)   # [1, 6]
        scores = self.decoder.forward(counts)                                    # [1, 4]
        probs  = torch.softmax(scores, dim=-1).squeeze(0).tolist()
        pred   = int(scores.argmax(dim=-1))
        truth  = self.true_class[self.cur_class]
        with self._lock:
            self.scores       = probs
            self.pred_class   = pred
            self.result_ready = True
            self.n_trials    += 1
            self.confusion[truth, pred] += 1
            if pred == truth:
                self.n_correct += 1

    def _next_class(self) -> int:
        if not self._requests.empty():
            return self._requests.get()
        c = self._cycle % N_CLASSES
        self._cycle += 1
        return c

    # -- trial state machine ---------------------------------------------
    def run(self) -> None:
        phase_left = 0
        self.phase = "settle"
        while self._running:
            while not self._config.empty():
                kind, val = self._config.get()
                if kind == "coupling":
                    self.source.set_coupling_gain(val)
                    with self._lock:
                        self.coupling = val
                elif kind == "auto":
                    with self._lock:
                        self.auto = val

            # phase transitions
            if phase_left <= 0:
                if self.phase == "settle":
                    if self.auto or not self._requests.empty():
                        c = self._next_class()
                        with self._lock:
                            self.cur_class    = c
                            self.cur_channel  = self.class_channel[c]
                            self.pred_class   = None
                            self.result_ready = False
                        self._counts[:] = 0.0
                        self.source.on_stims(self._encoded_burst(c, self._ts + 1))
                        self.phase, phase_left = "readout", READOUT_FRAMES
                    else:
                        phase_left = CHUNK_FRAMES   # idle until a request arrives
                elif self.phase == "readout":
                    self._decode()
                    self.phase, phase_left = "hold", HOLD_FRAMES
                elif self.phase == "hold":
                    self.phase, phase_left = "settle", SETTLE_FRAMES

            batch = self.source.read(self._ts, CHUNK_FRAMES)
            with self._lock:
                self.intensity *= DECAY
                for sp in batch.spikes:
                    self.intensity[sp.channel] += 1.0
                    if self.phase == "readout":
                        self._counts[sp.channel] += 1.0
                np.clip(self.intensity, 0.0, 8.0, out=self.intensity)
            self._ts   += CHUNK_FRAMES
            phase_left -= CHUNK_FRAMES
            time.sleep(STEP_SLEEP_S)

    def stop(self) -> None:
        self._running = False


SIM: ClassifierSim | None = None


PAGE = """<!DOCTYPE html><html><head><meta charset="utf-8">
<title>LIF Organoid — Classification</title>
<style>
  :root { --bg:#0b0e14; --panel:#141922; --line:#2a3444; --txt:#cdd6e4; --mut:#8895ab; }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--txt);
         font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif; }
  .wrap { max-width:960px; margin:0 auto; padding:22px 16px 48px; }
  h1 { font-size:1.25rem; font-weight:600; margin:0 0 3px; }
  .sub { color:var(--mut); font-size:.88rem; margin:0 0 18px; }
  .cols { display:grid; grid-template-columns:1fr 1fr; gap:18px; align-items:start; }
  @media (max-width:820px){ .cols { grid-template-columns:1fr; } }
  .panel { background:var(--panel); border:1px solid var(--line); border-radius:14px; padding:16px; }
  .bandhdr { display:grid; grid-template-columns:repeat(4,1fr); gap:6px; margin-bottom:8px; }
  .bandlbl { text-align:center; font-size:.72rem; padding:5px 0; border-radius:7px;
             color:#0b0e14; font-weight:600; opacity:.55; transition:opacity .15s; }
  .bandlbl.pred { opacity:1; box-shadow:0 0 0 2px #fff inset; }
  .grid { display:grid; grid-template-columns:repeat(8,1fr); gap:5px; }
  .cell { position:relative; aspect-ratio:1/1; border-radius:8px; background:#000;
          transition:background .05s linear, box-shadow .05s linear;
          display:flex; align-items:center; justify-content:center;
          font-size:.58rem; color:#3a4658; user-select:none; cursor:pointer; }
  .cell.dead { cursor:default; background:#0e1420; }
  .cell.inject { outline:3px solid #fff; z-index:2; }
  .h2 { font-size:.95rem; font-weight:600; margin:0 0 12px; }
  .scorebar { display:flex; align-items:center; gap:9px; margin:7px 0; font-size:.8rem; }
  .scorebar .nm { width:58px; color:var(--mut); }
  .track { flex:1; height:16px; background:#0e1420; border-radius:8px; overflow:hidden; }
  .fill { height:100%; width:0%; border-radius:8px; transition:width .2s; }
  .scorebar.win .nm { color:var(--txt); font-weight:700; }
  .verdict { font-size:1.1rem; font-weight:700; margin:14px 0 4px; }
  .metrics { display:flex; gap:22px; margin-top:6px; }
  .metric .v { font-size:1.6rem; font-weight:700; font-variant-numeric:tabular-nums; }
  .metric .k { font-size:.72rem; color:var(--mut); }
  table.cm { border-collapse:collapse; margin-top:14px; font-size:.72rem; }
  table.cm td, table.cm th { padding:5px 8px; text-align:center; color:var(--mut); }
  table.cm td.c { border-radius:5px; color:#fff; font-variant-numeric:tabular-nums; min-width:30px; }
  .controls { display:flex; align-items:center; gap:12px; margin-top:16px; flex-wrap:wrap; }
  .controls label { font-size:.82rem; color:var(--mut); }
  input[type=range]{ accent-color:#4cc9f0; width:150px; }
  button { background:#1c2534; color:var(--txt); border:1px solid var(--line);
           border-radius:8px; padding:7px 12px; cursor:pointer; font-size:.8rem; }
  button:hover { border-color:#4cc9f0; }
  button.active { background:#4cc9f0; color:#0b0e14; font-weight:600; }
  .status { display:flex; align-items:center; gap:8px; font-size:.8rem; color:var(--mut); margin-top:12px; }
  .dot { width:9px; height:9px; border-radius:50%; background:#e5484d; } .dot.on{ background:#30d158; }
  .phasetag { font-size:.72rem; padding:2px 8px; border-radius:6px; background:#1c2534; color:var(--mut); }
</style></head>
<body><div class="wrap">
  <h1>LIF Organoid — live classification</h1>
  <p class="sub">Fixed-channel encoder → organoid → linear (pooling) decoder. Each class is one electrode in its own column-band; the decoder reads which band responded.</p>
  <div class="cols">
    <div class="panel">
      <div class="bandhdr" id="bandhdr"></div>
      <div class="grid" id="grid"></div>
      <div class="controls">
        <button id="autobtn" class="active">Auto ▶</button>
        <span style="color:var(--mut);font-size:.8rem">present:</span>
        <span id="presentbtns"></span>
      </div>
      <div class="controls">
        <label>coupling</label>
        <input type="range" id="gain" min="0" max="18" step="0.5" value="2.5">
        <span id="gainval" style="font-size:.8rem">2.5 mV</span>
        <span style="color:var(--mut);font-size:.75rem">↑ higher = spreads across bands = worse</span>
      </div>
      <div class="status"><span class="dot" id="dot"></span><span id="stat">connecting…</span>
        <span class="phasetag" id="phase">—</span></div>
    </div>

    <div class="panel">
      <div class="h2">Decoder output (class scores)</div>
      <div id="scores"></div>
      <div class="verdict" id="verdict">—</div>
      <div class="metrics">
        <div class="metric"><div class="v" id="acc">0%</div><div class="k">running accuracy</div></div>
        <div class="metric"><div class="v" id="ntr">0</div><div class="k">trials</div></div>
      </div>
      <div class="h2" style="margin-top:18px">Confusion matrix</div>
      <table class="cm" id="cm"></table>
    </div>
  </div>
</div>
<script>
const NC = 4;
const CLASS_COL = ['#2a9d8f','#e9c46a','#e76f51','#9b5de5'];  // per-class colours
const DEAD = new Set([0,4,7,56,63]);
const cells = {};

// band header (4 labels over the 4 column-pairs)
const bandhdr = document.getElementById('bandhdr');
const bandlbls = [];
for (let b=0;b<NC;b++){ const d=document.createElement('div'); d.className='bandlbl';
  d.textContent='Class '+b; d.style.background=CLASS_COL[b]; bandhdr.appendChild(d); bandlbls.push(d); }

// grid: visual (row,col) -> channel = row + 8*col ; band = floor(col/2)
const grid = document.getElementById('grid');
for (let r=0;r<8;r++) for (let c=0;c<8;c++){
  const chan=r+8*c, band=Math.floor(c/2);
  const d=document.createElement('div');
  d.className='cell'+(DEAD.has(chan)?' dead':'');
  d.textContent=chan;
  d.style.border='1px solid '+hexA(CLASS_COL[band],0.30);
  if(!DEAD.has(chan)) d.onclick=()=>present(classForChannel(chan));
  grid.appendChild(d); cells[chan]=d;
}
function classForChannel(ch){ return Math.floor((ch%64)/16); }  // fallback (band)

// present buttons
const pbtns=document.getElementById('presentbtns');
for(let b=0;b<NC;b++){ const btn=document.createElement('button'); btn.textContent='C'+b;
  btn.style.borderColor=CLASS_COL[b]; btn.onclick=()=>present(b); pbtns.appendChild(btn); }

// score bars
const scoresDiv=document.getElementById('scores'); const fills=[], rows=[];
for(let b=0;b<NC;b++){
  const row=document.createElement('div'); row.className='scorebar';
  row.innerHTML=`<span class="nm">Class ${b}</span><div class="track"><div class="fill"></div></div><span class="pct" style="width:40px;text-align:right;color:var(--mut)">0%</span>`;
  const fill=row.querySelector('.fill'); fill.style.background=CLASS_COL[b];
  scoresDiv.appendChild(row); fills.push(fill); rows.push(row);
}

function hexA(hex,a){ const n=parseInt(hex.slice(1),16); return `rgba(${n>>16&255},${n>>8&255},${n&255},${a})`; }
const STOPS=[[0,0,0],[40,11,84],[159,42,99],[212,72,66],[245,125,21],[250,193,39],[252,255,164]];
function heat(t){ t=Math.max(0,Math.min(1,t)); const x=t*(STOPS.length-1),i=Math.floor(x),f=x-i;
  const a=STOPS[i],b=STOPS[Math.min(i+1,STOPS.length-1)],m=k=>Math.round(a[k]+(b[k]-a[k])*f);
  return `rgb(${m(0)},${m(1)},${m(2)})`; }

function present(c){ fetch('/present',{method:'POST',body:JSON.stringify({class_id:c})}); }

const autobtn=document.getElementById('autobtn');
let auto=true;
autobtn.onclick=()=>{ auto=!auto; autobtn.classList.toggle('active',auto);
  autobtn.textContent=auto?'Auto ▶':'Auto ❚❚';
  fetch('/config',{method:'POST',body:JSON.stringify({auto:auto})}); };

const gain=document.getElementById('gain'), gainval=document.getElementById('gainval');
gain.oninput=()=>{ gainval.textContent=parseFloat(gain.value).toFixed(1)+' mV';
  fetch('/config',{method:'POST',body:JSON.stringify({coupling_gain:parseFloat(gain.value)})}); };

// confusion table
const cm=document.getElementById('cm');
function buildCM(){ let h='<tr><th></th>'; for(let p=0;p<NC;p++) h+=`<th>p${p}</th>`; h+='</tr>';
  for(let t=0;t<NC;t++){ h+=`<tr><th>true ${t}</th>`; for(let p=0;p<NC;p++) h+=`<td class="c" id="cm_${t}_${p}">0</td>`; h+='</tr>'; }
  cm.innerHTML=h; }
buildCM();

const dot=document.getElementById('dot'), stat=document.getElementById('stat'), phaseEl=document.getElementById('phase');
const es=new EventSource('/stream');
es.onopen=()=>{ dot.classList.add('on'); stat.textContent='live'; };
es.onerror=()=>{ dot.classList.remove('on'); stat.textContent='disconnected'; };
es.onmessage=(e)=>{
  const s=JSON.parse(e.data);
  // grid
  for(let ch=0;ch<64;ch++){ if(DEAD.has(ch)) continue;
    const t=Math.min(1,s.intensity[ch]/4);
    cells[ch].style.background=heat(t);
    cells[ch].style.boxShadow=t>0.05?`0 0 ${Math.round(t*16)}px ${heat(t)}`:'none';
    cells[ch].classList.toggle('inject', ch===s.input_channel && (s.phase==='readout'));
  }
  phaseEl.textContent=s.phase+(s.input_class!=null?`  ·  input C${s.input_class}`:'');
  // scores + prediction
  const ready=s.result_ready && s.scores;
  for(let b=0;b<NC;b++){
    const v=ready?s.scores[b]:0;
    fills[b].style.width=(v*100).toFixed(0)+'%';
    rows[b].querySelector('.pct').textContent=ready?(v*100).toFixed(0)+'%':'—';
    rows[b].classList.toggle('win', ready && b===s.pred_class);
    bandlbls[b].classList.toggle('pred', ready && b===s.pred_class);
  }
  // verdict
  const vd=document.getElementById('verdict');
  if(ready){ const ok=s.pred_class===s.true_class;
    vd.textContent = ok ? `✓ predicted C${s.pred_class} (correct)` : `✗ predicted C${s.pred_class}, true C${s.true_class}`;
    vd.style.color = ok ? '#30d158' : '#e5484d';
  } else { vd.textContent='… presenting'; vd.style.color='var(--mut)'; }
  // metrics + confusion
  document.getElementById('acc').textContent=(s.accuracy*100).toFixed(0)+'%';
  document.getElementById('ntr').textContent=s.n_trials;
  let mx=1; for(const row of s.confusion) for(const v of row) mx=Math.max(mx,v);
  for(let t=0;t<NC;t++) for(let p=0;p<NC;p++){ const el=document.getElementById(`cm_${t}_${p}`);
    const v=s.confusion[t][p]; el.textContent=v;
    el.style.background = v? hexA(t===p?'#30d158':'#e5484d', Math.min(0.85,0.15+0.7*v/mx)) : '#0e1420'; }
};
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _body(self) -> dict:
        n = int(self.headers.get("Content-Length", 0))
        if not n:
            return {}
        try:
            return json.loads(self.rfile.read(n) or b"{}")
        except json.JSONDecodeError:
            return {}

    def do_GET(self):
        if self.path in ("/", "/index.html"):
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
            self.end_headers()
            try:
                while True:
                    self.wfile.write(f"data: {json.dumps(SIM.snapshot())}\n\n".encode())
                    self.wfile.flush()
                    time.sleep(1.0 / 25)
            except (BrokenPipeError, ConnectionResetError):
                pass
        else:
            self.send_error(404)

    def do_POST(self):
        data = self._body()
        if self.path == "/present" and "class_id" in data:
            SIM.present(int(data["class_id"]))
        elif self.path == "/config" and "coupling_gain" in data:
            SIM.set_coupling(float(data["coupling_gain"]))
        elif self.path == "/config" and "auto" in data:
            SIM.set_auto(bool(data["auto"]))
        else:
            return self.send_error(404)
        self.send_response(204)
        self.send_header("Content-Length", "0")
        self.end_headers()


def main():
    global SIM
    port = int(os.getenv("CL_LIVE_PORT", "8011"))
    SIM = ClassifierSim()
    threading.Thread(target=SIM.run, daemon=True).start()
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"\n  LIF classification viewer at:  http://127.0.0.1:{port}/")
    print(f"  Encoder mapping (class -> electrode -> band): "
          f"{ {c: (SIM.class_channel[c], SIM.true_class[c]) for c in range(N_CLASSES)} }")
    print("  Ctrl-C to stop.\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  Shutting down…")
    finally:
        SIM.stop()
        server.shutdown()


if __name__ == "__main__":
    main()
