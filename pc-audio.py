# pc-audio.py — Captura, processa e envia bandas PRONTAS (UDP) para o Raspberry Pi.
# Protocolos: A2 (áudio pronto) + B0 (config de parâmetros comuns)
# • Mantém tentativa de time-sync contínua (não encerra em timeout)
# • Envia config (burst inicial + periódico) e começa a transmitir assim que houver sync
# • Parâmetros “idênticos” centralizados aqui: --bands, --fps, --signal-hold, --vis-fps

import socket
import time
import numpy as np
import sounddevice as sd
from collections import deque
import sys
import platform
import threading
import argparse
from typing import Optional, Union

# --- Web UI imports ---
from flask import Flask, Response, request, jsonify
from flask import stream_with_context
import json

# ------------------------------- Constantes -----------------------------------
DEFAULT_RASPBERRY_IP = "192.168.66.71"
UDP_PORT = 5005
TCP_TIME_PORT = 5006

PKT_AUDIO_V2 = 0xA2  # [A2][8 ts_pc][bands(150)][beat][trans][dyn_floor][kick] => 163 bytes p/ 150 bandas
PKT_CFG = 0xB0       # [B0][ver u8][num_bands u16][fps u16][signal_hold_ms u16][vis_fps u16] => 10 bytes

# --------------------------- Utils de dispositivo -----------------------------
def resolve_device_index(name_or_index: Optional[Union[str, int]]) -> Optional[int]:
    if name_or_index is None:
        return None
    devs = sd.query_devices()
    try:
        idx = int(name_or_index)
        if 0 <= idx < len(devs):
            return idx
    except (TypeError, ValueError):
        pass
    needle = str(name_or_index).lower()
    for i, d in enumerate(devs):
        if needle in d["name"].lower():
            return i
    return None

def auto_candidate_outputs():
    devs = sd.query_devices()
    cands = []
    default_out = sd.default.device[1]
    if isinstance(default_out, int) and default_out >= 0:
        cands.append(default_out)
    keywords = ["speaker","alto-falante","headphone","fone","realtek","echo","dot","hdmi","display audio"]
    for i, d in enumerate(devs):
        if d.get("max_output_channels", 0) > 0 and any(k in d["name"].lower() for k in keywords):
            if i not in cands:
                cands.append(i)
    for i, d in enumerate(devs):
        if d.get("max_output_channels", 0) > 0 and i not in cands:
            cands.append(i)
    return cands

def choose_candidates(override: Optional[Union[str, int]] = None):
    devs = sd.query_devices()
    system = platform.system().lower()
    idx = resolve_device_index(override)
    if idx is not None:
        d = devs[idx]
        extra = None
        if system.startswith("win") and d.get("max_output_channels", 0) > 0:
            try:
                extra = sd.WasapiSettings(loopback=True)
            except Exception:
                extra = None
        return [(idx, extra, f"Override: idx={idx} '{d['name']}' (loopback={extra is not None})")]
    if system.startswith("win"):
        cands = []
        for i in auto_candidate_outputs():
            d = devs[i]
            try:
                extra = sd.WasapiSettings(loopback=True)
            except Exception:
                extra = None
            cands.append((i, extra, f"WASAPI loopback: '{d['name']}'"))
        if cands:
            return cands
    default_in = sd.default.device[0]
    if isinstance(default_in, int) and default_in >= 0:
        d = devs[default_in]
        return [(default_in, None, f"Default INPUT: idx={default_in} '{d['name']}'")]
    return [(None, None, "PortAudio default (fallback)")]

def open_with_probe(dev_idx, extra, callback, block_size):
    attempts = [(48000,2),(48000,1),(44100,2),(44100,1)]
    last_err = None
    for sr_try, ch_try in attempts:
        try:
            s = sd.InputStream(channels=ch_try, samplerate=sr_try, dtype="float32",
                               blocksize=block_size, device=dev_idx,
                               callback=callback, extra_settings=extra)
            return s, int(s.samplerate), ch_try, f"(opened sr={int(s.samplerate)} ch={ch_try})"
        except Exception as e:
            last_err = e
    try:
        s = sd.InputStream(channels=2, samplerate=44100, dtype="float32",
                           blocksize=block_size, device=None, callback=callback)
        return s, int(s.samplerate), 2, "(fallback default INPUT sr=44100 ch=2)"
    except Exception as e:
        last_err = e
    raise last_err if last_err else RuntimeError("Falha ao abrir InputStream.")

# ------------------------------ FFT / Bandas ----------------------------------
def make_bands_indices(nfft, sr, num_bands, fmin, fmax_limit):
    freqs = np.fft.rfftfreq(nfft, 1.0/sr)
    fmax = min(fmax_limit, sr/2.0)
    edges = np.geomspace(fmin, fmax, num_bands+1)
    edge_idx = np.searchsorted(freqs, edges, side="left").astype(np.int32)
    a_idx = edge_idx[:-1]
    b_idx = edge_idx[1:]
    b_idx = np.maximum(b_idx, a_idx + 1)
    return a_idx, b_idx

def make_compute_bands(sr, block_size, band_starts, band_ends, raw_ema_alpha, peak_ema_alpha):
    window = np.hanning(block_size).astype(np.float32)
    ema_bands = np.zeros(len(band_starts), dtype=np.float32)
    peak_ema = 1.0
    def compute(block):
        nonlocal ema_bands, peak_ema
        x = block * window
        fft_mag = np.abs(np.fft.rfft(x, n=block_size)).astype(np.float32)
        cs = np.concatenate(([0.0], np.cumsum(fft_mag, dtype=np.float32)))
        sums = cs[band_ends] - cs[band_starts]
        lens = (band_ends - band_starts).astype(np.float32)
        means = sums / np.maximum(lens, 1.0)
        vals = np.log1p(means)
        cur_max = float(np.max(vals)) if vals.size else 1.0
        peak_ema = (1.0 - peak_ema_alpha) * peak_ema + peak_ema_alpha * max(cur_max, 1e-6)
        norm = max(peak_ema, 1e-6)
        vals = (vals / norm).clip(0.0, 1.0)
        ema_bands = raw_ema_alpha * vals + (1.0 - raw_ema_alpha) * ema_bands
        return ema_bands  # 0..1
    return compute

# -------------------------------- Contextos -----------------------------------
class Shared:
    def __init__(self, n_bands: int):
        self.bands_eq_u8 = np.zeros(n_bands, dtype=np.uint8)
        self.beat = 0
        self.kick_intensity = 0
        self.avg = 0.0
        self.rms = 0.0
        self.avg_ema = 0.0
        self.rms_ema = 0.0
        self.block_max = 0.0
        self.last_update = 0.0
        # sinais extras p/ UI
        self.gate_active = False
        self.is_quiet = False
        self.resume_threshold = 0.0

class Ctx:
    def __init__(self, rpi_ip: str, max_fps: int):
        self.rpi_ip = rpi_ip
        self.max_fps = max_fps
        self.min_send_interval = 1.0 / max(1, max_fps)
        self.tx_count = 0
        self.last_status = 0.0
        self.time_sync_ok = False
        self.stop_flag = threading.Event()
        self.udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            self.udp_sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1 << 20)
        except Exception:
            pass

# ------------------------------- Time Sync ------------------------------------
def _recv_exact(sock, n):
    buf = b''
    while len(buf) < n:
        ch = sock.recv(n - len(buf))
        if not ch:
            raise ConnectionError("Conexão fechada durante receive.")
        buf += ch
    return buf

def time_sync_over_tcp(ctx, samples=12, timeout=0.6):
    """Faz 1 rodada de sync e retorna True/False (NÃO encerra o programa)."""
    results = []
    try:
        with socket.create_connection((ctx.rpi_ip, TCP_TIME_PORT), timeout=1.8) as s:
            s.settimeout(timeout)
            for _ in range(samples):
                t0 = time.monotonic_ns()
                s.sendall(b"TS1" + t0.to_bytes(8, 'little'))
                data = _recv_exact(s, 3 + 8 + 8)
                if data[:3] != b"TS2":
                    continue
                echo_t0 = int.from_bytes(data[3:11], 'little')
                tr_pi = int.from_bytes(data[11:19], 'little')
                if echo_t0 != t0:
                    continue
                t1 = time.monotonic_ns()
                rtt = t1 - t0
                mid = t0 + rtt // 2
                offset = tr_pi - mid
                results.append((rtt, offset))
            if not results:
                ctx.time_sync_ok = False
                print("\n[WARN] Time sync TCP falhou (sem amostras válidas).")
                return False
            rtt_min, offset_best = sorted(results, key=lambda x: x[0])[0]
            s.sendall(b"TS3" + int(offset_best).to_bytes(8, 'little', signed=True))
            ack = _recv_exact(s, 3 + 8)
            ok = (ack[:3] == b"TS3" and int.from_bytes(ack[3:11], 'little', signed=True) == int(offset_best))
            ctx.time_sync_ok = bool(ok)
            print(f"\n[INFO] Time sync TCP: RTT_min={rtt_min/1e6:.2f} ms, offset={offset_best/1e6:.3f} ms, ack={'OK' if ok else 'NOK'}")
            return ctx.time_sync_ok
    except Exception as e:
        ctx.time_sync_ok = False
        print(f"\n[WARN] Time sync TCP erro: {e}")
        return False

def resync_loop(ctx, interval_ok=60.0, interval_fail=2.0):
    """Loop que fica tentando sync para sempre (não bloqueia o main)."""
    while not ctx.stop_flag.is_set():
        ok = time_sync_over_tcp(ctx)
        wait = interval_ok if ok else interval_fail
        for _ in range(int(wait*10)):
            if ctx.stop_flag.is_set():
                return
            time.sleep(0.1)

# --------------------------------- Status -------------------------------------
def print_status(ctx, tag_extra: str = "", debug_line: str = ""):
    now = time.time()
    if (now - ctx.last_status) > 0.25:
        tag = "SYNC✓" if ctx.time_sync_ok else "WAITING SYNC"
        extra = f" {tag_extra}" if tag_extra else ""
        dbg = f" {debug_line}" if debug_line else ""
        sys.stdout.write(f"\rTX: {ctx.tx_count} {tag}{extra}{dbg}")
        sys.stdout.flush()
        ctx.last_status = now

# ----------------------------------- Main -------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--ip", type=str, default=DEFAULT_RASPBERRY_IP)
    parser.add_argument("--block-size", type=int, default=512)
    parser.add_argument("--fps", type=int, default=75)           # cadência de envio
    parser.add_argument("--bands", type=int, default=150)         # número de bandas
    parser.add_argument("--signal-hold", type=int, default=500)   # sustain no Pi (ms)
    parser.add_argument("--vis-fps", type=int, default=45)        # FPS de render do Pi
    # Equalização/normalização
    parser.add_argument("--eq-target", type=float, default=64.0)
    parser.add_argument("--eq-alpha", type=float, default=0.35)
    parser.add_argument("--tilt-min", type=float, default=0.9)
    parser.add_argument("--tilt-max", type=float, default=1.8)
    parser.add_argument("--norm-peak-ema", type=float, default=0.40)
    parser.add_argument("--raw-ema", type=float, default=1.0)
    # Pós-EQ
    parser.add_argument("--post-attack", type=float, default=1.0)
    parser.add_argument("--post-release", type=float, default=1.0)
    # Gate
    parser.add_argument("--silence-bands", type=float, default=3.0)
    parser.add_argument("--silence-rms", type=float, default=1e-5)
    parser.add_argument("--silence-duration", type=float, default=0.8)
    parser.add_argument("--resume-factor", type=float, default=2.0)
    parser.add_argument("--resume-stable", type=float, default=0.4)
    # Resync
    parser.add_argument("--resync", type=float, default=60.0)
    # Debug
    parser.add_argument("--debug", action="store_true")
    # Web UI
    parser.add_argument("--web-ui", action="store_true", help="Habilita dashboard web local")
    parser.add_argument("--web-port", type=int, default=8787, help="Porta do dashboard web")
    parser.add_argument("--web-fps", type=float, default=12.0, help="Taxa de atualização da UI (frames/s)")

    args = parser.parse_args()

    NUM_BANDS = int(args.bands)
    MAX_FPS = int(args.fps)
    MIN_SEND_INTERVAL = 1.0 / max(1, MAX_FPS)
    SIGNAL_HOLD_MS = int(args.signal_hold)
    VIS_FPS = int(args.vis_fps)

    # Contexto PC (estado)
    ctx = Ctx(args.ip, MAX_FPS)
    shared = Shared(NUM_BANDS)

    print(f"[CFG] bands={NUM_BANDS} fps={MAX_FPS} hold={SIGNAL_HOLD_MS}ms vis_fps={VIS_FPS} -> PC pipeline")
    print("[INFO] Iniciando time-sync TCP (porta 5006)...")

    # Thread de time-sync contínuo (não bloqueia o main)
    threading.Thread(target=resync_loop, args=(ctx, float(args.resync), 2.0), daemon=True).start()

    # ----------------------------- Envio de CONFIG (B0) ------------------------
    def send_config_packet():
        ver = 1
        nb = NUM_BANDS & 0xFFFF
        fps = MAX_FPS & 0xFFFF
        hold = SIGNAL_HOLD_MS & 0xFFFF
        vis = VIS_FPS & 0xFFFF
        payload = bytes([
            PKT_CFG, ver,
            nb & 0xFF, (nb >> 8) & 0xFF,
            fps & 0xFF, (fps >> 8) & 0xFF,
            hold & 0xFF, (hold >> 8) & 0xFF,
            vis & 0xFF, (vis >> 8) & 0xFF,
        ])
        ctx.udp_sock.sendto(payload, (ctx.rpi_ip, UDP_PORT))

    def cfg_sender():
        # Burst inicial: 5 pacotes em ~1s (cobre casos em que o Pi sobe depois)
        for _ in range(5):
            try: send_config_packet()
            except Exception: pass
            time.sleep(0.2)
        # Periódico: a cada 2 s
        while not ctx.stop_flag.is_set():
            try: send_config_packet()
            except Exception: pass
            time.sleep(2.0)

    threading.Thread(target=cfg_sender, daemon=True).start()

    # --------------------- Config dinâmica (runtime) + Web ---------------------
    CONFIG_LOCK = threading.Lock()
    CFG = {
        # Gate (defaults vindos dos args)
        "silence_bands": float(args.silence_bands),
        "silence_rms": float(args.silence_rms),
        "silence_duration": float(args.silence_duration),
        "resume_factor": float(args.resume_factor),
        "resume_stable": float(args.resume_stable),
        # Pós-EQ / smoothing (opcionais p/ ajuste fino visível)
        "eq_target": float(args.eq_target),
        "eq_alpha": float(args.eq_alpha),
        "post_attack": float(args.post_attack),
        "post_release": float(args.post_release),
        "tilt_min": float(args.tilt_min),
        "tilt_max": float(args.tilt_max),
    }

    def get_cfg():
        with CONFIG_LOCK:
            return dict(CFG)

    def update_cfg(patch: dict):
        changed = {}
        with CONFIG_LOCK:
            for k, v in patch.items():
                if k in CFG:
                    try:
                        v = float(v)
                    except Exception:
                        pass
                    CFG[k] = v
                    changed[k] = v
        return changed

    # ----------------------------- Web UI (Flask/SSE) -------------------------
    app = Flask(__name__)

    HTML_PAGE = """
<!doctype html><html lang="pt-br"><head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>PC-Audio Dashboard</title>
https://cdn.jsdelivr.net/npm/chart.js</script>
<style>
 body{font-family:system-ui,Segoe UI,Roboto,Arial,sans-serif;margin:16px;background:#0e1117;color:#e6edf3}
 .row{display:flex;gap:16px;flex-wrap:wrap}
 .card{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:12px;flex:1;min-width:320px}
 .bars{height:180px}
 .ok{color:#3fb950}.warn{color:#d29922}.bad{color:#f85149}
 label{display:block;margin:6px 0 2px}
 input[type=number]{width:140px}
 button{padding:6px 10px;border-radius:6px;border:1px solid #30363d;background:#21262d;color:#e6edf3;cursor:pointer}
 button:hover{background:#30363d}
 .grid{display:grid;grid-template-columns:repeat(2,minmax(180px,1fr));gap:8px}
 small{color:#8b949e}
</style>
</head><body>
<h2>PC-Audio Dashboard</h2>
<div class="row">
  <div class="card" style="flex:2">
    <canvas id="bars" class="bars"></canvas>
    <small>Barras (EQ normalizada 0..255)</small>
  </div>
  <div class="card" style="min-width:320px">
    <div><b>Status:</b> <span id="gate">–</span> • <span id="beat">Beat: 0</span> • <span id="kick">Kick: 0</span></div>
    <div style="margin-top:8px" class="grid">
      <div><b>avg_ema</b>: <span id="avg"></span></div>
      <div><b>rms_ema</b>: <span id="rms"></span></div>
      <div><b>silence_bands</b>: <span id="sb"></span></div>
      <div><b>silence_rms</b>: <span id="sr"></span></div>
    </div>
  </div>
</div>
<div class="row">
  <div class="card">
    <canvas id="series" style="height:200px"></canvas>
    <small>AVG/RMS vs thresholds</small>
  </div>
  <div class="card">
    <h3>Ajustes do Gate</h3>
    <div class="grid">
      <label>silence_bands <input id="i_sb" type="number" step="0.1"></label>
      <label>silence_rms <input id="i_sr" type="number" step="1e-6"></label>
      <label>silence_duration (s) <input id="i_sd" type="number" step="0.1"></label>
      <label>resume_factor <input id="i_rf" type="number" step="0.1"></label>
      <label>resume_stable (s) <input id="i_rs" type="number" step="0.1"></label>
    </div>
    <div style="margin-top:8px"><button id="apply">Aplicar</button> <button id="calib">Calibrar silêncio (2s)</button></div>
    <small id="msg"></small>
  </div>
</div>
<script>
let numBands = 150;
const barsCtx = document.getElementById('bars').getContext('2d');
const seriesCtx = document.getElementById('series').getContext('2d');

function makeLabels(n){ return [...Array(n).keys()].map(i=>'B'+(i+1)); }

const barsChart = new Chart(barsCtx, {
  type:'bar',
  data:{labels:makeLabels(numBands),datasets:[{label:'Bands',data:Array(numBands).fill(0),backgroundColor:'#58a6ff',borderWidth:0}]},
  options:{responsive:true,plugins:{legend:{display:false}},scales:{x:{display:false},y:{min:0,max:255}}}
});

const seriesChart = new Chart(seriesCtx,{
  type:'line',
  data:{labels:[],datasets:[
    {label:'avg_ema', data:[], borderColor:'#58a6ff', tension:.2},
    {label:'rms_ema', data:[], borderColor:'#a371f7', tension:.2, yAxisID:'y2'},
    {label:'silence_bands', data:[], borderColor:'#f85149', borderDash:[6,4], tension:0},
    {label:'silence_rms', data:[], borderColor:'#d29922', borderDash:[6,4], tension:0, yAxisID:'y2'}]},
  options:{animation:false, scales:{y:{beginAtZero:true}, y2:{beginAtZero:true, position:'right'}}}
});

const gateEl=document.getElementById('gate'),beatEl=document.getElementById('beat'),kickEl=document.getElementById('kick');
const avgEl=document.getElementById('avg'),rmsEl=document.getElementById('rms'),sbEl=document.getElementById('sb'),srEl=document.getElementById('sr');
const i_sb=document.getElementById('i_sb'),i_sr=document.getElementById('i_sr'),i_sd=document.getElementById('i_sd'),i_rf=document.getElementById('i_rf'),i_rs=document.getElementById('i_rs');
const msg=document.getElementById('msg');

function applyCfg(){
  const body={
    silence_bands: parseFloat(i_sb.value),
    silence_rms: parseFloat(i_sr.value),
    silence_duration: parseFloat(i_sd.value),
    resume_factor: parseFloat(i_rf.value),
    resume_stable: parseFloat(i_rs.value)
  };
  fetch('/api/config',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)})
    .then(r=>r.json()).then(j=>{msg.textContent='OK: '+JSON.stringify(j.changed)}).catch(e=>{msg.textContent='Erro: '+e});
}
document.getElementById('apply').onclick=applyCfg;
document.getElementById('calib').onclick=()=>fetch('/api/calibrate',{method:'POST'}).then(r=>r.json()).then(j=>{
  i_sb.value=j.suggest.silence_bands; i_sr.value=j.suggest.silence_rms; msg.textContent='Calibrado: '+JSON.stringify(j.suggest);
});

function setInputs(cfg){ i_sb.value=cfg.silence_bands; i_sr.value=cfg.silence_rms; i_sd.value=cfg.silence_duration; i_rf.value=cfg.resume_factor; i_rs.value=cfg.resume_stable; }

const evt=new EventSource('/api/stream');
evt.onmessage=(ev)=>{
  const s=JSON.parse(ev.data);
  // redimensiona se numBands mudar
  if (s.bands && s.bands.length && s.bands.length !== numBands){
    numBands = s.bands.length;
    barsChart.data.labels = makeLabels(numBands);
    barsChart.data.datasets[0].data = Array(numBands).fill(0);
  }
  // bars
  barsChart.data.datasets[0].data = s.bands; barsChart.update('none');
  // status
  gateEl.innerHTML = s.gate_active ? '<span class="ok">ACTIVE</span>' : (s.is_quiet ? '<span class="warn">IDLE (quiet)</span>' : '<span class="bad">IDLE</span>');
  beatEl.textContent='Beat: '+s.beat; kickEl.textContent='Kick: '+s.kick;
  avgEl.textContent=s.avg_ema.toFixed(2); rmsEl.textContent=s.rms_ema.toExponential(2);
  sbEl.textContent=s.cfg.silence_bands.toFixed(2); srEl.textContent=s.cfg.silence_rms.toExponential(2);

  // series (janela ~15s)
  const label = (new Date()).toLocaleTimeString();
  const keep=180;
  const dsAvg=seriesChart.data.datasets[0], dsRms=seriesChart.data.datasets[1];
  const dsSB =seriesChart.data.datasets[2], dsSR =seriesChart.data.datasets[3];
  dsAvg.data.push(s.avg_ema); dsRms.data.push(s.rms_ema);
  dsSB.data.push(s.cfg.silence_bands); dsSR.data.push(s.cfg.silence_rms);
  while(dsAvg.data.length>keep){ dsAvg.data.shift(); dsRms.data.shift(); dsSB.data.shift(); dsSR.data.shift();}
  seriesChart.data.labels.push(label); while(seriesChart.data.labels.length>keep) seriesChart.data.labels.shift();
  seriesChart.update('none');

  // init inputs na 1ª atualização
  if(!i_sb.value) setInputs(s.cfg);
};
</script>
</body></html>
"""

    @app.get("/")
    def index():
        return HTML_PAGE

    @app.get("/api/state")
    def api_state():
        snap = {
            "tx": ctx.tx_count,
            "gate_active": bool(getattr(shared, "gate_active", False)),
            "is_quiet": bool(getattr(shared, "is_quiet", False)),
            "avg": float(shared.avg), "rms": float(shared.rms),
            "avg_ema": float(shared.avg_ema), "rms_ema": float(shared.rms_ema),
            "beat": int(shared.beat), "kick": int(shared.kick_intensity),
            "bands": shared.bands_eq_u8.tolist(),
            "cfg": get_cfg(),
            "ts": time.time(),
        }
        return jsonify(snap)

    @app.post("/api/config")
    def api_config():
        patch = request.get_json(force=True, silent=True) or {}
        changed = update_cfg(patch)
        return jsonify({"changed": changed})

    @app.post("/api/calibrate")
    def api_calibrate():
        # mede 2s de ruído atual e sugere thresholds com margens
        samples = []
        t0 = time.time()
        while time.time() - t0 < 2.0:
            samples.append((shared.avg_ema, shared.rms_ema))
            time.sleep(0.05)
        if samples:
            avgs = [a for a, _ in samples]; rmss = [r for _, r in samples]
            p95_avg = float(np.percentile(avgs, 95))
            p95_rms = float(np.percentile(rmss, 95))
            suggest = {
                "silence_bands": round(p95_avg * 1.05, 2),
                "silence_rms": float(p95_rms * 1.10)
            }
            update_cfg(suggest)
        else:
            suggest = get_cfg()
        return jsonify({"suggest": suggest})

    @app.get("/api/stream")
    def api_stream():
        interval = max(0.02, 1.0 / float(args.web_fps))
        @stream_with_context
        def gen():
            while not ctx.stop_flag.is_set():
                snap = {
                    "gate_active": bool(getattr(shared, "gate_active", False)),
                    "is_quiet": bool(getattr(shared, "is_quiet", False)),
                    "avg_ema": float(shared.avg_ema),
                    "rms_ema": float(shared.rms_ema),
                    "beat": int(shared.beat),
                    "kick": int(shared.kick_intensity),
                    "bands": shared.bands_eq_u8.tolist(),
                    "cfg": get_cfg()
                }
                yield f"data: {json.dumps(snap)}\n\n"
                time.sleep(interval)
        return Response(gen(), mimetype="text/event-stream")

    def _run_web():
        # host 127.0.0.1 por padrão; mude para '0.0.0.0' se quiser abrir na LAN
        app.run(host="127.0.0.1", port=int(args.web_port), debug=False, threaded=True)

    if args.web_ui:
        threading.Thread(target=_run_web, daemon=True).start()
        print(f"\n[WEB] Dashboard em http://127.0.0.1:{int(args.web_port)}")

    # --------------------------- Pipeline de Áudio ------------------------------
    FMIN = 20.0; FMAX = 16000.0
    compute_bands = None
    raw_ema_alpha = float(args.raw_ema)
    peak_ema_alpha = float(args.norm_peak_ema)

    # Equalização (estado base)
    band_ema = np.ones(NUM_BANDS, dtype=np.float32) * (args.eq_target / 2.0)
    post_eq_prev = np.zeros(NUM_BANDS, dtype=np.float32)

    # Beat buffer
    ENERGY_BUFFER_SIZE = 10
    energy_buf = np.zeros(ENERGY_BUFFER_SIZE, dtype=np.float32)
    energy_idx = 0
    energy_count = 0
    last_energy = 0.0

    # Gate
    active = False
    silence_since = None
    resume_since = None
    last_tick = 0.0

    # Funções de envio de áudio
    def send_audio_packet(bands_u8: np.ndarray, beat_flag: int, transition_flag: int, dyn_floor: int, kick_intensity: int):
        ts_ns = time.monotonic_ns()
        payload = (
            bytes([PKT_AUDIO_V2]) +
            ts_ns.to_bytes(8, 'little') +
            bands_u8.tobytes() +
            bytes([beat_flag & 0xFF, transition_flag & 0xFF, dyn_floor & 0xFF, kick_intensity & 0xFF])
        )
        ctx.udp_sock.sendto(payload, (ctx.rpi_ip, UDP_PORT))
        ctx.tx_count += 1

    # Callback de áudio
    def audio_cb(indata, frames, time_info, status):
        nonlocal energy_idx, energy_count, last_energy, band_ema, post_eq_prev
        if status:
            sys.stdout.write("\n"); print(f"[WARN] Audio status: {status}")
        block = indata.mean(axis=1) if indata.ndim > 1 else indata
        if len(block) < args.block_size:
            block = np.pad(block, (0, args.block_size - len(block)), "constant")
        if compute_bands is None:
            return
        base_vals = compute_bands(block[:args.block_size])  # 0..1
        base_vals255 = (base_vals * 255.0).astype(np.float32)

        # snapshot dos parâmetros que afetam o render
        _cfg = get_cfg()
        eq_alpha = float(_cfg["eq_alpha"])
        # tilt pode mudar em runtime → recalcule curva
        tilt_curve_runtime = np.exp(
            np.linspace(np.log(_cfg["tilt_max"]), np.log(_cfg["tilt_min"]), NUM_BANDS)
        ).astype(np.float32)

        band_ema = (1.0 - eq_alpha) * band_ema + eq_alpha * base_vals255
        gain = _cfg["eq_target"] / np.maximum(band_ema, 1.0)
        eq = base_vals255 * gain * tilt_curve_runtime
        eq = np.clip(eq, 0.0, 255.0).astype(np.float32)

        if _cfg["post_attack"] < 1.0 or _cfg["post_release"] < 1.0:
            alpha = np.where(eq > post_eq_prev, _cfg["post_attack"], _cfg["post_release"]).astype(np.float32)
            eq = alpha * eq + (1.0 - alpha) * post_eq_prev
        post_eq_prev = eq

        bands_u8 = eq.clip(0, 255).astype(np.uint8)
        shared.bands_eq_u8 = bands_u8

        # AVG e RMS para o gate
        avg = float(np.mean(bands_u8))
        rms = float(np.sqrt(np.mean(block * block) + 1e-12))
        shared.avg_ema = shared.avg_ema * 0.85 + 0.15 * avg if shared.last_update > 0 else avg
        shared.rms_ema = shared.rms_ema * 0.85 + 0.15 * rms if shared.last_update > 0 else rms
        shared.avg = avg; shared.rms = rms; shared.last_update = time.time()

        # Beat/kick
        low_bands = bands_u8[:max(8, NUM_BANDS // 12)]
        energy = float(np.mean(low_bands)) / 255.0
        energy_buf[energy_idx] = energy
        energy_idx = (energy_idx + 1) % ENERGY_BUFFER_SIZE
        energy_count = min(energy_count + 1, ENERGY_BUFFER_SIZE)
        buf_view = energy_buf if energy_count == ENERGY_BUFFER_SIZE else energy_buf[:energy_count]
        avg_energy = float(np.mean(buf_view)) if energy_count > 0 else 0.0
        std_energy = float(np.std(buf_view)) if energy_count > 1 else 0.0
        dyn_thr = avg_energy + 1.20 * std_energy
        shared.beat = 1 if ((energy >= max(dyn_thr, 0.08)) and (energy >= last_energy)) else 0
        last_energy = energy

        if not hasattr(audio_cb, "_kick_ema"):
            audio_cb._kick_ema = 0.0
        audio_cb._kick_ema = 0.60 * audio_cb._kick_ema + 0.40 * (np.mean(low_bands) / 255.0)
        onset = max(0.0, (np.mean(low_bands) / 255.0) - audio_cb._kick_ema)
        ki = int(np.clip(onset * 255.0 * 2.2, 0, 255))
        if shared.beat == 1:
            ki = min(255, ki + 60)
        shared.kick_intensity = ki

    # Abrir stream
    cands = choose_candidates(override=args.device)
    stream = None; sr_eff = 44100
    for dev_idx, extra, desc in cands:
        try:
            stream, sr_eff, ch_eff, open_desc = open_with_probe(dev_idx, extra, audio_cb, args.block_size)
            sys.stdout.write(f"\n[INFO] Capturando de: {desc} {open_desc}\n")
            break
        except Exception as e:
            sys.stdout.write(f"\n[WARN] Falha ao abrir {desc}: {e}\n")
            continue
    if stream is None:
        sys.stdout.write("\n[FATAL] Não foi possível abrir nenhum dispositivo de áudio.\n")
        sys.exit(1)

    a_idx, b_idx = make_bands_indices(args.block_size, sr_eff, NUM_BANDS, FMIN, FMAX)
    compute_bands = make_compute_bands(sr_eff, args.block_size, a_idx, b_idx, raw_ema_alpha, peak_ema_alpha)

    # Inicia web se habilitado (avisar após stream pronto também)
    if args.web_ui:
        print(f"[WEB] Dashboard em http://127.0.0.1:{int(args.web_port)}")

    stream.start()

    try:
        while True:
            # Mesmo sem sync, o programa segue rodando — apenas não transmite A2
            if not ctx.time_sync_ok:
                print_status(ctx, " (paused)")
                time.sleep(0.05)
                continue

            bands_now = shared.bands_eq_u8
            avg = shared.avg; rms = shared.rms
            avg_ema = shared.avg_ema; rms_ema = shared.rms_ema
            beat = shared.beat; kick_intensity = shared.kick_intensity
            now = time.time()

            # Gate de silêncio (duplo critério)
            _cfg = get_cfg()
            is_quiet = (avg_ema < _cfg["silence_bands"]) and (rms_ema < _cfg["silence_rms"])
            resume_threshold = _cfg["silence_bands"] * _cfg["resume_factor"]
            shared.is_quiet = bool(is_quiet)
            shared.resume_threshold = float(resume_threshold)

            if active:
                if is_quiet:
                    if silence_since is None:
                        silence_since = now
                    elif (now - silence_since) >= _cfg["silence_duration"]:
                        # transição OFF
                        send_audio_packet(np.zeros(NUM_BANDS, dtype=np.uint8), 0, 1, 0, 0)
                        active = False; shared.gate_active = False
                        silence_since = None; resume_since = None
                        time.sleep(0.01)
                        continue
                else:
                    silence_since = None
            else:
                # Inativo -> verificar retomada
                if (avg_ema > resume_threshold) or (rms_ema > _cfg["silence_rms"] * 3.0):
                    if resume_since is None:
                        resume_since = now
                    elif (now - resume_since) >= _cfg["resume_stable"]:
                        mean_val = float(np.mean(bands_now))
                        dyn_floor = int(min(12, mean_val * 0.012)) if mean_val > 90 else 0
                        send_audio_packet(bands_now, beat, 1, dyn_floor, kick_intensity)
                        active = True; shared.gate_active = True
                        resume_since = None; last_tick = now
                        time.sleep(0.001)
                        continue
                else:
                    resume_since = None

            if not active:
                print_status(ctx, " (idle)")
                time.sleep(0.05)
                continue

            if (now - last_tick) < MIN_SEND_INTERVAL:
                time.sleep(0.001)
                continue
            last_tick = now

            mean_val = float(np.mean(bands_now))
            dyn_floor = int(min(12, mean_val * 0.012)) if mean_val > 90 else 0
            send_audio_packet(bands_now, beat, 0, dyn_floor, kick_intensity)
            print_status(ctx, "")
    except KeyboardInterrupt:
        pass
    finally:
        ctx.stop_flag.set()
        try:
            stream.stop(); stream.close()
        except Exception:
            pass
        sys.stdout.write("\n"); sys.stdout.flush()

if __name__ == "__main__":
    main()