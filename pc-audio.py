#!/usr/bin/env python3
# pc-audio.py - Web UI + WebSocket + Captura WASAPI loopback
# Escala LOG (geomspace) ou MEL (filterbank) via --scale.
# Envia A1 (padrão, mais reativo). A2 opcional (dyn_floor/kick com floor neutro por padrão).
# Empurra CFG (B0) p/ RPi (bands/fps/hold/vis).
# Fabio Bonilha + M365 Copilot - 2025-11-11

import asyncio
import socket
import time
import sys
import platform
import threading
import argparse
from collections import deque
from typing import Optional, Union, Deque, Tuple

import numpy as np
import sounddevice as sd
from aiohttp import web

# ============================= Config padrao =============================
RASPBERRY_IP = "192.168.66.71"
UDP_PORT     = 5005
TCP_TIME_PORT= 5006

DEFAULT_NUM_BANDS = 150
BLOCK_SIZE = 1024               # tamanho do bloco de áudio (samples)
NFFT       = 4096               # FFT (zero-padding): aumenta resolução em freq
FMIN, FMAX = 20.0, 16000.0
EMA_ALPHA  = 0.75               # suavização das bandas no tempo
PEAK_EMA_DEFAULT = 0.10         # normalização adaptativa (ajustável via CLI)

# Beat detection simples (graves)
ENERGY_BUFFER_SIZE = 10
BEAT_THRESHOLD     = 1.2
BEAT_HEIGHT_MIN    = 0.08

# Time sync (habilitado por padrao; pode ser desativado por --no-require-sync)
REQUIRE_TIME_SYNC     = True
RESYNC_INTERVAL_SEC   = 60.0
RESYNC_RETRY_INTERVAL = 2.0
TIME_SYNC_SAMPLES     = 12

# Protocolo UDP para o RPi
PKT_AUDIO_V2 = 0xA2  # [A2][8 ts_pc][bands][beat][trans][dyn_floor][kick]
PKT_AUDIO    = 0xA1  # [A1][8 ts_pc][bands][beat][trans]
PKT_CFG      = 0xB0  # [B0][1 ver][nb_lo][nb_hi][fps_lo][fps_hi][hold_lo][hold_hi][vis_lo][vis_hi]

# ============================= HTML (Frontend) =============================
HTML = """<!doctype html>
<html lang="pt-br">
<head>
  <meta charset="utf-8" />
  <title>Reactive LEDs - Monitor</title>
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <style>
    :root{--bg:#0d1117;--panel:#161b22;--ink:#c9d1d9;--muted:#8b949e;--chip:#21262d;--chip-b:#30363d}
    *{box-sizing:border-box} body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.5 system-ui,Segoe UI,Roboto,Arial}
    header{padding:16px 20px;background:linear-gradient(180deg,#0f1623,#0d1117)} .wrap{max-width:1100px;margin:0 auto;padding:20px}
    h1{margin:0 0 4px;font-size:18px}.sub{color:var(--muted);font-size:12px}
    .grid{display:grid;grid-template-columns:1fr;gap:16px}@media(min-width:900px){.grid{grid-template-columns:1.2fr .8fr}}
    .panel{background:var(--panel);border:1px solid var(--chip-b);border-radius:10px;padding:14px}
    .row{display:flex;gap:10px;flex-wrap:wrap;align-items:center}
    button{background:var(--chip);border:1px solid var(--chip-b);color:var(--ink);border-radius:8px;padding:10px 14px;cursor:pointer;transition:.15s}
    button:hover{border-color:#58a6ff;box-shadow:0 0 0 2px #1f6feb33 inset}
    .pill{padding:3px 8px;border-radius:999px;border:1px solid var(--chip-b);background:var(--chip);font-size:12px}
    .ok{color:#8cffc9;border-color:#2ea043}.muted{color:var(--muted)}
    #spec{width:100%;height:260px;background:#0a0f16;border:1px solid var(--chip-b);border-radius:8px;display:block}
    .kvs{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:10px}
    .kv{background:var(--chip);border:1px solid var(--chip-b);border-radius:8px;padding:8px 10px}.kv b{float:right;color:#fff}
  </style>
</head>
<body>
  <header>
    <div class="wrap">
      <h1>Reactive LEDs</h1>
      <div class="sub">Monitor de audio (log/mel) + Auto Mode</div>
    </div>
  </header>
  <div class="wrap grid">
    <section class="panel"><canvas id="spec"></canvas></section>
    <section class="panel">
      <div class="row">
        <span class="pill" id="badgeConn">Conectando...</span>
        <span class="pill" id="badgeAuto">Auto: OFF</span>
        <span class="pill" id="badgeState">Estado: -</span>
      </div>
      <div style="height:8px"></div>
      <div class="row">
        <button id="btnAuto">Ajuste automático</button>
        <button id="btnCalib">Calibrar silêncio (5s)</button>
        <button id="btnSilence">Forçar silêncio</button>
      </div>
      <div style="height:10px"></div>
      <div class="kvs">
        <div class="kv">Device <b id="dev">-</b></div>
        <div class="kv">SR/Ch <b id="srch">-</b></div>
        <div class="kv">AVG <b id="avg">0</b></div>
        <div class="kv">RMS <b id="rms">0</b></div>
        <div class="kv">TX <b id="tx">0</b></div>
        <div class="kv">FPS <b id="fps">0</b></div>
      </div>
      <div style="height:6px"></div>
      <div class="muted">/api/status para JSON detalhado.</div>
    </section>
  </div>
<script>
(()=> {
  const c = document.getElementById('spec'); const ctx = c.getContext('2d');
  const el=id=>document.getElementById(id);
  const badgeConn=el('badgeConn'), badgeAuto=el('badgeAuto'), badgeState=el('badgeState');
  const dev=el('dev'), srch=el('srch'), avg=el('avg'), rms=el('rms'), tx=el('tx'), fps=el('fps');
  function resize(){ c.width=c.clientWidth; c.height=260; } window.addEventListener('resize', resize); resize();
  const wsUrl=(location.protocol==='https:'?'wss://':'ws://')+location.host+'/ws';
  let ws;
  function draw(bands, beat){
    ctx.clearRect(0,0,c.width,c.height);
    if(!bands||!bands.length) return;
    const w=c.width/bands.length;
    for(let i=0;i<bands.length;i++){
      const v=bands[i]/255, h=(c.height-2)*v;
      ctx.fillStyle = `hsl(${Math.floor(180+140*v)}, 90%, ${35+45*v}%)`;
      ctx.fillRect(i*w, c.height-h, Math.max(1,w-1), h);
    }
    if(beat){ ctx.fillStyle='rgba(255,80,80,0.12)'; ctx.fillRect(0,0,c.width,c.height); }
  }
  function connect(){
    ws=new WebSocket(wsUrl);
    ws.onopen=()=>{badgeConn.textContent='Conectado';badgeConn.classList.add('ok');};
    ws.onclose=()=>{badgeConn.textContent='Reconectando...';badgeConn.classList.remove('ok');setTimeout(connect,1000);};
    ws.onmessage=(ev)=>{
      const d=JSON.parse(ev.data||'{}'); if(d.bands) draw(d.bands, d.beat);
      fps.textContent=d.fps??0; avg.textContent=d.avg??0; rms.textContent=d.rms??0; tx.textContent=d.tx_count??0;
      dev.textContent=d.device||'-'; srch.textContent=d.sr_ch||'-'; badgeState.textContent='Estado: '+(d.silence?'silêncio':'ativo');
      badgeAuto.textContent='Auto: '+(d.auto_mode?'ON':'OFF');
    };
  } connect();
  document.getElementById('btnAuto').onclick=async()=>{ await fetch('/api/mode',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({mode:'auto_on'})}); };
  document.getElementById('btnCalib').onclick=async()=>{ await fetch('/api/mode',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({mode:'calibrate_silence',duration_sec:5})}); };
  document.getElementById('btnSilence').onclick=async()=>{ await fetch('/api/mode',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({mode:'true_silence'})}); };
})();
</script>
</body></html>"""

# ============================= Helpers de Device (Windows loopback) =============================
def resolve_device_index(name_or_index: Optional[Union[str, int]]):
    if name_or_index is None:
        return None
    try:
        idx = int(name_or_index)
        devs = sd.query_devices()
        if 0 <= idx < len(devs): return idx
    except (TypeError, ValueError):
        pass
    needle = str(name_or_index).lower()
    for i, d in enumerate(sd.query_devices()):
        if needle in d["name"].lower():
            return i
    return None

def auto_candidate_outputs():
    devs = sd.query_devices()
    cands = []
    default_out = sd.default.device[1]
    if isinstance(default_out, int) and default_out >= 0: cands.append(default_out)
    keywords = ["speaker","alto-falante","headphone","fone","realtek","echo","dot","hdmi","display audio","nvidia","realtek hd"]
    for i, d in enumerate(devs):
        if d.get("max_output_channels",0)>0 and any(k in d["name"].lower() for k in keywords):
            if i not in cands: cands.append(i)
    for i, d in enumerate(devs):
        if d.get("max_output_channels",0)>0 and i not in cands: cands.append(i)
    return cands

def choose_candidates(override: Optional[Union[str,int]]=None):
    devs = sd.query_devices()
    system = platform.system().lower()
    idx = resolve_device_index(override)
    if idx is not None:
        d = devs[idx]
        extra = None; loopback = False
        if system.startswith("win") and d.get("max_output_channels",0)>0:
            try: extra = sd.WasapiSettings(loopback=True); loopback=True
            except Exception: extra=None; loopback=False
        tag = f"{'WASAPI loopback' if loopback else 'OUTPUT (no-loopback)'}: '{d['name']}'"
        return [(idx, extra, tag)]
    if system.startswith("win"):
        cands=[]
        for i in auto_candidate_outputs():
            d = devs[i]
            try: extra = sd.WasapiSettings(loopback=True); tag = f"WASAPI loopback: '{d['name']}'"
            except Exception: extra=None; tag = f"OUTPUT (no-loopback): '{d['name']}'"
            cands.append((i, extra, tag))
        if cands: return cands
    default_in = sd.default.device[0]
    if isinstance(default_in, int) and default_in >= 0:
        d = devs[default_in]
        return [(default_in, None, f"Default INPUT (mic): idx={default_in} '{d['name']}'")]
    return [(None, None, "PortAudio default (fallback)")]

# ============================= FFT / Bandas (LOG) =============================
def make_bands_indices(nfft, sr, num_bands, fmin, fmax_limit, min_bins=1):
    freqs = np.fft.rfftfreq(nfft, 1.0/sr)
    fmax = min(fmax_limit, sr/2.0)
    edges = np.geomspace(fmin, fmax, num_bands+1)
    edge_idx = np.searchsorted(freqs, edges, side="left").astype(np.int32)
    a = edge_idx[:-1]; b = edge_idx[1:]
    b = np.maximum(b, a + min_bins)
    for i in range(1, len(a)):
        if a[i] < b[i-1]:
            a[i] = b[i-1]
            b[i] = max(b[i], a[i] + min_bins)
    b = np.minimum(b, freqs.size-1); a = np.minimum(a, b-1)
    return a, b

def make_compute_bands_log(sr, block_size, nfft, band_starts, band_ends, ema_alpha, peak_ema_alpha):
    window = np.hanning(block_size).astype(np.float32)
    ema_bands = np.zeros(len(band_starts), dtype=np.float32)
    peak_ema  = 1.0
    def compute(block):
        nonlocal ema_bands, peak_ema
        x = block * window
        fft_mag = np.abs(np.fft.rfft(x, n=nfft)).astype(np.float32)  # zero-padding
        cs = np.concatenate(([0.0], np.cumsum(fft_mag, dtype=np.float32)))
        sums = cs[band_ends] - cs[band_starts]
        lens = (band_ends - band_starts).astype(np.float32)
        means = sums / np.maximum(lens, 1.0)
        vals  = np.log1p(means)
        cur_max = float(np.max(vals)) if vals.size else 1.0
        peak_ema = (1.0 - peak_ema_alpha) * peak_ema + peak_ema_alpha * max(cur_max, 1e-6)
        vals = (vals / max(peak_ema, 1e-6)).clip(0.0, 1.0)
        ema_bands = ema_alpha * vals + (1.0 - ema_alpha) * ema_bands
        return (ema_bands * 255.0).clip(0,255).astype(np.uint8)
    return compute

# ============================= MEL filterbank =============================
def hz_to_mel(f):
    return 2595.0 * np.log10(1.0 + f / 700.0)

def mel_to_hz(m):
    return 700.0 * (10.0**(m / 2595.0) - 1.0)

def build_mel_filterbank(sr, nfft, n_mels, fmin=20.0, fmax=None):
    if fmax is None: fmax = sr/2.0
    # mel points
    m_min = hz_to_mel(fmin); m_max = hz_to_mel(fmax)
    m_points = np.linspace(m_min, m_max, n_mels + 2)  # inclui extremidades
    f_points = mel_to_hz(m_points)
    # bins
    freqs = np.fft.rfftfreq(nfft, 1.0/sr)
    bins  = np.floor((nfft + 1) * f_points / sr).astype(int)
    bins = np.clip(bins, 0, freqs.size-1)
    fb = np.zeros((n_mels, freqs.size), dtype=np.float32)
    for m in range(1, n_mels+1):
        f_left, f_center, f_right = bins[m-1], bins[m], bins[m+1]
        if f_center == f_left: f_center = min(f_left+1, freqs.size-1)
        if f_right  == f_center: f_right = min(f_center+1, freqs.size-1)
        # subida
        if f_center > f_left:
            fb[m-1, f_left:f_center] = (np.arange(f_left, f_center) - f_left) / max(1, (f_center - f_left))
        # descida
        if f_right > f_center:
            fb[m-1, f_center:f_right] = (f_right - np.arange(f_center, f_right)) / max(1, (f_right - f_center))
    # normalização por área (opcional, aqui deixamos unit-area aproximada)
    return fb

def make_compute_bands_mel(sr, block_size, nfft, mel_fb, ema_alpha, peak_ema_alpha):
    window = np.hanning(block_size).astype(np.float32)
    ema_bands = np.zeros(mel_fb.shape[0], dtype=np.float32)
    peak_ema  = 1.0
    def compute(block):
        nonlocal ema_bands, peak_ema
        x = block * window
        spec = np.abs(np.fft.rfft(x, n=nfft)).astype(np.float32)   # magnitude linear
        # projeção mel (matmul)
        vals = mel_fb @ spec
        vals = np.log1p(vals)    # compressão
        cur_max = float(np.max(vals)) if vals.size else 1.0
        peak_ema = (1.0 - peak_ema_alpha) * peak_ema + peak_ema_alpha * max(cur_max, 1e-6)
        vals = (vals / max(peak_ema, 1e-6)).clip(0.0, 1.0)
        ema_bands = ema_alpha * vals + (1.0 - ema_alpha) * ema_bands
        return (ema_bands * 255.0).clip(0,255).astype(np.uint8)
    return compute

# ============================= Time Sync (TCP cliente) =============================
def _recv_exact(sock, n):
    buf = b''
    while len(buf) < n:
        ch = sock.recv(n - len(buf))
        if not ch:
            raise ConnectionError("Conexao fechada durante receive.")
        buf += ch
    return buf

_time_sync_ok = False
_stop_resync  = threading.Event()
_last_status  = 0.0

def time_sync_over_tcp(raspberry_ip, port=TCP_TIME_PORT, samples=TIME_SYNC_SAMPLES, timeout=0.6):
    global _time_sync_ok
    results = []
    try:
        with socket.create_connection((raspberry_ip, port), timeout=1.8) as s:
            s.settimeout(timeout)
            for _ in range(samples):
                t0 = time.monotonic_ns()
                s.sendall(b"TS1" + t0.to_bytes(8, 'little'))
                data = _recv_exact(s, 3 + 8 + 8)
                if data[:3] != b"TS2": continue
                echo_t0 = int.from_bytes(data[3:11], 'little')
                tr_pi   = int.from_bytes(data[11:19], 'little')
                if echo_t0 != t0: continue
                t1 = time.monotonic_ns()
                rtt = t1 - t0; mid = t0 + rtt//2; offset = tr_pi - mid
                results.append((rtt, offset))
            if not results:
                _time_sync_ok = False
                print("\n[WARN] Time sync TCP falhou (sem amostras validas)."); return False
            rtt_min, offset_best = sorted(results, key=lambda x: x[0])[0]
            s.sendall(b"TS3" + int(offset_best).to_bytes(8, 'little', signed=True))
            ack = _recv_exact(s, 3 + 8)
            ok = (ack[:3]==b"TS3" and int.from_bytes(ack[3:11],'little',signed=True)==int(offset_best))
            _time_sync_ok = bool(ok)
            print(f"\n[INFO] Time sync TCP: RTT_min={rtt_min/1e6:.2f} ms, offset={offset_best/1e6:.3f} ms, ack={'OK' if ok else 'NOK'}")
            return _time_sync_ok
    except Exception as e:
        _time_sync_ok = False
        print(f"\n[WARN] Time sync TCP erro: {e}")
        return False

def resync_worker(raspberry_ip, resync_interval):
    while not _stop_resync.is_set():
        if _time_sync_ok:
            if _stop_resync.wait(resync_interval): break
        else:
            if _stop_resync.wait(RESYNC_RETRY_INTERVAL): break
        time_sync_over_tcp(raspberry_ip)

# ============================= Estado compartilhado =============================
class Shared:
    def __init__(self, n_bands):
        self.bands = np.zeros(n_bands, dtype=np.uint8)
        self.beat  = 0
        self.avg   = 0.0
        self.rms   = 0.0
        self.active= False
        self.tx_count = 0
        self.device   = ""
        self.samplerate = 0
        self.channels   = 0
        self.last_update= 0.0

# ============================= UDP =============================
udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
try: udp_sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1<<20)
except Exception: pass

def send_packet_a1(bands_u8, beat_flag, transition_flag, rpi_ip, rpi_port, shared):
    ts_ns = time.monotonic_ns()
    payload = bytes([PKT_AUDIO]) + ts_ns.to_bytes(8,'little') + bands_u8.tobytes() + bytes([beat_flag, transition_flag])
    udp_sock.sendto(payload, (rpi_ip, rpi_port)); shared.tx_count += 1

def send_packet_a2(bands_u8, beat_flag, transition_flag, dyn_floor, kick, rpi_ip, rpi_port, shared):
    ts_ns = time.monotonic_ns()
    payload = bytes([PKT_AUDIO_V2]) + ts_ns.to_bytes(8,'little') + bands_u8.tobytes() + bytes([beat_flag, transition_flag, dyn_floor, kick])
    udp_sock.sendto(payload, (rpi_ip, rpi_port)); shared.tx_count += 1

def send_cfg_b0(rpi_ip, rpi_port, bands, fps, hold_ms, vis_fps):
    def le16(x): return [x & 0xFF, (x>>8) & 0xFF]
    pkt = bytearray([PKT_CFG, 1])
    pkt += bytes(le16(int(bands))); pkt += bytes(le16(int(fps))); pkt += bytes(le16(int(hold_ms))); pkt += bytes(le16(int(vis_fps)))
    udp_sock.sendto(bytes(pkt), (rpi_ip, rpi_port))

# ============================= Servidor Web/WS =============================
app = web.Application()

async def handle_root(request):
    return web.Response(body=HTML, content_type='text/html')
app.router.add_get('/', handle_root)

async def handle_status(request):
    sh = request.app["shared"]; ss = request.app["server_state"]
    return web.json_response({
        "device": sh.device, "samplerate": sh.samplerate, "channels": sh.channels,
        "avg": round(float(sh.avg),2), "rms": round(float(sh.rms),6),
        "active": bool(sh.active), "tx_count": sh.tx_count, "bands_len": len(sh.bands),
        "auto_mode": bool(ss["auto_mode"]),
        "th_silence_bands": round(ss["silence_bands"],1),
        "th_silence_rms":   round(ss["silence_rms"],6),
        "th_resume_factor": round(ss["resume_factor"],2),
        "time": time.time(),
    })
app.router.add_get('/api/status', handle_status)

async def handle_mode(request):
    data = await request.json(); ss = request.app["server_state"]
    if data.get("mode") == "auto_on":
        ss["auto_mode"] = True;  print("\n[AUTO] ON")
    elif data.get("mode") == "auto_off":
        ss["auto_mode"] = False; print("\n[AUTO] OFF")
    elif data.get("mode") == "calibrate_silence":
        dur = float(data.get("duration_sec", 5)); ss["calibrating"]=True
        ss["calib_until"] = time.time() + max(1.0, min(15.0, dur))
        ss["calib_avg"]=[]; ss["calib_rms"]=[]
        print(f"\n[CALIB] {dur:.1f}s...")
    elif data.get("mode") == "true_silence":
        ss["force_silence_once"] = True; print("\n[SILENCIO] Forçando apagamento...")
    return web.json_response({"status":"ok"})
app.router.add_post('/api/mode', handle_mode)

async def handle_ws(request):
    ws = web.WebSocketResponse(); await ws.prepare(request)
    print("[WS] Cliente conectado - monitor ao vivo ON"); last_sent=0.0
    sh = request.app["shared"]; ss = request.app["server_state"]
    try:
        while True:
            now=time.time()
            if now-last_sent>=0.06:
                srch=f"{sh.samplerate} / {sh.channels}"
                await ws.send_json({
                    "connected":True, "fps": round(1/max(0.001, now-last_sent),1),
                    "bands": sh.bands.tolist(), "beat": bool(sh.beat), "silence": not sh.active,
                    "avg": round(float(sh.avg),1), "rms": round(float(sh.rms),5),
                    "tx_count": sh.tx_count, "device": sh.device, "sr_ch": srch,
                    "auto_mode": bool(ss["auto_mode"]),
                    "th_silence_bands": round(ss["silence_bands"],1),
                    "th_silence_rms":   round(ss["silence_rms"],6),
                    "th_resume_factor": round(ss["resume_factor"],2),
                }); last_sent=now
            try:
                msg = await asyncio.wait_for(ws.receive(), timeout=0.01)
                if msg.type in (web.WSMsgType.CLOSE, web.WSMsgType.CLOSING, web.WSMsgType.CLOSED): break
            except asyncio.TimeoutError: pass
    except Exception as e:
        print(f"[WS] Erro: {e}")
    finally:
        print("[WS] Cliente desconectado")
    return ws
app.router.add_get('/ws', handle_ws)

# ============================= Util de status console =============================
def print_status(shared, tag_extra: str = "", require_sync=True):
    global _last_status, _time_sync_ok
    now = time.time()
    if (now - _last_status) > 0.25:
        tag = "SYNC✓" if _time_sync_ok else ("SYNC OFF" if not require_sync else "WAITING SYNC")
        sys.stdout.write(f"\rTX: {shared.tx_count} {tag}{tag_extra}")
        sys.stdout.flush()
        _last_status = now

# ============================= Main =============================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bind', type=str, default='0.0.0.0')
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--raspberry-ip', type=str, default=RASPBERRY_IP)
    parser.add_argument('--udp-port', type=int, default=UDP_PORT)
    parser.add_argument('--bands', type=int, default=DEFAULT_NUM_BANDS, help='Numero de bandas (log) ou default para MEL')
    parser.add_argument('--scale', type=str, choices=['log','mel'], default='log', help='log|mel (filterbank)')
    parser.add_argument('--mel-bands', type=int, default=None, help='Override de n_bandas para MEL')
    parser.add_argument('--pkt', type=str, choices=['a1','a2'], default='a1', help='Formato do audio (a1 padrao)')

    parser.add_argument('--tx-fps', type=int, default=75)
    parser.add_argument('--signal-hold-ms', type=int, default=600)
    parser.add_argument('--vis-fps', type=int, default=60)

    parser.add_argument('--require-sync', action='store_true')
    parser.add_argument('--no-require-sync', action='store_true')
    parser.add_argument('--resync', type=float, default=RESYNC_INTERVAL_SEC)

    # Gate de silencio
    parser.add_argument('--silence-bands', type=float, default=28.0)
    parser.add_argument('--silence-rms',   type=float, default=0.0015)
    parser.add_argument('--silence-duration', type=float, default=0.8)
    parser.add_argument('--resume-factor', type=float, default=1.8)
    parser.add_argument('--resume-stable', type=float, default=0.35)

    # Alphas
    parser.add_argument('--avg-ema', type=float, default=0.05)
    parser.add_argument('--rms-ema', type=float, default=0.05)
    parser.add_argument('--norm-peak-ema', type=float, default=PEAK_EMA_DEFAULT)

    args = parser.parse_args()

    # FPS TX efetivo
    TX_FPS = max(1, int(args.tx_fps))
    MIN_SEND_INTERVAL = 1.0 / TX_FPS

    # Estado do servidor
    server_state = {
        "silence_bands": float(args.silence_bands),
        "silence_rms":   float(args.silence_rms),
        "silence_duration": float(args.silence_duration),
        "resume_factor": float(args.resume_factor),
        "resume_stable": float(args.resume_stable),

        "avg_ema": float(args.avg_ema),
        "rms_ema": float(args.rms_ema),
        "norm_peak_ema": float(args.norm_peak_ema),

        "auto_mode": False,
        "force_silence_once": False,
        "calibrating": False,
        "calib_until": 0.0,
        "calib_avg": [], "calib_rms": [],
    }
    app["server_state"] = server_state

    # shared
    n_bands = int(args.bands)
    shared = Shared(n_bands)
    app["shared"] = shared

    # Servidor web
    def start_web():
        web.run_app(app, host=args.bind, port=args.port)
    threading.Thread(target=start_web, daemon=True).start()
    print(f"\n[WEB] http://{args.bind}:{args.port}")

    # Time-sync
    require_sync = REQUIRE_TIME_SYNC
    if args.no_require_sync: require_sync = False
    if args.require_sync:    require_sync = True
    if require_sync:
        print("[INFO] Time-sync TCP (5006)...")
        time_sync_over_tcp(args.raspberry_ip)
        threading.Thread(target=resync_worker, args=(args.raspberry_ip, float(args.resync)), daemon=True).start()

    # Beat buffers
    energy_buf = np.zeros(ENERGY_BUFFER_SIZE, dtype=np.float32)
    energy_idx = 0; energy_count = 0; last_energy = 0.0

    # Filtros mediana
    rms_med = deque(maxlen=7); avg_med = deque(maxlen=7)

    # Audio callback / compute
    compute_bands = None

    # Abrir stream (WASAPI loopback se possível)
    def open_with_probe(dev_idx, extra, callback):
        devs = sd.query_devices()
        d = devs[dev_idx] if isinstance(dev_idx, int) and 0 <= dev_idx < len(devs) else None

        ch_pref = int(d.get('max_output_channels', 2)) if d else 2
        if ch_pref <= 0:
            ch_pref = 2
        try:
            sr_pref = int(round(float(d.get('default_samplerate', 44100)))) if d else 44100
        except Exception:
            sr_pref = 44100

        attempts = [
            (sr_pref, ch_pref),
            (48000, 2), (48000, 1),
            (44100, 2), (44100, 1),
        ]

        # >>> aqui estava o problema: não use '; if ...' nem '&gt;'
        seen = set()
        ordered = []
        for sr, ch in attempts:
            k = (sr, ch)
            if k not in seen and sr > 0 and ch > 0:
                seen.add(k)
                ordered.append((sr, ch))

        last_err = None
        for sr_try, ch_try in ordered:
            try:
                s = sd.InputStream(
                    samplerate=sr_try, channels=ch_try, dtype='float32',
                    blocksize=BLOCK_SIZE, device=dev_idx,
                    callback=callback, latency='low',
                    extra_settings=extra
                )
                return s, int(s.samplerate), int(s.channels), f"(opened sr={int(s.samplerate)} ch={int(s.channels)})", (extra is not None)
            except Exception as e:
                last_err = e

    # Fallback (não-ideal): input default (mic)
    try:
        s = sd.InputStream(
            channels=2, samplerate=44100, dtype='float32',
            blocksize=BLOCK_SIZE, device=None, callback=callback
        )
        return s, int(s.samplerate), 2, "(fallback default INPUT sr=44100 ch=2)", False
    except Exception as e:
        last_err = e
        raise last_err

    dev_desc="desconhecido"; stream=None; opened_loopback=False
    for dev_idx, extra, desc in choose_candidates(override=args.device):
        try:
            stream, sr_eff, ch_eff, open_desc, opened_loopback = open_with_probe(dev_idx, extra, audio_cb=None)
            dev_desc = f"{desc} {open_desc}"
            break
        except Exception as e:
            print(f"\n[WARN] Falha ao abrir {desc}: {e}"); continue
    if stream is None:
        print("\n[FATAL] Nao foi possivel abrir nenhum dispositivo de audio."); sys.exit(1)

    # Definir callback agora que sabemos SR/Ch e podemos montar os mapeamentos
    shared.device=dev_desc; shared.samplerate=sr_eff; shared.channels=ch_eff

    # Preparar LOG ou MEL
    scale_mode = args.scale.lower()
    peak_ema_alpha = float(args.norm_peak_ema)

    if scale_mode == 'mel':
        mel_bands = int(args.mel_bands or n_bands)
        mel_fb = build_mel_filterbank(shared.samplerate, NFFT, mel_bands, FMIN, min(FMAX, shared.samplerate/2.0))
        compute_bands = make_compute_bands_mel(shared.samplerate, BLOCK_SIZE, NFFT, mel_fb, EMA_ALPHA, peak_ema_alpha)
        shared.bands = np.zeros(mel_bands, dtype=np.uint8)  # ajustar tamanho exposto
    else:
        a_idx, b_idx = make_bands_indices(NFFT, shared.samplerate, n_bands, FMIN, FMAX, min_bins=2)
        compute_bands = make_compute_bands_log(shared.samplerate, BLOCK_SIZE, NFFT, a_idx, b_idx, EMA_ALPHA, peak_ema_alpha)

    # Recriar stream com callback
    def audio_cb(indata, frames, time_info, status):
        nonlocal energy_idx, energy_count, last_energy, compute_bands
        if status: sys.stdout.write("\n"); print(f"[WARN] Audio status: {status}")
        block = indata.mean(axis=1) if indata.ndim>1 else indata
        if len(block) < BLOCK_SIZE:
            block = np.pad(block, (0, BLOCK_SIZE - len(block)), "constant")
        if compute_bands is None: return
        bands = compute_bands(block[:BLOCK_SIZE])
        shared.bands = bands
        avg = float(np.mean(bands)); rms = float(np.sqrt(np.mean(block*block) + 1e-12))
        shared.avg=avg; shared.rms=rms; shared.last_update=time.time()
        # Beat (graves)
        lb = bands[:max(8, len(bands)//12)]
        energy = float(np.mean(lb)) / 255.0 if lb.size>0 else 0.0
        energy_buf[energy_idx] = energy
        energy_idx = (energy_idx + 1) % ENERGY_BUFFER_SIZE
        energy_count = min(energy_count + 1, ENERGY_BUFFER_SIZE)
        buf_view = energy_buf if energy_count==ENERGY_BUFFER_SIZE else energy_buf[:energy_count]
        avg_energy = float(np.mean(buf_view)) if energy_count>0 else 0.0
        std_energy = float(np.std(buf_view)) if energy_count>1 else 0.0
        dyn_thr = avg_energy + BEAT_THRESHOLD * std_energy
        is_peak_now = (energy >= max(dyn_thr, BEAT_HEIGHT_MIN)) and (energy >= last_energy)
        shared.beat = 1 if is_peak_now else 0
        last_energy = energy

        # Coleta para Auto/Calib (feito no loop principal via shared)

    # Fechar stream sem callback e abrir com callback
    try:
        stream.close()
        # Reabrir com as mesmas config, agora com callback
        def reopen_with_same(sr, ch):
            s = sd.InputStream(samplerate=sr, channels=ch, dtype='float32',
                               blocksize=BLOCK_SIZE, device=None, callback=audio_cb)
            return s
        # Tentar novamente no mesmo device seria possível, mas é suficiente reabrir default com sr/ch efetivos.
        stream = reopen_with_same(shared.samplerate, shared.channels)
        stream.start()
    except Exception as e:
        print(f"\n[FATAL] Falha ao iniciar stream: {e}"); sys.exit(1)

    # Empurrar CFG para o RPi
    try:
        send_cfg_b0(args.raspberry_ip, args.udp_port,
                    len(shared.bands), TX_FPS, int(args.signal_hold_ms), int(args.vis_fps))
        print(f"[CFG->RPi] bands={len(shared.bands)} fps={TX_FPS} hold={int(args.signal_hold_ms)} vis_fps={int(args.vis_fps)}")
    except Exception as e:
        print(f"[WARN] Falha B0: {e}")

    print(f"[INFO] Capturando de: {dev_desc}")
    print("[PRONTO] Monitor online - abra o navegador.")

    # Loop principal
    try:
        last_tick = 0.0; silence_since=None
        # histórico para auto
        hist_avg: Deque[Tuple[float,float]] = deque(); hist_rms: Deque[Tuple[float,float]] = deque()
        last_auto_update = 0.0
        while True:
            now = time.time()
            if (REQUIRE_TIME_SYNC and not _time_sync_ok) and not args.no_require_sync:
                print_status(shared, " (paused)", require_sync=True); time.sleep(0.05); continue

            bands_now = shared.bands; avg = shared.avg; rms = shared.rms; beat = shared.beat
            rms_med.append(rms); avg_med.append(avg)
            if len(rms_med)==rms_med.maxlen:
                rms_filtered=float(np.median(rms_med)); avg_filtered=float(np.median(avg_med))
            else:
                rms_filtered=rms; avg_filtered=avg

            ss = app["server_state"]

            # EMAs
            if 'avg_ema_val' not in ss:
                ss['avg_ema_val']=avg_filtered; ss['rms_ema_val']=rms_filtered
            else:
                ss['avg_ema_val']=ss['avg_ema_val']*(1.0-ss['avg_ema']) + ss['avg_ema']*avg_filtered
                ss['rms_ema_val']=ss['rms_ema_val']*(1.0-ss['rms_ema']) + ss['rms_ema']*rms_filtered

            avg_ema = ss['avg_ema_val']; rms_ema = ss['rms_ema_val']

            # Histórico para Auto
            hist_avg.append((now, avg)); hist_rms.append((now, rms))
            cutoff = now - 8.0
            while hist_avg and hist_avg[0][0] < cutoff: hist_avg.popleft()
            while hist_rms and hist_rms[0][0] < cutoff: hist_rms.popleft()

            # Auto (opcional)
            if ss["auto_mode"] and (now - last_auto_update) >= 0.5:
                arr_avg = np.array([v for _,v in hist_avg], dtype=np.float32) if hist_avg else np.array([avg],dtype=np.float32)
                arr_rms = np.array([v for _,v in hist_rms], dtype=np.float32) if hist_rms else np.array([rms],dtype=np.float32)
                noise_avg_p = float(np.percentile(arr_avg, 20))
                noise_rms_p = float(np.percentile(arr_rms, 20))
                music_avg_p = float(np.percentile(arr_avg, 80))
                th_avg = max(1.0, noise_avg_p * 1.4)
                th_rms = max(1e-6, noise_rms_p * 1.6)
                base = max(th_avg, 1e-3)
                dyn_resume = max(1.3, min(4.0, (music_avg_p / base) * 0.9))
                ss["silence_bands"]=th_avg; ss["silence_rms"]=th_rms; ss["resume_factor"]=dyn_resume
                last_auto_update = now

            # Gate
            is_quiet = (avg_ema < ss['silence_bands']) and (rms_ema < ss['silence_rms'])
            resume_threshold = ss['silence_bands'] * ss['resume_factor']

            # /api/mode: força silêncio
            if ss.get("force_silence_once"):
                zero = np.zeros(len(bands_now), dtype=np.uint8)
                if args.pkt=='a2':
                    # A2 com piso neutro (não achata)
                    send_packet_a2(zero, 0, 1, 0, 0, args.raspberry_ip, args.udp_port, shared)
                else:
                    send_packet_a1(zero, 0, 1, args.raspberry_ip, args.udp_port, shared)
                shared.active=False; ss["force_silence_once"]=False; time.sleep(0.05); continue

            if shared.active:
                if is_quiet:
                    if silence_since is None: silence_since=now
                    elif (now - silence_since) >= ss['silence_duration']:
                        zero = np.zeros(len(bands_now), dtype=np.uint8)
                        if args.pkt=='a2': send_packet_a2(zero, 0, 1, 0, 0, args.raspberry_ip, args.udp_port, shared)
                        else:              send_packet_a1(zero, 0, 1, args.raspberry_ip, args.udp_port, shared)
                        print_status(shared, " (to silence)", require_sync=not args.no_require_sync)
                        shared.active=False; silence_since=None; time.sleep(0.01); continue
                else:
                    silence_since=None
            else:
                if (avg_ema > resume_threshold) or (rms_ema > ss['silence_rms']*3.0):
                    if 'resume_since' not in ss or ss['resume_since'] is None: ss['resume_since']=now
                    elif (now - ss['resume_since']) >= ss['resume_stable']:
                        # Piso dinamico neutro por padrão (A2 não achata)
                        dyn_floor = 0
                        lb = bands_now[:max(8, len(bands_now)//12)]
                        energy_norm = float(np.mean(lb))/255.0 if lb.size>0 else 0.0
                        kick_val = 220 if beat else int(max(0, min(255, round(energy_norm*255.0))))
                        if args.pkt=='a2': send_packet_a2(bands_now, beat, 1, dyn_floor, kick_val, args.raspberry_ip, args.udp_port, shared)
                        else:              send_packet_a1(bands_now, beat, 1, args.raspberry_ip, args.udp_port, shared)
                        print_status(shared, " (from silence)", require_sync=not args.no_require_sync)
                        shared.active=True; ss['resume_since']=None; last_tick=now; time.sleep(0.001); continue
                else:
                    ss['resume_since']=None

            if not shared.active:
                print_status(shared, " (idle)", require_sync=not args.no_require_sync); time.sleep(0.05); continue

            # Throttle TX
            if (now - last_tick) < MIN_SEND_INTERVAL:
                time.sleep(0.001); continue
            last_tick = now

            # Envio normal
            dyn_floor = 0  # <<<< piso neutro por padrão para A2 (preserva reatividade)
            lb = bands_now[:max(8, len(bands_now)//12)]
            energy_norm = float(np.mean(lb))/255.0 if lb.size>0 else 0.0
            kick_val = 220 if beat else int(max(0, min(255, round(energy_norm*255.0))))
            if args.pkt=='a2': send_packet_a2(bands_now, beat, 0, dyn_floor, kick_val, args.raspberry_ip, args.udp_port, shared)
            else:              send_packet_a1(bands_now, beat, 0, args.raspberry_ip, args.udp_port, shared)
            print_status(shared, "", require_sync=not args.no_require_sync)

    except KeyboardInterrupt:
        pass
    finally:
        _stop_resync.set()
        try:
            stream.stop(); stream.close()
        except Exception: pass
        sys.stdout.write("\n"); sys.stdout.flush()

if __name__ == '__main__':
    main()