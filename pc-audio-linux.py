#!/usr/bin/env python3
# pc-audio-linux.py - Web UI + WS + Captura de áudio + envio A1/A2 + CSV + Mirror
# Fabio Bonilha + M365 Copilot — 2025-11-14
#
# Recursos:
# - Captura com sounddevice (preferência) ou PyAudio (fallback/controlado por --backend)
# - FFT -> bandas (log ou mel), suavização EMA, detecção simples de beat
# - Time-sync TCP com o Raspberry (porta 5006) para latência/offset
# - Envio UDP A1/A2 (5005) e pacote de configuração B0/B1
# - Web UI básica via aiohttp (status + websocket)
# - CSV assíncrono com rotação por tempo/tamanho e mudança em #bands, com gzip opcional
# - Mirror opcional de pacotes A1/A2 para um host de coleta (--mirror-to IP:PORT)

import asyncio
import aiofiles
import socket
import time
import sys
import platform
import threading
import argparse
from collections import deque
from typing import Optional, Union, Deque, Tuple, List
import numpy as np
import sounddevice as sd
from aiohttp import web
import os
import csv
import gzip
from datetime import datetime
import tempfile
import shutil
import random
import subprocess
import importlib
from aiohttp import MultipartReader

# ---- PyAudio (opcional) ----
_PAW = None
_PA = None
try:
    import pyaudiowpatch as pyaudio  # Windows loopback
    _PAW = pyaudio
except Exception:
    try:
        import pyaudio
        _PA = pyaudio
    except Exception:
        _PAW = None
        _PA = None

# ====================== Config padrão ======================
RASPBERRY_IP = "192.168.66.71"
UDP_PORT = 5005
TCP_TIME_PORT = 5006

DEFAULT_NUM_BANDS = 150
BLOCK_SIZE = 1024
NFFT = 4096
FMIN, FMAX = 20.0, 4000.0
EMA_ALPHA = 0.75
PEAK_EMA_DEFAULT = 0.10

ENERGY_BUFFER_SIZE = 10
BEAT_THRESHOLD = 1.2
BEAT_HEIGHT_MIN = 0.08

REQUIRE_TIME_SYNC = True
RESYNC_INTERVAL_SEC = 60.0
RESYNC_RETRY_INTERVAL = 2.0
TIME_SYNC_SAMPLES = 12

PKT_AUDIO_V2 = 0xA2
PKT_AUDIO    = 0xA1
PKT_CFG      = 0xB0
PKT_RESET    = 0xB1

# ====================== Web UI HTML (enxuta) ======================
HTML = r"""
<!doctype html>
<html lang="pt-br"><meta charset="utf-8">
<title>Reactive LEDs — Monitor</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
body{font-family:system-ui,Segoe UI,Roboto,Helvetica,Arial,sans-serif;margin:12px;background:#111;color:#eee}
.card{display:block;width:100%;max-width:900px;margin:8px auto;border:1px solid #333;border-radius:8px;padding:12px;background:#181818}
h2{margin:4px 0} .mono{font-family:ui-monospace,Consolas,Menlo,monospace}
.bars{height:120px;background:#000;border:1px solid #333;border-radius:6px;display:flex;align-items:flex-end;padding:4px;gap:1px;overflow:hidden}
.bar{width:3px;background:#39f}
.bar{width:auto;flex:1 1 auto;background:#39f}
.kv{display:grid;grid-template-columns:130px 1fr;gap:6px;margin-top:6px}
.badge{padding:2px 6px;border-radius:999px;background:#333;display:inline-block}
.controls{display:grid;grid-template-columns:1fr;gap:8px;margin-top:8px}
.row{display:flex;gap:8px;flex-wrap:wrap}
.input-small{padding:6px;border-radius:6px;border:1px solid #333;background:#0f0f0f;color:#fff}
.btn{padding:8px 12px;border-radius:6px;border:0;background:#39f;color:#001;font-weight:600}
.status{margin-top:8px;padding:8px;border-radius:6px;background:#0b0b0b;border:1px solid #222}
@media(max-width:420px){ .kv{grid-template-columns:110px 1fr} .bar{width:2px} }
</style>
<div class="card">
<h2>Reactive LEDs — Monitor</h2>
<div class="kv">
    <div>Conexão:</div><div id="conn" class="badge">Desconectado</div>
    <div>FPS:</div><div id="fps">—</div>
    <div>Dispositivo:</div><div id="dev">—</div>
    <div>SR/CH:</div><div id="srch">—</div>
    <div>TX:</div><div id="tx">0</div>
    <div>Auto:</div><div id="auto">OFF</div>
</div>
<h3>Bandas</h3>
<div class="bars" id="bars"></div>
<div class="kv">
    <div>Bandas:</div><div id="nb">—</div>
    <div>Silêncio:</div><div id="sil">—</div>
    <div>AVG:</div><div id="avg">—</div>
    <div>RMS:</div><div id="rms">—</div>
</div>

<h3>Fonte de Áudio</h3>
<div class="controls">
    <div class="row">
        <label class="input-small">Fonte:
            <select id="src_select" class="input-small">
                <option value="system">Sistema (Loopback)</option>
                <option value="linein">Line In</option>
                <option value="mp3">MP3 (pasta)</option>
            </select>
        </label>
        <button id="apply_btn" class="btn">Aplicar Configuração</button>
    </div>
    <div id="mp3_area" style="display:none">
        <div class="row">
            <input id="mp3_server_path" class="input-small" placeholder="Caminho da pasta no servidor (ex: /home/user/mp3)" style="flex:1">
            <button id="use_server_btn" class="btn">Usar Pasta no Servidor</button>
        </div>
        <div style="font-size:12px;margin-top:6px;color:#aaa">Informe uma pasta existente no computador onde este servidor está rodando. O servidor lerá os arquivos .mp3 dessa pasta (shuffle, loop) e não reproduzirá som.</div>
        <div style="height:6px"></div>
        <div style="font-size:12px;color:#bbb">(Opcional) Enviar arquivos do cliente para o servidor:</div>
        <div class="row">
            <label class="btn" style="padding:6px;display:inline-block">Selecionar Pasta Local
                <input id="mp3_files" type="file" webkitdirectory directory multiple style="display:none">
            </label>
            <input id="mp3_folder_path" class="input-small" placeholder="Nenhuma pasta selecionada (upload)" readonly style="flex:1">
            <button id="upload_btn" class="btn">Upload</button>
        </div>
        <div style="margin-top:8px;">
            <progress id="upload_progress" value="0" max="100" style="width:100%;display:none"></progress>
            <div id="upload_info" style="font-size:12px;color:#ccc;margin-top:6px"></div>
        </div>
    </div>
    <div class="status" id="src_status">Fonte atual: —</div>
</div>
</div>

<script>
const bars = document.getElementById('bars');
for(let i=0;i<150;i++){ const d=document.createElement('div'); d.className='bar'; bars.appendChild(d);}
const els = {
    conn:document.getElementById('conn'), fps:document.getElementById('fps'), dev:document.getElementById('dev'),
    srch:document.getElementById('srch'), tx:document.getElementById('tx'), auto:document.getElementById('auto'),
    nb:document.getElementById('nb'), sil:document.getElementById('sil'), avg:document.getElementById('avg'),
    rms:document.getElementById('rms')
};
const srcSelect = document.getElementById('src_select');
const mp3Area = document.getElementById('mp3_area');
const mp3Files = document.getElementById('mp3_files');
const mp3Path = document.getElementById('mp3_folder_path');
const uploadBtn = document.getElementById('upload_btn');
const applyBtn = document.getElementById('apply_btn');
const statusDiv = document.getElementById('src_status');
const useServerBtn = document.getElementById('use_server_btn');

srcSelect.addEventListener('change', ()=>{ mp3Area.style.display = (srcSelect.value==='mp3')? 'block':'none'; });
mp3Files.addEventListener('change', ()=>{
    if(mp3Files.files.length>0){ document.getElementById('mp3_folder_path').value = mp3Files.files[0].webkitRelativePath ? mp3Files.files[0].webkitRelativePath.split('/')[0] : mp3Files.files[0].name; }
});

useServerBtn.addEventListener('click', async ()=>{
    const p = document.getElementById('mp3_server_path').value.trim();
    if(!p){ alert('Informe o caminho da pasta no servidor.'); return; }
    useServerBtn.disabled = true; useServerBtn.textContent='Verificando...';
    try{
        const r = await fetch('/api/set_mp3_dir', { method:'POST', headers:{'content-type':'application/json'}, body: JSON.stringify({path: p}) });
        const j = await r.json();
        if(r.ok){ statusDiv.textContent = 'Pasta no servidor definida — ' + (j.count||0) + ' arquivos'; }
        else { statusDiv.textContent = 'Erro: ' + j.error; }
    }catch(e){ statusDiv.textContent = 'Erro: '+e; }
    useServerBtn.disabled = false; useServerBtn.textContent='Usar Pasta no Servidor';
});

uploadBtn.addEventListener('click', async ()=>{
    if(!mp3Files.files.length){ alert('Selecione uma pasta com arquivos .mp3 primeiro.'); return; }
    const form = new FormData();
    for(const f of mp3Files.files){ form.append('files', f, f.name); }
    uploadBtn.disabled = true; uploadBtn.textContent='Enviando...';
    const progress = document.getElementById('upload_progress');
    const info = document.getElementById('upload_info');
    progress.style.display = 'block'; progress.value = 0; info.textContent = '';
    try{
        await new Promise((resolve, reject) => {
            const xhr = new XMLHttpRequest();
            xhr.open('POST', '/api/upload_mp3');
            xhr.upload.onprogress = (ev) => {
                if(ev.lengthComputable){
                    const pct = Math.round((ev.loaded/ev.total)*100);
                    progress.value = pct;
                    info.textContent = `Enviando... ${pct}% (${(ev.loaded/1024/1024).toFixed(1)} MB)`;
                } else {
                    info.textContent = 'Enviando...';
                }
            };
            xhr.onerror = () => { reject(new Error('Erro de rede durante upload')); };
            xhr.onload = () => {
                if(xhr.status >=200 && xhr.status < 300){
                    try{ const j = JSON.parse(xhr.responseText); resolve(j); }
                    catch(e){ resolve({}); }
                } else {
                    try{ const j = JSON.parse(xhr.responseText); reject(new Error(j.error||xhr.statusText)); }
                    catch(e){ reject(new Error(xhr.statusText)); }
                }
            };
            xhr.send(form);
        }).then((j)=>{
            statusDiv.textContent = 'MP3 upload OK — ' + (j.count||0) + ' arquivos';
            const info = document.getElementById('upload_info');
            if(j.mp3_dir) info.textContent = 'Salvo em: ' + j.mp3_dir;
        }).catch((err)=>{ statusDiv.textContent = 'Erro upload: '+err; info.textContent = '' });
    }catch(e){ statusDiv.textContent = 'Erro upload: '+e; }
    uploadBtn.disabled = false; uploadBtn.textContent='Upload'; progress.style.display='none';
});

applyBtn.addEventListener('click', async ()=>{
    const src = srcSelect.value;
    applyBtn.disabled = true; applyBtn.textContent='Aplicando...';
    const body = { source: src };
    try{
        const r = await fetch('/api/set_source', { method:'POST', headers:{'content-type':'application/json'}, body: JSON.stringify(body) });
        const j = await r.json();
        if(r.ok){ statusDiv.textContent = 'Fonte aplicada: ' + j.current; }
        else { statusDiv.textContent = 'Erro: ' + j.error; }
    }catch(e){ statusDiv.textContent = 'Erro: '+e; }
    applyBtn.disabled = false; applyBtn.textContent='Aplicar Configuração';
});

function connect(){
    const ws = new WebSocket((location.protocol==='https:'?'wss://':'ws://')+location.host+'/ws');
    ws.onopen=()=>{ els.conn.textContent='Conectado'; els.conn.style.background='#264'; };
    ws.onclose=()=>{ els.conn.textContent='Desconectado'; els.conn.style.background='#633'; setTimeout(connect,1000); };
    ws.onmessage=(ev)=>{
        const j=JSON.parse(ev.data);
        els.fps.textContent=j.fps; els.dev.textContent=j.device; els.srch.textContent=j.sr_ch;
        els.tx.textContent=j.tx_count; els.auto.textContent=j.auto_mode?'ON':'OFF';
        els.nb.textContent=j.bands.length; els.sil.textContent=j.silence?'Sim':'Não';
        els.avg.textContent=j.avg; els.rms.textContent=j.rms;
        // ensure bar count matches bands length
        const barsContainer = document.getElementById('bars');
        const cur = barsContainer.children.length;
        const want = j.bands.length || 0;
        if(cur !== want){
            // rebuild
            barsContainer.innerHTML = '';
            for(let i=0;i<want;i++){ const d=document.createElement('div'); d.className='bar'; barsContainer.appendChild(d); }
        }
        const b = barsContainer.children;
        const n=Math.min(b.length,j.bands.length);
        for(let i=0;i<n;i++){ b[i].style.height=(j.bands[i]/255.0*100)+'%';}
    };
}
connect();
</script>
"""

# ====================== utilidades ======================
def resolve_device_index(name_or_index: Optional[Union[str, int]]):
    if name_or_index is None:
        return None
    try:
        idx = int(name_or_index)
        devs = sd.query_devices()
        if 0 <= idx < len(devs):
            return idx
    except (TypeError, ValueError):
        pass
    needle = str(name_or_index).lower()
    for i, d in enumerate(sd.query_devices()):
        if needle in d.get("name","").lower():
            return i
    return None

def auto_candidate_outputs():
    devs = sd.query_devices()
    cands = []
    default_out = sd.default.device[1]
    if isinstance(default_out, int) and default_out >= 0:
        cands.append(default_out)
    keywords = ["speaker","alto-falante","headphone","fone","realtek","echo","dot","hdmi","display audio","nvidia","realtek hd"]
    for i, d in enumerate(devs):
        if d.get("max_output_channels",0)>0 and any(k in d["name"].lower() for k in keywords):
            if i not in cands:
                cands.append(i)
    for i, d in enumerate(devs):
        if d.get("max_output_channels",0)>0 and i not in cands:
            cands.append(i)
    return cands

def make_bands_indices(nfft, sr, num_bands, fmin, fmax_limit, min_bins=1):
    freqs = np.fft.rfftfreq(nfft, 1.0/sr)
    fmax = min(fmax_limit, sr/2.0)
    edges = np.geomspace(fmin, fmax, num_bands+1)
    edge_idx = np.searchsorted(freqs, edges, side="left").astype(np.int32)
    a = edge_idx[:-1]; b = edge_idx[1:]
    b = np.maximum(b, a+min_bins)
    for i in range(1, len(a)):
        if a[i] < b[i-1]:
            a[i] = b[i-1]
            b[i] = max(b[i], a[i]+min_bins)
    b = np.minimum(b, freqs.size-1)
    a = np.minimum(a, b-1)
    return a, b

def make_compute_bands_log(sr, block_size, nfft, band_starts, band_ends, ema_alpha, peak_ema_alpha):
    window = np.hanning(block_size).astype(np.float32)
    ema_b = np.zeros(len(band_starts), dtype=np.float32)
    peak_ema = 1.0
    def compute(block):
        nonlocal ema_b, peak_ema
        x = block * window
        fft_mag = np.abs(np.fft.rfft(x, n=nfft)).astype(np.float32)
        cs = np.concatenate(([0.0], np.cumsum(fft_mag, dtype=np.float32)))
        sums = cs[band_ends] - cs[band_starts]
        lens = (band_ends - band_starts).astype(np.float32)
        means = sums / np.maximum(lens, 1.0)
        vals = np.log1p(means)
        cur_max = float(np.max(vals)) if vals.size else 1.0
        peak_ema = (1.0 - peak_ema_alpha) * peak_ema + peak_ema_alpha * max(cur_max, 1e-6)
        vals = (vals / max(peak_ema, 1e-6)).clip(0.0, 1.0)
        ema_b = ema_alpha * vals + (1.0 - ema_alpha) * ema_b
        return (ema_b * 255.0).clip(0,255).astype(np.uint8)
    return compute

def hz_to_mel(f): return 2595.0 * np.log10(1.0 + f/700.0)
def mel_to_hz(m): return 700.0 * (10.0**(m/2595.0) - 1.0)

def build_mel_filterbank(sr, nfft, n_mels, fmin=20.0, fmax=None):
    if fmax is None: fmax = sr/2.0
    m_min = hz_to_mel(fmin); m_max = hz_to_mel(fmax)
    m_points = np.linspace(m_min, m_max, n_mels+2)
    f_points = mel_to_hz(m_points)
    freqs = np.fft.rfftfreq(nfft, 1.0/sr)
    bins = np.floor((nfft+1)*f_points/sr).astype(int)
    bins = np.clip(bins, 0, freqs.size-1)
    fb = np.zeros((n_mels, freqs.size), dtype=np.float32)
    for m in range(1, n_mels+1):
        fl, fc, fr = bins[m-1], bins[m], bins[m+1]
        if fc <= fl: fc = min(fl+1, freqs.size-1)
        if fr <= fc: fr = min(fc+1, freqs.size-1)
        if fc > fl:
            fb[m-1, fl:fc] = (np.arange(fl, fc)-fl) / max(1, (fc-fl))
        if fr > fc:
            fb[m-1, fc:fr] = (fr-np.arange(fc, fr)) / max(1, (fr-fc))
    return fb

def make_compute_bands_mel(sr, block_size, nfft, mel_fb, ema_alpha, peak_ema_alpha):
    window = np.hanning(block_size).astype(np.float32)
    ema_b = np.zeros(mel_fb.shape[0], dtype=np.float32)
    peak_ema = 1.0
    def compute(block):
        nonlocal ema_b, peak_ema
        x = block * window
        spec = np.abs(np.fft.rfft(x, n=nfft)).astype(np.float32)
        vals = mel_fb @ spec
        vals = np.log1p(vals)
        cur_max = float(np.max(vals)) if vals.size else 1.0
        peak_ema = (1.0 - peak_ema_alpha) * peak_ema + peak_ema_alpha * max(cur_max, 1e-6)
        vals = (vals / max(peak_ema, 1e-6)).clip(0.0, 1.0)
        ema_b = ema_alpha * vals + (1.0 - ema_alpha) * ema_b
        return (ema_b * 255.0).clip(0,255).astype(np.uint8)
    return compute

def _recv_exact(sock, n):
    buf = b''
    while len(buf) < n:
        ch = sock.recv(n - len(buf))
        if not ch:
            raise ConnectionError("Conexao fechada durante receive.")
        buf += ch
    return buf

_time_sync_ok = False
_stop_resync = threading.Event()
_last_status = 0.0

def time_sync_over_tcp(raspberry_ip, port=TCP_TIME_PORT, samples=TIME_SYNC_SAMPLES, timeout=0.6):
    global _time_sync_ok
    results = []
    try:
        with socket.create_connection((raspberry_ip, port), timeout=1.8) as s:
            s.settimeout(timeout)
            for _ in range(samples):
                t0 = time.monotonic_ns()
                s.sendall(b"TS1" + t0.to_bytes(8, 'little'))
                data = _recv_exact(s, 3+8+8)
                if data[:3] != b"TS2":
                    continue
                echo_t0 = int.from_bytes(data[3:11], 'little')
                tr_pi   = int.from_bytes(data[11:19], 'little')
                if echo_t0 != t0:
                    continue
                t1 = time.monotonic_ns()
                rtt = t1 - t0
                mid = t0 + rtt//2
                offset = tr_pi - mid
                results.append((rtt, offset))
        if not results:
            _time_sync_ok = False
            print("[WARN] Time sync TCP falhou (sem amostras válidas).")
            return False
        rtt_min, offset_best = sorted(results, key=lambda x: x[0])[0]
        with socket.create_connection((raspberry_ip, port), timeout=1.8) as s:
            s.settimeout(timeout)
            s.sendall(b"TS3" + int(offset_best).to_bytes(8, 'little', signed=True))
            ack = _recv_exact(s, 3+8)
            ok = (ack[:3] == b"TS3" and int.from_bytes(ack[3:11],'little',signed=True) == int(offset_best))
            _time_sync_ok = bool(ok)
            print(f"[INFO] Time sync TCP: RTT_min={rtt_min/1e6:.2f} ms, offset={offset_best/1e6:.3f} ms, ack={'OK' if ok else 'NOK'}")
            return _time_sync_ok
    except Exception as e:
        _time_sync_ok = False
        print(f"[WARN] Time sync TCP erro: {e}")
        return False

def resync_worker(raspberry_ip, resync_interval):
    while not _stop_resync.is_set():
        if _time_sync_ok:
            if _stop_resync.wait(resync_interval):
                break
        else:
            if _stop_resync.wait(RESYNC_RETRY_INTERVAL):
                break
        time_sync_over_tcp(raspberry_ip)

class Shared:
    def __init__(self, n_bands):
        self.bands = np.zeros(n_bands, dtype=np.uint8)
        self.beat = 0
        self.avg = 0.0
        self.rms = 0.0
        self.active = False
        self.tx_count = 0
        self.device = ""
        self.samplerate = 0
        self.channels = 0
        self.last_update = 0.0

udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
try:
    udp_sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1<<20)
except Exception:
    pass

def send_packet_a1(bands_u8, beat_flag, transition_flag, rpi_ip, rpi_port, shared):
    ts_ns = time.monotonic_ns()
    payload = bytes([PKT_AUDIO]) + ts_ns.to_bytes(8, 'little') + bands_u8.tobytes() + bytes([beat_flag, transition_flag])
    udp_sock.sendto(payload, (rpi_ip, rpi_port))
    shared.tx_count += 1
    # mirror (se configurado)
    try:
        if mirror_target is not None:
            udp_sock.sendto(payload, mirror_target)
    except Exception:
        pass

def send_packet_a2(bands_u8, beat_flag, transition_flag, dyn_floor, kick, rpi_ip, rpi_port, shared):
    ts_ns = time.monotonic_ns()
    payload = (bytes([PKT_AUDIO_V2]) + ts_ns.to_bytes(8, 'little') +
               bands_u8.tobytes() + bytes([beat_flag, transition_flag, dyn_floor, kick]))
    udp_sock.sendto(payload, (rpi_ip, rpi_port))
    shared.tx_count += 1
    # mirror (se configurado)
    try:
        if mirror_target is not None:
            udp_sock.sendto(payload, mirror_target)
    except Exception:
        pass

def send_cfg_b0(rpi_ip, rpi_port, bands, fps, hold_ms, vis_fps):
    def le16(x): return [x & 0xFF, (x >> 8) & 0xFF]
    pkt = bytearray([PKT_CFG, 1])
    pkt += bytes(le16(int(bands)))
    pkt += bytes(le16(int(fps)))
    pkt += bytes(le16(int(hold_ms)))
    pkt += bytes(le16(int(vis_fps)))
    udp_sock.sendto(bytes(pkt), (rpi_ip, rpi_port))

def send_reset_b1(rpi_ip, rpi_port):
    udp_sock.sendto(bytes([PKT_RESET]), (rpi_ip, rpi_port))

app = web.Application()

async def handle_root(request):
    return web.Response(text=HTML, content_type='text/html', charset='utf-8')
app.router.add_get('/', handle_root)

async def handle_status(request):
    sh = request.app["shared"]; ss = request.app["server_state"]
    now = time.time()
    calib_eta = max(0.0, ss.get("calib_until", 0.0) - now) if ss.get("calibrating", False) else 0.0
    return web.json_response({
        "device": sh.device, "samplerate": sh.samplerate, "channels": sh.channels,
        "avg": round(float(sh.avg), 2), "rms": round(float(sh.rms), 6),
        "active": bool(sh.active), "tx_count": sh.tx_count,
        "bands_len": len(sh.bands), "auto_mode": bool(ss["auto_mode"]),
        "current_source": request.app.get('current_source'),
        "mp3_dir": request.app.get('mp3_dir'),
        "hw_mode": request.app.get('hw_mode'),
        "th_silence_bands": round(ss["silence_bands"],1),
        "th_silence_rms": round(ss["silence_rms"],6),
        "th_resume_factor": round(ss["resume_factor"],2),
        "calibrating": bool(ss.get("calibrating", False)),
        "calib_eta": round(calib_eta, 2),
        "time": time.time(),
    })
app.router.add_get('/api/status', handle_status)

async def handle_mode(request):
    data = await request.json()
    ss = request.app["server_state"]
    if data.get("mode") == "auto_on":
        ss["auto_mode"] = True; print("[AUTO] ON")
    elif data.get("mode") == "auto_off":
        ss["auto_mode"] = False; print("[AUTO] OFF")
    elif data.get("mode") == "calibrate_silence":
        dur = float(data.get("duration_sec", 5))
        ss["calibrating"] = True
        ss["calib_started"] = time.time()
        ss["calib_until"] = time.time() + max(1.0, min(15.0, dur))
        ss["calib_avg"] = []; ss["calib_rms"] = []
        print(f"[CALIB] {dur:.1f}s...")
    elif data.get("mode") == "true_silence":
        ss["force_silence_once"] = True; print("[SILENCIO] Forçando apagamento...")
    return web.json_response({"status": "ok"})
app.router.add_post('/api/mode', handle_mode)


# ----------------- Fonte / MP3 handling -----------------
def ensure_pydub():
    try:
        import pydub
        return pydub
    except Exception:
        print('[INFO] pydub não encontrado — instalando via pip...')
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'pydub', 'imageio-ffmpeg'])
        except Exception as e:
            raise RuntimeError(f'Falha ao instalar pydub: {e}')
        import importlib
        pydub = importlib.import_module('pydub')
        try:
            img = importlib.import_module('imageio_ffmpeg')
            ff = img.get_ffmpeg_exe()
            pydub.AudioSegment.converter = ff
        except Exception:
            pass
        return pydub

def ensure_aiofiles():
    try:
        import aiofiles
        return aiofiles
    except Exception:
        print('[INFO] aiofiles não encontrado — instalando via pip...')
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'aiofiles'])
        except Exception as e:
            raise RuntimeError(f'Falha ao instalar aiofiles: {e}')
        import importlib
        aiofiles = importlib.import_module('aiofiles')
        return aiofiles
        try:
            img = importlib.import_module('imageio_ffmpeg')
            ff = img.get_ffmpeg_exe()
            pydub.AudioSegment.converter = ff
        except Exception:
            pass
        return pydub

def start_mp3_worker(app, folder_path):
    """Start a background thread that decodes mp3 files (silent) and feeds analysis callback."""
    if not os.path.isdir(folder_path):
        raise RuntimeError('Pasta MP3 inexistente')
    # collect mp3 files
    files = []
    for root, _, fnames in os.walk(folder_path):
        for f in fnames:
            if f.lower().endswith('.mp3'):
                files.append(os.path.join(root, f))
    if not files:
        raise RuntimeError('Nenhum MP3 encontrado na pasta')

    pydub = ensure_pydub()
    AudioSegment = pydub.AudioSegment

    stop_event = threading.Event()
    app['mp3_stop_event'] = stop_event

    def worker():
        shared = app['shared']
        # determine playback rate
        sr = int(shared.samplerate or 44100)
        block_size = BLOCK_SIZE
        while not stop_event.is_set():
            order = files[:]
            random.shuffle(order)
            for fp in order:
                if stop_event.is_set(): break
                try:
                    seg = AudioSegment.from_file(fp)
                    seg = seg.set_channels(1)
                    seg = seg.set_frame_rate(sr)
                    arr = np.array(seg.get_array_of_samples())
                    # pydub sample width handling
                    sw = seg.sample_width
                    if sw == 2:
                        arr = arr.astype(np.float32) / 32768.0
                    elif sw == 4:
                        arr = arr.astype(np.float32) / (2**31)
                    else:
                        # fallback normalization
                        arr = arr.astype(np.float32)
                        arr /= np.max(np.abs(arr)) if arr.size else 1.0
                    total = arr.shape[0]
                    pos = 0
                    # feed in blocks at realtime pace
                    t_block = float(block_size) / float(sr)
                    while pos < total and not stop_event.is_set():
                        chunk = arr[pos:pos+block_size]
                        if chunk.shape[0] < block_size:
                            chunk = np.pad(chunk, (0, block_size - chunk.shape[0]), 'constant')
                        try:
                            # call core audio callback (bypassing hardware guard)
                            cb = app.get('audio_cb_core')
                            if cb:
                                cb(chunk, chunk.shape[0], None, None)
                        except Exception:
                            pass
                        pos += block_size
                        time.sleep(t_block)
                except Exception as e:
                    print(f"[MP3] falha ao processar '{fp}': {e}")
            # loop infinite

    t = threading.Thread(target=worker, daemon=True)
    app['mp3_thread'] = t
    t.start()
    print(f"[MP3] Worker iniciado (pasta={folder_path}, arquivos={len(files)})")

def stop_mp3_worker(app):
    ev = app.get('mp3_stop_event')
    th = app.get('mp3_thread')
    app['mp3_thread'] = None
    app['mp3_stop_event'] = None
    if ev is not None:
        ev.set()
    if th is not None:
        th.join(timeout=1.0)
    # keep mp3_dir for possible reuse

async def handle_upload_mp3(request):
    """Receive multipart upload (streamed) and save files into persistent server folder.
    Writes are performed asynchronously using aiofiles to avoid blocking the aiohttp event loop.
    Filenames are sanitized to prevent path traversal; only .mp3 files are accepted.
    """
    # choose persistent destination: use `mp3` folder next to this script (easier to find)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dst = os.path.join(script_dir, 'mp3')
    os.makedirs(dst, exist_ok=True)
    print(f"[UPLOAD] recebendo arquivos MP3 -> {dst}")

    # Multipart streaming reader (does not buffer entire file in memory)
    reader = await request.multipart()
    aiofiles = ensure_aiofiles()
    import re, uuid
    count = 0
    try:
        async for part in reader:
            # only process file parts
            if part.filename is None:
                continue
            raw_name = part.filename
            # basic sanitization: basename + allow only safe chars
            name = os.path.basename(raw_name)
            # enforce .mp3
            if not name.lower().endswith('.mp3'):
                # consume and skip content
                while True:
                    chunk = await part.read_chunk()
                    if not chunk:
                        break
                continue
            # replace unsafe characters
            safe = re.sub(r'[^A-Za-z0-9._-]', '_', name)
            # avoid collisions: prefix with uuid
            safe = f"{uuid.uuid4().hex}_{safe}"
            out_path = os.path.join(dst, safe)
            # stream write asynchronously
            try:
                async with aiofiles.open(out_path, 'wb') as out_f:
                    while True:
                        chunk = await part.read_chunk()
                        if not chunk:
                            break
                        await out_f.write(chunk)
                count += 1
            except Exception as e:
                # on write error, remove partial file
                try: os.remove(out_path)
                except Exception: pass
                print(f"[UPLOAD] erro ao escrever '{out_path}': {e}")
                # continue processing other parts
                continue

        if count == 0:
            return web.json_response({'status':'error','error':'Nenhum MP3 encontrado nos arquivos enviados'}, status=400)

        # set as current mp3_dir (persistent)
        old = request.app.get('mp3_dir')
        old_was_temp = request.app.get('mp3_dir_is_temp', False)
        request.app['mp3_dir'] = dst
        request.app['mp3_dir_is_temp'] = False
        # remove old temp dir only
        if old and os.path.isdir(old) and old_was_temp and old != dst:
            try: shutil.rmtree(old)
            except Exception: pass

        return web.json_response({'status':'ok','count': count, 'mp3_dir': dst})
    except Exception as e:
        return web.json_response({'status':'error','error': str(e)}, status=500)

app.router.add_post('/api/upload_mp3', handle_upload_mp3)

async def handle_set_source(request):
    """Set source: {source: 'system'|'linein'|'mp3'}"""
    try:
        data = await request.json()
    except Exception:
        return web.json_response({'status':'error','error':'JSON inválido'}, status=400)
    src = data.get('source')
    if src not in ('system','linein','mp3'):
        return web.json_response({'status':'error','error':'Fonte inválida'}, status=400)
    # stop MP3 if running
    if src != 'mp3':
        stop_mp3_worker(request.app)
        request.app['current_source'] = 'hardware'
        request.app['hw_mode'] = src
        return web.json_response({'status':'ok','current': 'hardware'})
    # src == mp3
    mp3_dir = request.app.get('mp3_dir')
    if not mp3_dir or not os.path.isdir(mp3_dir):
        # fallback to linein
        request.app['current_source'] = 'hardware'
        request.app['hw_mode'] = 'linein'
        return web.json_response({'status':'error','error':'Pasta MP3 inexistente ou vazia - voltando para Line In'}, status=400)
    try:
        stop_mp3_worker(request.app)
        start_mp3_worker(request.app, mp3_dir)
        request.app['current_source'] = 'mp3'
        return web.json_response({'status':'ok','current':'mp3'})
    except Exception as e:
        request.app['current_source'] = 'hardware'
        request.app['hw_mode'] = 'linein'
        return web.json_response({'status':'error','error': str(e)}, status=500)

app.router.add_post('/api/set_source', handle_set_source)


async def handle_set_mp3_dir(request):
    """Set server-side mp3 directory: {path: '/path/to/folder'}"""
    try:
        data = await request.json()
    except Exception:
        return web.json_response({'status':'error','error':'JSON inválido'}, status=400)
    p = data.get('path')
    if not p:
        return web.json_response({'status':'error','error':'path ausente'}, status=400)
    if not os.path.isdir(p):
        return web.json_response({'status':'error','error':'pasta não encontrada no servidor'}, status=400)
    # count mp3
    files = []
    for root, _, fnames in os.walk(p):
        for f in fnames:
            if f.lower().endswith('.mp3'):
                files.append(os.path.join(root, f))
    if not files:
        return web.json_response({'status':'error','error':'nenhum mp3 na pasta'}, status=400)
    # set and respond
    old = request.app.get('mp3_dir')
    old_was_temp = request.app.get('mp3_dir_is_temp', False)
    request.app['mp3_dir'] = p
    request.app['mp3_dir_is_temp'] = False
    if old and os.path.isdir(old) and old_was_temp and old != p:
        try: shutil.rmtree(old)
        except Exception: pass
    return web.json_response({'status':'ok','mp3_dir': p, 'count': len(files)})

app.router.add_post('/api/set_mp3_dir', handle_set_mp3_dir)


async def handle_reset(request):
    cfg = request.app["cfg"]
    try:
        send_reset_b1(cfg["rpi_ip"], cfg["rpi_port"])
        return web.json_response({"status":"ok","sent":"B1 RESET"})
    except Exception as e:
        return web.json_response({"status":"error","error":str(e)}, status=500)
app.router.add_post('/api/reset', handle_reset)

async def handle_ws(request):
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    print("[WS] Cliente conectado - monitor ao vivo ON")
    last_sent = 0.0
    sh = request.app["shared"]; ss = request.app["server_state"]
    try:
        while True:
            now = time.time()
            if now - last_sent >= 0.062:  # ~16 fps
                srch = f"{sh.samplerate} / {sh.channels}"
                calib_eta = max(0.0, ss.get("calib_until",0.0) - now) if ss.get("calibrating", False) else 0.0
                await ws.send_json({
                    "connected": True,
                    "fps": round(1 / max(0.001, now - last_sent), 1),
                    "bands": sh.bands.tolist(),
                    "beat": bool(sh.beat),
                    "silence": not sh.active,
                    "avg": round(float(sh.avg), 1),
                    "rms": round(float(sh.rms), 5),
                    "tx_count": sh.tx_count,
                    "device": sh.device,
                    "sr_ch": srch,
                    "auto_mode": bool(ss["auto_mode"]),
                    "th_silence_bands": round(ss["silence_bands"],1),
                    "th_silence_rms": round(ss["silence_rms"],6),
                    "th_resume_factor": round(ss["resume_factor"],2),
                    "calibrating": bool(ss.get("calibrating", False)),
                    "calib_eta": round(calib_eta,2),
                })
                last_sent = now
            try:
                msg = await asyncio.wait_for(ws.receive(), timeout=0.01)
                if msg.type in (web.WSMsgType.CLOSE, web.WSMsgType.CLOSING, web.WSMsgType.CLOSED):
                    break
            except asyncio.TimeoutError:
                pass
    except Exception as e:
        print(f"[WS] Erro: {e}")
    finally:
        print("[WS] Cliente desconectado")
    return ws
app.router.add_get('/ws', handle_ws)

def print_status(shared, tag_extra: str = "", require_sync=True):
    global _last_status, _time_sync_ok
    now = time.time()
    if (now - _last_status) > 0.25:
        tag = "SYNCOK" if _time_sync_ok else ("SYNC OFF" if not require_sync else "WAITING SYNC")
        sys.stdout.write(f"\rTX: {shared.tx_count} {tag}{tag_extra} ")
        sys.stdout.flush()
        _last_status = now

# ============================= CSV Logger =============================
class CSVLogger:
    def __init__(self, base_dir: str, rotate_min: int, rotate_mb: int, gzip_enabled: bool):
        self.base_dir = base_dir
        self.rotate_min = max(1, int(rotate_min))
        self.rotate_bytes = max(1, int(rotate_mb)) * 1024 * 1024
        self.gzip_enabled = bool(gzip_enabled)
        os.makedirs(self.base_dir, exist_ok=True)
        self._q = deque()
        self._q_lock = threading.Lock()
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._worker, daemon=True)
        self._fh = None
        self._writer = None
        self._open_ts = 0.0
        self._written_bytes = 0
        self._cur_bands = None
        self._cur_path = None
        self._header = None
        self._flush_every = 0.75
        self._last_flush = 0.0
        self._t.start()

    def enqueue(self, row: dict):
        with self._q_lock:
            self._q.append(row)

    def stop(self):
        self._stop.set()
        self._t.join(timeout=2.0)
        try:
            if self._fh:
                self._fh.flush()
                self._fh.close()
        except Exception:
            pass

    def _need_rotate(self, bands_len: int) -> bool:
        now = time.time()
        if self._fh is None:
            return True
        if self._cur_bands is not None and bands_len != self._cur_bands:
            return True
        if (now - self._open_ts) >= (self.rotate_min * 60):
            return True
        if self._written_bytes >= self.rotate_bytes:
            return True
        return False

    def _open_new(self, bands_len: int):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = f"capture_b{bands_len}_{ts}.csv"
        path = os.path.join(self.base_dir, name + (".gz" if self.gzip_enabled else ""))
        if self._fh:
            try:
                self._fh.flush(); self._fh.close()
            except Exception:
                pass
        if self.gzip_enabled:
            self._fh = gzip.open(path, "at", encoding="utf-8", newline="")
        else:
            self._fh = open(path, "a", encoding="utf-8", newline="")
        self._writer = None
        self._open_ts = time.time()
        self._written_bytes = 0
        self._cur_bands = int(bands_len)
        self._cur_path = path
        self._header = None
        print(f"[CSV] {path} (rotate every {self.rotate_min} min / {self.rotate_bytes//(1024*1024)} MB)")

    @staticmethod
    def _make_header(bands_len: int) -> List[str]:
        base_cols = [
            "time_iso", "ts_pc_ns", "pkt_type", "bands_len",
            "beat", "transition", "dyn_floor", "kick",
            "avg", "rms", "device", "samplerate", "channels",
            "tx_fps", "scale_mode"
        ]
        band_cols = [f"band_{i}" for i in range(int(bands_len))]
        return base_cols + band_cols

    def _write_header(self, bands_len: int, meta: dict):
        self._header = self._make_header(bands_len)
        self._writer = csv.DictWriter(self._fh, fieldnames=self._header, extrasaction="ignore")
        self._writer.writeheader()
        meta_row = {k: meta.get(k, "") for k in ("device","samplerate","channels","tx_fps","scale_mode")}
        meta_row.update({
            "time_iso": datetime.now().isoformat(timespec="seconds"),
            "ts_pc_ns": 0, "pkt_type": "META", "bands_len": bands_len,
            "beat": "", "transition": "", "dyn_floor": "", "kick": "",
            "avg": "", "rms": ""
        })
        self._writer.writerow(meta_row)

    def _emit(self, row: dict):
        b = int(row.get("bands_len", 0))
        if self._need_rotate(b):
            self._open_new(b)
        if self._header is None or len(self._header) != (15 + b):  # 16 = base cols
            self._write_header(b, row)
        self._writer.writerow(row)
        self._written_bytes += sum(len(str(v)) for v in row.values()) + len(self._header)
        now = time.time()
        if (now - self._last_flush) >= self._flush_every:
            try: self._fh.flush()
            except Exception: pass
            self._last_flush = now

    def _worker(self):
        while not self._stop.is_set():
            row = None
            with self._q_lock:
                if self._q:
                    row = self._q.popleft()
            if row is None:
                time.sleep(0.01)
                continue
            try:
                self._emit(row)
            except Exception as e:
                sys.stderr.write(f"\n[CSV] erro ao escrever linha: {e}\n"); sys.stderr.flush()

# ============================= Backends de captura =============================

def _sd_choose_and_open(override, *, allow_mic: bool, mic_device: Optional[Union[str,int]], audio_cb, linein_substrings: List[str]):
    devs = sd.query_devices()
    system = platform.system().lower()
    def is_output(i): return devs[i].get("max_output_channels",0)>0
    def is_input(i):  return devs[i].get("max_input_channels",0)>0
    def is_linein_name(name: str) -> bool:
        n = name.lower()
        for s in linein_substrings:
            if s and s.lower() in n: return True
        return False

    if mic_device is not None:
        if not allow_mic:
            raise RuntimeError("Uso de microfone requer --allow-mic.")
        midx = resolve_device_index(mic_device)
        if midx is None or not is_input(midx):
            raise RuntimeError(f"--mic-device não encontrado ou não é entrada: {mic_device}")
        s = sd.InputStream(channels=2,
                           samplerate=int(devs[midx].get("default_samplerate", 44100)),
                           dtype='float32', blocksize=BLOCK_SIZE,
                           device=midx, callback=audio_cb)
        return s, int(s.samplerate), int(s.channels), f"(INPUT mic): '{devs[midx]['name']}'", False

    if override is not None:
        idx = resolve_device_index(override)
        if idx is None:
            raise RuntimeError(f"--device '{override}' não encontrado.")
        d = devs[idx]
        if is_output(idx) and system.startswith("win"):
            try:
                extra = sd.WasapiSettings(loopback=True)
                s = sd.InputStream(samplerate=int(d.get("default_samplerate", 44100)),
                                   channels=max(1, int(d.get("max_output_channels",2))),
                                   dtype='float32', blocksize=BLOCK_SIZE, device=idx,
                                   callback=audio_cb, latency='low', extra_settings=extra)
                return s, int(s.samplerate), int(s.channels), f"(WASAPI loopback): '{d['name']}'", True
            except Exception as e:
                raise RuntimeError(f"O dispositivo '{d['name']}' não abriu em loopback (SD): {e}")
        if is_input(idx):
            if not (is_linein_name(d['name']) or allow_mic):
                raise RuntimeError("Entrada parece microfone. Use --allow-mic para permitir.")
            s = sd.InputStream(samplerate=int(d.get("default_samplerate", 44100)),
                               channels=max(1, int(d.get("max_input_channels",2))),
                               dtype='float32', blocksize=BLOCK_SIZE, device=idx, callback=audio_cb)
            return s, int(s.samplerate), int(s.channels), f"(INPUT via --device): '{d['name']}'", False
        raise RuntimeError(f"Dispositivo '{d['name']}' não é válido para captura.")

    if system.startswith("win"):
        for i in auto_candidate_outputs():
            if not is_output(i): continue
            d = devs[i]
            try:
                extra = sd.WasapiSettings(loopback=True)
                s = sd.InputStream(samplerate=int(d.get("default_samplerate", 44100)),
                                   channels=max(1, int(d.get("max_output_channels",2))),
                                   dtype='float32', blocksize=BLOCK_SIZE, device=i, callback=audio_cb,
                                   latency='low', extra_settings=extra)
                return s, int(s.samplerate), int(s.channels), f"(WASAPI loopback): '{d['name']}'", True
            except Exception:
                continue
        raise RuntimeError("Nenhuma saída com WASAPI loopback disponível para o sounddevice.")

    preferred = None
    for i, d in enumerate(devs):
        if is_input(i) and is_linein_name(d['name']):
            preferred = i; break
    target = preferred
    if target is None:
        di = sd.default.device[0]
        if isinstance(di, int) and di>=0 and is_input(di):
            target = di
    if target is None:
        for i, d in enumerate(devs):
            if is_input(i):
                target = i; break
    if target is None:
        raise RuntimeError("Nenhum dispositivo de entrada disponível para captura.")
    d = devs[target]
    s = sd.InputStream(samplerate=int(d.get("default_samplerate", 44100)),
                       channels=max(1, int(d.get("max_input_channels",2))),
                       dtype='float32', blocksize=BLOCK_SIZE, device=target, callback=audio_cb)
    return s, int(s.samplerate), int(s.channels), f"(INPUT auto): '{d['name']}'", False

def _paw_choose_and_open(allow_mic: bool, mic_device: Optional[str], audio_cb):
    if _PAW is None and _PA is None:
        raise RuntimeError("PyAudioWPatch/PyAudio não encontrado. Instale com: pip install PyAudioWPatch ou PyAudio")
    pmod = _PAW or _PA
    pinst = pmod.PyAudio()

    if mic_device:
        if not allow_mic:
            pinst.terminate(); raise RuntimeError("Uso de microfone requer --allow-mic.")
        target_idx = None
        for i in range(pinst.get_device_count()):
            info = pinst.get_device_info_by_index(i)
            if mic_device.lower() in info.get("name"," ").lower() and info.get("maxInputChannels",0)>0:
                target_idx = i; break
        if target_idx is None:
            pinst.terminate(); raise RuntimeError(f"--mic-device '{mic_device}' não encontrado no PyAudio.")
        info = pinst.get_device_info_by_index(target_idx)
        rate = int(info.get("defaultSampleRate", 44100))
        ch = max(1, int(info.get("maxInputChannels", 1)))
        def _py_cb(in_data, frame_count, time_info, status):
            try:
                block = np.frombuffer(in_data, dtype=np.int16).astype(np.float32)/32768.0
                if ch>1: block = block.reshape(-1, ch).mean(axis=1)
                audio_cb(block, frame_count, time_info, status)
            except Exception: pass
            return (None, pmod.paContinue)
        stream = pinst.open(format=pmod.paInt16, channels=ch, rate=rate, input=True,
                            input_device_index=target_idx, frames_per_buffer=BLOCK_SIZE, stream_callback=_py_cb)
        return ("PyAudio (mic)", pinst, stream, rate, ch)

    # Loopback Windows com PyAudioWPatch (quando disponível)
    try:
        if _PAW is not None and platform.system().lower().startswith('win'):
            wasapi = pinst.get_host_api_info_by_type(pmod.paWASAPI)
            spk = pinst.get_device_info_by_index(wasapi["defaultOutputDevice"])
            if not spk.get("isLoopbackDevice", False):
                found = None
                for lb in pinst.get_loopback_device_info_generator():
                    if spk["name"] in lb["name"]:
                        found = lb; break
                if found is None:
                    pinst.terminate(); raise RuntimeError("Loopback padrão não encontrado no PyAudioWPatch.")
            rate = int(spk["defaultSampleRate"])
            ch   = max(1, int(spk.get("maxInputChannels", 2)))
            def _py_cb(in_data, frame_count, time_info, status):
                try:
                    block = np.frombuffer(in_data, dtype=np.int16).astype(np.float32)/32768.0
                    if ch>1: block = block.reshape(-1, ch).mean(axis=1)
                    audio_cb(block, frame_count, time_info, status)
                except Exception: pass
                return (None, pmod.paContinue)
            stream = pinst.open(format=pmod.paInt16, channels=ch, rate=rate, input=True,
                                input_device_index=spk["index"], frames_per_buffer=BLOCK_SIZE, stream_callback=_py_cb)
            return ("PyAudioWPatch (WASAPI loopback)", pinst, stream, rate, ch)
    except Exception:
        pass

    target_idx = None
    for i in range(pinst.get_device_count()):
        info = pinst.get_device_info_by_index(i)
        if info.get("maxInputChannels",0)>0:
            target_idx = i; break
    if target_idx is None:
        pinst.terminate(); raise RuntimeError("PyAudio: nenhum dispositivo de entrada disponível.")
    info = pinst.get_device_info_by_index(target_idx)
    rate = int(info.get("defaultSampleRate", 44100))
    ch   = max(1, int(info.get("maxInputChannels", 1)))
    def _py_cb(in_data, frame_count, time_info, status):
        try:
            block = np.frombuffer(in_data, dtype=np.int16).astype(np.float32)/32768.0
            if ch>1: block = block.reshape(-1, ch).mean(axis=1)
            audio_cb(block, frame_count, time_info, status)
        except Exception: pass
        return (None, pmod.paContinue)
    stream = pinst.open(format=pmod.paInt16, channels=ch, rate=rate, input=True,
                        input_device_index=target_idx, frames_per_buffer=BLOCK_SIZE, stream_callback=_py_cb)
    return ("PyAudio (input)", pinst, stream, rate, ch)

# ============================= Main =============================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bind', type=str, default='0.0.0.0')
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--backend', type=str, choices=['auto','sd','pyaudio'], default='auto')
    parser.add_argument('--no-pyaudio-fallback', action='store_true')
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--allow-mic', action='store_true')
    parser.add_argument('--mic-device', type=str, default=None)
    parser.add_argument('--linein-substr', type=str, default='ICUSBAUDIO7D,USB Audio,CM6206,CM106')
    parser.add_argument('--list-devices', action='store_true')

    parser.add_argument('--raspberry-ip', type=str, default=RASPBERRY_IP)
    parser.add_argument('--udp-port', type=int, default=UDP_PORT)

    parser.add_argument('--bands', type=int, default=DEFAULT_NUM_BANDS)
    parser.add_argument('--scale', type=str, choices=['log','mel'], default='log')
    parser.add_argument('--mel-bands', type=int, default=None)
    parser.add_argument('--mel-tilt', type=float, default=-0.25)
    parser.add_argument('--mel-no-area-norm', action='store_true')

    parser.add_argument('--pkt', type=str, choices=['a1','a2'], default='a2')
    parser.add_argument('--tx-fps', type=int, default=75)
    parser.add_argument('--signal-hold-ms', type=int, default=600)
    parser.add_argument('--vis-fps', type=int, default=60)

    parser.add_argument('--require-sync', action='store_true')
    parser.add_argument('--no-require-sync', action='store_true')
    parser.add_argument('--resync', type=float, default=RESYNC_INTERVAL_SEC)

    parser.add_argument('--silence-bands', type=float, default=28.0)
    parser.add_argument('--silence-rms', type=float, default=0.0015)
    parser.add_argument('--silence-duration', type=float, default=0.8)
    parser.add_argument('--resume-factor', type=float, default=1.8)
    parser.add_argument('--resume-stable', type=float, default=0.35)
    parser.add_argument('--avg-ema', type=float, default=0.05)
    parser.add_argument('--rms-ema', type=float, default=0.05)
    parser.add_argument('--norm-peak-ema', type=float, default=PEAK_EMA_DEFAULT)
    parser.add_argument('--no-reset-on-start', action='store_true')
    parser.add_argument('--noise-profile', action='store_true')
    parser.add_argument('--noise-headroom', type=float, default=1.12)
    parser.add_argument('--min-band', type=int, default=0)

    # Mirror
    parser.add_argument('--mirror-to', type=str, default=None,
                        help="IP:PORT para espelhar os pacotes A1/A2 para um host de treino/coleta")

    # CSV
    parser.add_argument('--csv-dir', type=str, default=None,
                        help="Diretório para salvar CSV com os pacotes enviados (A1/A2)")
    parser.add_argument('--csv-rotate-min', type=int, default=20,
                        help="Rotaciona o CSV a cada N minutos (default=20)")
    parser.add_argument('--csv-rotate-mb', type=int, default=256,
                        help="Rotaciona o CSV ao atingir N MB (default=256)")
    parser.add_argument('--csv-gzip', action='store_true',
                        help="Se presente, comprime os CSVs como .gz")

    args = parser.parse_args()

    if args.list_devices:
        devs = sd.query_devices()
        for i, d in enumerate(devs):
            print(f"{i:3d} {d['name']} in={d.get('max_input_channels',0)} out={d.get('max_output_channels',0)} sr={d.get('default_samplerate')}")
        return

    app["cfg"] = {"rpi_ip": args.raspberry_ip, "rpi_port": args.udp_port}
    TX_FPS = max(1, int(args.tx_fps))
    MIN_SEND_INTERVAL = 1.0 / TX_FPS

    server_state = {
        "silence_bands": float(args.silence_bands),
        "silence_rms": float(args.silence_rms),
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
        "calib_started": 0.0,
        "calib_avg": [],
        "calib_rms": [],
        "last_calib": {},
        "noise_profile": None,
        "noise_profile_frames": None,
        "noise_profile_path": None,
        "calib_max_frames": int(TX_FPS * 10),
    }
    app["server_state"] = server_state
    app["server_cfg"] = {
        "noise_profile_enabled": bool(args.noise_profile),
        "noise_headroom": float(args.noise_headroom),
        "min_band": int(args.min_band),
    }

    n_bands = int(args.bands)
    shared = Shared(n_bands)
    app["shared"] = shared
    # source control state (hardware vs mp3)
    app["current_source"] = 'hardware'
    app["hw_mode"] = 'system'
    app["mp3_dir"] = None
    app["mp3_thread"] = None
    app["mp3_stop_event"] = None
    app["audio_cb_core"] = None

    def start_web():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        web.run_app(app, host=args.bind, port=args.port, handle_signals=False)
    threading.Thread(target=start_web, daemon=True).start()
    print(f"[WEB] http://{args.bind}:{args.port}")

    require_sync = REQUIRE_TIME_SYNC
    if args.no_require_sync: require_sync = False
    if args.require_sync:    require_sync = True
    if require_sync:
        print("[INFO] Time-sync TCP (5006)...")
        time_sync_over_tcp(args.raspberry_ip)
    threading.Thread(target=resync_worker, args=(args.raspberry_ip, float(args.resync)), daemon=True).start()

    # Construção do extrator de bandas
    scale_mode = args.scale.lower()
    peak_ema_alpha = float(args.norm_peak_ema)
    compute_bands = None
    if scale_mode == 'mel':
        mel_bands = int(args.mel_bands or n_bands)
        mel_fb = build_mel_filterbank(44100, NFFT, mel_bands, FMIN, min(FMAX, 44100/2.0))
        if not args.mel_no_area_norm:
            row_sum = mel_fb.sum(axis=1, keepdims=True)
            mel_fb = mel_fb / np.maximum(row_sum, 1e-9)
        if abs(float(args.mel_tilt)) > 1e-9:
            k = np.arange(mel_bands, dtype=np.float32)
            tilt = ((k+1.0)/float(mel_bands)) ** float(args.mel_tilt)
            mel_fb = (mel_fb.T * tilt).T
        compute_bands = make_compute_bands_mel(44100, BLOCK_SIZE, NFFT, mel_fb, EMA_ALPHA, peak_ema_alpha)
        shared.bands = np.zeros(mel_bands, dtype=np.uint8)
    else:
        a_idx, b_idx = make_bands_indices(NFFT, 44100, n_bands, FMIN, FMAX, min_bins=2)
        compute_bands = make_compute_bands_log(44100, BLOCK_SIZE, NFFT, a_idx, b_idx, EMA_ALPHA, peak_ema_alpha)
        shared.bands = np.zeros(n_bands, dtype=np.uint8)

    # Buffer energia/beat
    energy_buf = deque(maxlen=ENERGY_BUFFER_SIZE)

    def make_audio_cb():
        last_energy = 0.0
        def audio_cb(block, frames, time_info, status):
            nonlocal last_energy
            b = block
            if b.shape[0] < BLOCK_SIZE:
                b = np.pad(b, (0, BLOCK_SIZE - b.shape[0]), 'constant')
            bands = compute_bands(b[:BLOCK_SIZE]) if compute_bands is not None else None
            if bands is None: return
            app["shared"].bands = bands
            avg = float(np.mean(bands))
            rms = float(np.sqrt(np.mean((b[:BLOCK_SIZE]*b[:BLOCK_SIZE])) + 1e-12))
            app["shared"].avg = avg
            app["shared"].rms = rms
            app["shared"].last_update = time.time()

            lb = bands[:max(8, len(bands)//12)]
            energy = float(np.mean(lb))/255.0 if lb.size>0 else 0.0
            energy_buf.append(energy)
            buf_view = list(energy_buf)
            avg_energy = float(np.mean(buf_view)) if buf_view else 0.0
            std_energy = float(np.std(buf_view)) if len(buf_view)>1 else 0.0
            dyn_thr = avg_energy + BEAT_THRESHOLD * std_energy
            is_peak_now = (energy >= max(dyn_thr, BEAT_HEIGHT_MIN)) and (energy >= last_energy)
            app["shared"].beat = 1 if is_peak_now else 0
            last_energy = energy
        return audio_cb

    backend_used = None; stream = None; pa_instance = None

    def _build_compute(sr):
        nonlocal compute_bands, shared
        if scale_mode == "mel":
            mel_bands = int(args.mel_bands or n_bands)
            mel_fb = build_mel_filterbank(sr, NFFT, mel_bands, FMIN, min(FMAX, sr/2.0))
            if not args.mel_no_area_norm:
                row_sum = mel_fb.sum(axis=1, keepdims=True)
                mel_fb = mel_fb / np.maximum(row_sum, 1e-9)
            if abs(float(args.mel_tilt)) > 1e-9:
                k = np.arange(mel_bands, dtype=np.float32)
                tilt = ((k+1.0)/float(mel_bands)) ** float(args.mel_tilt)
                mel_fb = (mel_fb.T * tilt).T
            compute_bands = make_compute_bands_mel(sr, BLOCK_SIZE, NFFT, mel_fb, EMA_ALPHA, peak_ema_alpha)
            shared.bands = np.zeros(mel_bands, dtype=np.uint8)
        else:
            a_idx, b_idx = make_bands_indices(NFFT, sr, n_bands, FMIN, FMAX, min_bins=2)
            compute_bands = make_compute_bands_log(sr, BLOCK_SIZE, NFFT, a_idx, b_idx, EMA_ALPHA, peak_ema_alpha)
            shared.bands = np.zeros(n_bands, dtype=np.uint8)

    linein_substrings = [s.strip() for s in (args.linein_substr or '').split(',') if s.strip()]

    if args.backend in ("auto","sd"):
        try:
            audio_cb = make_audio_cb()
            # expose core callback for MP3 worker
            app['audio_cb_core'] = audio_cb
            def sd_wrapper(indata, frames, time_info, status):
                # only feed hardware audio when current source is hardware
                if app.get('current_source', 'hardware') != 'hardware':
                    return
                arr = indata.copy().astype(np.float32)
                if arr.ndim>1 and arr.shape[1]>1:
                    arr = arr.mean(axis=1)
                audio_cb(arr.ravel(), frames, time_info, status)
            s, sr_eff, ch_eff, desc, is_loop = _sd_choose_and_open(
                args.device, allow_mic=args.allow_mic, mic_device=args.mic_device,
                audio_cb=sd_wrapper, linein_substrings=linein_substrings
            )
            _build_compute(sr_eff)
            stream = s; stream.start()
            shared.samplerate = sr_eff; shared.channels = ch_eff
            try:
                dev_info = sd.query_devices(resolve_device_index(args.device) if args.device is not None else stream.device)
                sr_def = dev_info.get("default_samplerate", sr_eff)
                name   = dev_info.get("name", desc)
            except Exception:
                sr_def = sr_eff; name = desc
            backend_used = f"sounddevice {desc}"
            print(f"[AUDIO] Dev: {name}\n SR(def)={sr_def} SR(opened)={sr_eff} CH={ch_eff}")
        except Exception as e_sd:
            if args.backend == "sd" or args.no_pyaudio_fallback:
                print(f"[FATAL] sounddevice falhou: {e_sd}"); sys.exit(1)
            else:
                print(f"[WARN] sounddevice falhou, tentando PyAudio... ({e_sd})")

    if stream is None and args.backend in ("auto","pyaudio"):
        try:
            audio_cb = make_audio_cb()
            app['audio_cb_core'] = audio_cb
            tag, pa_instance, pa_stream, sr_eff, ch_eff = _paw_choose_and_open(args.allow_mic, args.mic_device, audio_cb)
            _build_compute(sr_eff)
            pa_stream.start_stream()
            stream = pa_stream
            shared.samplerate = sr_eff; shared.channels = ch_eff
            backend_used = tag
        except Exception as e_pa:
            print(f"[FATAL] PyAudio falhou: {e_pa}"); sys.exit(1)

    if stream is None:
        print("[FATAL] Nenhum backend de áudio pôde ser aberto."); sys.exit(1)

    # Envia CFG e RESET (se habilitado)
    try:
        send_cfg_b0(args.raspberry_ip, args.udp_port, len(shared.bands), TX_FPS, int(args.signal_hold_ms), int(args.vis_fps))
        print(f"[CFG->RPi] bands={len(shared.bands)} fps={TX_FPS} hold={int(args.signal_hold_ms)} vis_fps={int(args.vis_fps)}")
    except Exception as e:
        print(f"[WARN] Falha B0: {e}")
    if not args.no_reset_on_start:
        try:
            send_reset_b1(args.raspberry_ip, args.udp_port)
            print("[RST->RPi] RESET (B1) enviado.")
        except Exception as e:
            print(f"[WARN] Falha ao enviar RESET B1: {e}")

    print(f"[INFO] Capturando de: {backend_used}")
    print("[PRONTO] Monitor online - abra o navegador.")

    # Mirror target
    global mirror_target
    mirror_target = None
    if args.mirror_to:
        try:
            _h, _p = args.mirror_to.split(":", 1)
            mirror_target = (_h, int(_p))
            print(f"[MIRROR] A1/A2 -> {_h}:{int(_p)}")
        except Exception as e:
            print(f"[WARN] --mirror-to inválido: {e}")
            mirror_target = None

    # CSV logger
    csv_logger = None
    if args.csv_dir:
        csv_logger = CSVLogger(
            base_dir=args.csv_dir,
            rotate_min=int(args.csv_rotate_min),
            rotate_mb=int(args.csv_rotate_mb),
            gzip_enabled=bool(args.csv_gzip)
        )

    rms_med = deque(maxlen=7)
    avg_med = deque(maxlen=7)
    last_tick = 0.0
    silence_since = None
    hist_avg: Deque[Tuple[float,float]] = deque()
    hist_rms: Deque[Tuple[float,float]] = deque()
    last_auto_update = 0.0

    try:
        while True:
            now = time.time()
            if (REQUIRE_TIME_SYNC and not _time_sync_ok) and not args.no_require_sync:
                print_status(shared, " (paused)", require_sync=True)
                time.sleep(0.05); continue

            bands_now = shared.bands
            avg = shared.avg
            rms = shared.rms
            beat = shared.beat

            rms_med.append(rms); avg_med.append(avg)
            if len(rms_med) == rms_med.maxlen:
                rms_filtered = float(np.median(rms_med))
                avg_filtered = float(np.median(avg_med))
            else:
                rms_filtered = rms; avg_filtered = avg

            ss = app["server_state"]; scfg = app["server_cfg"]
            if 'avg_ema_val' not in ss:
                ss['avg_ema_val'] = avg_filtered; ss['rms_ema_val'] = rms_filtered
            else:
                ss['avg_ema_val'] = ss['avg_ema_val']*(1.0-ss['avg_ema']) + ss['avg_ema']*avg_filtered
                ss['rms_ema_val'] = ss['rms_ema_val']*(1.0-ss['rms_ema']) + ss['rms_ema']*rms_filtered
            avg_ema = ss['avg_ema_val']; rms_ema = ss['rms_ema_val']

            hist_avg.append((now, avg_filtered))
            hist_rms.append((now, rms_filtered))
            cutoff = now - 8.0
            while hist_avg and hist_avg[0][0] < cutoff: hist_avg.popleft()
            while hist_rms and hist_rms[0][0] < cutoff: hist_rms.popleft()

            # auto-calib (opcional)
            if ss.get("calibrating", False) and now >= ss.get("calib_until",0.0):
                arr_avg = np.array(ss["calib_avg"] or [avg_filtered], dtype=np.float32)
                arr_rms = np.array(ss["calib_rms"] or [rms_filtered], dtype=np.float32)
                noise_avg_p = float(np.percentile(arr_avg, 80))
                noise_rms_p = float(np.percentile(arr_rms, 80))
                th_avg = max(1.0, noise_avg_p * 1.15)
                th_rms = max(1e-6, noise_rms_p * 1.25)
                base = max(th_avg, 1e-3)
                music_proxy = float(np.percentile(np.array([v for _,v in hist_avg], dtype=np.float32), 90)) if hist_avg else base*1.6
                resume_factor = float(np.clip((music_proxy/base) * 0.9, 1.4, 4.0))
                ss["silence_bands"] = th_avg; ss["silence_rms"] = th_rms; ss["resume_factor"] = resume_factor
                ss["last_calib"] = {"t": time.time(), "th_avg": float(th_avg), "th_rms": float(th_rms), "resume_factor": float(resume_factor)}
                ss["calibrating"] = False; ss["calib_until"]=0.0; ss["calib_started"]=0.0; ss["calib_avg"].clear(); ss["calib_rms"].clear()
                print(f"[CALIB] OK: th_avg={th_avg:.1f} th_rms={th_rms:.6f} resume_x={resume_factor:.2f}")

            # auto-mode (ajuste suave)
            if ss["auto_mode"] and (now - last_auto_update) >= 0.5:
                arr_avg = np.array([v for _,v in hist_avg], dtype=np.float32) if hist_avg else np.array([avg_filtered],dtype=np.float32)
                arr_rms = np.array([v for _,v in hist_rms], dtype=np.float32) if hist_rms else np.array([rms_filtered],dtype=np.float32)
                noise_avg_p = float(np.percentile(arr_avg, 20))
                noise_rms_p = float(np.percentile(arr_rms, 20))
                music_avg_p = float(np.percentile(arr_avg, 80))
                th_avg = max(1.0, noise_avg_p * 1.4)
                th_rms = max(1e-6, noise_rms_p * 1.6)
                base = max(th_avg, 1e-3)
                dyn_resume = max(1.3, min(4.0, (music_avg_p/base)*0.9))
                ss["silence_bands"]=th_avg; ss["silence_rms"]=th_rms; ss["resume_factor"]=dyn_resume
                last_auto_update = now

            avg_ema = ss['avg_ema_val']; rms_ema = ss['rms_ema_val']
            is_quiet = (avg_ema < ss['silence_bands']) and (rms_ema < ss['silence_rms'])
            resume_threshold = ss['silence_bands'] * ss['resume_factor']

            if ss.get("force_silence_once"):
                zero = np.zeros(len(bands_now), dtype=np.uint8)
                if args.pkt=='a2':
                    send_packet_a2(zero, 0, 1, 0, 0, args.raspberry_ip, args.udp_port, shared)
                else:
                    send_packet_a1(zero, 0, 1, args.raspberry_ip, args.udp_port, shared)
                shared.active=False; ss["force_silence_once"]=False; time.sleep(0.05); continue

            if shared.active:
                if is_quiet:
                    if silence_since is None: silence_since=now
                    elif (now - silence_since) >= ss['silence_duration']:
                        zero = np.zeros(len(bands_now), dtype=np.uint8)
                        if args.pkt=='a2':
                            send_packet_a2(zero, 0, 1, 0, 0, args.raspberry_ip, args.udp_port, shared)
                        else:
                            send_packet_a1(zero, 0, 1, args.raspberry_ip, args.udp_port, shared)
                        print_status(shared, " (to silence)", require_sync=not args.no_require_sync)
                        shared.active=False; silence_since=None; time.sleep(0.01); continue
                else:
                    silence_since=None
            else:
                if (avg_ema > resume_threshold) or (rms_ema > ss['silence_rms']*3.0):
                    if 'resume_since' not in ss or ss['resume_since'] is None:
                        ss['resume_since']=now
                    elif (now - ss['resume_since']) >= ss['resume_stable']:
                        dyn_floor = 0
                        lb = bands_now[:max(8, len(bands_now)//12)]
                        energy_norm = float(np.mean(lb))/255.0 if lb.size>0 else 0.0
                        kick_val = 220 if beat else int(max(0, min(255, round(energy_norm*255.0))))
                        b_send = bands_now
                        if app["server_cfg"]["noise_profile_enabled"] and ss.get("noise_profile") is not None:
                            prof = ss["noise_profile"] * float(app["server_cfg"]["noise_headroom"])
                            b_send = np.clip(bands_now.astype(np.float32)-prof, 0, 255).astype(np.uint8)
                        if app["server_cfg"]["min_band"] > 0:
                            b_send = np.maximum(b_send, app["server_cfg"]["min_band"]).astype(np.uint8)

                        if args.pkt=='a2':
                            send_packet_a2(b_send, beat, 1, dyn_floor, kick_val, args.raspberry_ip, args.udp_port, shared)
                            pkt_type="A2"; dyn_val=dyn_floor; kick_out=kick_val
                        else:
                            send_packet_a1(b_send, beat, 1, args.raspberry_ip, args.udp_port, shared)
                            pkt_type="A1"; dyn_val=0; kick_out=0

                        # CSV (transição from silence)
                        if csv_logger is not None:
                            try:
                                row = {
                                    "time_iso": datetime.now().isoformat(timespec="milliseconds"),
                                    "ts_pc_ns": time.monotonic_ns(),
                                    "pkt_type": pkt_type,
                                    "bands_len": int(b_send.size),
                                    "beat": int(beat),
                                    "transition": 1,
                                    "dyn_floor": int(dyn_val),
                                    "kick": int(kick_out),
                                    "avg": float(shared.avg),
                                    "rms": float(shared.rms),
                                    "device": shared.device or "",
                                    "samplerate": int(shared.samplerate or 0),
                                    "channels": int(shared.channels or 0),
                                    "tx_fps": int(args.tx_fps),
                                    "scale_mode": scale_mode
                                }
                                for i, v in enumerate(b_send.tolist()):
                                    row[f"band_{i}"] = int(v)
                                csv_logger.enqueue(row)
                            except Exception as e:
                                sys.stderr.write(f"\n[CSV] erro (from silence): {e}\n"); sys.stderr.flush()

                        print_status(shared, " (from silence)", require_sync=not args.no_require_sync)
                        shared.active=True; ss['resume_since']=None; last_tick=now; time.sleep(0.001); continue
                else:
                    ss['resume_since']=None

            if not shared.active:
                print_status(shared, " (idle)", require_sync=not args.no_require_sync)
                time.sleep(0.05); continue

            if (now - last_tick) < MIN_SEND_INTERVAL:
                time.sleep(0.001); continue
            last_tick = now

            dyn_floor = 0
            lb = bands_now[:max(8, len(bands_now)//12)]
            energy_norm = float(np.mean(lb))/255.0 if lb.size>0 else 0.0
            kick_val = 220 if beat else int(max(0, min(255, round(energy_norm*255.0))))
            b_send = bands_now
            if app["server_cfg"]["noise_profile_enabled"] and ss.get("noise_profile") is not None:
                prof = ss["noise_profile"] * float(app["server_cfg"]["noise_headroom"])
                b_send = np.clip(bands_now.astype(np.float32)-prof, 0, 255).astype(np.uint8)
            if app["server_cfg"]["min_band"] > 0:
                b_send = np.maximum(b_send, app["server_cfg"]["min_band"]).astype(np.uint8)

            if args.pkt=='a2':
                send_packet_a2(b_send, beat, 0, dyn_floor, kick_val, args.raspberry_ip, args.udp_port, shared)
                pkt_type="A2"; dyn_val=dyn_floor; kick_out=kick_val
            else:
                send_packet_a1(b_send, beat, 0, args.raspberry_ip, args.udp_port, shared)
                pkt_type="A1"; dyn_val=0; kick_out=0

            # CSV (frame ativo)
            if csv_logger is not None:
                try:
                    row = {
                        "time_iso": datetime.now().isoformat(timespec="milliseconds"),
                        "ts_pc_ns": time.monotonic_ns(),
                        "pkt_type": pkt_type,
                        "bands_len": int(b_send.size),
                        "beat": int(beat),
                        "transition": 0,
                        "dyn_floor": int(dyn_val),
                        "kick": int(kick_out),
                        "avg": float(shared.avg),
                        "rms": float(shared.rms),
                        "device": shared.device or "",
                        "samplerate": int(shared.samplerate or 0),
                        "channels": int(shared.channels or 0),
                        "tx_fps": int(args.tx_fps),
                        "scale_mode": scale_mode
                    }
                    for i, v in enumerate(b_send.tolist()):
                        row[f"band_{i}"] = int(v)
                    csv_logger.enqueue(row)
                except Exception as e:
                    sys.stderr.write(f"\n[CSV] erro ao montar linha: {e}\n"); sys.stderr.flush()

            print_status(shared, "", require_sync=not args.no_require_sync)

    except KeyboardInterrupt:
        pass
    finally:
        _stop_resync.set()
        try:
            if csv_logger is not None:
                csv_logger.stop()
        except Exception:
            pass
        try:
            if pa_instance is not None:
                try:
                    if stream.is_active(): stream.stop_stream()
                except Exception: pass
                try:
                    stream.close()
                except Exception: pass
                try:
                    pa_instance.terminate()
                except Exception: pass
            else:
                try: stream.stop()
                except Exception: pass
                try: stream.close()
                except Exception: pass
        except Exception:
            pass
        sys.stdout.write("\n"); sys.stdout.flush()

if __name__ == '__main__':
    main()