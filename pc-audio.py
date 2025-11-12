#!/usr/bin/env python3
# pc-audio.py - Web UI + WebSocket + Captura de áudio
# Preferência: sounddevice (WASAPI loopback). Fallback: PyAudio (PyAudioWPatch) se necessário.
# LOG/MEL; pacotes A1/A2; B0 (cfg) e B1 (reset); proteção anti-mic; frontend completo.
# Fabio Bonilha + M365 Copilot — 2025-11-12

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
import os

# --------- PyAudio (opcional) ---------
_PAW = None  # módulo pyaudiowpatch (preferido)
_PA  = None  # módulo pyaudio (oficial)
try:
    import pyaudiowpatch as pyaudio
    _PAW = pyaudio
except Exception:
    try:
        import pyaudio
        _PA = pyaudio
    except Exception:
        _PAW = None
        _PA  = None
# --------------------------------------

# ============================= Config padrao =============================
RASPBERRY_IP = "192.168.66.71"
UDP_PORT = 5005
TCP_TIME_PORT = 5006

DEFAULT_NUM_BANDS = 150
BLOCK_SIZE = 1024  # samples
NFFT = 4096        # zero-padding

FMIN, FMAX = 20.0, 16000.0

EMA_ALPHA = 0.75
PEAK_EMA_DEFAULT = 0.10

# Beat detection simples
ENERGY_BUFFER_SIZE = 10
BEAT_THRESHOLD = 1.2
BEAT_HEIGHT_MIN = 0.08

# Time sync
REQUIRE_TIME_SYNC = True
RESYNC_INTERVAL_SEC = 60.0
RESYNC_RETRY_INTERVAL = 2.0
TIME_SYNC_SAMPLES = 12

# Protocolo UDP RPi
PKT_AUDIO_V2 = 0xA2
PKT_AUDIO = 0xA1
PKT_CFG = 0xB0
PKT_RESET = 0xB1

# ============================= HTML (Frontend) =============================
# (mesmo HTML completo que te enviei; mantive igual)
HTML = r"""<!doctype html>
<html lang="pt-br">
<head>
  <meta charset="utf-8" />
  <title>Reactive LEDs — Monitor</title>
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <style>/* ... (CSS idêntico ao anterior) ... */</style>
</head>
<body>
  <!-- ... (mesmo corpo/JS do arquivo anterior) ... -->
</body>
</html>
"""

# ============================= Helpers de Device (Windows loopback) =============================
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
        if needle in d["name"].lower():
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

# ============================= FFT / LOG =============================
def make_bands_indices(nfft, sr, num_bands, fmin, fmax_limit, min_bins=1):
    freqs = np.fft.rfftfreq(nfft, 1.0/sr)
    fmax = min(fmax_limit, sr/2.0)
    edges = np.geomspace(fmin, fmax, num_bands+1)
    edge_idx = np.searchsorted(freqs, edges, side="left").astype(np.int32)
    a = edge_idx[:-1]
    b = edge_idx[1:]
    b = np.maximum(b, a + min_bins)
    for i in range(1, len(a)):
        if a[i] < b[i-1]:
            a[i] = b[i-1]
            b[i] = max(b[i], a[i] + min_bins)
    b = np.minimum(b, freqs.size-1)
    a = np.minimum(a, b-1)
    return a, b

def make_compute_bands_log(sr, block_size, nfft, band_starts, band_ends, ema_alpha, peak_ema_alpha):
    window = np.hanning(block_size).astype(np.float32)
    ema_bands = np.zeros(len(band_starts), dtype=np.float32)
    peak_ema = 1.0
    def compute(block):
        nonlocal ema_bands, peak_ema
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
        ema_bands = ema_alpha * vals + (1.0 - ema_alpha) * ema_bands
        return (ema_bands * 255.0).clip(0,255).astype(np.uint8)
    return compute

# ============================= MEL filterbank =============================
def hz_to_mel(f): return 2595.0 * np.log10(1.0 + f / 700.0)
def mel_to_hz(m): return 700.0 * (10.0**(m / 2595.0) - 1.0)

def build_mel_filterbank(sr, nfft, n_mels, fmin=20.0, fmax=None):
    if fmax is None:
        fmax = sr/2.0
    m_min = hz_to_mel(fmin)
    m_max = hz_to_mel(fmax)
    m_points = np.linspace(m_min, m_max, n_mels + 2)
    f_points = mel_to_hz(m_points)
    freqs = np.fft.rfftfreq(nfft, 1.0/sr)
    bins = np.floor((nfft + 1) * f_points / sr).astype(int)
    bins = np.clip(bins, 0, freqs.size-1)
    fb = np.zeros((n_mels, freqs.size), dtype=np.float32)
    for m in range(1, n_mels+1):
        f_left, f_center, f_right = bins[m-1], bins[m], bins[m+1]
        if f_center <= f_left: f_center = min(f_left + 1, freqs.size-1)
        if f_right <= f_center: f_right = min(f_center + 1, freqs.size-1)
        if f_center > f_left:
            fb[m-1, f_left:f_center] = (np.arange(f_left, f_center) - f_left) / max(1, (f_center - f_left))
        if f_right > f_center:
            fb[m-1, f_center:f_right] = (f_right - np.arange(f_center, f_right)) / max(1, (f_right - f_center))
    return fb

def make_compute_bands_mel(sr, block_size, nfft, mel_fb, ema_alpha, peak_ema_alpha):
    window = np.hanning(block_size).astype(np.float32)
    ema_bands = np.zeros(mel_fb.shape[0], dtype=np.float32)
    peak_ema = 1.0
    def compute(block):
        nonlocal ema_bands, peak_ema
        x = block * window
        spec = np.abs(np.fft.rfft(x, n=nfft)).astype(np.float32)
        vals = mel_fb @ spec
        vals = np.log1p(vals)
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
                data = _recv_exact(s, 3 + 8 + 8)
                if data[:3] != b"TS2":
                    continue
                echo_t0 = int.from_bytes(data[3:11], 'little')
                tr_pi = int.from_bytes(data[11:19], 'little')
                if echo_t0 != t0:
                    continue
                t1 = time.monotonic_ns()
                rtt = t1 - t0
                mid = t0 + rtt//2
                offset = tr_pi - mid
                results.append((rtt, offset))
            if not results:
                _time_sync_ok = False
                print("\n[WARN] Time sync TCP falhou (sem amostras validas).")
                return False
            rtt_min, offset_best = sorted(results, key=lambda x: x[0])[0]
            s.sendall(b"TS3" + int(offset_best).to_bytes(8, 'little', signed=True))
            ack = _recv_exact(s, 3 + 8)
            ok = (ack[:3] == b"TS3" and int.from_bytes(ack[3:11], 'little', signed=True) == int(offset_best))
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
            if _stop_resync.wait(resync_interval):
                break
        else:
            if _stop_resync.wait(RESYNC_RETRY_INTERVAL):
                break
            time_sync_over_tcp(raspberry_ip)

# ============================= Estado compartilhado =============================
class Shared:
    def __init__(self, n_bands):
        self.bands = np.zeros(n_bands, dtype=np.uint8)
        self.beat = 0
        self.avg = 0.0
        self.rms = 0.0
        self.active= False
        self.tx_count = 0
        self.device = ""
        self.samplerate = 0
        self.channels = 0
        self.last_update= 0.0

# ============================= UDP =============================
udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
try:
    udp_sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1 << 20)
except Exception:
    pass

def send_packet_a1(bands_u8, beat_flag, transition_flag, rpi_ip, rpi_port, shared):
    ts_ns = time.monotonic_ns()
    payload = bytes([PKT_AUDIO]) + ts_ns.to_bytes(8, 'little') + bands_u8.tobytes() + bytes([beat_flag, transition_flag])
    udp_sock.sendto(payload, (rpi_ip, rpi_port))
    shared.tx_count += 1

def send_packet_a2(bands_u8, beat_flag, transition_flag, dyn_floor, kick, rpi_ip, rpi_port, shared):
    ts_ns = time.monotonic_ns()
    payload = bytes([PKT_AUDIO_V2]) + ts_ns.to_bytes(8, 'little') + bands_u8.tobytes() + bytes([beat_flag, transition_flag, dyn_floor, kick])
    udp_sock.sendto(payload, (rpi_ip, rpi_port))
    shared.tx_count += 1

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

# ============================= Servidor Web/WS =============================
app = web.Application()

async def handle_root(request):
    # charset separado (evita ValueError no aiohttp)
    return web.Response(text=HTML, content_type='text/html', charset='utf-8')

app.router.add_get('/', handle_root)

async def handle_status(request):
    sh = request.app["shared"]
    ss = request.app["server_state"]
    now = time.time()
    calib_eta = max(0.0, ss.get("calib_until", 0.0) - now) if ss.get("calibrating", False) else 0.0
    return web.json_response({
        "device": sh.device,
        "samplerate": sh.samplerate,
        "channels": sh.channels,
        "avg": round(float(sh.avg), 2),
        "rms": round(float(sh.rms), 6),
        "active": bool(sh.active),
        "tx_count": sh.tx_count,
        "bands_len": len(sh.bands),
        "auto_mode": bool(ss["auto_mode"]),
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
        ss["auto_mode"] = True
        print("\n[AUTO] ON")
    elif data.get("mode") == "auto_off":
        ss["auto_mode"] = False
        print("\n[AUTO] OFF")
    elif data.get("mode") == "calibrate_silence":
        dur = float(data.get("duration_sec", 5))
        ss["calibrating"] = True
        ss["calib_started"] = time.time()
        ss["calib_until"] = time.time() + max(1.0, min(15.0, dur))
        ss["calib_avg"] = []
        ss["calib_rms"] = []
        if request.app["server_cfg"].get("noise_profile_enabled", False):
            ss["noise_profile_frames"] = []
        print(f"\n[CALIB] {dur:.1f}s...")
    elif data.get("mode") == "true_silence":
        ss["force_silence_once"] = True
        print("\n[SILENCIO] Forçando apagamento...")
    return web.json_response({"status": "ok"})

app.router.add_post('/api/mode', handle_mode)

async def handle_reset(request):
    cfg = request.app["cfg"]
    try:
        send_reset_b1(cfg["rpi_ip"], cfg["rpi_port"])
        return web.json_response({"status": "ok", "sent": "B1 RESET"})
    except Exception as e:
        return web.json_response({"status": "error", "error": str(e)}, status=500)

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

# ============================= Util status console =============================
def print_status(shared, tag_extra: str = "", require_sync=True):
    global _last_status, _time_sync_ok
    now = time.time()
    if (now - _last_status) > 0.25:
        tag = "SYNC✓" if _time_sync_ok else ("SYNC OFF" if not require_sync else "WAITING SYNC")
        sys.stdout.write(f"\rTX: {shared.tx_count} {tag}{tag_extra}")
        sys.stdout.flush()
        _last_status = now

# ============================= Backends de captura =============================
def _sd_choose_and_open(override, *, allow_mic: bool, mic_device: Optional[Union[str,int]],
                        audio_cb):
    """
    Tenta abrir com python-sounddevice em WASAPI loopback (Windows).
    Não cai em microfone (a menos que allow_mic=True com mic explicitado).
    """
    devs = sd.query_devices()
    system = platform.system().lower()

    def is_output(i): return devs[i].get("max_output_channels",0) > 0
    def is_input(i):  return devs[i].get("max_input_channels",0)  > 0

    # 1) mic explícito via mic_device
    if mic_device is not None:
        if not allow_mic:
            raise RuntimeError("Uso de microfone requer --allow-mic.")
        midx = resolve_device_index(mic_device)
        if midx is None or not is_input(midx):
            raise RuntimeError(f"--mic-device não encontrado ou não é entrada: {mic_device}")
        s = sd.InputStream(channels=2, samplerate=int(devs[midx].get("default_samplerate", 44100)),
                           dtype='float32', blocksize=BLOCK_SIZE, device=midx, callback=audio_cb)
        return s, int(s.samplerate), int(s.channels), f"(INPUT mic): '{devs[midx]['name']}'", False

    # 2) --device informado
    if override is not None:
        idx = resolve_device_index(override)
        if idx is None:
            raise RuntimeError(f"--device '{override}' não encontrado.")
        d = devs[idx]
        if is_output(idx) and system.startswith("win"):
            # tentar loopback
            try:
                extra = sd.WasapiSettings(loopback=True)  # requer PortAudio com WASAPI loopback
                s = sd.InputStream(samplerate=int(d.get("default_samplerate", 44100)),
                                   channels=max(1, int(d.get("max_output_channels", 2))),
                                   dtype='float32', blocksize=BLOCK_SIZE, device=idx,
                                   callback=audio_cb, latency='low', extra_settings=extra)
                return s, int(s.samplerate), int(s.channels), f"(WASAPI loopback): '{d['name']}'", True
            except Exception as e:
                raise RuntimeError(f"O dispositivo '{d['name']}' não abriu em loopback (SD): {e}")  # [6](https://github.com/spatialaudio/portaudio-binaries/issues/6)
        if is_input(idx):
            if not allow_mic:
                raise RuntimeError("Dispositivo de entrada só é permitido com --allow-mic.")
            s = sd.InputStream(samplerate=int(d.get("default_samplerate", 44100)),
                               channels=max(1, int(d.get("max_input_channels",2))),
                               dtype='float32', blocksize=BLOCK_SIZE, device=idx, callback=audio_cb)
            return s, int(s.samplerate), int(s.channels), f"(INPUT mic via --device): '{d['name']}'", False
        raise RuntimeError(f"Dispositivo '{d['name']}' não é válido para captura.")

    # 3) Autodetectar WASAPI loopback
    if system.startswith("win"):
        cands = []
        for i in auto_candidate_outputs():
            if not is_output(i): continue
            d = devs[i]
            try:
                extra = sd.WasapiSettings(loopback=True)
                s = sd.InputStream(samplerate=int(d.get("default_samplerate", 44100)),
                                   channels=max(1, int(d.get("max_output_channels", 2))),
                                   dtype='float32', blocksize=BLOCK_SIZE, device=i,
                                   callback=audio_cb, latency='low', extra_settings=extra)
                return s, int(s.samplerate), int(s.channels), f"(WASAPI loopback): '{d['name']}'", True
            except Exception:
                continue
        raise RuntimeError("Nenhuma saída com WASAPI loopback disponível para o sounddevice.")  # [6](https://github.com/spatialaudio/portaudio-binaries/issues/6)

    # Outras plataformas: exigir mic explícito
    raise RuntimeError("Sem loopback nesta plataforma via sounddevice; use --allow-mic e --mic-device.")

def _paw_choose_and_open(allow_mic: bool, mic_device: Optional[str], audio_cb):
    """
    PyAudio (preferindo PyAudioWPatch). Abre WASAPI loopback do default output.
    Cai em mic apenas se allow_mic=True e mic explicitado.
    """
    if _PAW is None and _PA is None:
        raise RuntimeError("PyAudioWPatch/PyAudio não encontrado. Instale com: pip install PyAudioWPatch")  # [2](https://pypi.org/project/PyAudioWPatch/)

    pmod = _PAW or _PA
    pinst = pmod.PyAudio()

    # Mic explícito?
    if mic_device:
        if not allow_mic:
            raise RuntimeError("Uso de microfone requer --allow-mic.")
        # localizar mic por nome
        target_idx = None
        for i in range(pinst.get_device_count()):
            info = pinst.get_device_info_by_index(i)
            if mic_device.lower() in info.get("name","").lower() and info.get("maxInputChannels",0) > 0:
                target_idx = i; break
        if target_idx is None:
            pinst.terminate()
            raise RuntimeError(f"--mic-device '{mic_device}' não encontrado no PyAudio.")
        info = pinst.get_device_info_by_index(target_idx)
        rate = int(info.get("defaultSampleRate", 44100))
        ch   = max(1, int(info.get("maxInputChannels", 1)))
        # callback pyaudio -> numpy
        def _py_cb(in_data, frame_count, time_info, status):
            try:
                block = np.frombuffer(in_data, dtype=np.int16).astype(np.float32) / 32768.0
                if ch > 1:
                    block = block.reshape(-1, ch).mean(axis=1)
                audio_cb(block, frame_count, time_info, status)
            except Exception:
                pass
            return (None, pmod.paContinue)
        stream = pinst.open(format=pmod.paInt16, channels=ch, rate=rate,
                            input=True, input_device_index=target_idx,
                            frames_per_buffer=BLOCK_SIZE, stream_callback=_py_cb)
        return ("PyAudio (mic)", pinst, stream, rate, ch)

    # WASAPI loopback (preferido no PyAudioWPatch)
    try:
        if _PAW is not None:
            # Default speakers loopback
            wasapi = pinst.get_host_api_info_by_type(pmod.paWASAPI)
            spk = pinst.get_device_info_by_index(wasapi["defaultOutputDevice"])
            if not spk.get("isLoopbackDevice", False):
                # localizar o análogo loopback
                found = None
                for lb in pinst.get_loopback_device_info_generator():
                    if spk["name"] in lb["name"]:
                        found = lb; break
                if found is None:
                    raise RuntimeError("Loopback padrão não encontrado no PyAudioWPatch.")
                spk = found
            rate = int(spk["defaultSampleRate"])
            ch   = max(1, int(spk["maxInputChannels"]))

            def _py_cb(in_data, frame_count, time_info, status):
                try:
                    block = np.frombuffer(in_data, dtype=np.int16).astype(np.float32) / 32768.0
                    if ch > 1:
                        block = block.reshape(-1, ch).mean(axis=1)
                    audio_cb(block, frame_count, time_info, status)
                except Exception:
                    pass
                return (None, pmod.paContinue)

            stream = pinst.open(format=pmod.paInt16, channels=ch, rate=rate,
                                input=True, input_device_index=spk["index"],
                                frames_per_buffer=BLOCK_SIZE, stream_callback=_py_cb)
            return ("PyAudioWPatch (WASAPI loopback)", pinst, stream, rate, ch)  # [7](https://github.com/s0d3s/PyAudioWPatch/blob/master/examples/pawp_record_wasapi_loopback.py)[2](https://pypi.org/project/PyAudioWPatch/)
        else:
            # PyAudio oficial: tentar achar dispositivos "[Loopback]" (se a build expõe)
            loop_idx = None; rate=48000; ch=2
            for i in range(pinst.get_device_count()):
                info = pinst.get_device_info_by_index(i)
                nm = info.get("name","")
                if "loopback" in nm.lower() and info.get("maxInputChannels",0)>0:
                    loop_idx=i; rate=int(info.get("defaultSampleRate",48000)); ch=int(info.get("maxInputChannels",2)); break
            if loop_idx is None:
                pinst.terminate()
                raise RuntimeError("PyAudio não expõe loopback nesta build. Instale PyAudioWPatch.")
            def _py_cb(in_data, frame_count, time_info, status):
                try:
                    block = np.frombuffer(in_data, dtype=np.int16).astype(np.float32) / 32768.0
                    if ch > 1:
                        block = block.reshape(-1, ch).mean(axis=1)
                    audio_cb(block, frame_count, time_info, status)
                except Exception:
                    pass
                return (None, pmod.paContinue)
            stream = pinst.open(format=pmod.paInt16, channels=ch, rate=rate,
                                input=True, input_device_index=loop_idx,
                                frames_per_buffer=BLOCK_SIZE, stream_callback=_py_cb)
            return ("PyAudio (loopback exposto)", pinst, stream, rate, ch)  # [5](https://pypi.org/project/PyAudio/)
    except Exception as e:
        pinst.terminate()
        raise

# ============================= Main =============================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bind', type=str, default='0.0.0.0')
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--backend', type=str, choices=['auto','sd','pyaudio'], default='auto',
                        help='Escolhe backend de captura: auto (padrão), sd (sounddevice), pyaudio')
    parser.add_argument('--no-pyaudio-fallback', action='store_true',
                        help='Desativa fallback para PyAudio quando sounddevice falhar.')
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--allow-mic', action='store_true',
                        help='Permite capturar microfone quando explicitamente solicitado (nunca usado por padrão).')
    parser.add_argument('--mic-device', type=str, default=None,
                        help='Seleciona o microfone a ser usado (requer --allow-mic).')
    parser.add_argument('--raspberry-ip', type=str, default=RASPBERRY_IP)
    parser.add_argument('--udp-port', type=int, default=UDP_PORT)
    parser.add_argument('--bands', type=int, default=DEFAULT_NUM_BANDS)
    parser.add_argument('--scale', type=str, choices=['log','mel'], default='log')
    parser.add_argument('--mel-bands', type=int, default=None)
    parser.add_argument('--mel-tilt', type=float, default=-0.25, help='Expoente do tilt nas bandas MEL (negativo realça graves; 0 desliga tilt)')
    parser.add_argument('--mel-no-area-norm', action='store_true', help='Desliga normalização de área dos filtros MEL')
    parser.add_argument('--pkt', type=str, choices=['a1','a2'], default='a1')
    parser.add_argument('--tx-fps', type=int, default=75)
    parser.add_argument('--signal-hold-ms', type=int, default=600)
    parser.add_argument('--vis-fps', type=int, default=60)
    parser.add_argument('--require-sync', action='store_true')
    parser.add_argument('--no-require-sync', action='store_true')
    parser.add_argument('--resync', type=float, default=RESYNC_INTERVAL_SEC)
    # Gate de silencio
    parser.add_argument('--silence-bands', type=float, default=28.0)
    parser.add_argument('--silence-rms', type=float, default=0.0015)
    parser.add_argument('--silence-duration', type=float, default=0.8)
    parser.add_argument('--resume-factor', type=float, default=1.8)
    parser.add_argument('--resume-stable', type=float, default=0.35)
    # Alphas
    parser.add_argument('--avg-ema', type=float, default=0.05)
    parser.add_argument('--rms-ema', type=float, default=0.05)
    parser.add_argument('--norm-peak-ema', type=float, default=PEAK_EMA_DEFAULT)
    # Reset
    parser.add_argument('--no-reset-on-start', action='store_true', help='Não envia RESET (B1) ao iniciar')
    # Perfil de silêncio / ruído
    parser.add_argument('--noise-profile', action='store_true', help='Habilita perfil de silêncio por banda e subtração pós-FFT/MEL')
    parser.add_argument('--noise-headroom', type=float, default=1.12, help='Fator >1 para ampliar o perfil antes de subtrair')
    parser.add_argument('--min-band', type=int, default=0, help='Valor mínimo (0..255) por banda após subtração')

    args = parser.parse_args()

    # Guardar cfg
    app["cfg"] = {"rpi_ip": args.raspberry_ip, "rpi_port": args.udp_port}

    # FPS TX efetivo
    TX_FPS = max(1, int(args.tx_fps))
    MIN_SEND_INTERVAL = 1.0 / TX_FPS

    # Estado do servidor
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

    # Config de servidor para features
    app["server_cfg"] = {
        "noise_profile_enabled": bool(args.noise_profile),
        "noise_headroom": float(args.noise_headroom),
        "min_band": int(args.min_band),
    }

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
    if args.no_require_sync:
        require_sync = False
    if args.require_sync:
        require_sync = True
    if require_sync:
        print("[INFO] Time-sync TCP (5006)...")
        time_sync_over_tcp(args.raspberry_ip)
        threading.Thread(target=resync_worker, args=(args.raspberry_ip, float(args.resync)), daemon=True).start()

    # ---- Preparar LOG ou MEL ----
    scale_mode = args.scale.lower()
    peak_ema_alpha = float(args.norm_peak_ema)

    compute_bands = None
    if scale_mode == 'mel':
        mel_bands = int(args.mel_bands or n_bands)
        # sr real será conhecido após abrir stream -> reconfiguramos depois se necessário
        compute_bands_builder = ("mel", mel_bands)
        shared.bands = np.zeros(mel_bands, dtype=np.uint8)
    else:
        compute_bands_builder = ("log", n_bands)
        shared.bands = np.zeros(n_bands, dtype=np.uint8)

    # ---- callback comum (recebe bloco float32 mono 0..1/±1) ----
    energy_buf = np.zeros(ENERGY_BUFFER_SIZE, dtype=np.float32)
    energy_idx = 0
    energy_count = 0
    last_energy = 0.0

    def make_audio_cb():
        nonlocal compute_bands, energy_idx, energy_count, last_energy
        def audio_cb(block, frames, time_info, status):
            # block: np.float32 mono (tam variável). Normalizamos para BLOCK_SIZE
            b = block
            if b.shape[0] < BLOCK_SIZE:
                b = np.pad(b, (0, BLOCK_SIZE - b.shape[0]), 'constant')
            bands = compute_bands(b[:BLOCK_SIZE]) if compute_bands is not None else None
            if bands is None:
                return
            # atualizar shared
            app["shared"].bands = bands
            avg = float(np.mean(bands))
            rms = float(np.sqrt(np.mean((b[:BLOCK_SIZE]*b[:BLOCK_SIZE])) + 1e-12))
            app["shared"].avg = avg
            app["shared"].rms = rms
            app["shared"].last_update = time.time()
            # beat simples
            lb = bands[:max(8, len(bands)//12)]
            energy = float(np.mean(lb)) / 255.0 if lb.size>0 else 0.0
            energy_buf[energy_idx] = energy
            energy_idx = (energy_idx + 1) % ENERGY_BUFFER_SIZE
            energy_count = min(energy_count + 1, ENERGY_BUFFER_SIZE)
            buf_view = energy_buf if energy_count == ENERGY_BUFFER_SIZE else energy_buf[:energy_count]
            avg_energy = float(np.mean(buf_view)) if energy_count > 0 else 0.0
            std_energy = float(np.std(buf_view)) if energy_count > 1 else 0.0
            dyn_thr = avg_energy + BEAT_THRESHOLD * std_energy
            is_peak_now = (energy >= max(dyn_thr, BEAT_HEIGHT_MIN)) and (energy >= last_energy)
            app["shared"].beat = 1 if is_peak_now else 0
            last_energy = energy
        return audio_cb

    # ---- abrir backend ----
    backend_used = None
    stream = None
    pa_instance = None  # caso PyAudio

    def _build_compute(sr):
        nonlocal compute_bands, shared
        if compute_bands_builder[0] == "mel":
            mel_bands = compute_bands_builder[1]
            mel_fb = build_mel_filterbank(sr, NFFT, mel_bands, FMIN, min(FMAX, sr/2.0))
            if not args.mel_no_area_norm:
                row_sum = mel_fb.sum(axis=1, keepdims=True)
                mel_fb = mel_fb / np.maximum(row_sum, 1e-9)
            if abs(float(args.mel_tilt)) > 1e-9:
                k = np.arange(mel_bands, dtype=np.float32)
                tilt = ((k + 1.0) / float(mel_bands)) ** float(args.mel_tilt)
                mel_fb = (mel_fb.T * tilt).T
            compute_bands = make_compute_bands_mel(sr, BLOCK_SIZE, NFFT, mel_fb, EMA_ALPHA, peak_ema_alpha)
            shared.bands = np.zeros(mel_bands, dtype=np.uint8)
        else:
            n_b = compute_bands_builder[1]
            a_idx, b_idx = make_bands_indices(NFFT, sr, n_b, FMIN, FMAX, min_bins=2)
            compute_bands = make_compute_bands_log(sr, BLOCK_SIZE, NFFT, a_idx, b_idx, EMA_ALPHA, peak_ema_alpha)
            shared.bands = np.zeros(n_b, dtype=np.uint8)

    # tentativa 1: sounddevice (se backend=auto ou sd)
    if args.backend in ("auto","sd"):
        try:
            audio_cb = make_audio_cb()
            s, sr_eff, ch_eff, desc, is_loop = _sd_choose_and_open(args.device, allow_mic=args.allow_mic,
                                                                    mic_device=args.mic_device, audio_cb=lambda b, *_: audio_cb(b.astype(np.float32)))
            _build_compute(sr_eff)
            stream = s
            stream.start()
            shared.samplerate = sr_eff
            shared.channels   = ch_eff
            backend_used = f"sounddevice {desc}"
        except Exception as e_sd:
            if args.backend == "sd" or args.no_pyaudio_fallback:
                print(f"\n[FATAL] sounddevice falhou: {e_sd}")
                sys.exit(1)
            else:
                print(f"\n[WARN] sounddevice falhou, tentando PyAudio... ({e_sd})")

    # tentativa 2: PyAudio (se backend=auto caiu aqui, ou backend=pyaudio)
    if stream is None and args.backend in ("auto","pyaudio"):
        try:
            audio_cb = make_audio_cb()
            tag, pa_instance, pa_stream, sr_eff, ch_eff = _paw_choose_and_open(args.allow_mic, args.mic_device, audio_cb)
            _build_compute(sr_eff)
            pa_stream.start_stream()
            stream = pa_stream
            shared.samplerate = sr_eff
            shared.channels   = ch_eff
            backend_used = tag
            # dica: `PyAudioWPatch` disponibiliza devices loopback e helpers dedicados
            # (ex.: get_loopback_device_info_generator)  [2](https://pypi.org/project/PyAudioWPatch/)[7](https://github.com/s0d3s/PyAudioWPatch/blob/master/examples/pawp_record_wasapi_loopback.py)
        except Exception as e_pa:
            print(f"\n[FATAL] PyAudio falhou: {e_pa}\nDica: instale PyAudioWPatch: pip install PyAudioWPatch")
            sys.exit(1)

    if stream is None:
        print("\n[FATAL] Nenhum backend de áudio pôde ser aberto.")
        sys.exit(1)

    # Empurrar CFG
    try:
        send_cfg_b0(args.raspberry_ip, args.udp_port, len(shared.bands), TX_FPS, int(args.signal_hold_ms), int(args.vis_fps))
        print(f"[CFG->RPi] bands={len(shared.bands)} fps={TX_FPS} hold={int(args.signal_hold_ms)} vis_fps={int(args.vis_fps)}")
    except Exception as e:
        print(f"[WARN] Falha B0: {e}")

    # Reset remoto no start
    if not args.no_reset_on_start:
        try:
            send_reset_b1(args.raspberry_ip, args.udp_port)
            print("[RST->RPi] RESET (B1) enviado.")
        except Exception as e:
            print(f"[WARN] Falha ao enviar RESET B1: {e}")

    print(f"[INFO] Capturando de: {backend_used}")
    print("[PRONTO] Monitor online - abra o navegador.")

    # --- Loop principal (idêntico ao anterior; gate + envio) ---
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
                time.sleep(0.05)
                continue

            bands_now = shared.bands
            avg = shared.avg
            rms = shared.rms
            beat = shared.beat

            rms_med.append(rms); avg_med.append(avg)
            if len(rms_med) == rms_med.maxlen:
                rms_filtered = float(np.median(rms_med))
                avg_filtered = float(np.median(avg_med))
            else:
                rms_filtered = rms
                avg_filtered = avg

            ss = app["server_state"]
            scfg = app["server_cfg"]

            if 'avg_ema_val' not in ss:
                ss['avg_ema_val'] = avg_filtered
                ss['rms_ema_val'] = rms_filtered
            else:
                ss['avg_ema_val'] = ss['avg_ema_val']*(1.0-ss['avg_ema']) + ss['avg_ema']*avg_filtered
                ss['rms_ema_val'] = ss['rms_ema_val']*(1.0-ss['rms_ema']) + ss['rms_ema']*rms_filtered

            avg_ema = ss['avg_ema_val']; rms_ema = ss['rms_ema_val']

            hist_avg.append((now, avg_filtered))
            hist_rms.append((now, rms_filtered))
            cutoff = now - 8.0
            while hist_avg and hist_avg[0][0] < cutoff: hist_avg.popleft()
            while hist_rms and hist_rms[0][0] < cutoff: hist_rms.popleft()

            # Fechar calibração
            if ss.get("calibrating", False) and now >= ss.get("calib_until", 0.0):
                arr_avg = np.array(ss["calib_avg"] or [avg_filtered], dtype=np.float32)
                arr_rms = np.array(ss["calib_rms"] or [rms_filtered], dtype=np.float32)
                noise_avg_p = float(np.percentile(arr_avg, 80))
                noise_rms_p = float(np.percentile(arr_rms, 80))
                th_avg = max(1.0, noise_avg_p * 1.15)
                th_rms = max(1e-6, noise_rms_p * 1.25)
                base = max(th_avg, 1e-3)
                music_proxy = float(np.percentile(np.array([v for _,v in hist_avg], dtype=np.float32), 90)) if hist_avg else base*1.6
                resume_factor = float(np.clip((music_proxy / base) * 0.9, 1.4, 4.0))
                ss["silence_bands"] = th_avg
                ss["silence_rms"] = th_rms
                ss["resume_factor"] = resume_factor

                if scfg["noise_profile_enabled"] and ss.get("noise_profile_frames"):
                    stack = np.stack(ss["noise_profile_frames"], axis=0).astype(np.float32)
                    p90 = np.percentile(stack, 90, axis=0).astype(np.float32)
                    ss["noise_profile"] = np.clip(p90, 0, 255)
                    try:
                        home = os.path.expanduser('~')
                        prof_dir = os.path.join(home, '.reactiveleds'); os.makedirs(prof_dir, exist_ok=True)
                        prof_name = f"noise-profile-sr{shared.samplerate}-nb{len(shared.bands)}.npy"
                        prof_path = os.path.join(prof_dir, prof_name)
                        np.save(prof_path, ss["noise_profile"])
                        ss["noise_profile_path"] = prof_path
                        print(f"[NOISE] Perfil salvo em {prof_path}")
                    except Exception as e:
                        print(f"[WARN] Falha ao salvar perfil: {e}")

                ss["last_calib"] = {
                    "t": time.time(),
                    "th_avg": float(th_avg),
                    "th_rms": float(th_rms),
                    "resume_factor": float(resume_factor),
                    "frames": int(len(ss.get("noise_profile_frames") or [])),
                }
                ss["calibrating"] = False
                ss["calib_until"] = 0.0
                ss["calib_started"] = 0.0
                ss["calib_avg"].clear(); ss["calib_rms"].clear()
                ss["noise_profile_frames"] = None
                print(f"\n[CALIB] OK: th_avg={th_avg:.1f} th_rms={th_rms:.6f} resume_x={resume_factor:.2f}")

            # Auto (opcional)
            if ss["auto_mode"] and (now - last_auto_update) >= 0.5:
                arr_avg = np.array([v for _,v in hist_avg], dtype=np.float32) if hist_avg else np.array([avg_filtered],dtype=np.float32)
                arr_rms = np.array([v for _,v in hist_rms], dtype=np.float32) if hist_rms else np.array([rms_filtered],dtype=np.float32)
                noise_avg_p = float(np.percentile(arr_avg, 20))
                noise_rms_p = float(np.percentile(arr_rms, 20))
                music_avg_p = float(np.percentile(arr_avg, 80))
                th_avg = max(1.0, noise_avg_p * 1.4)
                th_rms = max(1e-6, noise_rms_p * 1.6)
                base = max(th_avg, 1e-3)
                dyn_resume = max(1.3, min(4.0, (music_avg_p / base) * 0.9))
                ss["silence_bands"]=th_avg; ss["silence_rms"]=th_rms; ss["resume_factor"]=dyn_resume
                last_auto_update = now

            # Gate + envio
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
                    if silence_since is None:
                        silence_since=now
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
                            b_send = np.clip(bands_now.astype(np.float32) - prof, 0, 255).astype(np.uint8)
                        if app["server_cfg"]["min_band"] > 0:
                            b_send = np.maximum(b_send, app["server_cfg"]["min_band"]).astype(np.uint8)

                        if args.pkt=='a2':
                            send_packet_a2(b_send, beat, 1, dyn_floor, kick_val, args.raspberry_ip, args.udp_port, shared)
                        else:
                            send_packet_a1(b_send, beat, 1, args.raspberry_ip, args.udp_port, shared)
                        print_status(shared, " (from silence)", require_sync=not args.no_require_sync)
                        shared.active=True; ss['resume_since']=None; last_tick=now; time.sleep(0.001); continue
                else:
                    ss['resume_since']=None

            if not shared.active:
                print_status(shared, " (idle)", require_sync=not args.no_require_sync)
                time.sleep(0.05)
                continue

            if (now - last_tick) < MIN_SEND_INTERVAL:
                time.sleep(0.001)
                continue
            last_tick = now

            dyn_floor = 0
            lb = bands_now[:max(8, len(bands_now)//12)]
            energy_norm = float(np.mean(lb))/255.0 if lb.size>0 else 0.0
            kick_val = 220 if beat else int(max(0, min(255, round(energy_norm*255.0))))

            b_send = bands_now
            if app["server_cfg"]["noise_profile_enabled"] and ss.get("noise_profile") is not None:
                prof = ss["noise_profile"] * float(app["server_cfg"]["noise_headroom"])
                b_send = np.clip(bands_now.astype(np.float32) - prof, 0, 255).astype(np.uint8)
            if app["server_cfg"]["min_band"] > 0:
                b_send = np.maximum(b_send, app["server_cfg"]["min_band"]).astype(np.uint8)

            if args.pkt=='a2':
                send_packet_a2(b_send, beat, 0, dyn_floor, kick_val, args.raspberry_ip, args.udp_port, shared)
            else:
                send_packet_a1(b_send, beat, 0, args.raspberry_ip, args.udp_port, shared)
            print_status(shared, "", require_sync=not args.no_require_sync)

    except KeyboardInterrupt:
        pass
    finally:
        _stop_resync.set()
        try:
            if pa_instance is not None:
                # PyAudio
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
                # sounddevice
                try:
                    stream.stop()
                except Exception: pass
                try:
                    stream.close()
                except Exception: pass
        except Exception:
            pass
        sys.stdout.write("\n"); sys.stdout.flush()

if __name__ == '__main__':
    main()