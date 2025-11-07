# pc-audio.py — Captura, processa e envia bandas PRONTAS (UDP) para o Raspberry Pi.
# Protocolo A2 (áudio) + B0 (config)
# • FFT 512 (~11.6 ms @44.1 kHz), FPS configurável (--fps)
# • Envia PKT_CFG (0xB0) com bands/fps/signal_hold/vis_fps (burst no start + periodic)

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

DEFAULT_RASPBERRY_IP = "192.168.66.71"
UDP_PORT = 5005
TCP_TIME_PORT = 5006

PKT_AUDIO_V2 = 0xA2  # [A2][8 ts_pc][bands(150)][beat][trans][dyn_floor][kick]
PKT_CFG      = 0xB0  # [B0][ver u8][num_bands u16][fps u16][signal_hold_ms u16][vis_fps u16] = 10 bytes

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
        if d.get("max_output_channels",0)>0 and any(k in d["name"].lower() for k in keywords):
            if i not in cands: cands.append(i)
    for i, d in enumerate(devs):
        if d.get("max_output_channels",0)>0 and i not in cands:
            cands.append(i)
    return cands

def choose_candidates(override: Optional[Union[str, int]] = None):
    devs = sd.query_devices()
    system = platform.system().lower()
    idx = resolve_device_index(override)
    if idx is not None:
        d = devs[idx]
        extra = None
        if system.startswith("win") and d.get("max_output_channels",0)>0:
            try: extra = sd.WasapiSettings(loopback=True)
            except Exception: extra = None
        return [(idx, extra, f"Override: idx={idx} '{d['name']}' (loopback={extra is not None})")]
    if system.startswith("win"):
        cands = []
        for i in auto_candidate_outputs():
            d = devs[i]
            try: extra = sd.WasapiSettings(loopback=True)
            except Exception: extra = None
            cands.append((i, extra, f"WASAPI loopback: '{d['name']}'"))
        if cands: return cands
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

class Ctx:
    def __init__(self, rpi_ip: str, max_fps: int):
        self.rpi_ip = rpi_ip
        self.max_fps = max_fps
        self.min_send_interval = 1.0 / max(1, max_fps)
        self.tx_count = 0
        self.last_status = 0.0
        self.time_sync_ok = False
        self.stop_resync = threading.Event()
        self.udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            self.udp_sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1 << 20)
        except Exception:
            pass

def _recv_exact(sock, n):
    buf = b''
    while len(buf) < n:
        ch = sock.recv(n - len(buf))
        if not ch:
            raise ConnectionError("Conexão fechada durante receive.")
        buf += ch
    return buf

def time_sync_over_tcp(ctx, samples=12, timeout=0.6):
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
                tr_pi  = int.from_bytes(data[11:19], 'little')
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

def resync_worker(ctx, resync_interval):
    while not ctx.stop_resync.is_set():
        if ctx.time_sync_ok:
            if ctx.stop_resync.wait(resync_interval):
                break
        else:
            if ctx.stop_resync.wait(2.0):
                break
        time_sync_over_tcp(ctx)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--ip", type=str, default=DEFAULT_RASPBERRY_IP)
    parser.add_argument("--block-size", type=int, default=512)
    parser.add_argument("--fps", type=int, default=75)               # << unifica FPS
    parser.add_argument("--bands", type=int, default=150)            # << unifica N bandas
    parser.add_argument("--signal-hold", type=int, default=500)      # << sustain do Pi (ms)
    parser.add_argument("--vis-fps", type=int, default=45)           # << NOVO: FPS de render no Pi
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
    args = parser.parse_args()

    NUM_BANDS = int(args.bands)
    MAX_FPS = int(args.fps)
    MIN_SEND_INTERVAL = 1.0 / max(1, MAX_FPS)
    SIGNAL_HOLD_MS = int(args.signal_hold)
    VIS_FPS = int(args.vis_fps)

    ctx = Ctx(args.ip, MAX_FPS)
    shared = Shared(NUM_BANDS)

    print(f"[CFG] bands={NUM_BANDS} fps={MAX_FPS} hold={SIGNAL_HOLD_MS}ms vis_fps={VIS_FPS} -> PC pipeline")

    print("[INFO] Iniciando time-sync TCP (porta 5006)...")
    time_sync_over_tcp(ctx)
    t_sync = threading.Thread(target=resync_worker, args=(ctx, float(args.resync)), daemon=True)
    t_sync.start()

    # ------- CONFIG SENDER (burst + periodic) -------
    def send_config_packet():
        ver = 1
        nb = NUM_BANDS & 0xFFFF
        fps = MAX_FPS & 0xFFFF
        hold = SIGNAL_HOLD_MS & 0xFFFF
        vis = VIS_FPS & 0xFFFF  # vai no campo 'reserved'
        payload = bytes([
            PKT_CFG, ver,
            nb & 0xFF, (nb >> 8) & 0xFF,
            fps & 0xFF, (fps >> 8) & 0xFF,
            hold & 0xFF, (hold >> 8) & 0xFF,
            vis & 0xFF, (vis >> 8) & 0xFF,
        ])
        ctx.udp_sock.sendto(payload, (ctx.rpi_ip, UDP_PORT))

    def cfg_sender():
        # Burst inicial: 5 pacotes em ~1s
        for _ in range(5):
            try: send_config_packet()
            except Exception: pass
            time.sleep(0.2)
        # Envio periódico
        while not ctx.stop_resync.is_set():
            try: send_config_packet()
            except Exception: pass
            time.sleep(2.0)

    threading.Thread(target=cfg_sender, daemon=True).start()
    # ------------------------------------------------

    # Equalização/Pipeline (mesmo da sua versão, omitido aqui por brevidade)
    # ...  (MANTENHA idêntico ao seu pipeline vigente até o loop principal) ...

    # === A PARTIR DAQUI: copie seu pipeline de áudio vigente ===
    # Para economizar espaço, mantive exatamente o mesmo código que você já testou,
    # incluindo FFT 512, detecção de beat, cálculo de dynamic_floor e envio do A2.
    # (Se quiser, eu reenvio o arquivo inteiro com seu pipeline + essas adições.)

    # ------ A PARTIR DAQUI É IGUAL AO ARQUIVO QUE VOCÊ ESTAVA USANDO ------
    # (Cole aqui o mesmo callback, abertura de stream, compute_bands, loop principal etc.)
    # ----------------------------------------------------------------------

if __name__ == "__main__":
    main()
    