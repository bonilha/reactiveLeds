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

# ------------------------------- Constantes -------------------------------
DEFAULT_RASPBERRY_IP = "192.168.66.71"
UDP_PORT = 5005
TCP_TIME_PORT = 5006

PKT_AUDIO_V2 = 0xA2  # [A2][8 ts_pc][bands(150)][beat][trans][dyn_floor][kick] => 163 bytes
PKT_CFG      = 0xB0  # [B0][ver u8][num_bands u16][fps u16][signal_hold_ms u16][vis_fps u16] => 10 bytes

# ------------------------------- Utils de dispositivo -------------------------------
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

# ----------------------------- FFT / Bandas ------------------------------
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

# ------------------------------- Contextos -------------------------------
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
        self.stop_flag = threading.Event()
        self.udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            self.udp_sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1 << 20)
        except Exception:
            pass

# ------------------------------- Time Sync -------------------------------
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

def resync_loop(ctx, interval_ok=60.0, interval_fail=2.0):
    """Loop que fica tentando sync para sempre (não bloqueia o main)."""
    while not ctx.stop_flag.is_set():
        ok = time_sync_over_tcp(ctx)
        wait = interval_ok if ok else interval_fail
        for _ in range(int(wait*10)):
            if ctx.stop_flag.is_set():
                return
            time.sleep(0.1)

# ------------------------------- Status -----------------------------------
def print_status(ctx, tag_extra: str = "", debug_line: str = ""):
    now = time.time()
    if (now - ctx.last_status) > 0.25:
        tag = "SYNC✓" if ctx.time_sync_ok else "WAITING SYNC"
        extra = f" {tag_extra}" if tag_extra else ""
        dbg = f" {debug_line}" if debug_line else ""
        sys.stdout.write(f"\rTX: {ctx.tx_count} {tag}{extra}{dbg}")
        sys.stdout.flush()
        ctx.last_status = now

# ------------------------------- Main -------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--ip", type=str, default=DEFAULT_RASPBERRY_IP)
    parser.add_argument("--block-size", type=int, default=512)
    parser.add_argument("--fps", type=int, default=75)               # cadência de envio
    parser.add_argument("--bands", type=int, default=150)            # número de bandas
    parser.add_argument("--signal-hold", type=int, default=500)      # sustain no Pi (ms)
    parser.add_argument("--vis-fps", type=int, default=45)           # FPS de render do Pi
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

    # Contexto PC (estado)
    ctx = Ctx(args.ip, MAX_FPS)
    shared = Shared(NUM_BANDS)

    print(f"[CFG] bands={NUM_BANDS} fps={MAX_FPS} hold={SIGNAL_HOLD_MS}ms vis_fps={VIS_FPS} -> PC pipeline")
    print("[INFO] Iniciando time-sync TCP (porta 5006)...")

    # Thread de time-sync contínuo (não bloqueia o main)
    threading.Thread(target=resync_loop, args=(ctx, float(args.resync), 2.0), daemon=True).start()

    # ---------------- Envio de CONFIG (B0) ----------------
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
    # -----------------------------------------------------

    # ---------------- Pipeline de Áudio ------------------
    FMIN = 20.0; FMAX = 16000.0
    compute_bands = None
    raw_ema_alpha = float(args.raw_ema)
    peak_ema_alpha = float(args.norm_peak_ema)

    # Equalização e smoothing pós-EQ (iguais à versão estável)
    tilt_curve = np.exp(np.linspace(np.log(args.tilt_max), np.log(args.tilt_min), NUM_BANDS)).astype(np.float32)
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
    last_debug = 0.0

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

        eq_alpha = float(args.eq_alpha)
        band_ema = (1.0 - eq_alpha) * band_ema + eq_alpha * base_vals255
        gain = args.eq_target / np.maximum(band_ema, 1.0)
        eq = base_vals255 * gain * tilt_curve
        eq = np.clip(eq, 0.0, 255.0).astype(np.float32)

        if args.post_attack < 1.0 or args.post_release < 1.0:
            alpha = np.where(eq > post_eq_prev, args.post_attack, args.post_release).astype(np.float32)
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

    a_idx, b_idx = make_bands_indices(args.block_size, sr_eff, NUM_BANDS, 20.0, 16000.0)
    compute_bands = make_compute_bands(sr_eff, args.block_size, a_idx, b_idx, raw_ema_alpha, peak_ema_alpha)

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

            # Gate de silêncio
            is_quiet = (avg_ema < args.silence_bands) and (rms_ema < args.silence_rms)
            resume_threshold = args.silence_bands * args.resume_factor

            if active:
                if is_quiet:
                    if silence_since is None:
                        silence_since = now
                    elif (now - silence_since) >= args.silence_duration:
                        send_audio_packet(np.zeros(NUM_BANDS, dtype=np.uint8), 0, 1, 0, 0)
                        active = False; silence_since = None; resume_since = None
                        time.sleep(0.01)
                        continue
                else:
                    silence_since = None
            else:
                if (avg_ema > resume_threshold) or (rms_ema > args.silence_rms * 3.0):
                    if resume_since is None:
                        resume_since = now
                    elif (now - resume_since) >= args.resume_stable:
                        mean_val = float(np.mean(bands_now))
                        dyn_floor = int(min(12, mean_val * 0.012)) if mean_val > 90 else 0
                        send_audio_packet(bands_now, beat, 1, dyn_floor, kick_intensity)
                        active = True; resume_since = None; last_tick = now
                        time.sleep(0.001); continue
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