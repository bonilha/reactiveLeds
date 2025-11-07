# pc-audio.py - Captura áudio do PC e envia bandas (UDP) para o Raspberry Pi.
# MODO ESTRITO (sem legado/fallback):
# • NÃO transmite enquanto o time-sync NÃO estiver OK (✓).
# • Gate de silêncio NO LOOP PRINCIPAL:
#   - Ao detectar silêncio sustentado: envia 1 pacote zerado (transition=1) e PAUSA TX.
#   - Ao detectar retomada sustentada: envia 1º pacote real com transition=1 e retoma TX.
# • UDP/5005: PKT_AUDIO (0xA1) + [8 ts_pc(ns)] + [150 bands] + [beat] + [trans]
# • TCP/5006: time-sync cliente (PC) com offset signed=True + re-sync periódico.
#
# Áudio:
# • FFT 1024 (~23 ms @44.1 kHz), 50 FPS, EMA rápida, beat focado em sub/baixo.
#
# CLI:
# --debug                    : imprime métricas (RMS/AVG/estado) a cada ~250 ms
# --silence-bands <float>    : limiar de média de bandas para silêncio (default 8.0)
# --silence-rms <float>      : limiar de RMS para silêncio (default 3e-4)
# --silence-duration <float> : tempo (s) abaixo dos limiares para assumir silêncio (default 0.8)
# --resume-factor <float>    : histerese do AVG para retomar (default 2.0)
# --resume-stable <float>    : tempo (s) acima do limiar para retomar (default 0.4)
# --resync <int>             : re-sync periódico (s) (default 60)
# --avg-ema <float>          : alpha EMA do AVG usado no gate (default 0.20)
# --rms-ema <float>          : alpha EMA do RMS usado no gate (default 0.20)
# --norm-peak-ema <float>    : alpha EMA do pico de normalização espectral (default 0.10)

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
import logging  # Para debug log

# ------------------------------ Configs ------------------------------
RASPBERRY_IP = "192.168.66.71"
UDP_PORT = 5005
TCP_TIME_PORT = 5006

NUM_BANDS = 150
BLOCK_SIZE = 1024
FMIN = 20.0
FMAX = 16000.0
EMA_ALPHA = 0.35

# Beat detection
ENERGY_BUFFER_SIZE = 8
BEAT_THRESHOLD = 1.15
BEAT_HEIGHT_MIN = 0.08

# Envio
MAX_FPS = 75
MIN_SEND_INTERVAL = 1.0 / MAX_FPS

# Time-sync (estrito)
REQUIRE_TIME_SYNC = True
RESYNC_INTERVAL_SEC = 60.0
RESYNC_RETRY_INTERVAL = 2.0
TIME_SYNC_SAMPLES = 12

# Protocolo
PKT_AUDIO = 0xA1  # [A1][8 ts_pc][bands(150)][beat][trans] => 161 bytes

# UDP socket
udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
try:
    udp_sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1 << 20)
except Exception:
    pass

# Estado global
tx_count = 0
_last_status = 0.0
_time_sync_ok = False
_stop_resync = threading.Event()

def print_status(tag_extra: str = "", debug_line: str = ""):
    """Status de TX + sync (+ opcional debug em linha única)."""
    global _last_status, _time_sync_ok, tx_count
    now = time.time()
    if (now - _last_status) > 0.25:
        tag = "SYNC✓" if _time_sync_ok else "WAITING SYNC"
        extra = f" {tag_extra}" if tag_extra else ""
        dbg = f" {debug_line}" if debug_line else ""
        sys.stdout.write(f"\rTX: {tx_count} {tag}{extra}{dbg}")
        sys.stdout.flush()
        _last_status = now

# ------------------------------ Dispositivos (auto) ------------------------------
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
    keywords = ["speaker", "alto-falante", "headphone", "fone", "realtek", "echo", "dot", "hdmi", "display audio"]
    for i, d in enumerate(devs):
        if d.get("max_output_channels", 0) > 0 and any(k in d["name"].lower() for k in keywords):
            if i not in cands:
                cands.append(i)
    for i, d in enumerate(devs):
        if d.get("max_output_channels", 0) > 0 and i not in cands:
            cands.append(i)
    return cands

def choose_candidates(override: Optional[Union[str,int]]=None):
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
        if cands:
            return cands
    default_in = sd.default.device[0]
    if isinstance(default_in, int) and default_in >= 0:
        d = devs[default_in]
        return [(default_in, None, f"Default INPUT: idx={default_in} '{d['name']}'")]
    return [(None, None, "PortAudio default (fallback)")]

def open_with_probe(dev_idx, extra, callback):
    attempts = [(48000,2),(48000,1),(44100,2),(44100,1)]
    last_err = None
    for sr_try, ch_try in attempts:
        try:
            s = sd.InputStream(channels=ch_try, samplerate=sr_try, dtype="float32",
                               blocksize=BLOCK_SIZE, device=dev_idx,
                               callback=callback, extra_settings=extra)
            return s, int(s.samplerate), ch_try, f"(opened sr={int(s.samplerate)} ch={ch_try})"
        except Exception as e:
            last_err = e
    try:
        s = sd.InputStream(channels=2, samplerate=44100, dtype="float32",
                           blocksize=BLOCK_SIZE, device=None, callback=callback)
        return s, int(s.samplerate), 2, "(fallback default INPUT sr=44100 ch=2)"
    except Exception as e:
        last_err = e
    raise last_err if last_err else RuntimeError("Falha ao abrir InputStream.")

# ------------------------------ Bandas / FFT ------------------------------
def make_bands_indices(nfft, sr, num_bands, fmin, fmax_limit):
    """Retorna arrays vetorizados de inícios/fins por banda (inclusive-exclusivo)."""
    freqs = np.fft.rfftfreq(nfft, 1.0/sr)
    fmax = min(fmax_limit, sr/2.0)
    edges = np.geomspace(fmin, fmax, num_bands+1)
    # Vetorizado: busca dos índices via searchsorted para todas as bordas
    edge_idx = np.searchsorted(freqs, edges, side="left").astype(np.int32)
    a_idx = edge_idx[:-1]
    b_idx = edge_idx[1:]
    b_idx = np.maximum(b_idx, a_idx + 1)
    return a_idx, b_idx

def make_compute_bands(sr, block_size, band_starts, band_ends, ema_alpha, peak_ema_alpha):
    """
    Computa bandas de forma vetorizada:
    1) FFT real (rfft)
    2) magnitude -> médias por banda via soma cumulativa e diferenças (sem loops)
    3) log1p + normalização por EMA do pico (sem percentil caro)
    4) EMA por banda
    """
    window = np.hanning(block_size).astype(np.float32)
    ema_bands = np.zeros(len(band_starts), dtype=np.float32)
    # EMA do pico de normalização (começa em 1.0 para evitar divisão por 0)
    peak_ema = 1.0

    # Para cumsum eficiente, criamos um vetor de zeros inicial para acesso tipo cs[b]-cs[a]
    def compute(block):
        nonlocal ema_bands, peak_ema
        x = block * window
        fft_mag = np.abs(np.fft.rfft(x, n=block_size)).astype(np.float32)

        # Soma cumulativa prefixada (cs[0]=0), médias por banda (b-a)
        cs = np.concatenate(([0.0], np.cumsum(fft_mag, dtype=np.float32)))
        sums = cs[band_ends] - cs[band_starts]
        lens = (band_ends - band_starts).astype(np.float32)
        means = sums / np.maximum(lens, 1.0)

        vals = np.log1p(means)  # compressão
        # Normalização barata via EMA do pico (max)
        cur_max = float(np.max(vals)) if vals.size else 1.0
        peak_ema = (1.0 - peak_ema_alpha) * peak_ema + peak_ema_alpha * max(cur_max, 1e-6)
        norm = max(peak_ema, 1e-6)
        vals = (vals / norm).clip(0.0, 1.0)

        # EMA por banda
        ema_bands = ema_alpha * vals + (1.0 - ema_alpha) * ema_bands
        return (ema_bands * 255.0).clip(0,255).astype(np.uint8)

    return compute

# ------------------------------ Time sync TCP (cliente) ------------------------------
def _recv_exact(sock, n):
    buf = b''
    while len(buf) < n:
        ch = sock.recv(n - len(buf))
        if not ch:
            raise ConnectionError("Conexão fechada durante receive.")
        buf += ch
    return buf

def time_sync_over_tcp(samples=TIME_SYNC_SAMPLES, timeout=0.6):
    """
    Cliente TCP do time-sync:
    envia 'TS1'+t0_pc(8) N vezes e recebe 'TS2'+t0_pc+tr_pi(8) (ambos unsigned).
    escolhe offset do menor RTT; envia 'TS3'+offset(8 signed=True); espera eco 'TS3'+offset(8 signed=True).
    Atualiza _time_sync_ok e retorna True/False.
    """
    global _time_sync_ok
    results = []
    try:
        with socket.create_connection((RASPBERRY_IP, TCP_TIME_PORT), timeout=1.8) as s:
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
                offset = tr_pi - mid  # pode ser NEGATIVO
                results.append((rtt, offset))
            if not results:
                _time_sync_ok = False
                print("\n[WARN] Time sync TCP falhou (sem amostras válidas).")
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

def resync_worker(resync_interval):
    """Re-sync: se não sincronizado, tenta a cada 2 s; se OK, refaz a cada resync_interval."""
    while not _stop_resync.is_set():
        if _time_sync_ok:
            if _stop_resync.wait(resync_interval):
                break
        else:
            if _stop_resync.wait(RESYNC_RETRY_INTERVAL):
                break
        time_sync_over_tcp()

# ------------------------------ Estado compartilhado pelo callback ------------------------------
class Shared:
    def __init__(self, n_bands):
        self.bands = np.zeros(n_bands, dtype=np.uint8)
        self.beat = 0
        self.avg = 0.0
        self.rms = 0.0
        self.avg_ema = 0.0
        self.rms_ema = 0.0
        self.last_update = 0.0
        self.block_max = 0.0  # Novo: max abs de block raw

shared = Shared(NUM_BANDS)

# ------------------------------ Main ------------------------------
def main():
    global tx_count, RESYNC_INTERVAL_SEC

    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", help="Imprime RMS/AVG/estado do gate periodicamente.")
    parser.add_argument("--silence-bands", type=float, default=150.0, help="Limiar de média de bandas para silêncio (ajustado para spikes).")
    parser.add_argument("--silence-rms", type=float, default=0.003, help="Limiar de RMS para silêncio (ajustado para max spike).")
    parser.add_argument("--silence-duration", type=float, default=0.8)
    parser.add_argument("--resume-factor", type=float, default=2.0)
    parser.add_argument("--resume-stable", type=float, default=0.4)
    parser.add_argument("--resync", type=float, default=RESYNC_INTERVAL_SEC)
    # Novos ajustes
    parser.add_argument("--avg-ema", type=float, default=0.15, help="Alpha EMA do AVG usado no gate (reduzido para spikes).")
    parser.add_argument("--rms-ema", type=float, default=0.15, help="Alpha EMA do RMS usado no gate (reduzido para spikes).")
    parser.add_argument("--norm-peak-ema", type=float, default=0.25, help="Alpha da EMA do pico para normalização espectral.")
    parser.add_argument("--debug-rms", type=str, default=None, help="Log RMS/AVG/active/block_max to file.log")

    args = parser.parse_args()

    if args.debug_rms:
        logging.basicConfig(filename=args.debug_rms, level=logging.INFO, format='%(asctime)s %(message)s')

    SILENCE_THRESHOLD = args.silence_bands
    AMP_SILENCE_THRESHOLD = args.silence_rms
    SILENCE_DURATION = args.silence_duration
    RESUME_FACTOR = args.resume_factor
    RESUME_STABLE_TIME = args.resume_stable
    RESYNC_INTERVAL_SEC = float(args.resync)
    AVG_EMA_ALPHA = float(args.avg_ema)
    RMS_EMA_ALPHA = float(args.rms_ema)
    NORM_PEAK_EMA_ALPHA = float(args.norm_peak_ema)

    print("[INFO] Iniciando time-sync TCP (porta 5006)...")
    time_sync_over_tcp()
    t_sync = threading.Thread(target=resync_worker, args=(RESYNC_INTERVAL_SEC,), daemon=True); t_sync.start()

    # --------- Beat: buffer circular NumPy ---------
    energy_buf = np.zeros(ENERGY_BUFFER_SIZE, dtype=np.float32)
    energy_idx = 0
    energy_count = 0
    last_energy = 0.0  # para detecção de subida

    # Gate de silêncio (no loop principal)
    active = False
    silence_since = None
    resume_since = None
    last_tick = 0.0
    last_debug = 0.0

    # Filtros mediana para spikes (deque maxlen=7 para mais smoothing)
    rms_mediana = deque(maxlen=3)
    avg_mediana = deque(maxlen=3)

    def send_packet(bands_u8, beat_flag, transition_flag):
        """Envia um pacote (loop principal controla cadência e gating)."""
        global tx_count
        ts_ns = time.monotonic_ns()
        payload = bytes([PKT_AUDIO]) + ts_ns.to_bytes(8, 'little') \
                + bands_u8.tobytes() + bytes([beat_flag, transition_flag])
        udp_sock.sendto(payload, (RASPBERRY_IP, UDP_PORT))
        tx_count += 1

    # ------ callback apenas atualiza métricas e bands; NÃO decide envio ------
    band_starts = None
    band_ends = None
    compute_bands = None
    hann_window = np.hanning(BLOCK_SIZE).astype(np.float32)  # caso precise

    def audio_cb(indata, frames, time_info, status):
        nonlocal energy_idx, energy_count, last_energy
        if status:
            sys.stdout.write("\n"); print(f"[WARN] Audio status: {status}")

        # Mono mix
        block = indata.mean(axis=1) if indata.ndim > 1 else indata
        if len(block) < BLOCK_SIZE:
            block = np.pad(block, (0, BLOCK_SIZE - len(block)), "constant")

        if compute_bands is None:
            return

        bands = compute_bands(block[:BLOCK_SIZE])
        shared.bands = bands

        # AVG e RMS (com EMA para gate estável)
        avg = float(np.mean(bands))
        rms = float(np.sqrt(np.mean(block*block) + 1e-12))
        shared.block_max = float(np.max(np.abs(block))) if len(block) > 0 else 0.0  # Novo: max abs raw
        shared.avg_ema = shared.avg_ema*(1.0-AVG_EMA_ALPHA) + AVG_EMA_ALPHA*avg if shared.last_update>0 else avg
        shared.rms_ema = shared.rms_ema*(1.0-RMS_EMA_ALPHA) + RMS_EMA_ALPHA*rms if shared.last_update>0 else rms
        shared.avg = avg
        shared.rms = rms
        shared.last_update = time.time()

        # Beat (foco em graves) + buffer circular NumPy
        low_bands = bands[:max(8, NUM_BANDS//12)]
        energy = float(np.mean(low_bands)) / 255.0

        energy_buf[energy_idx] = energy
        energy_idx = (energy_idx + 1) % ENERGY_BUFFER_SIZE
        energy_count = min(energy_count + 1, ENERGY_BUFFER_SIZE)

        buf_view = energy_buf if energy_count == ENERGY_BUFFER_SIZE else energy_buf[:energy_count]
        avg_energy = float(np.mean(buf_view)) if energy_count > 0 else 0.0
        std_energy = float(np.std(buf_view)) if energy_count > 1 else 0.0
        dyn_thr = avg_energy + BEAT_THRESHOLD * std_energy

        # Pico "imediato": valor atual acima do anterior e acima dos limiares
        is_peak_now = (energy >= max(dyn_thr, BEAT_HEIGHT_MIN)) and (energy >= last_energy)
        shared.beat = 1 if is_peak_now else 0
        last_energy = energy

    # Abrir stream áudio
    cands = choose_candidates(override=None)
    stream = None; sr_eff = 44100
    for dev_idx, extra, desc in cands:
        try:
            stream, sr_eff, ch_eff, open_desc = open_with_probe(dev_idx, extra, audio_cb)
            sys.stdout.write(f"\n[INFO] Capturando de: {desc} {open_desc}\n")
            break
        except Exception as e:
            sys.stdout.write(f"\n[WARN] Falha ao abrir {desc}: {e}\n")
            continue
    if stream is None:
        sys.stdout.write("\n[FATAL] Não foi possível abrir nenhum dispositivo de áudio.\n")
        sys.exit(1)

    # Preparar bandas com SR efetivo (vetorizado)
    a_idx, b_idx = make_bands_indices(BLOCK_SIZE, sr_eff, NUM_BANDS, FMIN, FMAX)
    band_starts, band_ends = a_idx, b_idx
    compute_bands = make_compute_bands(sr_eff, BLOCK_SIZE, band_starts, band_ends, EMA_ALPHA, NORM_PEAK_EMA_ALPHA)

    stream.start()
    try:
        while True:
            # Sem time-sync: nenhum TX
            if REQUIRE_TIME_SYNC and (not _time_sync_ok):
                print_status(" (paused)")
                time.sleep(0.05)
                continue

            # Snapshot das métricas atuais
            bands_now = shared.bands
            avg = shared.avg
            rms = shared.rms
            avg_ema = shared.avg_ema
            rms_ema = shared.rms_ema
            beat = shared.beat
            block_max = shared.block_max
            now = time.time()

            # Aplicar filtro mediana (para spikes em rms/avg)
            rms_mediana.append(rms)
            avg_mediana.append(avg)
            if len(rms_mediana) == rms_mediana.maxlen:
                rms_filtered = np.median(rms_mediana)
                avg_filtered = np.median(avg_mediana)
            else:
                rms_filtered = rms
                avg_filtered = avg
            # Atualizar EMA com filtered
            avg_ema = avg_ema*(1.0-AVG_EMA_ALPHA) + AVG_EMA_ALPHA*avg_filtered if shared.last_update>0 else avg_filtered
            rms_ema = rms_ema*(1.0-RMS_EMA_ALPHA) + RMS_EMA_ALPHA*rms_filtered if shared.last_update>0 else rms_filtered

            # DEBUG: imprimir métricas periodicamente
            dbg = ""
            if args.debug and (now - last_debug) > 0.25:
                dbg = f"AVG={avg:5.1f} AVG_EMA={avg_ema:5.1f} RMS={rms:8.6f} RMS_EMA={rms_ema:8.6f} active={'Y' if active else 'N'}"
                last_debug = now
    
            # Gate de silêncio (usando EMA de AVG e RMS)
            is_quiet = (avg_ema < SILENCE_THRESHOLD) and (rms_ema < AMP_SILENCE_THRESHOLD)
            resume_threshold = SILENCE_THRESHOLD * RESUME_FACTOR

            if args.debug_rms:
                logging.info(f"RMS_raw={rms:.8f} RMS_ema={rms_ema:.8f} AVG_raw={avg:.1f} AVG_ema={avg_ema:.1f} active={active} is_quiet={is_quiet} block_max={block_max:.8f}")

            if active:
                if is_quiet:
                    if silence_since is None:
                        silence_since = now
                    elif (now - silence_since) >= SILENCE_DURATION:
                        # Borda de queda: envia 1 pacote zerado com transition e PAUSA
                        zero = np.zeros(NUM_BANDS, dtype=np.uint8)
                        send_packet(zero, 0, 1)
                        print_status(" (to silence)", dbg)
                        active = False
                        silence_since = None
                        resume_since = None
                        time.sleep(0.01)
                        continue
                else:
                    silence_since = None
            else:
                # Inativo → só verifica retomada sustentada (com EMA)
                if (avg_ema > resume_threshold) or (rms_ema > AMP_SILENCE_THRESHOLD * 3.0):
                    if resume_since is None:
                        resume_since = now
                    elif (now - resume_since) >= RESUME_STABLE_TIME:
                        # Borda de subida: 1º pacote com transition=1 e retoma TX
                        send_packet(bands_now, beat, 1)
                        print_status(" (from silence)", dbg)
                        active = True
                        resume_since = None
                        last_tick = now
                        time.sleep(0.001)
                        continue
                else:
                    resume_since = None

            # Se não está ativo, não transmite nada
            if not active:
                print_status(" (idle)", dbg)
                time.sleep(0.05)
                continue

            # TX ativo: respeita FPS
            if (now - last_tick) < MIN_SEND_INTERVAL:
                time.sleep(0.001)
                continue

            last_tick = now
            send_packet(bands_now, beat, 0)
            print_status("", dbg)

    except KeyboardInterrupt:
        pass
    finally:
        _stop_resync.set()
        try:
            stream.stop(); stream.close()
        except Exception:
            pass
        sys.stdout.write("\n"); sys.stdout.flush()

if __name__ == "__main__":
    main()