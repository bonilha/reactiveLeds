#!/usr/bin/env python3
# sync.py - Main completo com correções de reatividade e BUGFIXES críticos
# Alterações principais:
# - Kick boost MULTIPLICATIVO + LOCALIZADO nos graves (sem aplanar)
# - Smoothing com release mais rápido (0.28)
# - Equalização ágil (alpha 0.22)
# - Dynamic floor só em volume altíssimo (>90)
# - CORREÇÃO CRÍTICA: amplify_quad agora é QUADRÁTICO com ganho ajustado (evita saturação verde)
# - CORREÇÃO: hsv_to_rgb_bytes_vec usa uint8 corretamente (evita overflow em hue)
# - CORREÇÃO: segment_mean_from_cumsum robusto a bandas desalinhadas
# - CORREÇÃO: Rainbow Wave fixado (sem "correr" de 1 LED)
# - CORREÇÃO: VU Meter com escala dinâmica (mais reativo)
# - Pequenos ajustes de performance e robustez

import socket, time, board, neopixel, random, colorsys, threading, sys, select, tty, termios
import numpy as np
from collections import deque
from datetime import datetime
from fxcore import FXContext
from effects import build_effects
import argparse, os

PKT_AUDIO = 0xA1
UDP_PORT = 5005
TCP_TIME_PORT = 5006
LED_COUNT = 300
LED_PIN = board.D18
ORDER = neopixel.GRB
BRIGHTNESS = 0.7
pixels = neopixel.NeoPixel(LED_PIN, LED_COUNT, brightness=BRIGHTNESS, auto_write=False, pixel_order=ORDER)

# Sockets
udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
try:
    udp_sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 22)
except Exception:
    pass
udp_sock.bind(("0.0.0.0", UDP_PORT))
udp_sock.setblocking(True)

EXPECTED_BANDS = 150
rx_count = 0
_last_status = 0.0

# Time sync
time_offset_ns = 0
time_sync_ready = False
latency_ms_ema = None

def _recv_exact(conn, n):
    buf = b''
    while len(buf) < n:
        ch = conn.recv(n - len(buf))
        if not ch:
            raise ConnectionError("TCP encerrado")
        buf += ch
    return buf

def timesync_tcp_server():
    global time_offset_ns, time_sync_ready
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("0.0.0.0", TCP_TIME_PORT))
    srv.listen(1)
    while True:
        try:
            conn, addr = srv.accept()
            with conn:
                conn.settimeout(1.2)
                while True:
                    hdr = _recv_exact(conn, 3)
                    if hdr == b"TS1":
                        t0 = _recv_exact(conn, 8)
                        tr = time.monotonic_ns()
                        conn.sendall(b"TS2" + t0 + tr.to_bytes(8, 'little'))
                    elif hdr == b"TS3":
                        off_bytes = _recv_exact(conn, 8)
                        time_offset_ns = int.from_bytes(off_bytes, 'little', signed=True)
                        time_sync_ready = True
                        conn.sendall(b"TS3" + int(time_offset_ns).to_bytes(8, 'little', signed=True))
                        break
                    else:
                        break
        except Exception:
            time.sleep(0.05)

# Equalização mais ágil
EQ_ALPHA = 0.22
EQ_TARGET = 64.0
band_ema = np.ones(EXPECTED_BANDS, dtype=np.float32) * 32.0
TILT_MIN, TILT_MAX = 0.9, 1.8
tilt_curve = np.exp(np.linspace(np.log(TILT_MAX), np.log(TILT_MIN), EXPECTED_BANDS)).astype(np.float32)

def equalize_bands(bands_u8):
    global band_ema
    x = np.asarray(bands_u8, dtype=np.float32)
    band_ema = (1.0 - EQ_ALPHA) * band_ema + EQ_ALPHA * x
    gain = EQ_TARGET / np.maximum(band_ema, 1.0)
    eq = x * gain * tilt_curve
    return np.clip(eq, 0.0, 255.0).astype(np.uint8)

def compute_dynamic_floor(eq_bands, active):
    global dynamic_floor
    if not active:
        dynamic_floor = 0
        return
    mean_val = float(np.mean(eq_bands))
    if mean_val > 90:  # só em volume muito alto
        dynamic_floor = int(min(12, mean_val * 0.012))
    else:
        dynamic_floor = 0

# Paleta
STRATEGIES = ["complementar", "analoga", "triade", "tetrade", "split"]
COMPLEMENT_DELTA = 0.06

def clamp01(x): return x % 1.0

def build_palette_from_strategy(h0, strategy, num_colors=5):
    import colorsys, random
    def hsv_to_rgb_bytes(h, s, v):
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        return (int(r*255), int(g*255), int(b*255))
    if strategy == "complementar":
        hA = h0 % 1.0
        hB = (h0 + 0.5) % 1.0
        d = COMPLEMENT_DELTA
        hues = [hA, (hA+d)%1.0, (hA-d)%1.0, hB, (hB+d)%1.0, (hB-d)%1.0]
        s_choices = [0.60, 0.70, 0.80, 0.85]
        v_choices = [0.70, 0.80, 0.90, 1.00]
        pal = []
        while len(pal) < num_colors:
            h = random.choice(hues)
            if len(pal) == 0:
                s, v = 0.75, 0.95
            elif len(pal) == 1:
                s, v = 0.65, 0.75
            else:
                s = random.choice(s_choices)
                v = random.choice(v_choices)
            pal.append(hsv_to_rgb_bytes(h, s, v))
        return pal
    elif strategy == "analoga":
        step = 0.10
        hues = [clamp01(h0 + k*step) for k in (-2, -1, 0, 1, 2)]
    elif strategy == "triade":
        hues = [h0, clamp01(h0 + 1/3), clamp01(h0 + 2/3)]
    elif strategy == "tetrade":
        hues = [h0, clamp01(h0 + 0.25), clamp01(h0 + 0.5), clamp01(h0 + 0.75)]
    else:
        off = 1/6
        hues = [h0, clamp01(h0 + 0.5 - off), clamp01(h0 + 0.5 + off)]
    pal = []
    while len(pal) < num_colors:
        h = random.choice(hues)
        s = random.uniform(0.55, 0.85)
        v = random.uniform(0.70, 1.00)
        pal.append(hsv_to_rgb_bytes(h, s, v))
    return pal

base_hue_offset = random.randint(0, 255)
hue_seed = random.randint(0, 255)
base_saturation = 210
current_palette = [(255,255,255)] * 5
current_palette_name = "analoga"
last_palette_h0 = 0.0

PALETTE_BUFFER_MAX = 8
palette_queue = deque(maxlen=PALETTE_BUFFER_MAX)
palette_thread_stop = threading.Event()

def palette_worker():
    import random
    while not palette_thread_stop.is_set():
        try:
            if len(palette_queue) < PALETTE_BUFFER_MAX:
                strategy = random.choice(STRATEGIES)
                h0 = random.random()
                pal = build_palette_from_strategy(h0, strategy, num_colors=5)
                palette_queue.append((pal, strategy, h0))
            else:
                time.sleep(0.1)
        except Exception:
            time.sleep(0.05)

def get_next_palette():
    if palette_queue:
        return palette_queue.popleft()
    return None

# Efeitos
effects = build_effects()
current_effect = 0
last_effect_change = time.time()
effect_max_interval = 300  # 5 min

# Teclado
pending_key_change = None
def key_listener():
    global pending_key_change
    old_settings = termios.tcgetattr(sys.stdin)
    try:
        tty.setcbreak(sys.stdin.fileno())
        while True:
            if select.select([sys.stdin], [], [], 0)[0]:
                key = sys.stdin.read(1)
                if key == 'n':
                    pending_key_change = 'next'
                elif key == 'p':
                    pending_key_change = 'prev'
                elif key == 'q':
                    os._exit(0)
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)

# UDP receiver
latest_packet = None
latest_lock = threading.Lock()
stop_flag = False

def udp_receiver():
    global latest_packet, rx_count, _last_status
    while not stop_flag:
        try:
            data, _ = udp_sock.recvfrom(1024)
            if len(data) != 161 or data[0] != PKT_AUDIO:
                continue
            ts_pc = int.from_bytes(data[1:9], 'little')
            bands = np.frombuffer(data[9:159], dtype=np.uint8)
            beat = data[159]
            trans = data[160]
            with latest_lock:
                latest_packet = (PKT_AUDIO, bands, beat, trans, ts_pc)
            rx_count += 1
            now = time.time()
            if now - _last_status > 1.0:
                print(f"\rRX: {rx_count} packets", end="", flush=True)
                _last_status = now
        except Exception:
            time.sleep(0.001)

# Status
def print_status(effect_name, ctx, metrics):
    now = time.time()
    global _last_status
    if now - _last_status < 0.25:
        return
    curr = ctx.current_a_ema
    poww = ctx.power_w_ema
    cap = ctx.last_cap_scale
    lat = f"{latency_ms_ema:.1f}ms" if latency_ms_ema else "?"
    status = f"{effect_name} | {curr:.2f}A {poww:.1f}W cap:{cap:.2f} lat:{lat}"
    print(f"\r{status:<70}", end="", flush=True)
    _last_status = now

# Apply palette
def apply_new_colorset():
    global base_hue_offset, hue_seed, base_saturation, current_palette, current_palette_name, last_palette_h0
    pal_data = get_next_palette()
    if pal_data:
        current_palette, current_palette_name, last_palette_h0 = pal_data
    else:
        strategy = random.choice(STRATEGIES)
        h0 = random.random()
        current_palette = build_palette_from_strategy(h0, strategy, num_colors=5)
        current_palette_name = strategy
        last_palette_h0 = h0
    base_hue_offset = random.randint(0, 255)
    hue_seed = random.randint(0, 255)
    base_saturation = random.randint(190, 230)

# Main
def main():
    global current_effect, last_effect_change, stop_flag, dynamic_floor

    threading.Thread(target=timesync_tcp_server, daemon=True).start()
    threading.Thread(target=udp_receiver, daemon=True).start()
    threading.Thread(target=palette_worker, daemon=True).start()
    threading.Thread(target=key_listener, daemon=True).start()

    ctx = FXContext(
        pixels, LED_COUNT,
        base_hue_offset, hue_seed, base_saturation,
        current_budget_a=18.0, ma_per_channel=20.0, idle_ma_per_led=1.0
    )
    ctx.metrics = None

    last_bands = np.zeros(EXPECTED_BANDS, dtype=np.float32)
    last_beat = 0
    eq_cached = np.zeros(EXPECTED_BANDS, dtype=np.uint8)
    eq_valid = False

    SIGNAL_HOLD = 0.5
    signal_active_until = 0.0
    prev_active = False
    already_off = False
    bands_changed = False

    IDLE_TIMEOUT = 2.0
    last_rx_ts = time.time()
    FRAME_DT = 1/60
    next_frame = time.time()

    apply_new_colorset()

    try:
        while True:
            now = time.time()
            # troca por tecla
            global pending_key_change
            if pending_key_change is not None:
                prev_idx = current_effect
                if pending_key_change == 'next':
                    current_effect = (current_effect + 1) % len(effects)
                else:
                    current_effect = (current_effect - 1) % len(effects)
                pending_key_change = None
                apply_new_colorset()
                last_effect_change = now

            # idle
            if (now - last_rx_ts) > IDLE_TIMEOUT:
                if not already_off:
                    pixels.fill((0,0,0)); pixels.show()
                    already_off = True
                next_frame += FRAME_DT
                sl = next_frame - time.time()
                if sl > 0: time.sleep(sl)
                else: next_frame = time.time()
                continue

            pkt = None
            with latest_lock:
                if latest_packet is not None:
                    pkt = latest_packet
                    latest_packet = None
                    last_rx_ts = now
            if pkt is not None:
                _type, bands, beat_flag, transition_flag, ts_pc_ns = pkt
                bands_arr = np.asarray(bands, dtype=np.float32)
                avg_raw = float(np.mean(bands_arr))
                if transition_flag == 1 and avg_raw < 0.5:
                    signal_active_until = 0.0
                    last_bands[:] = 0.0
                    last_beat = 0
                    bands_changed = True
                else:
                    if avg_raw > 2.0:
                        signal_active_until = now + SIGNAL_HOLD
                    # attack/release por banda (release mais rápido)
                    if 'ar_prev' not in globals():
                        globals()['ar_prev'] = np.zeros(EXPECTED_BANDS, dtype=np.float32)
                    prev = globals()['ar_prev']
                    attack, release = 0.70, 0.28
                    alpha = np.where(bands_arr > prev, attack, release).astype(np.float32)
                    smoothed = (alpha * bands_arr + (1.0 - alpha) * prev).astype(np.float32)
                    globals()['ar_prev'] = smoothed
                    bands_changed = not np.allclose(smoothed, last_bands, atol=1e-3)
                    last_bands[:] = smoothed
                    last_beat = int(beat_flag)

                # troca de efeito
                time_up = (now - last_effect_change) > effect_max_interval
                if transition_flag or time_up:
                    current_effect = (current_effect + 1) % len(effects)
                    apply_new_colorset()
                    last_effect_change = now

                # latência
                global latency_ms_ema, time_sync_ready, time_offset_ns
                if time_sync_ready and ts_pc_ns is not None:
                    now_ns = time.monotonic_ns()
                    one_way_ns = now_ns - (ts_pc_ns + time_offset_ns)
                    if -5_000_000 <= one_way_ns <= 5_000_000_000:
                        lat_ms = one_way_ns / 1e6
                        latency_ms_ema = lat_ms if latency_ms_ema is None else 0.85*latency_ms_ema + 0.15*lat_ms

            active = (now < signal_active_until)
            active_rising = (not prev_active) and active

            if active:
                if active_rising or bands_changed or not eq_valid:
                    eq_cached = equalize_bands(last_bands.astype(np.uint8))
                    compute_dynamic_floor(eq_cached, True)
                    ctx.dynamic_floor = dynamic_floor
                    eq_valid = True
                else:
                    pass
            else:
                if eq_valid:
                    eq_cached[:] = 0
                    compute_dynamic_floor(eq_cached, False)
                    ctx.dynamic_floor = dynamic_floor
                    eq_valid = False

            b = eq_cached.copy()

            # KICK BOOST REATIVO (CORRIGIDO: multiplicativo + só nos graves)
            if eq_valid:
                if 'kick_ema' not in globals():
                    globals()['kick_ema'] = 0.0
                    globals()['kick_prev'] = 0.0
                    globals()['boost_decay'] = 0.0

                low_end = max(10, EXPECTED_BANDS // 10)
                low_cached = float(np.mean(b[:low_end]))
                kick_raw = max(0.0, low_cached - globals()['kick_prev'])
                globals()['kickUnion_prev'] = low_cached

                globals()['kick_ema'] = 0.6 * globals()['kick_ema'] + 0.4 * low_cached
                onset = max(0.0, low_cached - globals()['kick_ema'])

                kick_intensity = onset * 2.2
                if last_beat == 1:
                    kick_intensity += 60

                if kick_intensity > 0:
                    boost_factor = 1.0 + (kick_intensity / 255.0) * 0.8  # max ~1.8x
                    b[:low_end] = np.clip(b[:low_end].astype(np.float32) * boost_factor, 0, 255).astype(np.uint8)
                    globals()['boost_decay'] = kick_intensity
                else:
                    globals()['boost_decay'] *= 0.85

            # Render
            name, func = effects[current_effect]
            func(b if active else np.zeros_like(b), last_beat if active else 0, active)
            already_off = False
            prev_active = active
            bands_changed = False

            print_status(name, ctx, None)
            next_frame += FRAME_DT
            sl = next_frame - time.time()
            if sl > 0: time.sleep(sl)
            else: next_frame = time.time()

    except KeyboardInterrupt:
        pass
    finally:
        stop_flag = True
        palette_thread_stop.set()
        pixels.fill((0,0,0)); pixels.show()
        print("\n")

if __name__ == "__main__":
    main()