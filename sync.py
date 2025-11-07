#!/usr/bin/env python3
# sync.py — Raspberry Pi 3B renderer (compatível com A2 e A1)
# • Aceita PKT_AUDIO_V2 (0xA2, 163 bytes) e legado PKT_AUDIO (0xA1, 161 bytes).
# • As bandas já vêm equalizadas do PC. Render "on packet" para máxima reatividade.

import socket, time, board, neopixel, random, threading, sys, select, tty, termios, os
import numpy as np
from collections import deque
from fxcore import FXContext
from effects import build_effects  # build_effects(ctx)

PKT_AUDIO_V2 = 0xA2  # [A2][8 ts_pc][150 bands][beat][trans][dyn_floor][kick] => 163 bytes
PKT_AUDIO    = 0xA1  # [A1][8 ts_pc][150 bands][beat][trans]                   => 161 bytes
LEN_A2 = 163
LEN_A1 = 161

UDP_PORT = 5005
TCP_TIME_PORT = 5006

LED_COUNT = 300
LED_PIN = board.D18
ORDER = neopixel.GRB
BRIGHTNESS = 0.7
pixels = neopixel.NeoPixel(LED_PIN, LED_COUNT, brightness=BRIGHTNESS, auto_write=False, pixel_order=ORDER)

# ---- Debug & bypass helpers ----
# Se True, ignora o módulo de efeitos e renderiza um fallback em escala de cinza
BYPASS_EFFECTS = True
# Loga estatísticas simples 1x/seg
DEBUG_STATS = True

# Sockets
udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
try:
    udp_sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 22)
    udp_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
except Exception:
    pass
udp_sock.bind(("0.0.0.0", UDP_PORT))
udp_sock.setblocking(True)

EXPECTED_BANDS = 150
rx_count = 0
drop_len = 0
drop_hdr = 0
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

# Paleta
STRATEGIES = ["complementar", "analoga", "triade", "tetrade", "split"]
COMPLEMENT_DELTA = 0.06

def clamp01(x): return x % 1.0

def build_palette_from_strategy(h0, strategy, num_colors=5):
    import colorsys
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

# UDP receiver compatível (A2 e A1) + métricas de drop
latest_packet = None
latest_lock = threading.Lock()
stop_flag = False

def udp_receiver():
    global latest_packet, rx_count, drop_hdr, drop_len, _last_status
    print(f"[INFO] UDP receiver ligado em 0.0.0.0:{UDP_PORT} (aceita A2-163 e A1-161).")
    while not stop_flag:
        try:
            data, _ = udp_sock.recvfrom(2048)
            n = len(data)
            if n < 1:
                drop_len += 1
                continue

            hdr = data[0]
            if hdr == PKT_AUDIO_V2 and n == LEN_A2:
                ts_pc = int.from_bytes(data[1:9], 'little')
                bands = np.frombuffer(data[9:159], dtype=np.uint8)
                beat  = data[159]
                trans = data[160]
                dyn_floor = data[161]
                kick_intensity = data[162]
            elif hdr == PKT_AUDIO and n == LEN_A1:
                ts_pc = int.from_bytes(data[1:9], 'little')
                bands = np.frombuffer(data[9:159], dtype=np.uint8)
                beat  = data[159]
                trans = data[160]
                dyn_floor = 0
                kick_intensity = 0
            else:
                # Drop por tamanho ou header
                if hdr in (PKT_AUDIO, PKT_AUDIO_V2):
                    drop_len += 1
                else:
                    drop_hdr += 1
                continue

            with latest_lock:
                latest_packet = (bands.copy(), beat, trans, ts_pc, dyn_floor, kick_intensity)
            rx_count += 1

            now = time.time()
            if now - _last_status > 1.0:
                print(f"\rRX: {rx_count}  drops(len):{drop_len}  drops(hdr):{drop_hdr}", end="", flush=True)
                _last_status = now

        except Exception:
            time.sleep(0.001)

def print_status(effect_name, ctx):
    now = time.time()
    global _last_status, latency_ms_ema
    if now - _last_status < 0.25:
        return
    curr = ctx.current_a_ema
    poww = ctx.power_w_ema
    cap = ctx.last_cap_scale
    lat = f"{latency_ms_ema:.1f}ms" if latency_ms_ema else "?"
    status = f"{effect_name}  {curr:.2f}A {poww:.1f}W cap:{cap:.2f} lat:{lat}"
    print(f"\r{status:<70}", end="", flush=True)
    _last_status = now

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

def main():
    # latest_packet é atualizado pela thread udp_receiver e lido/limpo aqui:
    # precisa ser global para evitar UnboundLocalError ao reatribuir.
    global stop_flag, latency_ms_ema, latest_packet

    # Threads auxiliares
    threading.Thread(target=timesync_tcp_server, daemon=True).start()
    threading.Thread(target=udp_receiver, daemon=True).start()
    threading.Thread(target=palette_worker, daemon=True).start()
    threading.Thread(target=key_listener, daemon=True).start()

    # Contexto e efeitos (ctx antes; efeitos recebem ctx)
    ctx = FXContext(
        pixels, LED_COUNT,
        base_hue_offset, hue_seed, base_saturation,
        current_budget_a=18.0, ma_per_channel=20.0, idle_ma_per_led=1.0
    )
    ctx.metrics = None
    effects = build_effects(ctx)

    # Estado local do main
    current_effect = 0
    last_effect_change = time.time()
    effect_max_interval = 300  # 5 min

    last_bands = np.zeros(EXPECTED_BANDS, dtype=np.uint8)
    last_beat = 0

    SIGNAL_HOLD = 0.5
    signal_active_until = 0.0
    already_off = False
    last_dbg = 0.0

    IDLE_TIMEOUT = 2.0
    last_rx_ts = time.time()

    FRAME_DT = 1/75
    next_frame = time.time()
    RENDER_ON_PACKET = True

    apply_new_colorset()

    try:
        while True:
            now = time.time()

            # Teclas
            global pending_key_change
            if pending_key_change is not None:
                if pending_key_change == 'next':
                    current_effect = (current_effect + 1) % len(effects)
                else:
                    current_effect = (current_effect - 1) % len(effects)
                pending_key_change = None
                apply_new_colorset()
                last_effect_change = now

            # idle (sem RX recente)
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
            if pkt is not None:
                bands_u8, beat_flag, transition_flag, ts_pc_ns, dyn_floor, kick_intensity = pkt
                last_rx_ts = now

                # latência (um caminho)
                global time_sync_ready, time_offset_ns
                if time_sync_ready and ts_pc_ns is not None:
                    now_ns = time.monotonic_ns()
                    one_way_ns = now_ns - (ts_pc_ns + time_offset_ns)
                    if -5_000_000 <= one_way_ns <= 5_000_000_000:
                        lat_ms = one_way_ns / 1e6
                        latency_ms_ema = lat_ms if latency_ms_ema is None else 0.85*latency_ms_ema + 0.15*lat_ms

                # Estatísticas simples para debug
                avg_raw = float(np.mean(bands_u8))
                min_raw = int(bands_u8.min()) if bands_u8.size else 0
                max_raw = int(bands_u8.max()) if bands_u8.size else 0

                # Gating alinhado ao PC:
                # • transition=1 com bandas zeradas => desativa
                # • qualquer pacote "normal" => sustenta ativo por SIGNAL_HOLD
                if transition_flag == 1 and avg_raw < 0.5:
                    signal_active_until = 0.0
                    last_bands[:] = 0
                    last_beat = 0
                else:
                    signal_active_until = now + SIGNAL_HOLD
                    last_bands[:] = bands_u8
                    last_beat = int(beat_flag)

                # Dynamic floor vindo do PC (ou 0 no legado A1)
                ctx.dynamic_floor = int(dyn_floor)

                # troca de efeito por tempo/transition
                time_up = (now - last_effect_change) > effect_max_interval
                if transition_flag == 1 or time_up:
                    current_effect = (current_effect + 1) % len(effects)
                    apply_new_colorset()
                    last_effect_change = now

                active = (now < signal_active_until)

                b = last_bands.copy()

                # Render: efeitos ou fallback
                try:
                    if not BYPASS_EFFECTS:
                        name, func = effects[current_effect]
                        func(b if active else np.zeros_like(b), last_beat if active else 0, active)
                    else:
                        # Fallback ultra-simples: 150 bandas -> 300 LEDs (2 por banda), em cinza.
                        vals = np.repeat(b.astype(np.uint8), 2)
                        if vals.size < LED_COUNT:
                            # Se por algum motivo faltar, completa com zeros
                            vals = np.pad(vals, (0, LED_COUNT - vals.size), 'constant')
                        else:
                            vals = vals[:LED_COUNT]
                        rgb = np.stack([vals, vals, vals], axis=-1)  # tons de cinza
                        ctx.to_pixels_and_show(rgb)
                        name = "Fallback Gray (simple)"
                except Exception as e:
                    # Mesmo fallback no except, sem chamadas sofisticadas
                    vals = np.repeat(b.astype(np.uint8), 2)
                    if vals.size < LED_COUNT:
                        vals = np.pad(vals, (0, LED_COUNT - vals.size), 'constant')
                    else:
                        vals = vals[:LED_COUNT]
                    rgb = np.stack([vals, vals, vals], axis=-1)
                    ctx.to_pixels_and_show(rgb)
                    name = f"Fallback Gray (simple, err:{e.__class__.__name__})"

                already_off = False

                # Status + debug mínimo
                print_status(name, ctx)
                if DEBUG_STATS and (now - last_dbg) > 1.0:
                    sys.stdout.write(
                        f"\n[DBG] avg:{avg_raw:5.1f} min:{min_raw:3d} max:{max_raw:3d} active:{'Y' if active else 'N'}\n"
                    )
                    last_dbg = now

                if RENDER_ON_PACKET:
                    next_frame = time.time()
                    continue

            # pacing quando não tem pacote novo
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
# EOF   