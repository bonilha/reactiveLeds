#!/usr/bin/env python3
# sync.py — Raspberry Pi 3B renderer (A2/A1 + B0 config)
# • A2 (0xA2): áudio pronto (eq no PC)   • A1: legado
# • B0 (0xB0): config => bands/fps/signal_hold_ms/vis_fps
# • Saída unificada: uma única linha dinâmica no console

import socket, time, board, neopixel, random, threading, sys, select, tty, termios, os
import numpy as np
from collections import deque
from fxcore import FXContext
from effects import build_effects

# ---------- Protocolo ----------
PKT_AUDIO_V2 = 0xA2
PKT_AUDIO    = 0xA1
PKT_CFG      = 0xB0  # [B0][ver u8][num_bands u16][fps u16][hold_ms u16][vis_fps u16]
LEN_A2_MIN = 13   # 1 hdr + 8 ts + 0 bands + 4 flags = 13 bytes
LEN_A1_MIN = 11   # 1 hdr + 8 ts + 0 bands + 2 flags = 11 bytes
LEN_CFG    = 10

UDP_PORT = 5005
TCP_TIME_PORT = 5006

# ---------- LEDs ----------
LED_COUNT = 300
LED_PIN = board.D18
ORDER = neopixel.GRB
BRIGHTNESS = 0.8
LED_IS_RGBW = False

pixels = (neopixel.NeoPixel(LED_PIN, LED_COUNT, brightness=BRIGHTNESS, auto_write=False,
                            pixel_order=neopixel.GRBW, bpp=4)
          if LED_IS_RGBW else
          neopixel.NeoPixel(LED_PIN, LED_COUNT, brightness=BRIGHTNESS, auto_write=False, pixel_order=ORDER))

# ---------- Flags ----------
BYPASS_EFFECTS   = False   # efeitos habilitados
DEBUG_STATS      = False   # manter False para não poluir a linha única
ENABLE_SMOKE_TEST= False

# ---------- UDP ----------
udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
try:
    udp_sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 22)
    udp_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
except Exception: pass
udp_sock.bind(("0.0.0.0", UDP_PORT))
udp_sock.setblocking(True)

# ---------- Estado base (atualizados via B0 / A2) ----------
EXPECTED_BANDS = 150
CFG_FPS        = 75
SIGNAL_HOLD_MS = 500
CFG_VIS_FPS    = 45

# Contadores / métricas
rx_count = drop_len = drop_hdr = 0
time_offset_ns = 0
time_sync_ready = False
latency_ms_ema = None

# Teclado
pending_key_change = None

# Pacotes
latest_packet = None
latest_lock = threading.Lock()
stop_flag = False

# ---------- Util ----------
def _recv_exact(conn, n):
    buf = b''
    while len(buf) < n:
        ch = conn.recv(n - len(buf))
        if not ch: raise ConnectionError("TCP encerrado")
        buf += ch
    return buf

def timesync_tcp_server():
    """Servidor TCP de time-sync (PC = cliente)."""
    global time_offset_ns, time_sync_ready
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("0.0.0.0", TCP_TIME_PORT)); srv.listen(1)
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
                        conn.sendall(b"TS2" + t0 + tr.to_bytes(8,'little'))
                    elif hdr == b"TS3":
                        off_bytes = _recv_exact(conn, 8)
                        time_offset_ns = int.from_bytes(off_bytes,'little',signed=True)
                        time_sync_ready = True
                        conn.sendall(b"TS3" + int(time_offset_ns).to_bytes(8,'little',signed=True))
                        break
                    else:
                        break
        except Exception:
            time.sleep(0.05)

# ---------- Paleta / cores ----------
STRATEGIES = ["complementar","analoga","triade","tetrade","split"]
COMPLEMENT_DELTA = 0.06
def clamp01(x): return x % 1.0
def build_palette_from_strategy(h0, strategy, num_colors=5):
    import colorsys
    def hsv_to_rgb_bytes(h, s, v):
        r,g,b = colorsys.hsv_to_rgb(h,s,v); return (int(r*255), int(g*255), int(b*255))
    if strategy == "complementar":
        hA=h0%1.0; hB=(h0+0.5)%1.0; d=COMPLEMENT_DELTA
        hues=[hA,(hA+d)%1.0,(hA-d)%1.0,hB,(hB+d)%1.0,(hB-d)%1.0]
        s_choices=[0.60,0.70,0.80,0.85]; v_choices=[0.70,0.80,0.90,1.00]
        pal=[]
        while len(pal)<num_colors:
            h=random.choice(hues)
            if len(pal)==0: s,v=0.75,0.95
            elif len(pal)==1: s,v=0.65,0.75
            else: s=random.choice(s_choices); v=random.choice(v_choices)
            pal.append(hsv_to_rgb_bytes(h,s,v))
        return pal
    elif strategy == "analoga":
        step=0.10; hues=[clamp01(h0+k*step) for k in (-2,-1,0,1,2)]
    elif strategy == "triade":
        hues=[h0, clamp01(h0+1/3), clamp01(h0+2/3)]
    elif strategy == "tetrade":
        hues=[h0, clamp01(h0+0.25), clamp01(h0+0.5), clamp01(h0+0.75)]
    else:
        off=1/6; hues=[h0, clamp01(h0+0.5-off), clamp01(h0+0.5+off)]
    pal=[]
    while len(pal)<num_colors:
        h=random.choice(hues); s=random.uniform(0.55,0.85); v=random.uniform(0.70,1.00)
        pal.append(hsv_to_rgb_bytes(h,s,v))
    return pal

base_hue_offset = random.randint(0,255)
hue_seed = random.randint(0,255)
base_saturation = 210
current_palette = [(255,255,255)]*5
current_palette_name = "analoga"
last_palette_h0 = 0.0

PALETTE_BUFFER_MAX = 8
palette_queue = deque(maxlen=PALETTE_BUFFER_MAX)
palette_thread_stop = threading.Event()

def palette_worker():
    while not palette_thread_stop.is_set():
        try:
            if len(palette_queue) < PALETTE_BUFFER_MAX:
                strategy = random.choice(STRATEGIES); h0 = random.random()
                pal = build_palette_from_strategy(h0, strategy, num_colors=5)
                palette_queue.append((pal, strategy, h0))
            else:
                time.sleep(0.1)
        except Exception:
            time.sleep(0.05)

def get_next_palette():
    if palette_queue: return palette_queue.popleft()
    return None

def apply_new_colorset():
    global base_hue_offset, hue_seed, base_saturation, current_palette, current_palette_name, last_palette_h0
    pal_data = get_next_palette()
    if pal_data:
        current_palette, current_palette_name, last_palette_h0 = pal_data
    else:
        strategy = random.choice(STRATEGIES); h0 = random.random()
        current_palette = build_palette_from_strategy(h0, strategy, num_colors=5)
        current_palette_name = strategy; last_palette_h0 = h0
    base_hue_offset = random.randint(0,255)
    hue_seed = random.randint(0,255)
    base_saturation = random.randint(190,230)

# ---------- Teclado ----------
def key_listener():
    global pending_key_change
    old = termios.tcgetattr(sys.stdin)
    try:
        tty.setcbreak(sys.stdin.fileno())
        while True:
            if select.select([sys.stdin], [], [], 0)[0]:
                k = sys.stdin.read(1)
                if   k == 'n': pending_key_change = 'next'
                elif k == 'p': pending_key_change = 'prev'
                elif k == 'q': os._exit(0)
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old)

# ---------- UDP Receiver + CFG (dinâmico) ----------
def udp_receiver():
    global latest_packet, rx_count, drop_hdr, drop_len
    global EXPECTED_BANDS, CFG_FPS, SIGNAL_HOLD_MS, CFG_VIS_FPS
    print(f"[INFO] UDP receiver ligado em 0.0.0.0:{UDP_PORT} (aceita A2/A1 com bands variáveis e B0).")
    while not stop_flag:
        try:
            data, _ = udp_sock.recvfrom(4096)
            n = len(data)
            if n < 1:
                drop_len += 1
                continue

            hdr = data[0]

            # --- A2: 1+8+nb+4 (13+nb) ---
            if hdr == PKT_AUDIO_V2 and n >= LEN_A2_MIN:
                nb = n - LEN_A2_MIN
                if nb <= 0:
                    drop_len += 1
                    continue
                ts_pc = int.from_bytes(data[1:9],'little')
                bands = np.frombuffer(memoryview(data)[9:9+nb], dtype=np.uint8)
                beat  = data[9+nb]
                trans = data[10+nb]
                dyn_floor = data[11+nb]
                kick_intensity = data[12+nb]

                if EXPECTED_BANDS != nb:
                    EXPECTED_BANDS = int(nb)

                with latest_lock:
                    latest_packet = (bands.copy(), beat, trans, ts_pc, dyn_floor, kick_intensity)
                rx_count += 1
                continue

            # --- A1: 1+8+nb+2 (11+nb) ---
            if hdr == PKT_AUDIO and n >= LEN_A1_MIN:
                nb = n - LEN_A1_MIN
                if nb <= 0:
                    drop_len += 1
                    continue
                ts_pc = int.from_bytes(data[1:9],'little')
                bands = np.frombuffer(memoryview(data)[9:9+nb], dtype=np.uint8)
                beat  = data[9+nb]
                trans = data[10+nb]

                if EXPECTED_BANDS != nb:
                    EXPECTED_BANDS = int(nb)

                with latest_lock:
                    latest_packet = (bands.copy(), beat, trans, ts_pc, 0, 0)
                rx_count += 1
                continue

            # --- B0: config ---
            if hdr == PKT_CFG and n == LEN_CFG:
                ver = data[1]
                nb  = data[2] | (data[3] << 8)
                fps = data[4] | (data[5] << 8)
                hold= data[6] | (data[7] << 8)
                vis = data[8] | (data[9] << 8)
                if nb  > 0: EXPECTED_BANDS = int(nb)
                if fps > 0: CFG_FPS        = int(fps)
                if hold>=0: SIGNAL_HOLD_MS = int(hold)
                if vis > 0: CFG_VIS_FPS    = int(vis)
                continue

            # Desconhecido
            if hdr in (PKT_AUDIO, PKT_AUDIO_V2, PKT_CFG):
                drop_len += 1
            else:
                drop_hdr += 1

        except Exception:
            time.sleep(0.001)

# ---------- Linha unificada ----------
def unified_status_line(effect_name, ctx, active, bands, fps, vis_fps):
    curr = ctx.current_a_ema if ctx.current_a_ema is not None else 0.0
    poww = ctx.power_w_ema  if ctx.power_w_ema  is not None else 0.0
    cap  = ctx.last_cap_scale if ctx.last_cap_scale is not None else 1.0
    lat  = f"{latency_ms_ema:.1f}" if latency_ms_ema is not None else "?"
    line = (f"RX:{rx_count} dropL:{drop_len} dropH:{drop_hdr} | "
            f"Eff:{effect_name} | active:{'Y' if active else 'N'} | "
            f"bands:{bands} fps:{fps} vis:{vis_fps} | "
            f"I:{curr:.2f}A P:{poww:.1f}W cap:{cap:.2f} | lat:{lat}ms")
    sys.stdout.write("\r" + line.ljust(120))
    sys.stdout.flush()

# ---------- Smoke test (opcional) ----------
def hardware_smoke_test(ctx, seconds=0.6):
    print(f"[INFO] NeoPixel bpp={getattr(pixels,'bpp','n/a')} order={ORDER} is_rgbw={LED_IS_RGBW}")
    colors=[(255,0,0),(0,255,0),(0,0,255),(255,255,255),(0,0,0)]
    for rgb in colors:
        arr = np.tile(np.array(rgb, dtype=np.uint8), (LED_COUNT,1))
        ctx.to_pixels_and_show(arr); time.sleep(seconds)

# ---------- Main ----------
def main():
    global stop_flag, latency_ms_ema, latest_packet

    # Threads auxiliares
    threading.Thread(target=timesync_tcp_server, daemon=True).start()
    threading.Thread(target=udp_receiver,        daemon=True).start()
    threading.Thread(target=palette_worker,      daemon=True).start()
    threading.Thread(target=key_listener,        daemon=True).start()

    # Contexto e efeitos
    ctx = FXContext(pixels, LED_COUNT, base_hue_offset, hue_seed, base_saturation,
                    current_budget_a=18.0, ma_per_channel=20.0, idle_ma_per_led=1.0)
    ctx.metrics = None
    effects = build_effects(ctx)  # efeitos reais habilitados

    if ENABLE_SMOKE_TEST:
        hardware_smoke_test(ctx, seconds=0.6)

    # Estado principal
    current_effect = 0
    last_effect_change = time.time()
    effect_max_interval = 300.0  # 5 min

    last_bands = np.zeros(EXPECTED_BANDS, dtype=np.uint8)
    last_beat  = 0

    SIGNAL_HOLD = SIGNAL_HOLD_MS / 1000.0
    FRAME_DT    = 1.0 / max(1, CFG_FPS)
    RENDER_DT   = 1.0 / max(1, CFG_VIS_FPS)

    signal_active_until = 0.0
    already_off = False
    last_rx_ts = time.time()
    next_frame = time.time()
    last_render_ts = 0.0

    # Smoothing visual do fallback (só usado no fallback)
    VIS_SMOOTH_ATTACK  = 1.0
    VIS_SMOOTH_RELEASE = 0.60
    vis_prev_v = np.zeros(LED_COUNT, dtype=np.float32)

    try:
        while True:
            now = time.time()

            # Hot-apply de CFG (sem prints soltos)
            desired_frame_dt  = 1.0 / max(1, CFG_FPS)
            desired_hold      = SIGNAL_HOLD_MS / 1000.0
            desired_render_dt = 1.0 / max(1, CFG_VIS_FPS)
            if abs(desired_frame_dt - FRAME_DT) > 1e-6:  FRAME_DT  = desired_frame_dt
            if abs(desired_hold     - SIGNAL_HOLD) > 1e-6: SIGNAL_HOLD = desired_hold
            if abs(desired_render_dt- RENDER_DT)  > 1e-6:  RENDER_DT = desired_render_dt

            # Teclas (n/p)
            global pending_key_change
            if pending_key_change is not None:
                if pending_key_change == 'next':
                    current_effect = (current_effect + 1) % len(effects)
                else:
                    current_effect = (current_effect - 1) % len(effects)
                pending_key_change = None
                apply_new_colorset()
                last_effect_change = now

            # Idle (sem RX recente)
            if (now - last_rx_ts) > 2.0:
                if not already_off:
                    pixels.fill((0,0,0)); pixels.show()
                    already_off = True
                unified_status_line("Idle", ctx, False, EXPECTED_BANDS, CFG_FPS, CFG_VIS_FPS)
                next_frame += FRAME_DT
                sl = next_frame - time.time()
                if sl > 0: time.sleep(sl)
                else: next_frame = time.time()
                continue

            # Consumir pacote (se houver)
            pkt = None
            with latest_lock:
                if latest_packet is not None:
                    pkt = latest_packet
                    latest_packet = None

            if pkt is not None:
                bands_u8, beat_flag, transition_flag, ts_pc_ns, dyn_floor, kick_intensity = pkt
                last_rx_ts = now

                # Latência (um caminho)
                global time_sync_ready, time_offset_ns
                if time_sync_ready and ts_pc_ns is not None:
                    now_ns = time.monotonic_ns()
                    one_way_ns = now_ns - (ts_pc_ns + time_offset_ns)
                    if -5_000_000 <= one_way_ns <= 5_000_000_000:
                        lat_ms = one_way_ns / 1e6
                        latency_ms_ema = lat_ms if latency_ms_ema is None else 0.85*latency_ms_ema + 0.15*lat_ms

                # Gating alinhado ao PC
                avg_raw = float(np.mean(bands_u8))
                if transition_flag == 1 and avg_raw < 0.5:
                    signal_active_until = 0.0
                    # (re)zera o buffer no tamanho atual das bandas
                    if last_bands.size != bands_u8.size:
                        last_bands = np.zeros(bands_u8.size, dtype=np.uint8)
                    else:
                        last_bands[:] = 0
                    last_beat = 0
                else:
                    signal_active_until = now + SIGNAL_HOLD
                    # 🔧 FIX: realoca se o tamanho mudou (e então copia)
                    if last_bands.size != bands_u8.size:
                        last_bands = np.zeros(bands_u8.size, dtype=np.uint8)
                    last_bands[:] = bands_u8
                    last_beat = int(beat_flag)

                ctx.dynamic_floor = int(dyn_floor)

                # Auto-rotate a cada 5 min
                time_up = (now - last_effect_change) > effect_max_interval
                if transition_flag == 1 or time_up:
                    current_effect = (current_effect + 1) % len(effects)
                    apply_new_colorset()
                    last_effect_change = now

                active = (now < signal_active_until)
                b = last_bands.copy()

                # Decimator de render (vis_fps)
                if (now - last_render_ts) >= RENDER_DT:
                    last_render_ts = now
                    try:
                        if not BYPASS_EFFECTS:
                            name, func = effects[current_effect]
                            func(b if active else np.zeros_like(b),
                                 last_beat if active else 0,
                                 active)
                        else:
                            # Fallback HSV (debug)
                            led_idx = np.arange(LED_COUNT, dtype=np.int32)
                            band_idx = (led_idx * b.size) // LED_COUNT  # usa tamanho real de b
                            vals = b[band_idx].astype(np.uint8)
                            v = ((vals.astype(np.float32)/255.0)**1.6) * 255.0
                            alpha = np.where(v > vis_prev_v, VIS_SMOOTH_ATTACK, VIS_SMOOTH_RELEASE).astype(np.float32)
                            vis_prev_v = alpha*v + (1.0-alpha)*vis_prev_v
                            v = np.clip(vis_prev_v, 0, 255).astype(np.uint8)
                            h = ((led_idx * 256) // LED_COUNT).astype(np.int16)
                            s = np.full(LED_COUNT, 255, dtype=np.uint8)
                            rgb = ctx.hsv_to_rgb_bytes_vec(h, s, v)
                            ctx.to_pixels_and_show(rgb)
                        name = effects[current_effect][0] if not BYPASS_EFFECTS else "Fallback HSV"
                    except Exception as e:
                        vals = np.repeat(b.astype(np.uint8), 2)
                        if vals.size < LED_COUNT: vals = np.pad(vals, (0, LED_COUNT-vals.size), 'constant')
                        else: vals = vals[:LED_COUNT]
                        rgb = np.stack([vals, vals, vals], axis=-1)
                        ctx.to_pixels_and_show(rgb)
                        name = f"Fallback Gray ({e.__class__.__name__})"
                else:
                    name = effects[current_effect][0] if not BYPASS_EFFECTS else "Fallback HSV"

                # Linha única
                unified_status_line(name, ctx, active, b.size, CFG_FPS, CFG_VIS_FPS)
                continue

            # Pacing quando não chega pacote novo
            unified_status_line(effects[current_effect][0] if not BYPASS_EFFECTS else "Fallback HSV",
                                ctx, (now < signal_active_until), EXPECTED_BANDS, CFG_FPS, CFG_VIS_FPS)
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
        sys.stdout.write("\n"); sys.stdout.flush()

if __name__ == "__main__":
    main()