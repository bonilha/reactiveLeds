#!/usr/bin/env python3
# sync.py — Raspberry Pi 3B renderer (A2/A1 + B0 config)
# • A2 (0xA2): áudio pronto (eq no PC)   • A1: legado
# • B0 (0xB0): config => bands/fps/signal_hold_ms/vis_fps
# • Saída unificada: uma única linha dinâmica no console
# • Otimizações de CPU: caches (LED_IDX/H/S), LUT de gamma, band_idx por mudança
# • NOVO: threads separadas (audio / render), FPS reduzidos, palette dorme, status cache

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
ORDER = neopixel.GRB           # sua fita é GRB
BRIGHTNESS = 0.8
LED_IS_RGBW = False            # WS2812B: False

pixels = (neopixel.NeoPixel(LED_PIN, LED_COUNT, brightness=BRIGHTNESS, auto_write=False,
                            pixel_order=neopixel.GRBW, bpp=4)
          if LED_IS_RGBW else
          neopixel.NeoPixel(LED_PIN, LED_COUNT, brightness=BRIGHTNESS, auto_write=False, pixel_order=ORDER))

# ---------- Flags ----------
BYPASS_EFFECTS   = False       # efeitos habilitados
ENABLE_SMOKE_TEST= False       # smoke test já passou
STATUS_MAX_HZ    = 15          # atualizações de status no máx. 15 Hz (reduz I/O)

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
CFG_FPS        = 45          # <-- reduzido
SIGNAL_HOLD_MS = 500
CFG_VIS_FPS    = 30          # <-- reduzido

# Contadores / métricas
rx_count = drop_len = drop_hdr = 0
time_offset_ns = 0
time_sync_ready = False
latency_ms_ema = None

# Teclado
pending_key_change = None

# Pacotes (shared entre threads)
shared_bands = np.zeros(EXPECTED_BANDS, dtype=np.uint8)
shared_beat  = 0
shared_active = False
shared_dyn_floor = 0
shared_kick_intensity = 0
shared_lock = threading.Lock()
stop_flag = False

# ---------- Caches / LUTs p/ reduzir CPU ----------
LED_IDX  = np.arange(LED_COUNT, dtype=np.int32)
H_FIXED  = ((LED_IDX * 256) // LED_COUNT).astype(np.int16)   # hue 0..255
S_FIXED  = np.full(LED_COUNT, 255, dtype=np.uint8)           # s=255 fixo no fallback
GAMMA_LUT = (((np.arange(256, dtype=np.float32) / 255.0) ** 1.6) * 255.0).astype(np.uint8)

# cache do mapeamento LED->banda (recriado apenas quando o número de bandas muda)
_band_to_led = np.repeat(np.arange(EXPECTED_BANDS), (LED_COUNT // EXPECTED_BANDS) + 1)[:LED_COUNT]
_band_to_led_for = EXPECTED_BANDS

# throttle da linha única
_status_min_period = 1.0 / max(1, STATUS_MAX_HZ)
_last_status_ts = 0.0
_last_status_line = ""

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
palette_wake = threading.Event()

def palette_worker():
    while not palette_thread_stop.is_set():
        if len(palette_queue) < PALETTE_BUFFER_MAX:
            strategy = random.choice(STRATEGIES); h0 = random.random()
            pal = build_palette_from_strategy(h0, strategy, num_colors=5)
            palette_queue.append((pal, strategy, h0))
        else:
            palette_wake.wait(2.0)
            palette_wake.clear()

def get_next_palette():
    if palette_queue:
        palette_wake.set()
        return palette_queue.popleft()
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
    global rx_count, drop_hdr, drop_len
    global EXPECTED_BANDS, CFG_FPS, SIGNAL_HOLD_MS, CFG_VIS_FPS
    global shared_bands, shared_beat, shared_active, shared_dyn_floor, shared_kick_intensity
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

                with shared_lock:
                    shared_bands[:nb] = bands
                    shared_beat = int(beat)
                    shared_active = True
                    shared_dyn_floor = int(dyn_floor)
                    shared_kick_intensity = int(kick_intensity)
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

                with shared_lock:
                    shared_bands[:nb] = bands
                    shared_beat = int(beat)
                    shared_active = True
                    shared_dyn_floor = 0
                    shared_kick_intensity = 0
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
                # recria mapeamento se mudou número de bandas
                global _band_to_led, _band_to_led_for
                if _band_to_led_for != EXPECTED_BANDS:
                    _band_to_led = np.repeat(np.arange(EXPECTED_BANDS),
                                             (LED_COUNT // EXPECTED_BANDS) + 1)[:LED_COUNT]
                    _band_to_led_for = EXPECTED_BANDS
                continue

            # Desconhecido
            if hdr in (PKT_AUDIO, PKT_AUDIO_V2, PKT_CFG):
                drop_len += 1
            else:
                drop_hdr += 1

        except Exception:
            time.sleep(0.001)

# ---------- Linha unificada (com Paleta) ----------
def unified_status_line(effect_name, ctx, active, bands, fps, vis_fps, pal_name):
    global _last_status_ts, _last_status_line
    now = time.time()
    if (now - _last_status_ts) < _status_min_period:
        return
    _last_status_ts = now

    curr = ctx.current_a_ema if ctx.current_a_ema is not None else 0.0
    poww = ctx.power_w_ema  if ctx.power_w_ema  is not None else 0.0
    cap  = ctx.last_cap_scale if ctx.last_cap_scale is not None else 1.0
    lat  = f"{latency_ms_ema:.1f}" if latency_ms_ema is not None else "?"
    line = (f"RX:{rx_count} dropL:{drop_len} dropH:{drop_hdr} | "
            f"Eff:{effect_name} Pal:{pal_name} | active:{'Y' if active else 'N'} | "
            f"bands:{bands} fps:{fps} vis:{vis_fps} | "
            f"I:{curr:.2f}A P:{poww:.1f}W cap:{cap:.2f} | lat:{lat}ms")
    if line == _last_status_line and (now - _last_status_ts) < 1.0:
        return
    _last_status_line = line
    sys.stdout.write("\r" + line.ljust(140))
    sys.stdout.flush()

# ---------- Smoke test (opcional) ----------
def hardware_smoke_test(ctx, seconds=0.6):
    print(f"[INFO] NeoPixel bpp={getattr(pixels,'bpp','n/a')} order={ORDER} is_rgbw={LED_IS_RGBW}")
    colors=[(255,0,0),(0,255,0),(0,0,255),(255,255,255),(0,0,0)]
    for rgb in colors:
        arr = np.tile(np.array(rgb, dtype=np.uint8), (LED_COUNT,1))
        ctx.to_pixels_and_show(arr); time.sleep(seconds)

# ---------- Render Thread ----------
def render_thread(ctx, effects):
    global stop_flag, latency_ms_ema, shared_active, shared_dyn_floor
    current_effect = 0
    last_effect_change = time.time()
    effect_max_interval = 300.0  # 5 min

    SIGNAL_HOLD = SIGNAL_HOLD_MS / 1000.0
    RENDER_DT   = 1.0 / max(1, CFG_VIS_FPS)

    signal_active_until = 0.0
    last_rx_ts = time.time()
    last_render_ts = 0.0

    # fallback otimizado
    def fallback_hsv(bands_u8, active):
        if not active:
            return np.zeros((LED_COUNT, 3), dtype=np.uint8)
        v = bands_u8[_band_to_led]
        v = GAMMA_LUT[v]
        return ctx.hsv_to_rgb_bytes_vec(H_FIXED, S_FIXED, v)

    while not stop_flag:
        now = time.time()

        # hot-reload CFG
        desired_hold = SIGNAL_HOLD_MS / 1000.0
        desired_render_dt = 1.0 / max(1, CFG_VIS_FPS)
        if abs(desired_hold - SIGNAL_HOLD) > 1e-6:
            SIGNAL_HOLD = desired_hold
        if abs(desired_render_dt - RENDER_DT) > 1e-6:
            RENDER_DT = desired_render_dt

        # teclas
        global pending_key_change
        if pending_key_change is not None:
            if pending_key_change == 'next':
                current_effect = (current_effect + 1) % len(effects)
            else:
                current_effect = (current_effect - 1) % len(effects)
            pending_key_change = None
            apply_new_colorset()
            last_effect_change = now

        # idle
        if (now - last_rx_ts) > 2.0:
            if shared_active:
                with shared_lock:
                    shared_active = False
            if now >= signal_active_until:
                pixels.fill((0,0,0)); pixels.show()
                unified_status_line("Idle", ctx, False, EXPECTED_BANDS, CFG_FPS, CFG_VIS_FPS, current_palette_name)
                time.sleep(0.01)
                continue

        # copia estado
        with shared_lock:
            b = shared_bands.copy()
            beat = shared_beat
            active = shared_active
            dyn_floor = shared_dyn_floor
            kick_intensity = shared_kick_intensity
        last_rx_ts = now
        signal_active_until = now + SIGNAL_HOLD
        ctx.dynamic_floor = dyn_floor

        # auto-rotate
        if (now - last_effect_change) > effect_max_interval:
            current_effect = (current_effect + 1) % len(effects)
            apply_new_colorset()
            last_effect_change = now

        # render a cada RENDER_DT
        if (now - last_render_ts) >= RENDER_DT:
            last_render_ts = now
            try:
                if not BYPASS_EFFECTS:
                    name, func = effects[current_effect]
                    rgb = func(b if active else np.zeros_like(b),
                               beat if active else 0,
                               active)
                else:
                    name = "Fallback HSV"
                    rgb = fallback_hsv(b, active)
                ctx.to_pixels_and_show(rgb)
            except Exception as e:
                rgb = np.tile(np.array([b.mean(), b.mean(), b.mean()], dtype=np.uint8), (LED_COUNT, 1))
                ctx.to_pixels_and_show(rgb)
                name = f"Fallback Gray ({e.__class__.__name__})"
        else:
            name = effects[current_effect][0] if not BYPASS_EFFECTS else "Fallback HSV"

        unified_status_line(name, ctx, active, EXPECTED_BANDS, CFG_FPS, CFG_VIS_FPS, current_palette_name)
        time.sleep(max(0.0, RENDER_DT - (time.time() - now)))

# ---------- Main ----------
def main():
    global stop_flag

    # Threads auxiliares
    threading.Thread(target=timesync_tcp_server, daemon=True).start()
    threading.Thread(target=udp_receiver,        daemon=True).start()
    threading.Thread(target=palette_worker,      daemon=True).start()
    threading.Thread(target=key_listener,        daemon=True).start()

    # Contexto e efeitos
    ctx = FXContext(pixels, LED_COUNT, base_hue_offset, hue_seed, base_saturation,
                    current_budget_a=12.0, ma_per_channel=20.0, idle_ma_per_led=1.0)  # <-- orçamento mais baixo
    ctx.metrics = None
    effects = build_effects(ctx)

    if ENABLE_SMOKE_TEST:
        hardware_smoke_test(ctx, seconds=0.6)

    # Inicia render thread
    render_t = threading.Thread(target=render_thread, args=(ctx, effects), daemon=True)
    render_t.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        stop_flag = True
        palette_thread_stop.set()
        palette_wake.set()
        pixels.fill((0,0,0)); pixels.show()
        sys.stdout.write("\n"); sys.stdout.flush()

if __name__ == "__main__":
    main()