#!/usr/bin/env python3
# sync.py — Raspberry Pi 3B renderer (A2/A1 + B0 config) — VERSÃO CORRIGIDA
# • CPU < 30% com 45/30 FPS
# • Threads separadas (UDP + Render)
# • Fallback HSV otimizado
# • Palette dorme
# • Status cache
# • Compatível com effects.py original

import socket, time, board, neopixel, random, threading, sys, select, tty, termios, os
import numpy as np
from collections import deque
from fxcore import FXContext
from effects import build_effects

# ---------- Protocolo ----------
PKT_AUDIO_V2 = 0xA2
PKT_AUDIO    = 0xA1
PKT_CFG      = 0xB0
LEN_A2_MIN = 13
LEN_A1_MIN = 11
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
BYPASS_EFFECTS   = False
ENABLE_SMOKE_TEST= False
STATUS_MAX_HZ    = 15

# ---------- UDP ----------
udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
try:
    udp_sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 22)
    udp_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
except Exception: pass
udp_sock.bind(("0.0.0.0", UDP_PORT))
udp_sock.setblocking(False)  # <-- non-blocking para não travar

# ---------- Estado base ----------
EXPECTED_BANDS = 150
CFG_FPS        = 45
SIGNAL_HOLD_MS = 500
CFG_VIS_FPS    = 30

rx_count = drop_len = drop_hdr = 0
time_offset_ns = 0
time_sync_ready = False
latency_ms_ema = None

# ---------- Shared state (UDP → Render) ----------
class SharedState:
    def __init__(self):
        self.bands = np.zeros(EXPECTED_BANDS, dtype=np.uint8)
        self.beat = 0
        self.active = False
        self.dyn_floor = 0
        self.kick_intensity = 0
        self.last_update = 0.0

shared = SharedState()
shared_lock = threading.Lock()
stop_flag = threading.Event()

# ---------- Caches ----------
LED_IDX = np.arange(LED_COUNT, dtype=np.int32)
H_FIXED = ((LED_IDX * 256) // LED_COUNT).astype(np.uint8)
S_FIXED = np.full(LED_COUNT, 255, dtype=np.uint8)
GAMMA_LUT = (((np.arange(256, dtype=np.float32) / 255.0) ** 1.6) * 255.0).astype(np.uint8)

# mapeamento LED -> banda (recriado só se mudar)
_band_to_led = np.repeat(np.arange(EXPECTED_BANDS), (LED_COUNT // EXPECTED_BANDS) + 1)[:LED_COUNT]
_band_to_led_for = EXPECTED_BANDS

# status
_status_min_period = 1.0 / STATUS_MAX_HZ
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
    global time_offset_ns, time_sync_ready
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("0.0.0.0", TCP_TIME_PORT)); srv.listen(1)
    while not stop_flag.is_set():
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

# ---------- Paleta ----------
STRATEGIES = ["complementar","analoga","triade","tetrade","split"]
def build_palette_from_strategy(h0, strategy, num_colors=5):
    import colorsys
    def hsv_to_rgb_bytes(h, s, v):
        r,g,b = colorsys.hsv_to_rgb(h,s,v); return (int(r*255), int(g*255), int(b*255))
    if strategy == "complementar":
        hA=h0%1.0; hB=(h0+0.5)%1.0; d=0.06
        hues=[hA,(hA+d)%1.0,(hA-d)%1.0,hB,(hB+d)%1.0,(hB-d)%1.0]
        pal = []
        while len(pal)<num_colors:
            h=random.choice(hues)
            s = random.choice([0.60,0.70,0.80,0.85]) if len(pal)>1 else (0.75 if len(pal)==0 else 0.65)
            v = random.choice([0.70,0.80,0.90,1.00]) if len(pal)>1 else (0.95 if len(pal)==0 else 0.75)
            pal.append(hsv_to_rgb_bytes(h,s,v))
        return pal
    # ... (outras estratégias simplificadas)
    step = 0.10; hues = [(h0 + k*step)%1.0 for k in (-2,-1,0,1,2)]
    pal = []
    while len(pal)<num_colors:
        h = random.choice(hues)
        s = random.uniform(0.55,0.85)
        v = random.uniform(0.70,1.00)
        pal.append(hsv_to_rgb_bytes(h,s,v))
    return pal

current_palette_name = "analoga"
palette_queue = deque(maxlen=8)
palette_wake = threading.Event()

def palette_worker():
    while not stop_flag.is_set():
        if len(palette_queue) < 8:
            strategy = random.choice(STRATEGIES)
            h0 = random.random()
            pal = build_palette_from_strategy(h0, strategy, 5)
            palette_queue.append((pal, strategy, h0))
        else:
            palette_wake.wait(2.0)
            palette_wake.clear()

def apply_new_colorset():
    global current_palette_name
    if palette_queue:
        palette_wake.set()
        _, current_palette_name, _ = palette_queue.popleft()

# ---------- Teclado ----------
def key_listener():
    old = termios.tcgetattr(sys.stdin)
    try:
        tty.setcbreak(sys.stdin.fileno())
        while not stop_flag.is_set():
            if select.select([sys.stdin], [], [], 0)[0]:
                k = sys.stdin.read(1)
                if k == 'n': shared.pending_key = 'next'
                elif k == 'p': shared.pending_key = 'prev'
                elif k == 'q': os._exit(0)
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old)

# ---------- UDP Receiver ----------
def udp_receiver():
    global rx_count, drop_len, drop_hdr, EXPECTED_BANDS, CFG_FPS, SIGNAL_HOLD_MS, CFG_VIS_FPS
    global _band_to_led, _band_to_led_for
    while not stop_flag.is_set():
        try:
            data, _ = udp_sock.recvfrom(4096)
            if not data: continue
            n = len(data)
            if n < 1: drop_len += 1; continue
            hdr = data[0]

            # A2
            if hdr == PKT_AUDIO_V2 and n >= LEN_A2_MIN:
                nb = n - LEN_A2_MIN
                if nb <= 0: drop_len += 1; continue
                bands = np.frombuffer(data[9:9+nb], dtype=np.uint8)
                beat = data[9+nb]
                dyn_floor = data[11+nb]
                kick = data[12+nb]
                if nb != EXPECTED_BANDS:
                    EXPECTED_BANDS = nb
                    _band_to_led = np.repeat(np.arange(nb), (LED_COUNT // nb) + 1)[:LED_COUNT]
                    _band_to_led_for = nb
                with shared_lock:
                    shared.bands[:nb] = bands
                    shared.beat = beat
                    shared.active = True
                    shared.dyn_floor = dyn_floor
                    shared.kick_intensity = kick
                    shared.last_update = time.time()
                rx_count += 1
                continue

            # A1 (legado)
            if hdr == PKT_AUDIO and n >= LEN_A1_MIN:
                nb = n - LEN_A1_MIN
                if nb <= 0: drop_len += 1; continue
                bands = np.frombuffer(data[9:9+nb], dtype=np.uint8)
                beat = data[9+nb]
                if nb != EXPECTED_BANDS:
                    EXPECTED_BANDS = nb
                    _band_to_led = np.repeat(np.arange(nb), (LED_COUNT // nb) + 1)[:LED_COUNT]
                    _band_to_led_for = nb
                with shared_lock:
                    shared.bands[:nb] = bands
                    shared.beat = beat
                    shared.active = True
                    shared.dyn_floor = 0
                    shared.kick_intensity = 0
                    shared.last_update = time.time()
                rx_count += 1
                continue

            # B0 config
            if hdr == PKT_CFG and n == LEN_CFG:
                nb = data[2] | (data[3] << 8)
                fps = data[4] | (data[5] << 8)
                hold = data[6] | (data[7] << 8)
                vis = data[8] | (data[9] << 8)
                if nb > 0 and nb != EXPECTED_BANDS:
                    EXPECTED_BANDS = nb
                    _band_to_led = np.repeat(np.arange(nb), (LED_COUNT // nb) + 1)[:LED_COUNT]
                    _band_to_led_for = nb
                if fps > 0: CFG_FPS = fps
                if hold >= 0: SIGNAL_HOLD_MS = hold
                if vis > 0: CFG_VIS_FPS = vis
                continue

            if hdr in (PKT_AUDIO, PKT_AUDIO_V2, PKT_CFG):
                drop_len += 1
            else:
                drop_hdr += 1
        except BlockingIOError:
            time.sleep(0.001)
        except Exception:
            time.sleep(0.001)

# ---------- Status ----------
def unified_status_line(effect_name, ctx, active, bands, fps, vis_fps, pal_name):
    global _last_status_ts, _last_status_line
    now = time.time()
    if (now - _last_status_ts) < _status_min_period:
        return
    _last_status_ts = now
    curr = ctx.current_a_ema if ctx.current_a_ema is not None else 0.0
    poww = ctx.power_w_ema if ctx.power_w_ema is not None else 0.0
    cap = ctx.last_cap_scale if ctx.last_cap_scale is not None else 1.0
    lat = f"{latency_ms_ema:.1f}" if latency_ms_ema is not None else "?"
    line = f"RX:{rx_count} dropL:{drop_len} dropH:{drop_hdr} | Eff:{effect_name} Pal:{pal_name} | active:{'Y' if active else 'N'} | bands:{bands} fps:{fps} vis:{vis_fps} | I:{curr:.2f}A P:{poww:.1f}W cap:{cap:.2f} | lat:{lat}ms"
    if line == _last_status_line: return
    _last_status_line = line
    sys.stdout.write("\r" + line.ljust(140))
    sys.stdout.flush()

# ---------- Render Thread ----------
def render_thread(ctx, effects):
    current_effect = 0
    last_effect_change = time.time()
    effect_max_interval = 300.0
    RENDER_DT = 1.0 / max(1, CFG_VIS_FPS)
    SIGNAL_HOLD = SIGNAL_HOLD_MS / 1000.0
    signal_active_until = 0.0
    last_render_ts = 0.0
    shared.pending_key = None

    def fallback_hsv(bands_u8):
        v = bands_u8[_band_to_led]
        v = GAMMA_LUT[v]
        return ctx.hsv_to_rgb_bytes_vec(H_FIXED, S_FIXED, v)

    while not stop_flag.is_set():
        now = time.time()

        # CFG hot-reload
        SIGNAL_HOLD = SIGNAL_HOLD_MS / 1000.0
        RENDER_DT = 1.0 / max(1, CFG_VIS_FPS)

        # teclas
        if hasattr(shared, 'pending_key') and shared.pending_key:
            if shared.pending_key == 'next':
                current_effect = (current_effect + 1) % len(effects)
            else:
                current_effect = (current_effect - 1) % len(effects)
            shared.pending_key = None
            apply_new_colorset()
            last_effect_change = now

        # idle
        with shared_lock:
            active = shared.active and (now - shared.last_update) < 2.0
            b = shared.bands.copy()
            beat = shared.beat
            dyn_floor = shared.dyn_floor
        if not active:
            if now >= signal_active_until:
                pixels.fill((0,0,0)); pixels.show()
                unified_status_line("Idle", ctx, False, EXPECTED_BANDS, CFG_FPS, CFG_VIS_FPS, current_palette_name)
                time.sleep(0.01)
                continue
        else:
            signal_active_until = now + SIGNAL_HOLD
            ctx.dynamic_floor = dyn_floor

        # auto-rotate
        if (now - last_effect_change) > effect_max_interval:
            current_effect = (current_effect + 1) % len(effects)
            apply_new_colorset()
            last_effect_change = now

        # render
        if (now - last_render_ts) >= RENDER_DT:
            last_render_ts = now
            try:
                if not BYPASS_EFFECTS:
                    name, func = effects[current_effect]
                    rgb = func(b, beat, active)
                else:
                    name = "Fallback HSV"
                    rgb = fallback_hsv(b)
                ctx.to_pixels_and_show(rgb)
            except Exception as e:
                rgb = np.tile([100,100,100], (LED_COUNT,1)).astype(np.uint8)
                ctx.to_pixels_and_show(rgb)
                name = f"ERR ({e.__class__.__name__})"
        else:
            name = effects[current_effect][0] if not BYPASS_EFFECTS and effects else "Fallback HSV"

        unified_status_line(name, ctx, active, EXPECTED_BANDS, CFG_FPS, CFG_VIS_FPS, current_palette_name)
        time.sleep(max(0, RENDER_DT - (time.time() - now)))

# ---------- Main ----------
def main():
    global stop_flag
    stop_flag.clear()

    threading.Thread(target=timesync_tcp_server, daemon=True).start()
    threading.Thread(target=udp_receiver, daemon=True).start()
    threading.Thread(target=palette_worker, daemon=True).start()
    threading.Thread(target=key_listener, daemon=True).start()

    ctx = FXContext(pixels, LED_COUNT, 0, 0, 210,
                    current_budget_a=12.0, ma_per_channel=20.0, idle_ma_per_led=1.0)
    effects = build_effects(ctx)

    if ENABLE_SMOKE_TEST:
        colors = [(255,0,0),(0,255,0),(0,0,255),(255,255,255),(0,0,0)]
        for c in colors:
            ctx.to_pixels_and_show(np.tile(c, (LED_COUNT,1)))
            time.sleep(0.6)

    render_t = threading.Thread(target=render_thread, args=(ctx, effects), daemon=True)
    render_t.start()

    try:
        while True: time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        stop_flag.set()
        palette_wake.set()
        pixels.fill((0,0,0)); pixels.show()
        print("\n[INFO] Encerrado.")

if __name__ == "__main__":
    main()