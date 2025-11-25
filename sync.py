#!/usr/bin/env python3
# sync.py — Raspberry Pi renderer (A2/A1 + B0 + B1 reset) com log single/multi linha
import socket, time, board, neopixel, random, threading, sys, select, tty, termios, os, argparse
import numpy as np
from datetime import datetime, date, timedelta
from collections import deque
from fxcore import FXContext
from effects import build_effects

# -------------------- Protocolo --------------------
PKT_AUDIO_V2 = 0xA2
PKT_AUDIO = 0xA1
PKT_CFG = 0xB0
PKT_RESET = 0xB1
LEN_A2_MIN = 13
LEN_A1_MIN = 11
LEN_CFG = 10
UDP_PORT = 5005
TCP_TIME_PORT = 5006

# -------------------- LEDs --------------------
LED_COUNT = 300
LED_PIN = board.D18
ORDER = neopixel.GRB
BRIGHTNESS = 0.8
LED_IS_RGBW = False
pixels = (
    neopixel.NeoPixel(LED_PIN, LED_COUNT, brightness=BRIGHTNESS, auto_write=False,
                      pixel_order=neopixel.GRBW, bpp=4)
    if LED_IS_RGBW else
    neopixel.NeoPixel(LED_PIN, LED_COUNT, brightness=BRIGHTNESS, auto_write=False,
                      pixel_order=ORDER)
)
BYPASS_EFFECTS = False
ENABLE_SMOKE_TEST = False
STATUS_MAX_HZ = 10

# -------------------- Sockets --------------------
udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
try:
    udp_sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 22)
    udp_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
except Exception:
    pass
udp_sock.bind(("0.0.0.0", UDP_PORT))
udp_sock.setblocking(True)

# -------------------- Solar schedule (sunrise/sunset) --------------------
# Default location: Curitiba, Paraná, Brazil
# Latitude/Longitude values can be overridden by editing these variables
SOLAR_DEFAULT_NAME = "Curitiba"
SOLAR_DEFAULT_REGION = "Paraná"
SOLAR_DEFAULT_COUNTRY = "Brazil"
SOLAR_LATITUDE = -25.4278
SOLAR_LONGITUDE = -49.2733
SOLAR_TIMEZONE = "America/Sao_Paulo"
# When True, the schedule will turn LEDs OFF at sunrise and ON at (sunset - 30min)
ENABLE_SOLAR_SCHEDULE = True

# Try to import astral/zoneinfo; if unavailable, solar scheduling will be disabled
try:
    from astral import Observer
    from astral.sun import sun as astral_sun
    try:
        from zoneinfo import ZoneInfo
    except Exception:
        ZoneInfo = None
    ASTRAL_AVAILABLE = True
except Exception:
    Observer = None
    astral_sun = None
    ZoneInfo = None
    ASTRAL_AVAILABLE = False

def _compute_sun_times_for(d: date):
    """Return (sunrise_dt, sunset_dt) for date d as timezone-aware datetimes.
    Returns (None, None) if calculation is not available."""
    # prefer astral when available, otherwise use simple fallback (06:00/18:00 local)
    tz = ZoneInfo(SOLAR_TIMEZONE) if ZoneInfo is not None else None
    if ASTRAL_AVAILABLE:
        try:
            observer = Observer(latitude=SOLAR_LATITUDE, longitude=SOLAR_LONGITUDE)
            s = astral_sun(observer, date=d, tzinfo=tz)
            sunrise = s.get('sunrise')
            sunset = s.get('sunset')
            return sunrise, sunset
        except Exception:
            # fall through to fallback heuristic
            pass
    # fallback times (naive but predictable): sunrise at 06:00, sunset at 18:00
    try:
        sunrise = datetime(d.year, d.month, d.day, 6, 0, tzinfo=tz)
        sunset = datetime(d.year, d.month, d.day, 18, 0, tzinfo=tz)
        return sunrise, sunset
    except Exception:
        return None, None

# -------------------- Estado dinâmico --------------------
EXPECTED_BANDS = 150
CFG_FPS = 75
SIGNAL_HOLD_MS = 500
CFG_VIS_FPS = 45
rx_count = drop_len = drop_hdr = 0

time_offset_ns = 0
time_sync_ready = False
latency_ms_ema = None
pending_key_change = None
latest_packet = None
latest_lock = threading.Lock()
stop_flag = False
reset_flag = False

# -------------------- Log mode --------------------
LOG_MODE_SINGLE_LINE = True  # ajustado via argparse
def _w(s: str, *, same_line: bool):
    if same_line:
        sys.stdout.write("\r" + s)
        sys.stdout.flush()
    else:
        print(s, flush=True)
def log_info(msg: str):
    if LOG_MODE_SINGLE_LINE:
        _w(msg.ljust(140), same_line=True)
    else:
        print(msg)
def log_event_inline(msg: str):
    if LOG_MODE_SINGLE_LINE:
        _w(msg.ljust(140), same_line=True)
    else:
        print(msg)

# -------------------- Caches / LUTs --------------------
LED_IDX = np.arange(LED_COUNT, dtype=np.int32)
H_FIXED = ((LED_IDX * 256) // LED_COUNT).astype(np.uint16)  # 0..255 u16
GAMMA_LUT = (((np.arange(256, dtype=np.float32) / 255.0) ** 1.6) * 255.0).astype(np.uint8)
_band_idx = None
_band_idx_for = -1
_status_min_period = 1.0 / max(1, STATUS_MAX_HZ)
_last_status_ts = 0.0
HSV_LUT_FLAT = None
def _build_hv_lut_flat(s_fixed: int = 255):
    h_vals = np.arange(256, dtype=np.uint8)
    v_vals = np.arange(256, dtype=np.uint8)
    hh, vv = np.meshgrid(h_vals, v_vals, indexing='ij')
    h = hh.reshape(-1)
    v = vv.reshape(-1)
    s = np.full_like(h, s_fixed, dtype=np.uint8)
    rgb_flat = FXContext.hsv_to_rgb_bytes_vec(h, s, v)
    return rgb_flat
HSV_LUT_FLAT = _build_hv_lut_flat(255)
H_IDX_BASE_U16= (H_FIXED << 8).astype(np.uint16)
_vals_buf = np.empty(LED_COUNT, dtype=np.uint8)
_v_buf = np.empty(LED_COUNT, dtype=np.uint8)
_idx_buf_u16 = np.empty(LED_COUNT, dtype=np.uint16)
_ZERO_CACHE = {}
def _zeros_of(n):
    arr = _ZERO_CACHE.get(n)
    if arr is None:
        arr = np.zeros(n, dtype=np.uint8)
        _ZERO_CACHE[n] = arr
    return arr

# -------------------- Utils --------------------
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
    srv.bind(("0.0.0.0", TCP_TIME_PORT)); srv.listen(1)
    while True:
        try:
            conn, _ = srv.accept()
            with conn:
                conn.settimeout(1.2)
                while True:
                    hdr = _recv_exact(conn, 3)
                    if hdr == b"TS1":
                        t0 = _recv_exact(conn, 8)
                        tr = time.monotonic_ns()
                        conn.sendall(b"TS2" + t0 + tr.to_bytes(8, 'little'))
                    elif hdr == b"TS3":
                        off = _recv_exact(conn, 8)
                        time_offset_ns = int.from_bytes(off, 'little', signed=True)
                        time_sync_ready = True
                        conn.sendall(b"TS3" + int(time_offset_ns).to_bytes(8, 'little', signed=True))
                        break
                    else:
                        break
        except Exception:
            time.sleep(0.05)

# -------------------- Paleta / cores --------------------
STRATEGIES = ["complementar", "analoga", "triade", "tetrade", "split"]
def clamp01(x): return x % 1.0
def build_palette_from_strategy(h0, strategy, num_colors=5):
    import colorsys
    def hsv_to_rgb_bytes(h, s, v):
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        return (int(r * 255), int(g * 255), int(b * 255))
    if strategy == "complementar":
        d = 0.06
        hA = h0 % 1.0; hB = (h0 + 0.5) % 1.0
        hues = [hA, (hA + d)%1.0, (hA - d)%1.0, hB, (hB + d)%1.0, (hB - d)%1.0]
        s_choices = [0.60, 0.70, 0.80, 0.85]; v_choices = [0.70, 0.80, 0.90, 1.00]
        pal = []
        while len(pal) < num_colors:
            h = random.choice(hues)
            s = 0.75 if len(pal)==0 else (0.65 if len(pal)==1 else random.choice(s_choices))
            v = 0.95 if len(pal)==0 else (0.75 if len(pal)==1 else random.choice(v_choices))
            pal.append(hsv_to_rgb_bytes(h, s, v))
        return pal
    elif strategy == "analoga":
        step = 0.10
        hues = [clamp01(h0 + k * step) for k in (-2, -1, 0, 1, 2)]
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
current_palette = [(255, 255, 255)] * 5
current_palette_name = "analoga"
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
    if palette_queue:
        return palette_queue.popleft()
    return None
def apply_new_colorset():
    global base_hue_offset, hue_seed, base_saturation, current_palette, current_palette_name
    pal_data = get_next_palette()
    if pal_data:
        current_palette, current_palette_name, _ = pal_data
    else:
        strategy = random.choice(STRATEGIES); h0 = random.random()
        current_palette = build_palette_from_strategy(h0, strategy, num_colors=5)
        current_palette_name = strategy
    base_hue_offset = random.randint(0, 255)
    hue_seed = random.randint(0, 255)
    base_saturation = random.randint(190, 230)
def push_palette_to_ctx(ctx: FXContext):
    ctx.base_hue_offset = int(base_hue_offset)
    ctx.hue_seed = int(hue_seed)
    ctx.base_saturation = int(base_saturation)

# -------------------- Teclado --------------------
def key_listener():
    global pending_key_change, reset_flag
    try:
        old = termios.tcgetattr(sys.stdin)
    except Exception:
        return
    try:
        tty.setcbreak(sys.stdin.fileno())
        while True:
            if select.select([sys.stdin], [], [], 0.05)[0]:
                k = sys.stdin.read(1)
                if k == 'n':
                    pending_key_change = 'next'
                elif k == 'p':
                    pending_key_change = 'prev'
                elif k == 'r':
                    reset_flag = True
                elif k == 'q':
                    os._exit(0)
    finally:
        try:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old)
        except Exception:
            pass

# -------------------- UDP Receiver + CFG --------------------
def udp_receiver():
    global latest_packet, rx_count, drop_hdr, drop_len
    global EXPECTED_BANDS, CFG_FPS, SIGNAL_HOLD_MS, CFG_VIS_FPS, reset_flag
    log_info(f"[INFO] UDP 0.0.0.0:{UDP_PORT} (A2/A1 + B0 + B1)")
    while not stop_flag:
        try:
            data, _ = udp_sock.recvfrom(4096)
            n = len(data)
            if n < 1:
                drop_len += 1
                continue
            hdr = data[0]
            if hdr == PKT_RESET:
                reset_flag = True
                log_info("[RST] B1 recebido")
                continue
            if hdr == PKT_AUDIO_V2 and n >= LEN_A2_MIN:
                nb = n - LEN_A2_MIN
                if nb <= 0:
                    drop_len += 1
                    continue
                ts_pc = int.from_bytes(data[1:9], 'little')
                bands = np.frombuffer(memoryview(data)[9:9+nb], dtype=np.uint8)
                beat = data[9 + nb]
                trans = data[10 + nb]
                dyn_floor = data[11 + nb]
                kick_intensity = data[12 + nb]
                if EXPECTED_BANDS != nb:
                    EXPECTED_BANDS = int(nb)
                with latest_lock:
                    latest_packet = (bands.copy(), beat, trans, ts_pc, dyn_floor, kick_intensity)
                rx_count += 1
                continue
            if hdr == PKT_AUDIO and n >= LEN_A1_MIN:
                nb = n - LEN_A1_MIN
                if nb <= 0:
                    drop_len += 1
                    continue
                ts_pc = int.from_bytes(data[1:9], 'little')
                bands = np.frombuffer(memoryview(data)[9:9+nb], dtype=np.uint8)
                beat = data[9 + nb]
                trans = data[10 + nb]
                if EXPECTED_BANDS != nb:
                    EXPECTED_BANDS = int(nb)
                with latest_lock:
                    latest_packet = (bands.copy(), beat, trans, ts_pc, 0, 0)
                rx_count += 1
                continue
            if hdr == PKT_CFG and n == LEN_CFG:
                # le16 helper explícito
                nb   = data[2] | (data[3] << 8)
                fps  = data[4] | (data[5] << 8)
                hold = data[6] | (data[7] << 8)
                vis  = data[8] | (data[9] << 8)
                if nb  > 0: EXPECTED_BANDS = int(nb)
                if fps > 0: CFG_FPS = int(fps)
                if hold>=0: SIGNAL_HOLD_MS = int(hold)
                if vis > 0: CFG_VIS_FPS = int(vis)
                log_info(f"[CFG] bands={EXPECTED_BANDS} fps={CFG_FPS} hold={SIGNAL_HOLD_MS} vis_fps={CFG_VIS_FPS}")
                continue
            if hdr in (PKT_AUDIO, PKT_AUDIO_V2, PKT_CFG):
                drop_len += 1
            else:
                drop_hdr += 1
        except Exception:
            time.sleep(0.001)

# -------------------- Status --------------------
_status_min_period = 1.0 / max(1, STATUS_MAX_HZ)
_last_status_ts = 0.0
def unified_status_line(effect_name, ctx, active, bands, fps, vis_fps, pal_name):
    global _last_status_ts
    now = time.time()
    if (now - _last_status_ts) < _status_min_period:
        return
    _last_status_ts = now
    curr = ctx.current_a_ema if ctx.current_a_ema is not None else 0.0
    poww = ctx.power_w_ema if ctx.power_w_ema is not None else 0.0
    cap = ctx.last_cap_scale if ctx.last_cap_scale is not None else 1.0
    lat = f"{latency_ms_ema:.1f}" if latency_ms_ema is not None else "?"
    line = (f"RX:{rx_count} dropL:{drop_len} dropH:{drop_hdr} "
            f"Eff:{effect_name} Pal:{pal_name} "
            f"active:{'Y' if active else 'N'} "
            f"bands:{bands} fps:{fps} vis:{vis_fps} "
            f"I:{curr:.2f}A P:{poww:.1f}W cap:{cap:.2f} "
            f"lat:{lat}ms")
    if LOG_MODE_SINGLE_LINE:
        _w(line.ljust(140), same_line=True)
    else:
        print(line)

# -------------------- Smoke --------------------
def hardware_smoke_test(ctx, seconds=0.6):
    log_info(f"[INFO] NeoPixel bpp={getattr(pixels,'bpp','n/a')} order={ORDER} rgbw={LED_IS_RGBW}")
    colors = [(255,0,0),(0,255,0),(0,0,255),(255,255,255),(0,0,0)]
    for rgb in colors:
        arr = np.tile(np.array(rgb, dtype=np.uint8), (LED_COUNT, 1))
        ctx.to_pixels_and_show(arr); time.sleep(seconds)

# -------------------- Main --------------------
def main():
    global stop_flag, latency_ms_ema, latest_packet, _band_idx, _band_idx_for, reset_flag, LOG_MODE_SINGLE_LINE
    ap = argparse.ArgumentParser()
    ap.add_argument('--log-mode', choices=['single','multi'], default='single')
    args, _unknown = ap.parse_known_args()
    LOG_MODE_SINGLE_LINE = (args.log_mode == 'single')

    threading.Thread(target=timesync_tcp_server, daemon=True).start()
    threading.Thread(target=udp_receiver, daemon=True).start()
    threading.Thread(target=palette_worker, daemon=True).start()
    threading.Thread(target=key_listener, daemon=True).start()

    ctx = FXContext(pixels, LED_COUNT, base_hue_offset, hue_seed, base_saturation,
                    current_budget_a=18.0, ma_per_channel=20.0, idle_ma_per_led=1.0)
    ctx.metrics = None
    effects = build_effects(ctx)
    push_palette_to_ctx(ctx)

    if ENABLE_SMOKE_TEST:
        hardware_smoke_test(ctx, seconds=0.6)

    current_effect = 0
    last_effect_change = time.time()
    effect_max_interval = 300.0

    last_bands = np.zeros(EXPECTED_BANDS, dtype=np.uint8)
    last_beat = 0

    SIGNAL_HOLD = SIGNAL_HOLD_MS / 1000.0
    FRAME_DT = 1.0 / max(1, CFG_FPS)
    RENDER_DT = 1.0 / max(1, CFG_VIS_FPS)

    signal_active_until = 0.0
    already_off = False
    last_rx_ts = time.time()
    next_frame = time.time()
    last_render_ts = 0.0
    _last_solar_log = 0.0
    # --- Solar schedule state ---
    solar_date = None
    solar_sunrise = None
    solar_on_time = None
    if ENABLE_SOLAR_SCHEDULE:
        solar_date = date.today()
        sr, ss = _compute_sun_times_for(solar_date)
        if sr is not None and ss is not None:
            solar_sunrise = sr
            solar_on_time = ss - timedelta(minutes=30)
        else:
            solar_date = None
            solar_sunrise = None
            solar_on_time = None
    # Diagnostic log about solar scheduling
    try:
        if ENABLE_SOLAR_SCHEDULE:
            log_info(f"[SOLAR] scheduling enabled; astral={'yes' if ASTRAL_AVAILABLE else 'no'}")
            if solar_sunrise is not None and solar_on_time is not None:
                tzname = SOLAR_TIMEZONE if ZoneInfo is not None else 'local'
                log_info(f"[SOLAR] sunrise={solar_sunrise.isoformat()} on_time={solar_on_time.isoformat()} tz={tzname}")
            else:
                log_info("[SOLAR] sunrise/on_time unavailable (scheduling will use fallback times)")
    except Exception:
        pass

    # Enforce initial LED state according to solar schedule (startup behavior)
    try:
        if ENABLE_SOLAR_SCHEDULE and solar_sunrise is not None and solar_on_time is not None:
            tz = ZoneInfo(SOLAR_TIMEZONE) if ZoneInfo is not None else None
            now_dt = datetime.now(tz) if tz is not None else datetime.now()
            if solar_sunrise <= now_dt < solar_on_time:
                pixels.fill((0,0,0)); pixels.show()
                already_off = True
                log_event_inline(f"[SOLAR] LEDs desligados na inicialização ({solar_date.isoformat()})")
            else:
                already_off = False
                log_event_inline(f"[SOLAR] LEDs ativos na inicialização ({solar_date.isoformat()})")
    except Exception:
        # don't prevent startup on any error here
        pass

    try:
        while True:
            now = time.time()
            # check solar schedule (if available) and force LEDs off between
            # sunrise and (sunset - 30 minutes)
            if ENABLE_SOLAR_SCHEDULE:
                try:
                    tz = ZoneInfo(SOLAR_TIMEZONE) if ZoneInfo is not None else None
                    now_dt = datetime.now(tz) if tz is not None else datetime.now()
                    # recompute if date changed
                    if solar_date is None or now_dt.date() != solar_date:
                        solar_date = now_dt.date()
                        sr, ss = _compute_sun_times_for(solar_date)
                        if sr is not None and ss is not None:
                            solar_sunrise = sr
                            solar_on_time = ss - timedelta(minutes=30)
                        else:
                            solar_sunrise = None
                            solar_on_time = None
                    # if we have valid sunrise/on-time, enforce off window
                    if solar_sunrise is not None and solar_on_time is not None:
                        # log diagnostics at most once every 60s
                        try:
                            if (time.time() - _last_solar_log) > 60.0:
                                cond = (solar_sunrise <= now_dt < solar_on_time)
                                _last_solar_log = time.time()
                                log_info(f"[SOLARDBG] now={now_dt.isoformat()} sunrise={solar_sunrise.isoformat()} on_time={solar_on_time.isoformat()} cond={cond} tz={getattr(now_dt,'tzinfo',None)}")
                        except Exception:
                            pass

                        if solar_sunrise <= now_dt < solar_on_time:
                            # within forced-off window
                            if not already_off:
                                pixels.fill((0,0,0)); pixels.show()
                                already_off = True
                                log_event_inline(f"[SOLAR] LEDs desligados ({solar_date.isoformat()})")
                            unified_status_line("SolarOff", ctx, False, EXPECTED_BANDS, CFG_FPS, CFG_VIS_FPS, current_palette_name)
                            next_frame += FRAME_DT
                            sl = next_frame - time.time()
                            if sl > 0: time.sleep(sl)
                            else: next_frame = time.time()
                            continue
                        else:
                            # if we are past on_time and were off, re-enable
                            if already_off and now_dt >= solar_on_time:
                                already_off = False
                                log_event_inline(f"[SOLAR] LEDs ligados (após {solar_on_time.time().isoformat()})")
                except Exception:
                    # ignore solar scheduling errors and proceed normally
                    pass
            desired_frame_dt = 1.0 / max(1, CFG_FPS)
            desired_hold = SIGNAL_HOLD_MS / 1000.0
            desired_render_dt = 1.0 / max(1, CFG_VIS_FPS)
            if abs(desired_frame_dt - FRAME_DT) > 1e-6: FRAME_DT = desired_frame_dt
            if abs(desired_hold - SIGNAL_HOLD) > 1e-6: SIGNAL_HOLD = desired_hold
            if abs(desired_render_dt - RENDER_DT) > 1e-6: RENDER_DT = desired_render_dt

            global pending_key_change
            if pending_key_change is not None:
                if pending_key_change == 'next':
                    current_effect = (current_effect + 1) % len(effects)
                else:
                    current_effect = (current_effect - 1) % len(effects)
                pending_key_change = None
                apply_new_colorset()
                push_palette_to_ctx(ctx)
                last_effect_change = now
                log_event_inline("[KEY] effect change")

            if reset_flag:
                reset_flag = False
                with latest_lock:
                    latest_packet = None
                pixels.fill((0,0,0)); pixels.show()
                last_bands = np.zeros(EXPECTED_BANDS, dtype=np.uint8)
                last_beat = 0
                signal_active_until = 0.0
                already_off = False
                _band_idx_for = -1
                apply_new_colorset()
                push_palette_to_ctx(ctx)
                last_effect_change = now
                log_event_inline("[RST] reset aplicado")

            # ----------------------------------------------------------------
            # PATCH #1: Antes de concluir "2s sem pacotes", veja se há pacote pendente.
            pkt = None
            with latest_lock:
                if latest_packet is not None:
                    pkt = latest_packet
                    latest_packet = None

            if (now - last_rx_ts) > 2.0 and pkt is None:
                if not already_off:
                    pixels.fill((0,0,0)); pixels.show()
                    already_off = True
                unified_status_line("Idle", ctx, False, EXPECTED_BANDS, CFG_FPS, CFG_VIS_FPS, current_palette_name)
                next_frame += FRAME_DT
                sl = next_frame - time.time()
                if sl > 0: time.sleep(sl)
                else: next_frame = time.time()
                continue
            # ----------------------------------------------------------------

            if pkt is not None:
                bands_u8, beat_flag, transition_flag, ts_pc_ns, dyn_floor, kick_intensity = pkt
                last_rx_ts = now
                if already_off:
                    already_off = False

                if time_sync_ready and ts_pc_ns is not None:
                    now_ns = time.monotonic_ns()
                    one_way_ns = now_ns - (ts_pc_ns + time_offset_ns)
                    if -5_000_000 <= one_way_ns <= 5_000_000_000:
                        lat_ms = one_way_ns / 1e6
                        global latency_ms_ema
                        latency_ms_ema = lat_ms if latency_ms_ema is None else 0.85 * latency_ms_ema + 0.15 * lat_ms

                avg_raw = float(np.mean(bands_u8))

                # -------------------- PATCH #2: histerese de transição silenciosa --------------------
                if transition_flag == 1 and avg_raw < 0.5:
                    if last_bands.size != bands_u8.size:
                        last_bands = np.zeros(bands_u8.size, dtype=np.uint8)
                    else:
                        last_bands[:] = 0
                    last_beat = 0
                    # não altera signal_active_until aqui (não derruba hold)
                else:
                    signal_active_until = now + (SIGNAL_HOLD_MS / 1000.0)
                    if last_bands.size != bands_u8.size:
                        last_bands = np.zeros(bands_u8.size, dtype=np.uint8)
                    last_bands[:] = bands_u8
                    last_beat = int(beat_flag)
                # -------------------- /PATCH #2 --------------------

                # troca automática de efeito
                time_up = (now - last_effect_change) > effect_max_interval
                if transition_flag == 1 or time_up:
                    current_effect = (current_effect + 1) % len(effects)
                    apply_new_colorset()
                    push_palette_to_ctx(ctx)
                    last_effect_change = now

            active = (now < signal_active_until)
            b = last_bands

            if _band_idx_for != b.size:
                _band_idx = (LED_IDX * b.size) // LED_COUNT
                _band_idx_for = b.size

            name = effects[current_effect][0] if not BYPASS_EFFECTS else "Fallback HSV (LUT)"

            if (now - last_render_ts) >= RENDER_DT:
                last_render_ts = now
                try:
                    if not BYPASS_EFFECTS:
                        _, func = effects[current_effect]
                        b_in = b if active else _zeros_of(b.size)
                        func(b_in, last_beat if active else 0, active)
                    else:
                        np.take(b, _band_idx, out=_vals_buf)
                        np.take(GAMMA_LUT, _vals_buf, out=_v_buf)
                        np.add(H_IDX_BASE_U16, _v_buf.astype(np.uint16), out=_idx_buf_u16, dtype=np.uint16)
                        rgb = HSV_LUT_FLAT[_idx_buf_u16]
                        ctx.to_pixels_and_show(rgb)
                except Exception as e:
                    vals = np.repeat(b, 2)
                    if vals.size < LED_COUNT: vals = np.pad(vals, (0, LED_COUNT - vals.size), 'constant')
                    else: vals = vals[:LED_COUNT]
                    rgb = np.stack([vals, vals, vals], axis=-1)
                    ctx.to_pixels_and_show(rgb)
                    name = f"Fallback Gray ({e.__class__.__name__})"

            unified_status_line(name, ctx, active, b.size, CFG_FPS, CFG_VIS_FPS, current_palette_name)

            next_frame += FRAME_DT
            sl = next_frame - time.time()
            if sl > 0: time.sleep(sl)
            else: next_frame = time.time()

    except KeyboardInterrupt:
        pass
    finally:
        stop_flag = True
        palette_thread_stop.set()
        pixels.fill((0, 0, 0)); pixels.show()
        if LOG_MODE_SINGLE_LINE:
            _w("\r".ljust(140), same_line=True)
        sys.stdout.write("\n"); sys.stdout.flush()

if __name__ == "__main__":
    main()