#!/usr/bin/env python3
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

# Equalização / piso dinâmico (mesmo princípio do original)
EQ_ALPHA = 0.28 #0.15
EQ_TARGET = 64.0
band_ema = np.ones(EXPECTED_BANDS, dtype=np.float32) * 32.0
TILT_MIN, TILT_MAX = 0.9, 1.8
tilt_curve = np.exp(np.linspace(np.log(TILT_MAX), np.log(TILT_MIN), EXPECTED_BANDS)).astype(np.float32)
FLOOR_FACTOR = 0.02 #0.05
dynamic_floor = int(min(20, max(0, mean_val * FLOOR_FACTOR))) #0

def equalize_bands(bands_u8):
    global band_ema
    x = np.asarray(bands_u8, dtype=np.float32)
    # REDUZIDO: alpha mais lento pra não "engolir" picos
    band_ema = (1.0 - 0.15) * band_ema + 0.15 * x
    gain = 64.0 / np.maximum(band_ema, 1.0)
    eq = x * gain * tilt_curve
    return np.clip(eq, 0.0, 255.0).astype(np.uint8)

def compute_dynamic_floor(eq_bands, active):
    global dynamic_floor
    if not active:
        dynamic_floor = 0
        return
    # Só aplica floor se a música for MUITO alta
    mean_val = float(np.mean(eq_bands))
    if mean_val > 80:  # era 0, agora só ativa em volume alto
        dynamic_floor = int(min(15, mean_val * 0.015))  # max 15, era 28
    else:
        dynamic_floor = 0

# Paletas (simplificado / o suficiente para manter fluxo)
STRATEGIES = ["complementar", "analoga", "triade", "tetrade", "split"]
COMPLEMENT_DELTA = 0.06

def clamp01(x): return x % 1.0

def build_palette_from_strategy(h0, strategy, num_colors=5):
    import colorsys, random
    
    def hsv_to_rgb_bytes(h, s, v):
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        return (int(r*255), int(g*255), int(b*255))
    
    # Complementar: 2 hues opostos + variações
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
    
    # Análoga: 5 hues próximos (±20°)
    elif strategy == "analoga":
        step = 0.10
        hues = [clamp01(h0 + k*step) for k in (-2, -1, 0, 1, 2)]
    
    # Tríade: 3 hues equidistantes (120° cada)
    elif strategy == "triade":
        hues = [h0, clamp01(h0 + 1/3), clamp01(h0 + 2/3)]
    
    # Tétrade: 4 hues equidistantes (90° cada)
    elif strategy == "tetrade":
        hues = [h0, clamp01(h0 + 0.25), clamp01(h0 + 0.5), clamp01(h0 + 0.75)]
    
    # Split-complementar: base + 2 adjacentes ao complemento (padrão)
    else:  # "split"
        off = 1/6
        hues = [h0, clamp01(h0 + 0.5 - off), clamp01(h0 + 0.5 + off)]
    
    # Gera paleta com variações de saturação e valor
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
                strategy = random.choice(STRATEGIES); h0 = random.random()
                pal = build_palette_from_strategy(h0, strategy, num_colors=5)
                palette_queue.append((pal, strategy, h0))
            else:
                time.sleep(0.1)
        except Exception:
            time.sleep(0.05)

def get_next_palette():
    import random
    if palette_queue:
        pal, strategy, h0 = palette_queue.popleft()
        return pal, strategy, h0
    strategy = random.choice(STRATEGIES); h0 = random.random()
    pal = build_palette_from_strategy(h0, strategy, num_colors=5)
    return pal, strategy, h0

# Métricas históricas
class MetricsCollector:
    def __init__(self, interval_s=1.0, windows_s=(60,300,900), csv_path=None):
        self.interval_s = float(interval_s)
        self.windows_s = tuple(int(w) for w in windows_s)
        self.csv_path = csv_path
        self._sec_t0 = None
        self._n = 0
        self._i_sum = self._p_sum = self._cap_sum = 0.0
        self._i_min = float('+inf'); self._i_max = float('-inf')
        self._p_min = float('+inf'); self._p_max = float('-inf')
        self._cap_max = 0.0
        self._buffers = {w: deque(maxlen=w) for w in self.windows_s}
        if self.csv_path:
            self._ensure_header()
    def _ensure_header(self):
        need = (not os.path.exists(self.csv_path)) or (os.path.getsize(self.csv_path)==0)
        if need:
            with open(self.csv_path, 'a', encoding='utf-8') as f:
                f.write('ts_iso,i_avg,i_min,i_max,p_avg,p_min,p_max,cap_avg,cap_max\n')
    def on_frame(self, now_s, i_A, p_W, cap_scale):
        if self._sec_t0 is None:
            self._sec_t0 = now_s
        self._n += 1
        self._i_sum += i_A; self._p_sum += p_W
        cut = (1.0 - float(cap_scale))
        self._cap_sum += cut
        self._i_min = min(self._i_min, i_A); self._i_max = max(self._i_max, i_A)
        self._p_min = min(self._p_min, p_W); self._p_max = max(self._p_max, p_W)
        self._cap_max = max(self._cap_max, cut)
        if (now_s - self._sec_t0) >= self.interval_s:
            self._flush_second(now_s)
    def _flush_second(self, now_s):
        if self._n <= 0:
            self._sec_t0 = now_s; return
        i_avg = self._i_sum / self._n; p_avg = self._p_sum / self._n; cap_avg = self._cap_sum / self._n
        point = (now_s, i_avg, self._i_min, self._i_max, p_avg, self._p_min, self._p_max, cap_avg, self._cap_max)
        for w,dq in self._buffers.items():
            dq.append(point)
        if self.csv_path:
            ts_iso = datetime.fromtimestamp(now_s).strftime('%Y-%m-%d %H:%M:%S')
            with open(self.csv_path, 'a', encoding='utf-8') as f:
                f.write(f"{ts_iso},{i_avg:.4f},{self._i_min:.4f},{self._i_max:.4f},{p_avg:.4f},{self._p_min:.4f},{self._p_max:.4f},{cap_avg:.4f},{self._cap_max:.4f}\n")
        self._sec_t0 = now_s
        self._n = 0
        self._i_sum = self._p_sum = self._cap_sum = 0.0
        self._i_min = float('+inf'); self._i_max = float('-inf')
        self._p_min = float('+inf'); self._p_max = float('-inf')
        self._cap_max = 0.0
    def get_window_stats(self, w):
        dq = self._buffers.get(w)
        if not dq: return None
        n = len(dq)
        if n == 0: return None
        i_av = [t[1] for t in dq]; i_mn=[t[2] for t in dq]; i_mx=[t[3] for t in dq]
        p_av = [t[4] for t in dq]; p_mn=[t[5] for t in dq]; p_mx=[t[6] for t in dq]
        c_av = [t[7] for t in dq]; c_mx=[t[8] for t in dq]
        mean = lambda a: sum(a)/len(a)
        return (mean(i_av), min(i_mn), max(i_mx), mean(p_av), min(p_mn), max(p_mx), mean(c_av), max(c_mx))

# RX thread
latest_packet = None
latest_lock = threading.Lock()
stop_flag = False
last_rx_ts = 0.0

def rx_thread():
    global latest_packet, rx_count, last_rx_ts, stop_flag
    while not stop_flag:
        try:
            data, addr = udp_sock.recvfrom(4096)
            if not data:
                continue
            if data[0] == PKT_AUDIO and len(data) >= 1+8+EXPECTED_BANDS+2:
                ts_pc_ns = int.from_bytes(data[1:9], 'little')
                bands = list(data[9:9+EXPECTED_BANDS])
                beat  = int(data[9+EXPECTED_BANDS])
                trans = int(data[9+EXPECTED_BANDS+1])
                with latest_lock:
                    latest_packet = ('audio', bands, beat, trans, ts_pc_ns)
                rx_count += 1
                last_rx_ts = time.time()
        except Exception:
            pass

# Teclado
pending_key_change = None

def input_thread():
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        while not stop_flag:
            r,_,_ = select.select([sys.stdin], [], [], 0.05)
            if r:
                ch = sys.stdin.read(1)
                if ch.lower() == 'n': set_pending_change('next')
                elif ch.lower() == 'p': set_pending_change('prev')
    except Exception:
        pass
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)

def set_pending_change(kind):
    global pending_key_change
    pending_key_change = kind

TARGET_FPS = 50
FRAME_DT = 1.0 / TARGET_FPS
IDLE_TIMEOUT = 1.5

# Status

# --- STATUS: 1 linha, sem wrap, com truncamento à largura do terminal ---
import shutil

def print_status(effect_name, ctx, metrics):
    """
    Escreve o status em **uma única linha**:
      - Limpa a linha atual (ESC[2K) e escreve com '\r' no início;
      - Trunca ao número de colunas do terminal, evitando quebra automática;
      - Mantém I/P/CAP instantâneos e a janela de 60s.
    """
    global _last_status, rx_count, current_palette_name, latency_ms_ema, time_sync_ready
    now = time.time()
    if (now - _last_status) <= 0.25:
        return

    # Campos instantâneos
    lat_txt = "-.-" if (not time_sync_ready or latency_ms_ema is None) else f"{latency_ms_ema:4.1f}"
    sync_tag = "✓" if time_sync_ready else " "
    i_txt = "--.-" if ctx.current_a_ema is None else f"{ctx.current_a_ema:3.1f}"
    p_txt = "--.-" if ctx.power_w_ema   is None else f"{ctx.power_w_ema:4.1f}"
    cap_txt = f" CAP:{int((1.0-ctx.last_cap_scale)*100):2d}%" if ctx.last_cap_scale < 0.999 else ""

    # Janela 60s (curta)
    i1m_txt = ""
    try:
        st = metrics.get_window_stats(60) if metrics else None
        if st:
            i1m_avg, _i1m_min, i1m_max, p1m_avg, _p1m_min, p1m_max, cap1m_avg, cap1m_max = st
            i1m_txt = (f"  I1m:{i1m_avg:3.1f}/{i1m_max:3.1f}  "
                       f"P1m:{p1m_avg:4.1f}/{p1m_max:4.1f}  "
                       f"CAP1m:{int(cap1m_avg*100):2d}%/{int(cap1m_max*100):2d}%")
    except Exception:
        pass

    # Linha base (rótulos compactos)
    line = (f"RX:{rx_count:4d}  Ef:{effect_name:.28s}  Pal:{current_palette_name:.10s}  "
            f"LAT:{lat_txt} ms {sync_tag}  I:{i_txt}A  P:{p_txt}W{cap_txt}{i1m_txt}")

    # Largura do terminal + truncamento (evita wrap)
    cols = shutil.get_terminal_size(fallback=(120, 20)).columns
    ellipsis = "…"
    if len(line) > max(1, cols):
        # Deixa 1 coluna “de respiro” e adiciona reticências
        cut = max(0, cols - len(ellipsis))
        line = (line[:cut] + ellipsis) if cut > 0 else line[:cols]

    # Escreve 1 linha só: \r + ESC[2K (clear) + conteúdo
    sys.stdout.write("\r\033[2K" + line)
    sys.stdout.flush()
    _last_status = now

SIGNAL_HOLD = 0.5

def main():
    global current_palette, current_palette_name, base_hue_offset, hue_seed, last_palette_h0
    global stop_flag, latency_ms_ema, dynamic_floor, latest_packet  # <— importante!

    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics-log", type=str, default=None,
                        help="CSV 1 linha/s com I/P/CAP (ex.: /var/tmp/led-metrics.csv)")
    parser.add_argument("--metrics-interval", type=float, default=1.0,
                        help="Janela de agregação (s).")
    parser.add_argument("--metrics-windows", type=str, default="60,300,900",
                        help="Janelas rolling em s, separadas por vírgula (ex.: 60,300,900)")
    parser.add_argument("--current-budget", type=float, default=18.0,
                        help="Orçamento de corrente (A) para o limitador por frame.")
    args = parser.parse_args()

    # Métricas
    windows = tuple(int(x) for x in str(args.metrics_windows).split(',') if x.strip())
    metrics = MetricsCollector(interval_s=args.metrics_interval, windows_s=windows, csv_path=args.metrics_log)

    # Threads
    tsrv = threading.Thread(target=timesync_tcp_server, daemon=True); tsrv.start()
    t = threading.Thread(target=rx_thread, daemon=True); t.start()
    pt = threading.Thread(target=palette_worker, daemon=True); pt.start()
    it = threading.Thread(target=input_thread, daemon=True); it.start()

    # Contexto dos efeitos (power-cap + métricas)
    ctx = FXContext(
        pixels=pixels,
        led_count=LED_COUNT,
        base_hue_offset=base_hue_offset,
        hue_seed=hue_seed,
        base_saturation=base_saturation,
        current_budget_a=args.current_budget,
        ma_per_channel=20.0,
        idle_ma_per_led=1.0
    )
    ctx.metrics = metrics

    last_effect_change = time.time()
    effect_max_interval = 300.0
    current_effect = 0

    # Paleta inicial
    def apply_new_colorset():
        global current_palette_name  # ← ADICIONAR ESTA LINHA!
        nonlocal current_effect, last_effect_change
        
        pal, strategy, h0 = get_next_palette()
        current_palette[:] = pal
        current_palette_name = strategy  # ← ADICIONAR ESTA LINHA!
        
        # Atualiza seeds no ctx
        ctx.base_hue_offset = int(h0 * 255)
        ctx.hue_seed = random.randint(0, 255)
        
        # Rotação leve para tríades
        if strategy == "triade" and len(current_palette) > 1:
            rot = random.randint(1, len(current_palette) - 1)
            current_palette[:] = current_palette[rot:] + current_palette[:rot]
        
        last_effect_change = time.time()

    apply_new_colorset()

    effects = build_effects(ctx)
    name0, _ = effects[current_effect]
    print_status(name0, ctx, metrics)

    last_bands = np.zeros(EXPECTED_BANDS, dtype=np.float32)
    last_beat = 0
    bands_changed = False
    eq_cached = np.zeros(EXPECTED_BANDS, dtype=np.uint8)
    eq_valid = False

    SIGNAL_HOLD = 0.5
    signal_active_until = 0.0
    prev_active = False
    already_off = False

    next_frame = time.time()
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

            # idle
            if (now - last_rx_ts) > IDLE_TIMEOUT:
                if not already_off:
                    pixels.fill((0,0,0)); pixels.show()
                    already_off = True
                print_status(effects[current_effect][0], ctx, metrics)
                next_frame += FRAME_DT
                sl = next_frame - time.time()
                if sl > 0: time.sleep(sl)
                else: next_frame = time.time()
                continue

            pkt = None
            with latest_lock:
                if latest_packet is not None:
                    pkt = latest_packet; 
                    # consome o pacote
                    latest_packet = None
            if pkt is not None:
                _type, bands, beat_flag, transition_flag, ts_pc_ns = pkt
                bands_arr = np.asarray(bands, dtype=np.float32)
                avg_raw = float(np.mean(bands_arr))
                if transition_flag==1 and avg_raw < 0.5:
                    signal_active_until = 0.0
                    last_bands[:] = 0.0
                    last_beat = 0
                    bands_changed = True
                else:
                    if avg_raw > 2.0:
                        signal_active_until = now + SIGNAL_HOLD
                    # attack/release por banda
                    if 'ar_prev' not in globals():
                        globals()['ar_prev'] = np.zeros(EXPECTED_BANDS, dtype=np.float32)
                    prev = globals()['ar_prev']
                    attack, release = 0.70, 0.15
                    alpha = np.where(bands_arr > prev, attack, release).astype(np.float32)
                    smoothed = (alpha * bands_arr + (1.0 - alpha) * prev).astype(np.float32)
                    globals()['ar_prev'] = smoothed
                    bands_changed = not np.allclose(smoothed, last_bands, atol=1e-3)
                    last_bands[:] = smoothed
                    last_beat = int(beat_flag)
                    if last_beat == 1:
                        last_beat = 0  # será sobrescrito se vier novo

                # troca de efeito por transição ou 5min
                time_up = (now - last_effect_change) > effect_max_interval
                if transition_flag or time_up:
                    current_effect = (current_effect + 1) % len(effects)
                    apply_new_colorset()

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
            # kick boost simples
# === sync.py === SUBSTITUA o bloco do kick boost (dentro do main loop) ===
            # KICK BOOST REATIVO (agora com onset + beat_flag)
            if eq_valid:
                if 'kick_ema' not in globals(): 
                    globals()['kick_ema'] = 0.0
                    globals()['kick_prev'] = 0.0

                low_cached = float(np.mean(b[:max(6, EXPECTED_BANDS//15)]))  # mais foco em 40-80Hz
                kick_raw = max(0.0, low_cached - globals()['kick_prev'])
                globals()['kick_prev'] = low_cached

                # EMA mais rápida + boost só em onset
                globals()['kick_ema'] = 0.6 * globals()['kick_ema'] + 0.4 * low_cached
                onset = max(0.0, low_cached - globals()['kick_ema'])

                # Boost extra se beat_flag == 1
                kick_intensity = onset * 2.2
                if last_beat == 1:
                    kick_intensity += 60  # explosão no beat detectado

                if kick_intensity > 0:
                    boost = np.clip(kick_intensity, 0, 120)
                    b = np.clip(b.astype(np.float32) + boost, 0, 255).astype(np.uint8)
                    
            # Render efeito atual
            name, func = effects[current_effect]
            func(b if active else np.zeros_like(b), last_beat if active else 0, active)
            already_off = False
            prev_active = active
            bands_changed = False

            print_status(name, ctx, metrics)
            next_frame += FRAME_DT
            sl = next_frame - time.time()
            if sl > 0: time.sleep(sl)
            else: next_frame = time.time()

    except KeyboardInterrupt:
        pass
    finally:
        global stop_flag
        stop_flag = True
        palette_thread_stop.set()
        pixels.fill((0,0,0)); pixels.show()
        sys.stdout.write("\n"); sys.stdout.flush()

if __name__ == "__main__":
    main()