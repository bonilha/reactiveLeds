# effects/dynamics.py
import numpy as np

_peaks = None
_dec_tick = 0

def effect_peak_hold_columns(ctx, bands_u8, beat_flag, active):
    global _peaks, _dec_tick
    n = len(bands_u8)
    if _peaks is None or _peaks.shape[0] != n:
        _peaks = np.zeros(n, dtype=np.uint32)
    bands_u32 = np.asarray(bands_u8, dtype=np.uint32)
    _peaks = np.maximum(_peaks, bands_u32)
    _dec_tick = (_dec_tick + 1) % 4
    if _dec_tick == 0:
        decay = 1 + (5 if beat_flag else 0)
        _peaks = np.maximum(0, _peaks - decay)
    scaled = np.clip(_peaks * 140 // 100, 0, 255).astype(np.uint8)
    v_raw = ctx.amplify_quad(scaled.astype(np.uint16))
    v = ctx.apply_floor_vec(v_raw, active, None)
    hue = (ctx.base_hue_offset + ((np.arange(n) * 3) % 256) + (ctx.hue_seed & 0x3F) + (v >> 3)) % 256
    if beat_flag:
        hue = (hue + 32) % 256
    sat = np.maximum(0, ctx.base_saturation - (v >> 2)).astype(np.uint8)
    starts = (ctx.I_ALL * n) // ctx.LED_COUNT
    band_per_led = starts.clip(0, n - 1)
    val_per_led = v[band_per_led]
    hue_per_led = hue[band_per_led]
    sat_per_led = sat[band_per_led]
    rgb = ctx.hsv_to_rgb_bytes_vec(hue_per_led.astype(np.uint8), sat_per_led, val_per_led.astype(np.uint8))
    ctx.to_pixels_and_show(rgb)

def effect_full_strip_pulse(ctx, bands_u8, beat_flag, active):
    # Soma grave simples
    n = len(bands_u8)
    limit = min(8, n)
    raw = int(np.sum(np.asarray(bands_u8[:limit], dtype=np.uint32)) // max(1,limit))
    lvl = ctx.amplify_quad(np.array([raw], dtype=np.uint16))[0]
    if beat_flag:
        lvl = int(lvl * 1.4)
    v = np.full(ctx.LED_COUNT, min(255, lvl), dtype=np.uint8)
    v = ctx.apply_floor_vec(v, active, None)
    hue = (ctx.base_hue_offset + (ctx.I_ALL >> 4) + (ctx.hue_seed >> 2)) % 256
    sat = np.full(ctx.LED_COUNT, ctx.base_saturation, dtype=np.uint8)
    rgb = ctx.hsv_to_rgb_bytes_vec(hue.astype(np.uint8), sat, v)
    ctx.to_pixels_and_show(rgb)

# Waterfall reativo melhorado: shift descendente com rainbow por freq + fade suave (fix anti-branco)
_water = None
_water_decay = None

def effect_waterfall(ctx, bands_u8, beat_flag, active):
    global _water, _water_decay
    if _water is None or _water.shape[0] != ctx.LED_COUNT:
        _water = np.zeros((ctx.LED_COUNT, 3), dtype=np.uint8)
        _water_decay = np.ones(ctx.LED_COUNT, dtype=np.float32)  # buffer de decaimento por linha
        print("[DEBUG] Waterfall init: buffers criados para {} LEDs".format(ctx.LED_COUNT))  # <-- Debug init

    n = len(bands_u8)
    if n == 0:
        # Idle: fade out suave
        _water *= 0.92
        _water_decay *= 0.95
        ctx.to_pixels_and_show(_water)
        return

    print("[DEBUG] Waterfall frame: n_bands={}, beat={}, active={}, sample_max={}".format(  # <-- Debug frame
        n, beat_flag, active, np.max(bands_u8) if n > 0 else 0))  # Mostra pico pra checar reatividade

    # Usa mais bandas pra reatividade full (não só graves)
    limit = min(32, n)  # mids/highs inclusos
    arr = np.asarray(bands_u8[:limit], dtype=np.float32)
    maxv = float(np.max(arr)) if limit else 0.0
    meanv = float(np.mean(arr)) if limit else 0.0
    combined = (maxv * 1.5 + meanv) / 2.5
    scaled = np.clip(combined * 1.2, 0, 255)
    v_base = min(255, int(scaled))
    if beat_flag:
        v_base = min(255, int(v_base * 1.3))

    # Saturação FIXA alta (200-255) pra ZERAR branco – cai só 10 em idle
    base_sat = 220 if active else 180  # Força cor vibrante, ignora base_saturation se baixo
    sat = np.clip(base_sat, 200, 255)  # Mínimo 200: HSV nunca vira branco

    # Hue: base + rainbow por freq + variação por LED no shift (descendente colorido)
    hue_base = (ctx.base_hue_offset + (ctx.hue_seed >> 1) + (v_base >> 3)) % 256
    freq_hue_offset = int((np.mean(np.arange(limit) * arr / np.sum(arr + 1e-6)) * 3) % 256) if np.sum(arr) > 0 else 0
    hue = (hue_base + freq_hue_offset) % 256

    # Cor do "topo" (input atual) – agora com sat forçada
    color = ctx.hsv_to_rgb_bytes_vec(
        np.array([hue], dtype=np.uint8),
        np.array([sat], dtype=np.uint8),  # Sat fixa alta
        np.array([v_base], dtype=np.uint8)
    )[0]
    print("[DEBUG] Cor topo: hue={}, sat={}, v={}, RGB={}".format(hue, sat, v_base, color))  # <-- Debug cor (deve ter R/G/B !=255 todos)

    # Shift descendente variável (mais rápido em beat)
    shift = 2 + (1 if beat_flag else 0)  # Aumentei pra 2-3: mais "queda" fluida no serial WS2812B
    if shift > 0:
        _water[shift:] = _water[:-shift]
        _water_decay[shift:] = _water_decay[:-shift]

    # Preenche topo com fade linear + rainbow por linha (evita step, adiciona cor descendente)
    for j in range(shift):
        fade_factor = (shift - j) / float(shift)  # 1.0 recente, 0.0 velho
        line_hue = (hue + (j * 8)) % 256  # Rainbow descendente: +8 por linha pra cor gradual
        line_color = ctx.hsv_to_rgb_bytes_vec(
            np.array([line_hue], dtype=np.uint8),
            np.array([sat], dtype=np.uint8),
            np.array([int(v_base * fade_factor)], dtype=np.uint8)
        )[0]
        faded_color = (line_color.astype(np.float32) * _water_decay[j] * 0.95).astype(np.uint8)  # Atenua extra
        _water[j] = faded_color

    # Decaimento global suave por linha (simula "queda" e evita acúmulo)
    if active:
        decay_rate = 0.94  # Lento pra rastro, mas mais agressivo que antes
    else:
        decay_rate = 0.82  # Rápido em idle
    _water_decay[:shift] = 1.0  # Reset topo
    _water *= decay_rate
    _water_decay *= decay_rate

    # Floor e show (cap power no fxcore.py cuida do resto)
    _water = ctx.apply_floor_vec(_water.astype(np.uint16), active, None).astype(np.uint8)
    ctx.to_pixels_and_show(_water)

# Bass Ripple Pulse v2 (anel gaussiano)
class _Ripple:
    __slots__ = ("r","v","spd","thick","hue_shift")
    def __init__(self, r, v, spd, thick, hue_shift):
        self.r = float(r); self.v = float(v); self.spd = float(spd); self.thick = float(thick); self.hue_shift = int(hue_shift)

_brp_active = []
_brp_env = 0.0
_brp_prev_low = 0.0

def effect_bass_ripple_pulse_v2(ctx, bands_u8, beat_flag, active):
    global _brp_active, _brp_env, _brp_prev_low
    n = len(bands_u8)
    if n == 0:
        ctx.to_pixels_and_show(np.zeros((ctx.LED_COUNT,3), dtype=np.uint8)); return
    low_n = max(8, n // 8)
    low_mean = float(np.mean(np.asarray(bands_u8[:low_n], dtype=np.float32)))
    # envelope
    attack, release = 0.6, 0.18
    _brp_env = (attack if low_mean > _brp_env else release) * low_mean + (1.0 - (attack if low_mean > _brp_env else release)) * _brp_env
    spawn = bool(beat_flag)
    if not spawn and (low_mean - _brp_prev_low) > 6.0:
        spawn = True
    _brp_prev_low = low_mean
    if spawn and active:
        amp = ctx.amplify_quad(np.array([int(_brp_env)], dtype=np.uint16))[0]
        amp = int(np.clip(amp * 1.20, 60, 255))
        spd = 1.2 + 2.8 * (_brp_env / 255.0)
        thick = 2.5 + 3.5 * (_brp_env / 255.0)
        hue_shift = (int(_brp_env) >> 3) & 0x1F
        _brp_active.append(_Ripple(0.0, amp, spd, thick, hue_shift))
        if len(_brp_active) > 4:
            _brp_active = _brp_active[-4:]
    v_acc = np.zeros(ctx.LED_COUNT, dtype=np.float32)
    if active and _brp_active:
        d = np.abs(ctx.I_ALL.astype(np.float32) - float(ctx.CENTER))
        survivors = []
        for rp in _brp_active:
            diff = np.abs(d - rp.r)
            shape = np.exp(-0.5 * (diff / (rp.thick + 1e-6)) ** 2)
            v_acc += shape * rp.v
            rp.r += rp.spd
            rp.v *= 0.92
            rp.thick *= 0.98
            if rp.r < (ctx.CENTER + ctx.LED_COUNT) and rp.v > 3.0:
                survivors.append(rp)
        _brp_active = survivors
    else:
        _brp_active.clear()
    v = np.clip(v_acc, 0, 255).astype(np.uint8)
    v = ctx.apply_floor_vec(v, active, None)
    hue = (ctx.base_hue_offset + (ctx.I_ALL >> 2) + (ctx.hue_seed >> 1)) % 256
    sat = np.full(ctx.LED_COUNT, max(180, ctx.base_saturation), dtype=np.uint8)
    rgb = ctx.hsv_to_rgb_bytes_vec(hue.astype(np.uint8), sat, v)
    ctx.to_pixels_and_show(rgb)
