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
    raw = int(np.sum(np.asarray(bands_u8[:limit], dtype=np.uint32)) // max(1, limit))
    lvl = ctx.amplify_quad(np.array([raw], dtype=np.uint16))[0]
    if beat_flag:
        lvl = int(lvl * 1.4)
    v = np.full(ctx.LED_COUNT, min(255, lvl), dtype=np.uint8)
    v = ctx.apply_floor_vec(v, active, None)
    hue = (ctx.base_hue_offset + (ctx.I_ALL >> 4) + (ctx.hue_seed >> 2)) % 256
    sat = np.full(ctx.LED_COUNT, ctx.base_saturation, dtype=np.uint8)
    rgb = ctx.hsv_to_rgb_bytes_vec(hue.astype(np.uint8), sat, v)
    ctx.to_pixels_and_show(rgb)


# Waterfall clássico reativo: espectro full mapeado por LED, shift down com decay longo (enche strip)
_water = None


def effect_waterfall(ctx, bands_u8, beat_flag, active):
    global _water
    if _water is None or _water.shape[0] != ctx.LED_COUNT:
        _water = np.zeros((ctx.LED_COUNT, 3), dtype=np.uint8)
        print("[INFO] Waterfall init: buffers para {} LEDs WS2812B".format(ctx.LED_COUNT))

    n = len(bands_u8)
    if n == 0:
        _water = (_water.astype(np.float32) * 0.92).astype(np.uint8)
        ctx.to_pixels_and_show(_water)
        return

    # Mapeia bands para LEDs: interpola valores por posição
    arr = np.asarray(bands_u8, dtype=np.float32)
    if n < 2:
        v_row = np.full(ctx.LED_COUNT, arr[0] if n else 0, dtype=np.float32)
    else:
        # Posições das bandas (logspace para mais resolução em low freq) – aqui linear simples
        band_pos = np.linspace(0, ctx.LED_COUNT - 1, n)
        led_pos = np.arange(ctx.LED_COUNT, dtype=np.float32)
        v_row = np.interp(led_pos, band_pos, arr)

    # BOOST AGRESSIVO para reatividade forte
    v_row = v_row * 2.0  # Dobra ganho base

    # Beat flag com boost massivo
    if beat_flag:
        v_row *= 1.8  # Era 1.3, agora 1.8

    # Clip antes do quad para evitar saturação
    v_row = np.clip(v_row, 0, 255).astype(np.uint16)

    # Amplify quad para resposta não-linear (graves explodem)
    v_row = ctx.amplify_quad(v_row)

    # BOOST ADICIONAL pós-quad para frequências baixas (graves)
    low_boost = np.linspace(1.4, 1.0, ctx.LED_COUNT)  # 40% boost na esquerda (graves)
    v_row = np.clip(v_row.astype(np.float32) * low_boost, 0, 255).astype(np.uint16)

    # Hue por posição: baixa freq (esquerda) = vermelho/laranja, alta freq (direita) = azul/ciano
    hue_base = (np.arange(ctx.LED_COUNT, dtype=np.float32) / ctx.LED_COUNT) * 200  # Mais range
    hue_row = (ctx.base_hue_offset + hue_base.astype(np.int32) + (ctx.hue_seed >> 2)) % 256

    # Beat shift no hue para pulsar cores
    if beat_flag:
        hue_row = (hue_row + 20) % 256
    hue_row = hue_row.astype(np.uint8)

    # --------- AJUSTE: saturação baseada na paleta (ctx.base_saturation) ----------
    sat_base = int(np.clip(getattr(ctx, "base_saturation", 220), 0, 255))
    sat_row = np.full(
        ctx.LED_COUNT,
        sat_base if active else max(100, sat_base - 60),
        dtype=np.uint8
    )
    # -----------------------------------------------------------------------------

    # Converte para RGB
    new_row = ctx.hsv_to_rgb_bytes_vec(hue_row, sat_row, v_row.astype(np.uint8))

    # Decay RÁPIDO para resposta instantânea (não acumula demais)
    _water = (_water.astype(np.float32) * 0.75).astype(np.uint8)  # Era 0.96, agora 0.75

    # Adiciona nova linha com ADIÇÃO (não maximum) para picos explosivos
    _water = np.clip(_water.astype(np.uint16) + new_row.astype(np.uint16), 0, 255).astype(np.uint8)

    # Aplica floor e renderiza
    _water_out = ctx.apply_floor_vec(_water.astype(np.uint16), active, None).astype(np.uint8)
    ctx.to_pixels_and_show(_water_out)


# Bass Ripple Pulse v2 (anel gaussiano)
class _Ripple:
    __slots__ = ("r", "v", "spd", "thick", "hue_shift")

    def __init__(self, r, v, spd, thick, hue_shift):
        self.r = float(r)
        self.v = float(v)
        self.spd = float(spd)
        self.thick = float(thick)
        self.hue_shift = int(hue_shift)


_brp_active = []
_brp_env = 0.0
_brp_prev_low = 0.0


def effect_bass_ripple_pulse_v2(ctx, bands_u8, beat_flag, active):
    global _brp_active, _brp_env, _brp_prev_low
    n = len(bands_u8)
    if n == 0:
        ctx.to_pixels_and_show(np.zeros((ctx.LED_COUNT, 3), dtype=np.uint8)); return
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