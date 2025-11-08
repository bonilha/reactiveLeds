# effects/clean.py
import numpy as np

__all__ = [
    "effect_spectral_blade",
    "effect_bass_center_bloom",
    "effect_bass_center",
    "effect_peak_dots",
    "effect_centroid_comet",
    "effect_beat_outward_burst",
    "effect_quantized_sections",
]

# ---- Spectral Blade ----
def effect_spectral_blade(ctx, bands_u8, beat_flag, active):
    arr = np.asarray(bands_u8, dtype=np.float32)
    v_raw = ctx.segment_mean_from_cumsum(arr, ctx.SEG_STARTS_HALF, ctx.SEG_ENDS_HALF)
    v = ctx.amplify_quad(v_raw.astype(np.uint16))
    v = ctx.apply_floor_vec(v, active, None)
    hue = (ctx.base_hue_offset + (ctx.I_LEFT >> 1) + (ctx.hue_seed >> 2) + (v >> 3)) % 256
    sat = np.maximum(0, ctx.base_saturation - (v >> 2)).astype(np.uint8)
    val = np.clip(v, 0, 255).astype(np.uint8)
    left_rgb = ctx.hsv_to_rgb_bytes_vec(hue.astype(np.uint8), sat, val)
    rgb = np.zeros((ctx.LED_COUNT, 3), dtype=np.uint8)
    rgb[ctx.CENTER - 1 - ctx.I_LEFT] = left_rgb
    rgb[ctx.CENTER + ctx.I_LEFT] = left_rgb
    ctx.to_pixels_and_show(rgb)

# -- Bass Center Bloom (full-strip, palette, anti-NaN/overflow, mais reativo) --
_bass_radius_ema = 0.0
_beat_flash = 0.0
_env_ema = 0.0

# -- Bass Center Bloom (full-strip, palette, anti-NaN/overflow, mais reativo) --
_bass_radius_ema = 0.0
_beat_flash = 0.0
_env_ema = 0.0

def effect_bass_center_bloom(ctx, bands_u8, beat_flag, active):
    """
    Bloom central reativo aos graves com cobertura da fita inteira.
    - Corrige NaN/Inf (np.nan_to_num) e evita bases negativas em potências.
    - Reatividade maior: ganho = base + env + transiente (+beat).
    - Usa ctx.current_palette se existir; fallback HSV senão.
    """
    import numpy as np, time

    global _bass_radius_ema, _beat_flash, _env_ema

    # ----- BEAT FLASH (0..1) -----
    _beat_flash = 1.0 if beat_flag else _beat_flash * 0.85
    _beat_flash = float(np.clip(_beat_flash, 0.0, 1.0))

    # ----- GUARDAS -----
    n = len(bands_u8)
    L = int(getattr(ctx, "LED_COUNT", 0))
    if n == 0 or L <= 0:
        ctx.to_pixels_and_show(np.zeros((L, 3), dtype=np.uint8))
        return

    # ----- ENERGIA: ENVELOPE + TRANSIENTE -----
    low_n = max(8, n // 8)
    low = np.asarray(bands_u8[:low_n], dtype=np.float32)
    low_mean = float(np.mean(low))                    # 0..255
    _env_ema = 0.75 * float(_env_ema) + 0.25 * low_mean
    env01 = float(np.clip(_env_ema / 255.0, 0.0, 1.0))
    trans = float(np.clip((low_mean - _env_ema) / 255.0, 0.0, 1.0))

    # ----- RAIO (EMA) -----
    target_radius = float(np.clip((low_mean / 255.0) * (ctx.CENTER * 0.95),
                                  0.0, float(ctx.CENTER)))
    _bass_radius_ema = 0.80 * float(_bass_radius_ema) + 0.20 * target_radius
    radius = float(np.clip(_bass_radius_ema, 0.0, float(ctx.CENTER)))

    # ----- PERFIS ESPACIAIS (float32) -----
    idx = ctx.I_ALL.astype(np.float32)
    d = np.abs(idx - float(ctx.CENTER))

    # Core: triângulo 0..1
    base_w = max(1.0, radius)
    body = np.clip(1.0 - (d / base_w), 0.0, 1.0).astype(np.float32)

    # Halo: gaussiano 0..1 (divisor mínimo evita NaN)
    sigma = 0.35 * max(6.0, radius)
    halo = np.exp(-0.5 * (d / max(1e-3, sigma))**2).astype(np.float32)

    # Wide bed: raised-cosine 0..1 (garante base >=0 antes do power)
    cos_arg = (d / max(1.0, float(ctx.CENTER))) * (np.pi * 0.5)
    cos_clip = np.clip(cos_arg, 0.0, np.pi * 0.5)
    cos_val  = np.cos(cos_clip)
    cos_val  = np.nan_to_num(cos_val, nan=0.0, posinf=0.0, neginf=0.0)
    cos_val  = np.maximum(cos_val, 0.0).astype(np.float32)
    wide = (cos_val ** 1.2).astype(np.float32)  # expoente mais suave (evita warning)

    # ----- GANHOS / COMPOSIÇÃO (mais reativos) -----
    # Ajuste fácil: aumente/diminua estes coeficientes para calibrar o "punch"
    AMP_BASE = 50.0
    AMP_ENV  = 170.0   # ganho pelo envelope (lento)
    AMP_TRANS= 180.0   # ganho pelo transiente (rápido)
    BEAT_BOOST = 1.12

    amp = AMP_BASE + AMP_ENV * env01 + AMP_TRANS * trans
    if beat_flag:
        amp *= BEAT_BOOST
    amp = float(np.clip(amp, 0.0, 240.0))

    CORE_W = 0.55 + 0.35 * trans + 0.10 * env01   # 0.55..1.0 (abre no ataque)
    HALO_W = 0.22 + 0.20 * env01                  # 0.22..0.42 (enche em energia)
    BED_GAIN = (10.0 + 55.0 * env01)              # cama larga mais discreta
    BG_GAIN  = (4.0  + 12.0 * env01)              # fundo uniforme leve

    v_core = amp * (CORE_W * body + HALO_W * halo)
    v_bed  = BED_GAIN * wide
    v_bg   = BG_GAIN

    v = np.clip(v_core + v_bed + v_bg, 0, 255).astype(np.float32)

    # ----- BOOST DE REATIVIDADE -----
    # Aumentar amplificação geral para mais "punch" nos graves
    v = v * 1.5  # Multiplicador global para intensificar o bloom
    v = np.clip(v, 0, 255).astype(np.float32)

    # Reduzir decay do EMA para resposta mais rápida
    _env_ema = 0.60 * float(_env_ema) + 0.40 * low_mean  # Ajuste alpha para mais attack

    # Aumentar raio para mais "center bloom"
    target_radius = float(np.clip((low_mean / 255.0) * (ctx.CENTER * 1.2), 0.0, float(ctx.CENTER)))  # Raio maior
    _bass_radius_ema = 0.70 * float(_bass_radius_ema) + 0.30 * target_radius  # Mais responsivo

    # ----- COR: PALETTE OU HSV -----
    v_u8 = np.clip(v, 0, 255).astype(np.uint8)
    v_u8 = ctx.apply_floor_vec(v_u8, active, None)

    # Hue baseado na posição para variação (reativo ao seed)
    frac = d / max(1.0, float(ctx.CENTER))
    hue_base = ctx.base_hue_offset + (ctx.hue_seed >> 2) + (frac * 48.0).astype(np.int32)
    hue = (hue_base + (v_u8 >> 3)) % 256
    sat = np.full(L, ctx.base_saturation, dtype=np.uint8)

    # Boost no beat: flash mais forte
    if beat_flag:
        v_u8 = np.clip(v_u8 * 1.3, 0, 255).astype(np.uint8)
        hue = (hue + 32) % 256  # Shift de cor no beat para feedback visual

    rgb = ctx.hsv_to_rgb_bytes_vec(hue.astype(np.uint8), sat, v_u8)

    # ----- RENDER -----
    ctx.to_pixels_and_show(rgb)


# Alias para compatibilidade
def effect_bass_center(ctx, bands_u8, beat_flag, active):
    return effect_bass_center_bloom(ctx, bands_u8, beat_flag, active)

# ---- Peak Dots ----
_peak_levels = None
def effect_peak_dots(ctx, bands_u8, beat_flag, active, k=6):
    global _peak_levels
    n = len(bands_u8)
    if n == 0:
        ctx.to_pixels_and_show(np.zeros((ctx.LED_COUNT, 3), dtype=np.uint8))
        return
    if _peak_levels is None or len(_peak_levels) != n:
        _peak_levels = np.zeros(n, dtype=np.float32)
    x = np.asarray(bands_u8, dtype=np.float32)
    _peak_levels = np.maximum(_peak_levels * (0.90 if active else 0.80), x)
    k = min(k, n)
    idx = np.argpartition(_peak_levels, -k)[-k:]
    idx = idx[np.argsort(-_peak_levels[idx])]
    pos = (idx.astype(np.int32) * ctx.LED_COUNT) // n
    pos = np.clip(pos, 0, ctx.LED_COUNT - 1)
    rgb = np.zeros((ctx.LED_COUNT, 3), dtype=np.uint8)
    val = np.clip((_peak_levels[idx] * 1.3), 0, 255).astype(np.uint8)
    val = np.maximum(val, ctx.dynamic_floor).astype(np.uint8)
    hue = (ctx.base_hue_offset + (idx * 5) + (ctx.hue_seed & 0x3F)) % 256
    sat = np.full(k, ctx.base_saturation, dtype=np.uint8)
    dots = ctx.hsv_to_rgb_bytes_vec(hue.astype(np.uint8), sat, val)
    rgb[pos] = dots
    ctx.to_pixels_and_show(rgb)

# ---- Centroid Comet ----
_centroid_pos = None
_trail_buf = None
def effect_centroid_comet(ctx, bands_u8, beat_flag, active):
    global _centroid_pos, _trail_buf
    n = len(bands_u8)
    if _trail_buf is None or _trail_buf.shape[0] != ctx.LED_COUNT:
        _trail_buf = np.zeros(ctx.LED_COUNT, dtype=np.float32)
    if _centroid_pos is None:
        _centroid_pos = float(ctx.CENTER)
    if n == 0:
        ctx.to_pixels_and_show(np.zeros((ctx.LED_COUNT, 3), dtype=np.uint8)); return
    w = np.asarray(bands_u8, dtype=np.float32)
    s = float(np.sum(w)) + 1e-6
    idx = np.arange(n, dtype=np.float32)
    centroid_band = float(np.sum(idx * w) / s)
    target_pos = (centroid_band / max(1.0, n - 1.0)) * (ctx.LED_COUNT - 1)
    alpha_pos = 0.25 if active else 0.15
    _centroid_pos = (1.0 - alpha_pos) * _centroid_pos + alpha_pos * target_pos
    head_pos = int(round(_centroid_pos))
    mean_energy = float(np.mean(w))
    inj = ctx.amplify_quad(np.array([int(mean_energy)], dtype=np.uint16))[0]
    if beat_flag:
        inj = int(min(255, inj * 1.25))
    width = 6
    dist = np.abs(ctx.I_ALL - head_pos)
    head_shape = np.clip((width - dist), 0, width).astype(np.float32) / float(width)
    head_val = np.clip(head_shape * float(inj), 0, 255).astype(np.float32)
    decay = 0.90 if active else 0.80
    _trail_buf = _trail_buf * decay + head_val
    v = np.clip(_trail_buf, 0, 255).astype(np.uint8)
    v = ctx.apply_floor_vec(v, active, None)
    hue = (ctx.base_hue_offset + (head_pos >> 1) + (ctx.hue_seed >> 2)) % 256
    hue_arr = np.full(ctx.LED_COUNT, hue, dtype=np.uint8)
    sat = np.full(ctx.LED_COUNT, max(96, ctx.base_saturation - 32), dtype=np.uint8)
    rgb = ctx.hsv_to_rgb_bytes_vec(hue_arr, sat, v)
    ctx.to_pixels_and_show(rgb)

# ---- Beat Outward Burst ----
_burst_left = None
_burst_right = None
_burst_buf = None
def effect_beat_outward_burst(ctx, bands_u8, beat_flag, active):
    global _burst_left, _burst_right, _burst_buf
    if _burst_buf is None or _burst_buf.shape[0] != ctx.LED_COUNT:
        _burst_buf = np.zeros(ctx.LED_COUNT, dtype=np.float32)
        _burst_left = ctx.CENTER
        _burst_right = ctx.CENTER
    n = len(bands_u8)
    low_n = max(8, n // 8) if n > 0 else 8
    low_mean = float(np.mean(np.asarray(bands_u8[:low_n], dtype=np.float32))) if n > 0 else 0.0
    if beat_flag and active:
        amp = ctx.amplify_quad(np.array([int(low_mean)], dtype=np.uint16))[0]
        peak = np.clip(float(amp) * 1.2, 80, 255)
        k = 4
        kdist = np.abs(ctx.I_ALL - ctx.CENTER)
        kshape = np.clip((k - kdist), 0, k).astype(np.float32) / float(k)
        _burst_buf += kshape * peak
        _burst_left = ctx.CENTER - 1
        _burst_right = ctx.CENTER
    if active:
        _burst_left = max(0, _burst_left - 2)
        _burst_right = min(ctx.LED_COUNT - 1, _burst_right + 2)
    tri_w = 3
    ldist = np.abs(ctx.I_ALL - _burst_left)
    rdist = np.abs(ctx.I_ALL - _burst_right)
    ltri = np.clip((tri_w - ldist), 0, tri_w).astype(np.float32) / float(tri_w)
    rtri = np.clip((tri_w - rdist), 0, tri_w).astype(np.float32) / float(tri_w)
    decay = 0.86 if active else 0.78
    _burst_buf = _burst_buf * decay + (ltri + rtri) * 200.0
    v = np.clip(_burst_buf, 0, 255).astype(np.uint8)
    v = ctx.apply_floor_vec(v, active, None)
    hue = (ctx.base_hue_offset + (ctx.I_ALL >> 2) + (ctx.hue_seed & 0x1F)) % 256
    sat = np.full(ctx.LED_COUNT, ctx.base_saturation, dtype=np.uint8)
    rgb = ctx.hsv_to_rgb_bytes_vec(hue.astype(np.uint8), sat, v)
    ctx.to_pixels_and_show(rgb)

# ---- Quantized Sections ----
_QS_SECTIONS = 10
_QS_LEVELS = 8
def effect_quantized_sections(ctx, bands_u8, beat_flag, active):
    arr = np.asarray(bands_u8, dtype=np.float32)
    v_full = ctx.segment_mean_from_cumsum(arr, ctx.SEG_STARTS_FULL, ctx.SEG_ENDS_FULL)
    block = max(1, ctx.LED_COUNT // _QS_SECTIONS)
    v_led = v_full.astype(np.uint16)
    sec_idx = ctx.I_ALL // block
    sec_idx = np.clip(sec_idx, 0, _QS_SECTIONS - 1)
    sums = np.zeros(_QS_SECTIONS, dtype=np.float32)
    cnts = np.zeros(_QS_SECTIONS, dtype=np.int32)
    np.add.at(sums, sec_idx, v_led.astype(np.float32))
    np.add.at(cnts, sec_idx, 1)
    means = np.divide(sums, np.maximum(1, cnts), dtype=np.float32)
    step = 255.0 / float(max(1, _QS_LEVELS - 1))
    qvals = np.clip((np.rint(means / step) * step), 0, 255).astype(np.uint8)
    v = qvals[sec_idx]
    v = ctx.apply_floor_vec(v, active, None)
    hue = (ctx.base_hue_offset + (sec_idx * 7) + (ctx.hue_seed >> 1)) % 256
    sat = np.full(ctx.LED_COUNT, max(80, ctx.base_saturation - 20), dtype=np.uint8)
    rgb = ctx.hsv_to_rgb_bytes_vec(hue.astype(np.uint8), sat, v.astype(np.uint8))
    ctx.to_pixels_and_show(rgb)