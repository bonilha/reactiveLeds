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

# ---- Bass Center Bloom (robusto, usa paleta, anti-overflow) ----
_bass_radius_ema = 0.0
_beat_flash = 0.0

def effect_bass_center_bloom(ctx, bands_u8, beat_flag, active):
    """
    Bloom central reativo aos graves.
    Implementação segura (float32) com caps em todas as etapas para evitar OverflowError,
    usando a fita toda, e com colorização por paleta (se ctx.current_palette existir).
    """
    import time

    global _bass_radius_ema, _beat_flash

    # flash curto no beat (clamp 0..1)
    if beat_flag:
        _beat_flash = 1.0
    else:
        _beat_flash *= 0.85
    _beat_flash = float(np.clip(_beat_flash, 0.0, 1.0))

    # guarda contra pacote vazio
    n = len(bands_u8)
    if n == 0 or ctx.LED_COUNT <= 0:
        ctx.to_pixels_and_show(np.zeros((ctx.LED_COUNT, 3), dtype=np.uint8))
        return

    # energia de graves -> raio alvo
    low_n = max(8, n // 8)
    low = np.asarray(bands_u8[:low_n], dtype=np.float32)
    low_mean = float(np.mean(low))  # 0..255

    target_radius = float(np.clip((low_mean / 255.0) * (ctx.CENTER * 0.95), 0.0, float(ctx.CENTER)))
    _bass_radius_ema = 0.80 * float(_bass_radius_ema) + 0.20 * target_radius
    radius = float(np.clip(_bass_radius_ema, 0.0, float(ctx.CENTER)))

    # perfil radial: corpo (triângulo) + halo (gaussiano)
    idx = ctx.I_ALL.astype(np.float32)
    d = np.abs(idx - float(ctx.CENTER))  # distância ao centro

    base_w = max(1.0, radius)
    body = np.clip(1.0 - (d / base_w), 0.0, 1.0).astype(np.float32)

    sigma = 0.35 * max(6.0, radius)
    halo = np.exp(-0.5 * (d / max(1e-3, sigma))**2).astype(np.float32)

    e01 = float(np.clip(low_mean / 255.0, 0.0, 1.0))
    shape = np.clip(0.65 * body + 0.35 * halo, 0.0, 1.0).astype(np.float32)

    # ganho: 80..220 + pulso no beat
    base_amp = 80.0 + 140.0 * e01
    amp = float(np.clip(base_amp * (1.0 + 0.25 * _beat_flash), 0.0, 235.0))

    # brilho por LED (float32 0..255)
    v_f = np.clip(shape * amp, 0.0, 235.0).astype(np.float32)

    # piso dinâmico (canal-agnóstico)
    floor = float(getattr(ctx, "dynamic_floor", 0))
    if floor > 0:
        v_f = np.maximum(v_f, floor).astype(np.float32)

    # ---------- COR ----------
    # 1) paleta se houver, 2) fallback HSV
    pal = getattr(ctx, "current_palette", None)
    use_pal = isinstance(pal, (list, tuple)) and len(pal) >= 2

    if use_pal:
        pal_arr = np.asarray(pal, dtype=np.uint8)
        m = pal_arr.shape[0]

        t = time.time()
        phase = (t * 0.08) * m
        pos = ((idx / max(1.0, float(ctx.LED_COUNT))) * m + phase) % m

        i0 = np.floor(pos).astype(np.int32) % m
        i1 = (i0 + 1) % m
        frac = (pos - np.floor(pos)).astype(np.float32)[:, None]

        c0 = pal_arr[i0].astype(np.float32)
        c1 = pal_arr[i1].astype(np.float32)
        base_rgb = (c0 * (1.0 - frac) + c1 * frac)  # 0..255

        sat_base = int(np.clip(getattr(ctx, "base_saturation", 220), 0, 255))
        s = sat_base / 255.0
        v_col = (v_f[:, None] / 255.0) * base_rgb
        gray = v_f[:, None]
        out = gray * (1.0 - s) + v_col * s
        rgb = np.clip(out, 0, 255).astype(np.uint8)

    else:
        t = time.time()
        h_time = int((t * 35.0) % 256)
        h_spatial = ((ctx.I_ALL.astype(np.int32) * 256) // max(1, ctx.LED_COUNT))
        hue = (int(ctx.base_hue_offset) + (int(ctx.hue_seed) >> 2) + h_time + h_spatial) % 256

        sat = np.full(ctx.LED_COUNT, int(np.clip(getattr(ctx, "base_saturation", 220), 0, 255)),
                      dtype=np.uint8)
        v_u8 = np.clip(v_f, 0.0, 255.0).astype(np.uint8)
        rgb = ctx.hsv_to_rgb_bytes_vec(hue.astype(np.uint8), sat, v_u8)

    # leve cap final
    rgb = np.clip(rgb, 0, 250).astype(np.uint8)
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