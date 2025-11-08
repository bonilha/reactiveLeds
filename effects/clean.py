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

def effect_bass_center_bloom(ctx, bands_u8, beat_flag, active):
    """
    Bloom central reativo aos graves com cobertura da fita inteira.
    - Anti-NaN/overflow: caps e np.nan_to_num antes de cast.
    - Mais reativo: ganho guiado por envelope + transiente (subida de graves).
    - Usa ctx.current_palette se existir; fallback HSV senão.
    """
    import numpy as np, time

    global _bass_radius_ema, _beat_flash, _env_ema

    # --- beat flash (0..1) ---
    _beat_flash = 1.0 if beat_flag else _beat_flash * 0.85
    _beat_flash = float(np.clip(_beat_flash, 0.0, 1.0))

    # --- guardas ---
    n = len(bands_u8)
    L = int(getattr(ctx, "LED_COUNT", 0))
    if n == 0 or L <= 0:
        ctx.to_pixels_and_show(np.zeros((L, 3), dtype=np.uint8))
        return

    # --- graves: média + envelope + transiente ---
    low_n = max(8, n // 8)
    low = np.asarray(bands_u8[:low_n], dtype=np.float32)
    low_mean = float(np.mean(low))                         # 0..255
    _env_ema = 0.75 * float(_env_ema) + 0.25 * low_mean    # envelope mais estável
    env01 = float(np.clip(_env_ema / 255.0, 0.0, 1.0))
    # transiente = “subida” instantânea acima do envelope (0..1)
    trans = float(np.clip((low_mean - _env_ema) / 255.0, 0.0, 1.0))

    # --- raio (suavizado) ---
    target_radius = float(np.clip((low_mean / 255.0) * (ctx.CENTER * 0.95), 0.0, float(ctx.CENTER)))
    _bass_radius_ema = 0.80 * float(_bass_radius_ema) + 0.20 * target_radius
    radius = float(np.clip(_bass_radius_ema, 0.0, float(ctx.CENTER)))

    # --- perfis espaciais (float32) ---
    idx = ctx.I_ALL.astype(np.float32)
    d = np.abs(idx - float(ctx.CENTER))

    # Core (triângulo 0..1)
    base_w = max(1.0, radius)
    body = np.clip(1.0 - (d / base_w), 0.0, 1.0).astype(np.float32)

    # Halo (gauss 0..1) — usa divisor mínimo para evitar NaN
    sigma = 0.35 * max(6.0, radius)
    halo = np.exp(-0.5 * (d / max(1e-3, sigma))**2).astype(np.float32)

    # Wide bed (raised-cosine suave 0..1)
    cos_arg = (d / max(1.0, float(ctx.CENTER))) * (np.pi * 0.5)
    wide = (np.cos(np.clip(cos_arg, 0.0, np.pi * 0.5)) ** 1.4).astype(np.float32)

    # --- ganhos / composição (mais reativos) ---
    # ganho principal: base + envelope + transiente; leve boost no beat
    amp = 60.0 + 160.0 * env01 + 120.0 * trans
    if beat_flag:
        amp *= 1.15
    amp = float(np.clip(amp, 0.0, 235.0))

    # pesos do core/halo dependem do transiente (abre no "ataque")
    core_w = 0.50 + 0.35 * trans + 0.10 * env01   # 0.5..0.95
    halo_w = 0.25 + 0.20 * env01                  # 0.25..0.45

    # cama larga e fundo menores para não “lavar” o contraste
    bed_gain = (12.0 + 70.0 * env01)              # 12..82
    bg_gain  = (6.0  + 16.0 * env01)              # 6..22

    v_core = amp * (core_w * body + halo_w * halo)
    v_bed  = bed_gain * wide
    v_bg   = bg_gain

    v_f = v_core + v_bed + v_bg
    if not active:
        v_f *= 0.90

    # piso dinâmico
    floor = float(getattr(ctx, "dynamic_floor", 0))
    if floor > 0.0:
        v_f = np.maximum(v_f, floor)

    # ----- Anti-NaN/Inf + caps antes do cast -----
    v_f = np.nan_to_num(v_f, nan=0.0, posinf=240.0, neginf=0.0).astype(np.float32)
    v_f = np.clip(v_f, 0.0, 240.0).astype(np.float32)

    # ---------- COR ----------
    pal = getattr(ctx, "current_palette", None)
    use_pal = isinstance(pal, (list, tuple)) and len(pal) >= 2

    if use_pal:
        pal_arr = np.asarray(pal, dtype=np.uint8)
        m = pal_arr.shape[0]

        # varredura leve da paleta + micro-salto com o beat
        t = time.time()
        phase = (t * 0.07) * m + (0.20 * m * _beat_flash)
        posc = ((idx / max(1.0, float(L))) * m + phase) % m

        i0 = np.floor(posc).astype(np.int32) % m
        i1 = (i0 + 1) % m
        frac = (posc - np.floor(posc)).astype(np.float32)[:, None]

        c0 = pal_arr[i0].astype(np.float32)
        c1 = pal_arr[i1].astype(np.float32)
        base_rgb = (c0 * (1.0 - frac) + c1 * frac)  # 0..255 float32

        sat_base = int(np.clip(getattr(ctx, "base_saturation", 220), 0, 255))
        s = sat_base / 255.0

        v_col = (v_f[:, None] / 255.0) * base_rgb
        gray  = v_f[:, None]
        out   = gray * (1.0 - s) + v_col * s
        rgb   = np.clip(out, 0, 255).astype(np.uint8)

    else:
        # HSV com variação temporal + espacial para evitar monocromia
        t = time.time()
        h_time = int((t * 32.0) % 256)
        h_spatial = ((ctx.I_ALL.astype(np.int32) * 256) // max(1, L))
        hue = (int(ctx.base_hue_offset) + (int(ctx.hue_seed) >> 2) + h_time + h_spatial) % 256

        sat = np.full(L, int(np.clip(getattr(ctx, "base_saturation", 220), 0, 255)), dtype=np.uint8)
        # aqui garantimos que não há NaN antes do cast:
        v_u8 = np.clip(np.nan_to_num(v_f, nan=0.0, posinf=255.0, neginf=0.0), 0.0, 255.0).astype(np.uint8)
        rgb = ctx.hsv_to_rgb_bytes_vec(hue.astype(np.uint8), sat, v_u8)

    # cap final moderado
    rgb = np.clip(rgb, 0, 245).astype(np.uint8)
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