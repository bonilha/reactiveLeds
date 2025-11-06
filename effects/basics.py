# effects/basics.py
import numpy as np

# Estado para rainbow
_rainbow_wave_pos = 0

def effect_line_spectrum(ctx, bands_u8, beat_flag, active):
    arr = np.asarray(bands_u8, dtype=np.float32)
    v_raw = ctx.segment_mean_from_cumsum(arr, ctx.SEG_STARTS_FULL, ctx.SEG_ENDS_FULL)
    v_lin = np.clip(v_raw.astype(np.uint16), 0, 255)
    v_quad = ctx.amplify_quad(v_lin)  # v^2/255
    v = ((v_quad + v_lin) // 2)       # média dos dois -> curva mais suave
    hue = (ctx.base_hue_offset + (ctx.I_ALL * 3) + (ctx.hue_seed >> 1) + (v >> 3)) % 256
    sat = np.maximum(0, ctx.base_saturation - (v >> 2)).astype(np.uint8)
    val = np.clip(v, 0, 255).astype(np.uint8)
    rgb = ctx.hsv_to_rgb_bytes_vec(hue.astype(np.uint8), sat, val)
    ctx.to_pixels_and_show(rgb)

def effect_mirror_spectrum(ctx, bands_u8, beat_flag, active):
    arr = np.asarray(bands_u8, dtype=np.float32)
    v_raw = ctx.segment_mean_from_cumsum(arr, ctx.SEG_STARTS_HALF, ctx.SEG_ENDS_HALF)
    v = ctx.amplify_quad(v_raw.astype(np.uint16))
    v = ctx.apply_floor_vec(v, active, None)
    hue = (ctx.base_hue_offset + (ctx.I_LEFT * 3) + (ctx.hue_seed >> 1) + (v >> 3)) % 256
    sat = np.maximum(0, ctx.base_saturation - (v >> 2)).astype(np.uint8)
    val = np.clip(v, 0, 255).astype(np.uint8)
    left_rgb = ctx.hsv_to_rgb_bytes_vec(hue.astype(np.uint8), sat, val)
    rgb = np.zeros((ctx.LED_COUNT, 3), dtype=np.uint8)
    rgb[ctx.CENTER - 1 - ctx.I_LEFT] = left_rgb
    rgb[ctx.CENTER + ctx.I_LEFT]     = left_rgb
    ctx.to_pixels_and_show(rgb)

def effect_rainbow_wave(ctx, bands_u8, beat_flag, active):
    global _rainbow_wave_pos
    _rainbow_wave_pos = (_rainbow_wave_pos + 5) % 256
    n = max(1, len(bands_u8))
    idx = ctx.I_ALL % n
    b0 = np.asarray(bands_u8, dtype=np.uint8)[idx]
    v = ctx.amplify_quad(np.clip(b0.astype(np.uint16) * 150 // 100, 0, 255))
    v = ctx.apply_floor_vec(v, active, None)
    hue = (ctx.base_hue_offset + _rainbow_wave_pos + (ctx.I_ALL % 256) + ctx.hue_seed) % 256
    sat = np.maximum(0, ctx.base_saturation - (v >> 2)).astype(np.uint8)
    val = np.clip(v, 0, 255).astype(np.uint8)
    rgb = ctx.hsv_to_rgb_bytes_vec(hue.astype(np.uint8), sat, val)
    ctx.to_pixels_and_show(rgb)

def effect_vu_meter(ctx, bands_u8, beat_flag, active):
    third = max(1, ctx.LED_COUNT // 3)
    n = len(bands_u8)
    pos = (np.arange(n) * 3 // max(1, n)).astype(np.int32)
    arr = np.asarray(bands_u8, dtype=np.uint8)
    level = []
    for k in (0,1,2):
        m = (pos == k)
        s = int(np.sum(arr[m].astype(np.uint32)))
        c = int(np.sum(m)) or 1
        level.append(int((s // c)))
    level = [min(255, x) for x in level]
    # boost para níveis baixos
    mx = max(level)
    scale = 255 / max(1, mx) if mx < 128 else 1.0
    level = [int(x * scale) for x in level]
    lit = [int(level[i] * third // 255) for i in range(3)]
    rgb = np.zeros((ctx.LED_COUNT, 3), dtype=np.uint8)
    for i in range(lit[0]):
        v = ctx.amplify_quad(np.array([32 + (i * 223 // max(1,third-1))], dtype=np.uint16))[0]
        hue = (ctx.base_hue_offset + (ctx.hue_seed >> 1) + (i * 3)) % 256
        rgb[i] = ctx.hsv_to_rgb_bytes_vec(np.array([hue], dtype=np.uint8),
                                          np.array([ctx.base_saturation], dtype=np.uint8),
                                          np.array([min(255,v)], dtype=np.uint8))[0]
    for i in range(lit[1]):
        v = ctx.amplify_quad(np.array([32 + (i * 223 // max(1,third-1))], dtype=np.uint16))[0]
        hue = (ctx.base_hue_offset + (ctx.hue_seed >> 1) + (i * 3) + 32) % 256
        j = third + i
        if j < ctx.LED_COUNT:
            rgb[j] = ctx.hsv_to_rgb_bytes_vec(np.array([hue], dtype=np.uint8),
                                              np.array([ctx.base_saturation], dtype=np.uint8),
                                              np.array([min(255,v)], dtype=np.uint8))[0]
    for i in range(lit[2]):
        v = ctx.amplify_quad(np.array([32 + (i * 223 // max(1,third-1))], dtype=np.uint16))[0]
        hue = (ctx.base_hue_offset + (ctx.hue_seed >> 1) + (i * 3) + 64) % 256
        j = 2*third + i
        if j < ctx.LED_COUNT:
            rgb[j] = ctx.hsv_to_rgb_bytes_vec(np.array([hue], dtype=np.uint8),
                                              np.array([ctx.base_saturation], dtype=np.uint8),
                                              np.array([min(255,v)], dtype=np.uint8))[0]
    ctx.to_pixels_and_show(rgb)
