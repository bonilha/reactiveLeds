# effects/fire.py
import numpy as np

_fire_heat_edge_v3 = None

def effect_clean_fire_edge_v3(ctx, bands_u8, beat_flag, active):
    global _fire_heat_edge_v3
    if _fire_heat_edge_v3 is None or _fire_heat_edge_v3.shape[0] != ctx.LED_COUNT:
        _fire_heat_edge_v3 = np.zeros(ctx.LED_COUNT, dtype=np.float32)
    n = len(bands_u8)
    low_n = max(8, n // 8)
    low_arr = np.asarray(bands_u8[:low_n], dtype=np.float32)
    low_mean = float(np.mean(low_arr))
    low_std  = float(np.std(low_arr))
    # env rápido + derivada
    if not hasattr(effect_clean_fire_edge_v3, 'env'): effect_clean_fire_edge_v3.env = 0.0
    env_prev = effect_clean_fire_edge_v3.env
    attack, release = 0.85, 0.28
    alpha = attack if low_mean > env_prev else release
    env = alpha * low_mean + (1.0 - alpha) * env_prev
    effect_clean_fire_edge_v3.env = env
    if not hasattr(effect_clean_fire_edge_v3, 'low_prev'): effect_clean_fire_edge_v3.low_prev = low_mean
    dlow = max(0.0, low_mean - effect_clean_fire_edge_v3.low_prev)
    effect_clean_fire_edge_v3.low_prev = low_mean
    env01 = np.clip(env / 255.0, 0.0, 1.0)
    d01   = np.clip(dlow / 40.0, 0.0, 1.0)
    std01 = np.clip(low_std / 64.0, 0.0, 1.0)
    speed   = 0.8 + 3.2 * env01 + 1.6 * d01
    decay   = (0.90 if active else 0.80) - 0.04 * env01
    cooling = 0.03 + 0.07 * (1.0 - env01)
    jitter  = (8.0 + 20.0 * std01) * (1.25 if beat_flag else 1.0)
    src = np.zeros(ctx.LED_COUNT, dtype=np.float32)
    base_inj = (50.0 + 210.0 * env01 + 120.0 * d01) * (1.35 if beat_flag else 1.0)
    src[0]  += base_inj
    src[-1] += base_inj
    k = 3
    randL = (np.random.rand(k).astype(np.float32) - 0.5) * jitter * 2.2
    randR = (np.random.rand(k).astype(np.float32) - 0.5) * jitter * 2.2
    src[:k]  += randL
    src[-k:] += randR
    h = _fire_heat_edge_v3
    padded = np.pad(h, (2,2), mode='edge')
    b = (padded[0:-4]*0.06 + padded[1:-3]*0.18 + padded[2:-2]*0.52 + padded[3:-1]*0.18 + padded[4:]*0.06)
    x = np.arange(ctx.LED_COUNT, dtype=np.float32)
    vel = np.where(x < ctx.CENTER, +speed, -speed)
    x_src = np.clip(x - vel, 0.0, ctx.LED_COUNT - 1.0)
    adv = np.interp(x_src, x, b).astype(np.float32)
    noise = (np.random.rand(ctx.LED_COUNT).astype(np.float32) - 0.5) * jitter
    h_new = adv * decay + src + noise
    h_new = np.clip(h_new * (1.0 - cooling), 0.0, 255.0)
    _fire_heat_edge_v3 = h_new
    v = np.clip(h_new, 0, 255).astype(np.uint8)
    v = ctx.apply_floor_vec(v, active, None)
    h0 = int(10.0 * 255.0 / 360.0)
    h1 = int(38.0 * 255.0 / 360.0)
    frac = np.power(v.astype(np.float32) / 255.0, 0.55)
    hue = (h0 + (h1 - h0) * frac).astype(np.uint8)
    sat = np.full(ctx.LED_COUNT, max(210, ctx.base_saturation), dtype=np.uint8)
    rgb = ctx.hsv_to_rgb_bytes_vec(hue, sat, v)
    ctx.to_pixels_and_show(rgb)

_fire_heat_center_v3 = None

def effect_clean_fire_center_v3(ctx, bands_u8, beat_flag, active):
    global _fire_heat_center_v3
    if _fire_heat_center_v3 is None or _fire_heat_center_v3.shape[0] != ctx.LED_COUNT:
        _fire_heat_center_v3 = np.zeros(ctx.LED_COUNT, dtype=np.float32)
    n = len(bands_u8)
    low_n   = max(8, n // 8)
    low_arr = np.asarray(bands_u8[:low_n], dtype=np.float32)
    low_mean = float(np.mean(low_arr))
    low_std  = float(np.std(low_arr))
    if not hasattr(effect_clean_fire_center_v3, 'env'): effect_clean_fire_center_v3.env = 0.0
    env_prev = effect_clean_fire_center_v3.env
    attack, release = 0.85, 0.28
    alpha = attack if low_mean > env_prev else release
    env = alpha * low_mean + (1.0 - alpha) * env_prev
    effect_clean_fire_center_v3.env = env
    if not hasattr(effect_clean_fire_center_v3, 'low_prev'): effect_clean_fire_center_v3.low_prev = low_mean
    dlow = max(0.0, low_mean - effect_clean_fire_center_v3.low_prev)
    effect_clean_fire_center_v3.low_prev = low_mean
    env01 = np.clip(env / 255.0, 0.0, 1.0)
    d01   = np.clip(dlow / 40.0, 0.0, 1.0)
    std01 = np.clip(low_std / 64.0, 0.0, 1.0)
    speed   = 0.7 + 3.0 * env01 + 1.5 * d01
    decay   = (0.90 if active else 0.80) - 0.03 * env01
    cooling = 0.04 + 0.08 * (1.0 - env01)
    jitter  = (6.0 + 18.0 * std01) * (1.25 if beat_flag else 1.0)
    src = np.zeros(ctx.LED_COUNT, dtype=np.float32)
    base_inj = (60.0 + 200.0 * env01 + 110.0 * d01) * (1.30 if beat_flag else 1.0)
    src[ctx.CENTER] += base_inj
    if ctx.CENTER-1 >= 0:       src[ctx.CENTER-1] += base_inj * 0.75
    if ctx.CENTER+1 < ctx.LED_COUNT: src[ctx.CENTER+1] += base_inj * 0.75
    for off in (-2,-1,0,1,2):
        i = ctx.CENTER + off
        if 0 <= i < ctx.LED_COUNT:
            src[i] += (np.random.rand() - 0.5) * jitter * 2.0
    h = _fire_heat_center_v3
    padded = np.pad(h, (2,2), mode='edge')
    b = (padded[0:-4]*0.06 + padded[1:-3]*0.18 + padded[2:-2]*0.52 + padded[3:-1]*0.18 + padded[4:]*0.06)
    x = np.arange(ctx.LED_COUNT, dtype=np.float32)
    vel = np.where(x < ctx.CENTER, -speed, +speed)  # para fora
    x_src = np.clip(x - vel, 0.0, ctx.LED_COUNT - 1.0)
    adv = np.interp(x_src, x, b).astype(np.float32)
    noise = (np.random.rand(ctx.LED_COUNT).astype(np.float32) - 0.5) * jitter
    h_new = adv * decay + src + noise
    h_new = np.clip(h_new * (1.0 - cooling), 0.0, 255.0)
    _fire_heat_center_v3 = h_new
    v = np.clip(h_new, 0, 255).astype(np.uint8)
    v = ctx.apply_floor_vec(v, active, None)
    h0 = int(10.0 * 255.0 / 360.0)
    h1 = int(38.0 * 255.0 / 360.0)
    frac = np.power(v.astype(np.float32) / 255.0, 0.55)
    hue = (h0 + (h1 - h0) * frac).astype(np.uint8)
    sat = np.full(ctx.LED_COUNT, max(210, ctx.base_saturation), dtype=np.uint8)
    rgb = ctx.hsv_to_rgb_bytes_vec(hue, sat, v)
    ctx.to_pixels_and_show(rgb)
