# effects/fire.py
import numpy as np

# === ESTADO GLOBAL ===
_fire_heat_edge_v4 = None
_fire_sparks_edge = []

_fire_heat_center_v4 = None
_fire_sparks_center = []

def effect_clean_fire_edge_v4(ctx, bands_u8, beat_flag, active):
    global _fire_heat_edge_v4, _fire_sparks_edge
    L = ctx.LED_COUNT
    if _fire_heat_edge_v4 is None or _fire_heat_edge_v4.shape[0] != L:
        _fire_heat_edge_v4 = np.zeros(L, dtype=np.float32)
        _fire_sparks_edge = []

    n = len(bands_u8)
    if n == 0:
        _fire_heat_edge_v4 *= 0.9
        ctx.to_pixels_and_show(np.zeros((L, 3), dtype=np.uint8))
        return

    low_n = max(8, n // 8)
    mid_n = max(low_n, n // 4)
    low = np.asarray(bands_u8[:low_n], dtype=np.float32)
    mid = np.asarray(bands_u8[low_n:mid_n], dtype=np.float32)
    high = np.asarray(bands_u8[mid_n:], dtype=np.float32)

    low_mean = np.mean(low)
    mid_mean = np.mean(mid)
    high_mean = np.mean(high)
    low_std = np.std(low)

    # Envelope
    if not hasattr(effect_clean_fire_edge_v4, 'env'):
        effect_clean_fire_edge_v4.env = 0.0
    env = 0.8 * low_mean + 0.2 * effect_clean_fire_edge_v4.env
    effect_clean_fire_edge_v4.env = env
    dlow = max(0, low_mean - getattr(effect_clean_fire_edge_v4, 'prev', 0))
    effect_clean_fire_edge_v4.prev = low_mean

    e01 = np.clip(env / 255, 0, 1)
    d01 = np.clip(dlow / 50, 0, 1)
    m01 = np.clip(mid_mean / 255, 0, 1)
    h01 = np.clip(high_mean / 255, 0, 1)

    speed = 1.5 + 4.5 * e01 + 2.0 * d01 + 1.8 * m01
    decay = 0.93 - 0.06 * e01
    cooling = 0.015 + 0.05 * (1 - e01) - 0.03 * m01
    jitter = (12 + 30 * (low_std / 64) + 20 * h01) * (1.5 if beat_flag else 1.0)

    src = np.zeros(L, dtype=np.float32)
    base_inj = (70 + 230 * e01 + 140 * d01) * (1.5 if beat_flag else 1.0)
    src[0] += base_inj
    src[-1] += base_inj
    k = 5
    src[:k] += (np.random.rand(k) - 0.5) * jitter * 3
    src[-k:] += (np.random.rand(k) - 0.5) * jitter * 3

    h = _fire_heat_edge_v4
    padded = np.pad(h, (2, 2), mode='edge')
    kernel = np.array([0.06, 0.18, 0.52, 0.18, 0.06])
    b = np.convolve(padded, kernel, mode='valid')
    x = np.arange(L, dtype=np.float32)
    vel = np.where(x < ctx.CENTER, speed, -speed)
    x_src = np.clip(x - vel, 0, L - 1)
    adv = np.interp(x_src, x, b)
    noise = (np.random.rand(L) - 0.5) * jitter
    h_new = adv * decay + src + noise
    h_new = np.clip(h_new * (1 - cooling), 0, 255)
    _fire_heat_edge_v4 = h_new

    if beat_flag and len(_fire_sparks_edge) < 10:
        for _ in range(2 + int(4 * m01)):
            pos = np.random.choice([0, L-1]) + np.random.uniform(-8, 8)
            vel = 100 + 150 * e01
            life = 1.0
            _fire_sparks_edge.append([pos, vel, life])

    for sp in _fire_sparks_edge[:]:
        sp[0] += sp[1] * 0.018
        sp[2] *= 0.93
        if sp[2] < 0.1 or not (0 <= sp[0] < L):
            _fire_sparks_edge.remove(sp)
            continue
        i = int(sp[0])
        if 0 <= i < L:
            h_new[i] = min(255, h_new[i] + 120 * sp[2])

    v = np.clip(h_new, 0, 255).astype(np.uint8)
    v = ctx.apply_floor_vec(v, active, None)

    pal = getattr(ctx, "current_palette", None)
    if pal and len(pal) >= 3:
        pal_arr = np.asarray(pal, dtype=np.uint8)
        m = len(pal_arr)
        idx = (np.power(v / 255.0, 0.7) * (m - 1)).astype(int) % m
        rgb = pal_arr[idx]
    else:
        hue = (5 + 45 * (v / 255.0)**0.6 + 10 * h01).astype(np.uint8)
        sat = np.full(L, max(200, ctx.base_saturation), dtype=np.uint8)
        rgb = ctx.hsv_to_rgb_bytes_vec(hue, sat, v)

    ctx.to_pixels_and_show(rgb)


def effect_clean_fire_center_v4(ctx, bands_u8, beat_flag, active):
    global _fire_heat_center_v4, _fire_sparks_center
    L = ctx.LED_COUNT
    if _fire_heat_center_v4 is None or _fire_heat_center_v4.shape[0] != L:
        _fire_heat_center_v4 = np.zeros(L, dtype=np.float32)
        _fire_sparks_center = []

    n = len(bands_u8)
    if n == 0:
        _fire_heat_center_v4 *= 0.88
        ctx.to_pixels_and_show(np.zeros((L, 3), dtype=np.uint8))
        return

    low_n = max(8, n // 8)
    mid_n = max(low_n, n // 4)
    low = np.asarray(bands_u8[:low_n], dtype=np.float32)
    mid = np.asarray(bands_u8[low_n:mid_n], dtype=np.float32)
    high = np.asarray(bands_u8[mid_n:], dtype=np.float32)

    low_mean = np.mean(low)
    mid_mean = np.mean(mid)
    high_mean = np.mean(high)
    low_std = np.std(low)

    if not hasattr(effect_clean_fire_center_v4, 'env'):
        effect_clean_fire_center_v4.env = 0.0
    env = 0.8 * low_mean + 0.2 * effect_clean_fire_center_v4.env
    effect_clean_fire_center_v4.env = env
    dlow = max(0, low_mean - getattr(effect_clean_fire_center_v4, 'prev', 0))
    effect_clean_fire_center_v4.prev = low_mean

    e01 = np.clip(env / 255, 0, 1)
    d01 = np.clip(dlow / 50, 0, 1)
    m01 = np.clip(mid_mean / 255, 0, 1)
    h01 = np.clip(high_mean / 255, 0, 1)

    speed = 1.0 + 3.8 * e01 + 1.8 * d01 + 2.0 * m01
    decay = 0.92 - 0.05 * e01
    cooling = 0.02 + 0.06 * (1 - e01) - 0.02 * m01
    jitter = (10 + 28 * (low_std / 64) + 18 * h01) * (1.4 if beat_flag else 1.0)

    src = np.zeros(L, dtype=np.float32)
    base_inj = (80 + 210 * e01 + 130 * d01) * (1.4 if beat_flag else 1.0)
    c = ctx.CENTER
    for off in range(-2, 3):
        i = c + off
        if 0 <= i < L:
            src[i] += base_inj * (1.0 if off == 0 else 0.7)
    src[c-3:c+4] += (np.random.rand(7) - 0.5) * jitter * 2.5

    h = _fire_heat_center_v4
    padded = np.pad(h, (2, 2), mode='edge')
    kernel = np.array([0.06, 0.18, 0.52, 0.18, 0.06])
    b = np.convolve(padded, kernel, mode='valid')
    x = np.arange(L, dtype=np.float32)
    vel = np.where(x < c, -speed, speed)
    x_src = np.clip(x - vel, 0, L - 1)
    adv = np.interp(x_src, x, b)
    noise = (np.random.rand(L) - 0.5) * jitter
    h_new = adv * decay + src + noise
    h_new = np.clip(h_new * (1 - cooling), 0, 255)
    _fire_heat_center_v4 = h_new

    if beat_flag and len(_fire_sparks_center) < 12:
        for _ in range(3 + int(5 * m01)):
            pos = c + np.random.uniform(-10, 10)
            vel = 90 + 140 * e01
            life = 1.0
            _fire_sparks_center.append([pos, vel, life])

    for sp in _fire_sparks_center[:]:
        sp[0] += sp[1] * 0.02
        sp[2] *= 0.94
        if sp[2] < 0.1 or not (0 <= sp[0] < L):
            _fire_sparks_center.remove(sp)
            continue
        i = int(sp[0])
        if 0 <= i < L:
            h_new[i] = min(255, h_new[i] + 110 * sp[2])

    v = np.clip(h_new, 0, 255).astype(np.uint8)
    v = ctx.apply_floor_vec(v, active, None)

    pal = getattr(ctx, "current_palette", None)
    if pal and len(pal) >= 3:
        pal_arr = np.asarray(pal, dtype=np.uint8)
        m = len(pal_arr)
        idx = (np.power(v / 255.0, 0.75) * (m - 1)).astype(int) % m
        rgb = pal_arr[idx]
    else:
        hue = (8 + 42 * (v / 255.0)**0.6 + 12 * h01).astype(np.uint8)
        sat = np.full(L, max(195, ctx.base_saturation), dtype=np.uint8)
        rgb = ctx.hsv_to_rgb_bytes_vec(hue, sat, v)

    ctx.to_pixels_and_show(rgb)
# EOF --- IGNORE ---
