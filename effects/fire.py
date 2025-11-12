# effects/fire.py
import numpy as np

_fire_heat_edge_v4 = None
_fire_heat_center_v4 = None
_fire_sparks_edge = []
_fire_sparks_center = []

def _apply_palette(ctx, v):
    pal = getattr(ctx, "current_palette", None)
    if pal and len(pal) >= 3:
        pal_arr = np.asarray(pal, dtype=np.uint8)
        m = len(pal_arr)
        idx = (np.power(v / 255.0, 0.85) * (m - 1)).astype(int) % m
        return pal_arr[idx]
    else:
        hue = (10 + 40 * (v / 255.0)**0.7).astype(np.uint8)
        sat = np.full(v.shape[0], max(190, ctx.base_saturation), dtype=np.uint8)
        return ctx.hsv_to_rgb_bytes_vec(hue, sat, v)

def effect_clean_fire_edge_v4(ctx, bands_u8, beat_flag, active):
    global _fire_heat_edge_v4, _fire_sparks_edge
    L = ctx.LED_COUNT
    if _fire_heat_edge_v4 is None or _fire_heat_edge_v4.shape[0] != L:
        _fire_heat_edge_v4 = np.zeros(L, dtype=np.float32)
        _fire_sparks_edge = []

    if len(bands_u8) == 0:
        _fire_heat_edge_v4 *= 0.85
        ctx.to_pixels_and_show(np.zeros((L, 3), dtype=np.uint8))
        return

    # Dinâmica baseada em graves e médios
    low = np.mean(bands_u8[:max(8, len(bands_u8)//8)])
    mid = np.mean(bands_u8[len(bands_u8)//8:len(bands_u8)//4])
    env = 0.7 * low + 0.3 * mid
    e01 = np.clip(env / 255, 0, 1)

    # Parâmetros
    decay = 0.94
    cooling = 0.02 + 0.02 * (1 - e01)
    jitter = 3.0 + 5.0 * e01

    # Injeta calor nas extremidades
    src = np.zeros(L, dtype=np.float32)
    inj = (120 + 250 * e01) * (1.5 if beat_flag else 1.0)
    src[0:4] += inj
    src[-4:] += inj

    # Propagação
    h = _fire_heat_edge_v4
    padded = np.pad(h, (1, 1), mode='edge')
    kernel = np.array([0.2, 0.6, 0.2])
    adv = np.convolve(padded, kernel, mode='valid')

    noise = (np.random.rand(L) - 0.5) * jitter
    h_new = adv * decay + src + noise
    h_new = np.clip(h_new * (1 - cooling), 0, 255)
    _fire_heat_edge_v4 = h_new

    # Sparks discretos
    if beat_flag and len(_fire_sparks_edge) < 5:
        for _ in range(2):
            pos = np.random.choice([0, L-1])
            _fire_sparks_edge.append([pos, 0.9])

    for sp in _fire_sparks_edge[:]:
        sp[1] *= 0.88
        if sp[1] < 0.1:
            _fire_sparks_edge.remove(sp)
            continue
        i = sp[0]
        if 0 <= i < L:
            h_new[i] = min(255, h_new[i] + 90 * sp[1])

    v = np.clip(h_new, 0, 255).astype(np.uint8)
    v = ctx.apply_floor_vec(v, active, None)
    rgb = _apply_palette(ctx, v)
    ctx.to_pixels_and_show(rgb)

def effect_clean_fire_center_v4(ctx, bands_u8, beat_flag, active):
    global _fire_heat_center_v4, _fire_sparks_center
    L = ctx.LED_COUNT
    if _fire_heat_center_v4 is None or _fire_heat_center_v4.shape[0] != L:
        _fire_heat_center_v4 = np.zeros(L, dtype=np.float32)
        _fire_sparks_center = []

    if len(bands_u8) == 0:
        _fire_heat_center_v4 *= 0.85
        ctx.to_pixels_and_show(np.zeros((L, 3), dtype=np.uint8))
        return

    low = np.mean(bands_u8[:max(8, len(bands_u8)//8)])
    mid = np.mean(bands_u8[len(bands_u8)//8:len(bands_u8)//4])
    env = 0.7 * low + 0.3 * mid
    e01 = np.clip(env / 255, 0, 1)

    decay = 0.93
    cooling = 0.02 + 0.02 * (1 - e01)
    jitter = 4.0 + 6.0 * e01

    src = np.zeros(L, dtype=np.float32)
    c = ctx.CENTER
    inj = (140 + 260 * e01) * (1.5 if beat_flag else 1.0)
    src[c-3:c+4] += inj

    h = _fire_heat_center_v4
    padded = np.pad(h, (1, 1), mode='edge')
    kernel = np.array([0.2, 0.6, 0.2])
    adv = np.convolve(padded, kernel, mode='valid')

    noise = (np.random.rand(L) - 0.5) * jitter
    h_new = adv * decay + src + noise
    h_new = np.clip(h_new * (1 - cooling), 0, 255)
    _fire_heat_center_v4 = h_new

    if beat_flag and len(_fire_sparks_center) < 6:
        for _ in range(3):
            pos = c + np.random.randint(-6, 7)
            _fire_sparks_center.append([pos, 0.9])

    for sp in _fire_sparks_center[:]:
        sp[1] *= 0.88
        if sp[1] < 0.1:
            _fire_sparks_center.remove(sp)
            continue
        i = sp[0]
        if 0 <= i < L:
            h_new[i] = min(255, h_new[i] + 80 * sp[1])

    v = np.clip(h_new, 0, 255).astype(np.uint8)
    v = ctx.apply_floor_vec(v, active, None)
    rgb = _apply_palette(ctx, v)
    ctx.to_pixels_and_show(rgb)