import numpy as np

_fire_heat_edge_v3 = None
_sparks = []  # Nova: lista de [pos, vel_up, life] para faíscas

def effect_clean_fire_edge_v3(ctx, bands_u8, beat_flag, active):
    global _fire_heat_edge_v3, _sparks
    if _fire_heat_edge_v3 is None or _fire_heat_edge_v3.shape[0] != ctx.LED_COUNT:
        _fire_heat_edge_v3 = np.zeros(ctx.LED_COUNT, dtype=np.float32)
        _sparks = []

    n = len(bands_u8)
    low_n = max(8, n // 8)
    mid_n = max(low_n, n // 4)  # Novo: mids para turbulência
    low_arr = np.asarray(bands_u8[:low_n], dtype=np.float32)
    mid_arr = np.asarray(bands_u8[low_n:mid_n], dtype=np.float32)
    low_mean = float(np.mean(low_arr))
    mid_mean = float(np.mean(mid_arr))
    low_std = float(np.std(low_arr))
    mid_std = float(np.std(mid_arr))

    # Envelope (mantido, mas adicione mids para jitter extra)
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
    d01 = np.clip(dlow / 40.0, 0.0, 1.0)
    std01 = np.clip(low_std / 64.0, 0.0, 1.0)
    mid01 = np.clip(mid_mean / 255.0, 0.0, 1.0)  # Novo: mids controlam turbulência

    # Parâmetros dinâmicos (mais variados)
    speed = 1.2 + 4.0 * env01 + 2.0 * d01 + 1.5 * mid01  # Mids aumentam speed
    decay = (0.92 if active else 0.82) - 0.05 * env01
    cooling = 0.02 + 0.06 * (1.0 - env01) - 0.02 * mid01  # Mids reduzem cooling (mais chamas)
    jitter = (10.0 + 25.0 * std01 + 15.0 * mid_std / 64.0) * (1.4 if beat_flag else 1.0)

    # Injeção de calor (mantida, mas boost em mids)
    src = np.zeros(ctx.LED_COUNT, dtype=np.float32)
    base_inj = (60.0 + 220.0 * env01 + 130.0 * d01) * (1.4 if beat_flag else 1.0)
    src[0] += base_inj + mid_mean * 0.5
    src[-1] += base_inj + mid_mean * 0.5
    k = 4  # Aumentado para mais jitter
    randL = (np.random.rand(k).astype(np.float32) - 0.5) * jitter * 2.5
    randR = (np.random.rand(k).astype(np.float32) - 0.5) * jitter * 2.5
    src[:k] += randL
    src[-k:] += randR

    # Propagação (mantida)
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

    # Novo: Sparks em beats (faíscas "voando" para cima)
    if beat_flag and len(_sparks) < 8:
        for _ in range(2 + int(3 * mid01)):  # Mais sparks em mids altos
            pos = np.random.choice([0, ctx.LED_COUNT-1]) + np.random.uniform(-5, 5)
            vel_up = 80.0 + 120.0 * env01  # Para cima (positivo)
            life = 1.0
            _sparks.append([pos, vel_up, life])

    # Atualiza sparks
    for sp in _sparks[:]:
        sp[0] += sp[1] * 0.02  # Movimento lento
        sp[2] *= 0.95  # Decay
        if sp[2] < 0.1 or sp[0] < 0 or sp[0] >= ctx.LED_COUNT:
            _sparks.remove(sp)
        else:
            idx = int(sp[0])
            if 0 <= idx < ctx.LED_COUNT:
                h_new[idx] = min(255, h_new[idx] + 100 * sp[2])  # Adiciona brilho branco

    # Valor final
    v = np.clip(h_new, 0, 255).astype(np.uint8)
    v = ctx.apply_floor_vec(v, active, None)

    # Cores: Integre palette (novo!)
    pal = getattr(ctx, "current_palette", None)
    if pal and len(pal) >= 3:  # Ex: paleta com azul (frio), laranja, vermelho (quente)
        pal_arr = np.asarray(pal, dtype=np.uint8)
        frac = np.power(v.astype(np.float32) / 255.0, 0.65)  # Mais saturado em altos
        idx = (frac * (len(pal) - 1)).astype(np.int32)
        rgb = pal_arr[idx]  # Mapear heat para paleta
    else:  # Fallback HSV (variado)
        h0 = int(5.0 * 255.0 / 360.0)  # Azul escuro para baixa heat
        h1 = int(50.0 * 255.0 / 360.0)  # Vermelho para alta
        frac = np.power(v.astype(np.float32) / 255.0, 0.65)
        hue = (h0 + (h1 - h0) * frac + (mid_mean / 10)).astype(np.uint8)  # Mids variam hue
        sat = np.full(ctx.LED_COUNT, max(200, ctx.base_saturation), dtype=np.uint8)
        rgb = ctx.hsv_to_rgb_bytes_vec(hue, sat, v)

    ctx.to_pixels_and_show(rgb)