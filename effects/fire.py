# effects/fire.py
import numpy as np

_fire_heat_edge_v4 = None
_fire_heat_center_v4 = None
_fire_sparks_edge = []
_fire_sparks_center = []

def _apply_palette(ctx, v):
    """Aplica paleta com gamma ajustado para fogo"""
    pal = getattr(ctx, "current_palette", None)
    if pal and len(pal) >= 3:
        pal_arr = np.asarray(pal, dtype=np.uint8)
        m = len(pal_arr)
        idx = (np.power(v / 255.0, 0.7) * (m - 1)).astype(int) % m
        return pal_arr[idx]
    else:
        # Paleta fogo: preto -> vermelho -> laranja -> amarelo -> branco
        fire_pal = np.array([
            [0, 0, 0],       [64, 0, 0],      [255, 0, 0],
            [255, 64, 0],    [255, 128, 0],   [255, 200, 0],
            [255, 255, 0],   [255, 255, 128], [255, 255, 255]
        ], dtype=np.uint8)
        m = len(fire_pal)
        idx = (np.power(v / 255.0, 0.75) * (m - 1)).astype(int) % m
        return fire_pal[idx]

def effect_clean_fire_edge_v4(ctx, bands_u8, beat_flag, active):
    """
    Fogo nas bordas com envelope attack/release + análise espectral multiband
    - Graves: altura e intensidade
    - Médios: turbulência
    - Agudos: sparks
    - Derivada temporal para transientes
    """
    global _fire_heat_edge_v4, _fire_sparks_edge
    L = ctx.LED_COUNT
    
    if _fire_heat_edge_v4 is None or _fire_heat_edge_v4.shape[0] != L:
        _fire_heat_edge_v4 = np.zeros(L, dtype=np.float32)
        _fire_sparks_edge = []
    
    if len(bands_u8) == 0 or not active:
        _fire_heat_edge_v4 *= 0.78
        if np.max(_fire_heat_edge_v4) < 1.0:
            ctx.to_pixels_and_show(np.zeros((L, 3), dtype=np.uint8))
        return
    
    n = len(bands_u8)
    
    # === ANÁLISE MULTIBAND ===
    low_n = max(8, n // 8)
    mid_n = max(12, n // 4)
    high_n = max(6, n // 12)
    
    low_arr = np.asarray(bands_u8[:low_n], dtype=np.float32)
    mid_arr = np.asarray(bands_u8[low_n:mid_n], dtype=np.float32) if mid_n > low_n else low_arr
    high_arr = np.asarray(bands_u8[-high_n:], dtype=np.float32) if high_n > 0 else low_arr
    
    low_mean = float(np.mean(low_arr))
    low_std = float(np.std(low_arr))
    mid_mean = float(np.mean(mid_arr))
    high_mean = float(np.mean(high_arr))
    
    # === ENVELOPE COM ATTACK/RELEASE (mantém resposta rápida) ===
    if not hasattr(effect_clean_fire_edge_v4, 'env'):
        effect_clean_fire_edge_v4.env = 0.0
    env_prev = effect_clean_fire_edge_v4.env
    attack, release = 0.88, 0.25  # Attack mais rápido para responsividade
    alpha = attack if low_mean > env_prev else release
    env = alpha * low_mean + (1.0 - alpha) * env_prev
    effect_clean_fire_edge_v4.env = env
    
    # === DERIVADA TEMPORAL (detecta transientes) ===
    if not hasattr(effect_clean_fire_edge_v4, 'low_prev'):
        effect_clean_fire_edge_v4.low_prev = low_mean
    dlow = max(0.0, low_mean - effect_clean_fire_edge_v4.low_prev)
    effect_clean_fire_edge_v4.low_prev = low_mean
    
    # Normalização
    env01 = np.clip(env / 255.0, 0.0, 1.0)
    d01 = np.clip(dlow / 45.0, 0.0, 1.0)  # Transientes
    std01 = np.clip(low_std / 60.0, 0.0, 1.0)  # Variação espectral
    mid01 = np.clip(mid_mean / 255.0, 0.0, 1.0)
    high01 = np.clip(high_mean / 255.0, 0.0, 1.0)
    
    # === PARÂMETROS REATIVOS ===
    # Velocidade: graves + transientes
    speed = 0.9 + 3.5 * env01 + 2.0 * d01
    
    # Decay: menos decay em alta energia (fogo persiste)
    decay = (0.91 if active else 0.78) - 0.05 * env01
    
    # Cooling: graves reduzem (mais calor), agudos aumentam (esfria)
    cooling = 0.020 + 0.065 * (1.0 - env01) + 0.025 * high01
    cooling = np.clip(cooling, 0.010, 0.090)
    
    # Jitter: std dos graves + médios + boost no beat
    jitter = (6.0 + 22.0 * std01 + 12.0 * mid01) * (1.4 if beat_flag else 1.0)
    
    # === INJEÇÃO DE CALOR NAS BORDAS ===
    src = np.zeros(L, dtype=np.float32)
    
    # Intensidade: envelope + transientes + beat
    base_inj = (60.0 + 230.0 * env01 + 140.0 * d01) * (1.5 if beat_flag else 1.0)
    
    # Injeção direta nas extremidades
    src[0] += base_inj
    src[-1] += base_inj
    
    # Ruído local nas extremidades (controlado por jitter)
    k = int(2 + 4 * env01)  # Área de injeção cresce com energia
    randL = (np.random.rand(k).astype(np.float32) - 0.5) * jitter * 2.5
    randR = (np.random.rand(k).astype(np.float32) - 0.5) * jitter * 2.5
    src[:k] += randL
    src[-k:] += randR
    
    # === PROPAGAÇÃO COM ADVECÇÃO ===
    h = _fire_heat_edge_v4
    
    # Blur (difusão térmica)
    padded = np.pad(h, (2, 2), mode='edge')
    b = (padded[0:-4] * 0.06 + padded[1:-3] * 0.18 + 
         padded[2:-2] * 0.52 + padded[3:-1] * 0.18 + padded[4:] * 0.06)
    
    # Advecção: fogo se move das bordas para o centro
    x = np.arange(L, dtype=np.float32)
    vel = np.where(x < ctx.CENTER, +speed, -speed)
    x_src = np.clip(x - vel, 0.0, L - 1.0)
    adv = np.interp(x_src, x, b).astype(np.float32)
    
    # Turbulência (ruído global)
    noise = (np.random.rand(L).astype(np.float32) - 0.5) * jitter
    
    # Atualização
    h_new = adv * decay + src + noise
    h_new = np.clip(h_new * (1.0 - cooling), 0.0, 255.0)
    _fire_heat_edge_v4 = h_new
    
    # === SPARKS REATIVOS A AGUDOS ===
    spark_prob = 0.10 + 0.35 * high01
    max_sparks = int(3 + 9 * env01)
    
    if (beat_flag or np.random.random() < spark_prob) and len(_fire_sparks_edge) < max_sparks:
        num_sparks = 1 + int(2 * high01)
        for _ in range(num_sparks):
            # Sparks nas extremidades
            pos = 0 if np.random.random() < 0.5 else L - 1
            intensity = 0.75 + 0.25 * np.random.random()
            velocity = int(1 + 3 * env01) * (1 if pos == 0 else -1)
            _fire_sparks_edge.append([pos, intensity, velocity])
    
    # Atualizar sparks
    for sp in _fire_sparks_edge[:]:
        sp[1] *= 0.84  # Decay
        sp[0] += sp[2]  # Movimento
        
        if sp[1] < 0.06 or sp[0] < 0 or sp[0] >= L:
            _fire_sparks_edge.remove(sp)
            continue
        
        i = int(sp[0])
        if 0 <= i < L:
            h_new[i] = min(255.0, h_new[i] + 110 * sp[1])
            # Difusão do spark
            if i > 0:
                h_new[i-1] = min(255.0, h_new[i-1] + 35 * sp[1])
            if i < L - 1:
                h_new[i+1] = min(255.0, h_new[i+1] + 35 * sp[1])
    
    # === RENDERIZAÇÃO ===
    v = np.clip(h_new, 0, 255).astype(np.uint8)
    v = ctx.apply_floor_vec(v, active, None)
    rgb = _apply_palette(ctx, v)
    ctx.to_pixels_and_show(rgb)

def effect_clean_fire_center_v4(ctx, bands_u8, beat_flag, active):
    """
    Fogo no centro com envelope attack/release + análise espectral multiband
    Similar ao edge mas propagação para fora (centro -> bordas)
    """
    global _fire_heat_center_v4, _fire_sparks_center
    L = ctx.LED_COUNT
    
    if _fire_heat_center_v4 is None or _fire_heat_center_v4.shape[0] != L:
        _fire_heat_center_v4 = np.zeros(L, dtype=np.float32)
        _fire_sparks_center = []
    
    if len(bands_u8) == 0 or not active:
        _fire_heat_center_v4 *= 0.78
        if np.max(_fire_heat_center_v4) < 1.0:
            ctx.to_pixels_and_show(np.zeros((L, 3), dtype=np.uint8))
        return
    
    n = len(bands_u8)
    
    # === ANÁLISE MULTIBAND ===
    low_n = max(8, n // 8)
    mid_n = max(12, n // 4)
    high_n = max(6, n // 12)
    
    low_arr = np.asarray(bands_u8[:low_n], dtype=np.float32)
    mid_arr = np.asarray(bands_u8[low_n:mid_n], dtype=np.float32) if mid_n > low_n else low_arr
    high_arr = np.asarray(bands_u8[-high_n:], dtype=np.float32) if high_n > 0 else low_arr
    
    low_mean = float(np.mean(low_arr))
    low_std = float(np.std(low_arr))
    mid_mean = float(np.mean(mid_arr))
    high_mean = float(np.mean(high_arr))
    
    # === ENVELOPE COM ATTACK/RELEASE ===
    if not hasattr(effect_clean_fire_center_v4, 'env'):
        effect_clean_fire_center_v4.env = 0.0
    env_prev = effect_clean_fire_center_v4.env
    attack, release = 0.88, 0.25
    alpha = attack if low_mean > env_prev else release
    env = alpha * low_mean + (1.0 - alpha) * env_prev
    effect_clean_fire_center_v4.env = env
    
    # === DERIVADA TEMPORAL ===
    if not hasattr(effect_clean_fire_center_v4, 'low_prev'):
        effect_clean_fire_center_v4.low_prev = low_mean
    dlow = max(0.0, low_mean - effect_clean_fire_center_v4.low_prev)
    effect_clean_fire_center_v4.low_prev = low_mean
    
    # Normalização
    env01 = np.clip(env / 255.0, 0.0, 1.0)
    d01 = np.clip(dlow / 45.0, 0.0, 1.0)
    std01 = np.clip(low_std / 60.0, 0.0, 1.0)
    mid01 = np.clip(mid_mean / 255.0, 0.0, 1.0)
    high01 = np.clip(high_mean / 255.0, 0.0, 1.0)
    
    # === PARÂMETROS REATIVOS ===
    speed = 0.8 + 3.2 * env01 + 1.8 * d01
    decay = (0.91 if active else 0.78) - 0.04 * env01
    cooling = 0.025 + 0.070 * (1.0 - env01) + 0.020 * high01
    cooling = np.clip(cooling, 0.015, 0.095)
    jitter = (5.0 + 20.0 * std01 + 10.0 * mid01) * (1.35 if beat_flag else 1.0)
    
    # === INJEÇÃO NO CENTRO ===
    src = np.zeros(L, dtype=np.float32)
    base_inj = (70.0 + 220.0 * env01 + 130.0 * d01) * (1.45 if beat_flag else 1.0)
    
    c = ctx.CENTER
    src[c] += base_inj
    
    # Injeção em vizinhos do centro
    if c - 1 >= 0:
        src[c - 1] += base_inj * 0.70
    if c + 1 < L:
        src[c + 1] += base_inj * 0.70
    
    # Ruído local no centro
    for off in range(-2, 3):
        i = c + off
        if 0 <= i < L:
            src[i] += (np.random.rand() - 0.5) * jitter * 2.2
    
    # === PROPAGAÇÃO COM ADVECÇÃO ===
    h = _fire_heat_center_v4
    
    # Blur
    padded = np.pad(h, (2, 2), mode='edge')
    b = (padded[0:-4] * 0.06 + padded[1:-3] * 0.18 + 
         padded[2:-2] * 0.52 + padded[3:-1] * 0.18 + padded[4:] * 0.06)
    
    # Advecção: fogo se move do centro para fora
    x = np.arange(L, dtype=np.float32)
    vel = np.where(x < c, -speed, +speed)
    x_src = np.clip(x - vel, 0.0, L - 1.0)
    adv = np.interp(x_src, x, b).astype(np.float32)
    
    # Turbulência
    noise = (np.random.rand(L).astype(np.float32) - 0.5) * jitter
    
    # Atualização
    h_new = adv * decay + src + noise
    h_new = np.clip(h_new * (1.0 - cooling), 0.0, 255.0)
    _fire_heat_center_v4 = h_new
    
    # === SPARKS REATIVOS ===
    spark_prob = 0.15 + 0.40 * high01
    max_sparks = int(4 + 11 * env01)
    
    if (beat_flag or np.random.random() < spark_prob) and len(_fire_sparks_center) < max_sparks:
        num_sparks = 2 + int(3 * high01)
        for _ in range(num_sparks):
            # Sparks emanam do centro
            offset = int(np.random.randn() * 5)
            pos = np.clip(c + offset, 0, L - 1)
            intensity = 0.70 + 0.30 * np.random.random()
            velocity = int(np.sign(offset if offset != 0 else 1) * (1 + 2 * env01))
            _fire_sparks_center.append([pos, intensity, velocity])
    
    # Atualizar sparks
    for sp in _fire_sparks_center[:]:
        sp[1] *= 0.82
        sp[0] += sp[2]
        
        if sp[1] < 0.06 or sp[0] < 0 or sp[0] >= L:
            _fire_sparks_center.remove(sp)
            continue
        
        i = int(sp[0])
        if 0 <= i < L:
            h_new[i] = min(255.0, h_new[i] + 100 * sp[1])
            if i > 0:
                h_new[i-1] = min(255.0, h_new[i-1] + 30 * sp[1])
            if i < L - 1:
                h_new[i+1] = min(255.0, h_new[i+1] + 30 * sp[1])
    
    # === RENDERIZAÇÃO ===
    v = np.clip(h_new, 0, 255).astype(np.uint8)
    v = ctx.apply_floor_vec(v, active, None)
    rgb = _apply_palette(ctx, v)
    ctx.to_pixels_and_show(rgb)