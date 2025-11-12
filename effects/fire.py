# effects/fire.py
import numpy as np

_fire_heat_edge_v4 = None
_fire_heat_center_v4 = None
_fire_sparks_edge = []
_fire_sparks_center = []
_fire_explosion_edge = 0.0
_fire_explosion_center = 0.0
_fire_beat_history_edge = []
_fire_beat_history_center = []

def _apply_palette(ctx, v):
    """Aplica paleta de cores com mapeamento dinâmico baseado em intensidade"""
    pal = getattr(ctx, "current_palette", None)
    if pal and len(pal) >= 3:
        pal_arr = np.asarray(pal, dtype=np.uint8)
        m = len(pal_arr)
        # Gamma ajustado para realçar cores quentes
        idx = (np.power(v / 255.0, 0.65) * (m - 1)).astype(int) % m
        return pal_arr[idx]
    else:
        # Paleta de fogo realista: preto -> vermelho escuro -> vermelho -> laranja -> amarelo -> branco
        fire_pal = np.array([
            [0, 0, 0],         # 0: preto (sem calor)
            [32, 0, 0],        # 1: vermelho muito escuro
            [96, 0, 0],        # 2: vermelho escuro
            [192, 0, 0],       # 3: vermelho
            [255, 32, 0],      # 4: vermelho-laranja
            [255, 96, 0],      # 5: laranja escuro
            [255, 160, 0],     # 6: laranja
            [255, 224, 0],     # 7: laranja-amarelo
            [255, 255, 64],    # 8: amarelo
            [255, 255, 192],   # 9: amarelo claro
            [255, 255, 255]    # 10: branco (calor extremo)
        ], dtype=np.uint8)
        m = len(fire_pal)
        idx = (np.power(v / 255.0, 0.7) * (m - 1)).astype(int) % m
        return fire_pal[idx]

def _analyze_spectrum(bands_u8):
    """Analisa o espectro de frequências para extrair características da música"""
    if len(bands_u8) == 0:
        return {
            'bass': 0.0, 'mid_low': 0.0, 'mid': 0.0, 'mid_high': 0.0, 'high': 0.0,
            'energy': 0.0, 'bass_punch': 0.0, 'brightness': 0.0, 'complexity': 0.0
        }
    
    n = len(bands_u8)
    # Divide espectro em bandas de frequência
    bass_end = max(4, n // 10)           # ~20-100 Hz
    mid_low_end = max(8, n // 5)         # ~100-300 Hz
    mid_end = max(12, n // 3)            # ~300-1000 Hz
    mid_high_end = max(20, 2 * n // 3)   # ~1000-4000 Hz
    
    bass = float(np.mean(bands_u8[:bass_end]))
    mid_low = float(np.mean(bands_u8[bass_end:mid_low_end]))
    mid = float(np.mean(bands_u8[mid_low_end:mid_end]))
    mid_high = float(np.mean(bands_u8[mid_end:mid_high_end]))
    high = float(np.mean(bands_u8[mid_high_end:]))
    
    # Métricas derivadas
    energy = float(np.mean(bands_u8))
    bass_punch = bass / max(1.0, energy)  # Quanto dos graves domina
    brightness = (mid_high + high) / max(1.0, energy)  # Quanto de agudos
    complexity = float(np.std(bands_u8)) / max(1.0, energy)  # Variação espectral
    
    return {
        'bass': bass, 'mid_low': mid_low, 'mid': mid, 'mid_high': mid_high, 'high': high,
        'energy': energy, 'bass_punch': bass_punch, 'brightness': brightness, 
        'complexity': complexity
    }

def effect_clean_fire_edge_v4(ctx, bands_u8, beat_flag, active):
    """
    Efeito de fogo nas bordas (extremidades) com reatividade musical avançada:
    - Graves (bass) controlam altura/intensidade das chamas
    - Médios controlam movimento e oscilação
    - Agudos criam sparks e faíscas
    - Beats criam explosões de fogo
    - Complexidade espectral afeta turbulência
    """
    global _fire_heat_edge_v4, _fire_sparks_edge, _fire_explosion_edge, _fire_beat_history_edge
    
    L = ctx.LED_COUNT
    if _fire_heat_edge_v4 is None or _fire_heat_edge_v4.shape[0] != L:
        _fire_heat_edge_v4 = np.zeros(L, dtype=np.float32)
        _fire_sparks_edge = []
        _fire_explosion_edge = 0.0
        _fire_beat_history_edge = []

    # Decay mais rápido quando inativo
    if len(bands_u8) == 0 or not active:
        _fire_heat_edge_v4 *= 0.75
        _fire_explosion_edge *= 0.85
        if np.max(_fire_heat_edge_v4) < 1.0 and _fire_explosion_edge < 0.1:
            ctx.to_pixels_and_show(np.zeros((L, 3), dtype=np.uint8))
        return

    # Análise espectral detalhada
    spectrum = _analyze_spectrum(bands_u8)
    bass = spectrum['bass']
    mid = spectrum['mid']
    high = spectrum['high']
    energy = spectrum['energy']
    bass_punch = spectrum['bass_punch']
    brightness = spectrum['brightness']
    complexity = spectrum['complexity']
    
    # Normalização para [0, 1]
    energy_01 = np.clip(energy / 255.0, 0, 1)
    bass_01 = np.clip(bass / 255.0, 0, 1)
    mid_01 = np.clip(mid / 255.0, 0, 1)
    high_01 = np.clip(high / 255.0, 0, 1)
    
    # Detecção de beat com histórico
    _fire_beat_history_edge.append(1 if beat_flag else 0)
    if len(_fire_beat_history_edge) > 5:
        _fire_beat_history_edge.pop(0)
    beat_strength = sum(_fire_beat_history_edge) / len(_fire_beat_history_edge)
    
    # === PARÂMETROS REATIVOS ===
    
    # Decay: mais lento com energia alta (fogo persiste)
    decay = 0.88 + 0.08 * (1 - energy_01)
    
    # Cooling: graves reduzem cooling (mais calor), agudos aumentam (esfria)
    cooling = 0.010 + 0.030 * brightness - 0.015 * bass_punch
    cooling = np.clip(cooling, 0.005, 0.050)
    
    # Jitter/turbulência: controlado por complexidade e médios
    jitter = 1.0 + 8.0 * complexity + 4.0 * mid_01
    
    # Velocidade de propagação: graves fazem subir mais rápido
    propagation_speed = 0.85 + 0.15 * bass_01
    
    # === INJEÇÃO DE CALOR NAS EXTREMIDADES ===
    src = np.zeros(L, dtype=np.float32)
    
    # Intensidade base controlada por graves
    base_intensity = 80 + 200 * bass_01
    
    # Boost no beat (explosão momentânea)
    if beat_flag:
        _fire_explosion_edge = min(1.0, _fire_explosion_edge + 0.6 * (0.5 + 0.5 * energy_01))
    
    # Decaimento da explosão
    _fire_explosion_edge *= 0.88
    
    # Multiplicador total
    multiplier = (1.0 + 2.0 * _fire_explosion_edge) * (1.0 + 0.5 * beat_strength)
    injection = base_intensity * multiplier
    
    # Área de injeção aumenta com energia
    inject_width = int(3 + 8 * energy_01)
    src[0:inject_width] += injection
    src[-inject_width:] += injection
    
    # Assimetria sutil baseada em mid_low vs mid_high
    mid_low = spectrum['mid_low'] / 255.0
    mid_high = spectrum['mid_high'] / 255.0
    asymmetry = (mid_high - mid_low) * 50
    src[0:inject_width] += asymmetry
    src[-inject_width:] -= asymmetry
    
    # === PROPAGAÇÃO COM ADVECÇÃO ===
    h = _fire_heat_edge_v4
    
    # Kernel de difusão adaptativo (mais largo com energia alta)
    kernel_size = 3 if energy_01 < 0.3 else (5 if energy_01 < 0.6 else 7)
    if kernel_size == 3:
        kernel = np.array([0.25, 0.50, 0.25])
        pad_size = 1
    elif kernel_size == 5:
        kernel = np.array([0.1, 0.25, 0.3, 0.25, 0.1])
        pad_size = 2
    else:
        kernel = np.array([0.05, 0.15, 0.20, 0.20, 0.20, 0.15, 0.05])
        pad_size = 3
    
    padded = np.pad(h, (pad_size, pad_size), mode='edge')
    diffused = np.convolve(padded, kernel, mode='valid')
    
    # Advecção (movimento ascendente) controlada por propagation_speed
    adv = diffused * propagation_speed
    
    # Ruído/turbulência
    noise = (np.random.rand(L) - 0.5) * jitter
    
    # Atualização
    h_new = adv * decay + src + noise
    h_new = np.clip(h_new * (1 - cooling), 0, 255)
    _fire_heat_edge_v4 = h_new
    
    # === SPARKS/FAÍSCAS CONTROLADAS POR AGUDOS ===
    spark_probability = 0.08 + 0.25 * high_01
    max_sparks = int(4 + 8 * energy_01)
    
    if (beat_flag or np.random.random() < spark_probability) and len(_fire_sparks_edge) < max_sparks:
        num_new_sparks = 1 + int(3 * high_01)
        for _ in range(num_new_sparks):
            # Sparks nas extremidades
            pos = np.random.choice([0, L-1]) if np.random.random() < 0.7 else np.random.randint(0, min(15, L))
            if np.random.random() > 0.5:
                pos = max(L - 15, 0) + np.random.randint(0, min(15, L))
            
            intensity = 0.7 + 0.3 * np.random.random()
            velocity = int(1 + 3 * energy_01) * (1 if pos < L // 2 else -1)
            _fire_sparks_edge.append([pos, intensity, velocity])
    
    # Atualizar sparks
    for sp in _fire_sparks_edge[:]:
        sp[1] *= 0.82  # Decay de intensidade
        sp[0] += sp[2]  # Movimento
        
        if sp[1] < 0.05 or sp[0] < 0 or sp[0] >= L:
            _fire_sparks_edge.remove(sp)
            continue
        
        i = int(sp[0])
        if 0 <= i < L:
            # Sparks adicionam calor local
            spark_heat = 120 * sp[1]
            h_new[i] = min(255, h_new[i] + spark_heat)
            # Difusão do spark
            if i > 0:
                h_new[i-1] = min(255, h_new[i-1] + spark_heat * 0.3)
            if i < L - 1:
                h_new[i+1] = min(255, h_new[i+1] + spark_heat * 0.3)
    
    # === RENDERIZAÇÃO ===
    v = np.clip(h_new, 0, 255).astype(np.uint8)
    v = ctx.apply_floor_vec(v, active, None)
    rgb = _apply_palette(ctx, v)
    ctx.to_pixels_and_show(rgb)

def effect_clean_fire_center_v4(ctx, bands_u8, beat_flag, active):
    """
    Efeito de fogo no centro com reatividade musical avançada:
    - Comportamento similar ao edge mas focado no centro
    - Propagação bilateral a partir do centro
    - Reage mais intensamente a graves profundos
    """
    global _fire_heat_center_v4, _fire_sparks_center, _fire_explosion_center, _fire_beat_history_center
    
    L = ctx.LED_COUNT
    if _fire_heat_center_v4 is None or _fire_heat_center_v4.shape[0] != L:
        _fire_heat_center_v4 = np.zeros(L, dtype=np.float32)
        _fire_sparks_center = []
        _fire_explosion_center = 0.0
        _fire_beat_history_center = []

    if len(bands_u8) == 0 or not active:
        _fire_heat_center_v4 *= 0.75
        _fire_explosion_center *= 0.85
        if np.max(_fire_heat_center_v4) < 1.0 and _fire_explosion_center < 0.1:
            ctx.to_pixels_and_show(np.zeros((L, 3), dtype=np.uint8))
        return

    # Análise espectral
    spectrum = _analyze_spectrum(bands_u8)
    bass = spectrum['bass']
    mid = spectrum['mid']
    high = spectrum['high']
    energy = spectrum['energy']
    bass_punch = spectrum['bass_punch']
    brightness = spectrum['brightness']
    complexity = spectrum['complexity']
    
    energy_01 = np.clip(energy / 255.0, 0, 1)
    bass_01 = np.clip(bass / 255.0, 0, 1)
    mid_01 = np.clip(mid / 255.0, 0, 1)
    high_01 = np.clip(high / 255.0, 0, 1)
    
    # Histórico de beats
    _fire_beat_history_center.append(1 if beat_flag else 0)
    if len(_fire_beat_history_center) > 5:
        _fire_beat_history_center.pop(0)
    beat_strength = sum(_fire_beat_history_center) / len(_fire_beat_history_center)
    
    # === PARÂMETROS REATIVOS ===
    decay = 0.87 + 0.09 * (1 - energy_01)
    cooling = 0.012 + 0.028 * brightness - 0.018 * bass_punch
    cooling = np.clip(cooling, 0.005, 0.050)
    jitter = 1.5 + 9.0 * complexity + 5.0 * mid_01
    propagation_speed = 0.83 + 0.17 * bass_01
    
    # === INJEÇÃO NO CENTRO ===
    src = np.zeros(L, dtype=np.float32)
    c = ctx.CENTER
    
    # Centro reage mais intensamente a graves
    base_intensity = 100 + 220 * bass_01
    
    if beat_flag:
        _fire_explosion_center = min(1.0, _fire_explosion_center + 0.7 * (0.5 + 0.5 * energy_01))
    
    _fire_explosion_center *= 0.86
    
    multiplier = (1.0 + 2.5 * _fire_explosion_center) * (1.0 + 0.6 * beat_strength)
    injection = base_intensity * multiplier
    
    # Área de injeção no centro
    inject_width = int(4 + 10 * energy_01)
    start = max(0, c - inject_width)
    end = min(L, c + inject_width + 1)
    
    # Perfil gaussiano de injeção (mais intenso no centro)
    center_profile = np.exp(-0.5 * ((np.arange(start, end) - c) / (inject_width * 0.5)) ** 2)
    src[start:end] += injection * center_profile
    
    # === PROPAGAÇÃO ===
    h = _fire_heat_center_v4
    
    kernel_size = 3 if energy_01 < 0.3 else (5 if energy_01 < 0.6 else 7)
    if kernel_size == 3:
        kernel = np.array([0.25, 0.50, 0.25])
        pad_size = 1
    elif kernel_size == 5:
        kernel = np.array([0.1, 0.25, 0.3, 0.25, 0.1])
        pad_size = 2
    else:
        kernel = np.array([0.05, 0.15, 0.20, 0.20, 0.20, 0.15, 0.05])
        pad_size = 3
    
    padded = np.pad(h, (pad_size, pad_size), mode='edge')
    diffused = np.convolve(padded, kernel, mode='valid')
    adv = diffused * propagation_speed
    
    noise = (np.random.rand(L) - 0.5) * jitter
    
    h_new = adv * decay + src + noise
    h_new = np.clip(h_new * (1 - cooling), 0, 255)
    _fire_heat_center_v4 = h_new
    
    # === SPARKS ===
    spark_probability = 0.12 + 0.30 * high_01
    max_sparks = int(5 + 10 * energy_01)
    
    if (beat_flag or np.random.random() < spark_probability) and len(_fire_sparks_center) < max_sparks:
        num_new_sparks = 2 + int(4 * high_01)
        for _ in range(num_new_sparks):
            # Sparks emanam do centro
            offset = int(np.random.randn() * inject_width * 0.8)
            pos = np.clip(c + offset, 0, L - 1)
            intensity = 0.75 + 0.25 * np.random.random()
            velocity = int(np.sign(offset) * (1 + 2 * energy_01))
            _fire_sparks_center.append([pos, intensity, velocity])
    
    for sp in _fire_sparks_center[:]:
        sp[1] *= 0.80
        sp[0] += sp[2]
        
        if sp[1] < 0.05 or sp[0] < 0 or sp[0] >= L:
            _fire_sparks_center.remove(sp)
            continue
        
        i = int(sp[0])
        if 0 <= i < L:
            spark_heat = 110 * sp[1]
            h_new[i] = min(255, h_new[i] + spark_heat)
            if i > 0:
                h_new[i-1] = min(255, h_new[i-1] + spark_heat * 0.35)
            if i < L - 1:
                h_new[i+1] = min(255, h_new[i+1] + spark_heat * 0.35)
    
    # === RENDERIZAÇÃO ===
    v = np.clip(h_new, 0, 255).astype(np.uint8)
    v = ctx.apply_floor_vec(v, active, None)
    rgb = _apply_palette(ctx, v)
    ctx.to_pixels_and_show(rgb)