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
        idx = (np.power(v / 255.0, 0.7) * (m - 1)).astype(int) % m  # Ajustado gamma para melhor distribuição
        return pal_arr[idx]
    else:
        # Fallback para paleta de fogo clássica: preto -> vermelho -> laranja -> amarelo -> branco
        fire_pal = np.array([
            [0, 0, 0],       # 0: preto
            [64, 0, 0],      # baixo: vermelho escuro
            [255, 0, 0],     # vermelho
            [255, 64, 0],    # laranja escuro
            [255, 128, 0],   # laranja
            [255, 200, 0],   # amarelo-laranja
            [255, 255, 0],   # amarelo
            [255, 255, 128], # amarelo claro
            [255, 255, 255]  # branco (alto calor)
        ], dtype=np.uint8)
        m = len(fire_pal)
        idx = (np.power(v / 255.0, 0.75) * (m - 1)).astype(int) % m  # Gamma ajustado para mais vermelho/laranja
        return fire_pal[idx]

def effect_clean_fire_edge_v4(ctx, bands_u8, beat_flag, active):
    global _fire_heat_edge_v4, _fire_sparks_edge
    L = ctx.LED_COUNT
    if _fire_heat_edge_v4 is None or _fire_heat_edge_v4.shape[0] != L:
        _fire_heat_edge_v4 = np.zeros(L, dtype=np.float32)
        _fire_sparks_edge = []

    if len(bands_u8) == 0 or not active:
        _fire_heat_edge_v4 *= 0.80  # Decay mais rápido quando inativo
        if np.max(_fire_heat_edge_v4) < 1.0:  # Apaga completamente se baixo
            ctx.to_pixels_and_show(np.zeros((L, 3), dtype=np.uint8))
        return

    # Dinâmica baseada em graves e médios (mais reativo)
    low = np.mean(bands_u8[:max(8, len(bands_u8)//8)])
    mid = np.mean(bands_u8[len(bands_u8)//8:len(bands_u8)//4])
    env = 0.6 * low + 0.4 * mid  # Mais peso nos graves para "chamas" fortes
    e01 = np.clip(env / 255, 0, 1)

    # Parâmetros ajustados para mais variação e reatividade
    decay = 0.92 + 0.03 * (1 - e01)  # Decay variável: mais lento em alta energia
    cooling = 0.015 + 0.025 * (1 - e01)  # Menos cooling em alta energia
    jitter = 2.0 + 6.0 * e01  # Mais ruído em alta energia para "chamas dançantes"

    # Injeta calor nas extremidades (mais forte em beat)
    src = np.zeros(L, dtype=np.float32)
    inj = (100 + 280 * e01) * (1.6 if beat_flag else 1.0)  # Boost maior no beat
    src[0:5] += inj  # Injeta mais pixels para chamas mais visíveis
    src[-5:] += inj

    # Propagação com kernel mais suave para chamas "ascendentes"
    h = _fire_heat_edge_v4
    padded = np.pad(h, (2, 2), mode='edge')  # Padding maior para propagação
    kernel = np.array([0.1, 0.3, 0.5, 0.3, 0.1])  # Kernel mais largo para suavidade
    adv = np.convolve(padded, kernel, mode='valid')

    noise = (np.random.rand(L) - 0.5) * jitter
    h_new = adv * decay + src + noise
    h_new = np.clip(h_new * (1 - cooling), 0, 255)
    _fire_heat_edge_v4 = h_new

    # Sparks discretos (mais frequentes e variados)
    if (beat_flag or np.random.random() < 0.15 * e01) and len(_fire_sparks_edge) < 6:
        for _ in range(1 + int(2 * e01)):  # Mais sparks em alta energia
            pos = np.random.choice([0, L-1])
            intensity = 0.8 + 0.2 * np.random.random()
            _fire_sparks_edge.append([pos, intensity])

    for sp in _fire_sparks_edge[:]:
        sp[1] *= 0.85  # Decay de sparks ajustado
        if sp[1] < 0.08:
            _fire_sparks_edge.remove(sp)
            continue
        i = sp[0]
        if 0 <= i < L:
            h_new[i] = min(255, h_new[i] + 100 * sp[1])  # Sparks mais brilhantes

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

    if len(bands_u8) == 0 or not active:
        _fire_heat_center_v4 *= 0.80  # Decay mais rápido quando inativo
        if np.max(_fire_heat_center_v4) < 1.0:  # Apaga completamente se baixo
            ctx.to_pixels_and_show(np.zeros((L, 3), dtype=np.uint8))
        return

    low = np.mean(bands_u8[:max(8, len(bands_u8)//8)])
    mid = np.mean(bands_u8[len(bands_u8)//8:len(bands_u8)//4])
    env = 0.6 * low + 0.4 * mid  # Mais peso nos graves
    e01 = np.clip(env / 255, 0, 1)

    decay = 0.91 + 0.03 * (1 - e01)  # Decay variável
    cooling = 0.015 + 0.025 * (1 - e01)
    jitter = 3.0 + 7.0 * e01  # Mais jitter para movimento

    src = np.zeros(L, dtype=np.float32)
    c = ctx.CENTER
    inj = (120 + 280 * e01) * (1.6 if beat_flag else 1.0)  # Boost no beat
    src[c-4:c+5] += inj  # Injeta em área maior no centro

    h = _fire_heat_center_v4
    padded = np.pad(h, (2, 2), mode='edge')
    kernel = np.array([0.1, 0.3, 0.5, 0.3, 0.1])  # Kernel suave
    adv = np.convolve(padded, kernel, mode='valid')

    noise = (np.random.rand(L) - 0.5) * jitter
    h_new = adv * decay + src + noise
    h_new = np.clip(h_new * (1 - cooling), 0, 255)
    _fire_heat_center_v4 = h_new

    if (beat_flag or np.random.random() < 0.2 * e01) and len(_fire_sparks_center) < 7:
        for _ in range(2 + int(2 * e01)):
            pos = c + np.random.randint(-8, 9)
            intensity = 0.8 + 0.2 * np.random.random()
            _fire_sparks_center.append([pos, intensity])

    for sp in _fire_sparks_center[:]:
        sp[1] *= 0.85
        if sp[1] < 0.08:
            _fire_sparks_center.remove(sp)
            continue
        i = sp[0]
        if 0 <= i < L:
            h_new[i] = min(255, h_new[i] + 90 * sp[1])

    v = np.clip(h_new, 0, 255).astype(np.uint8)
    v = ctx.apply_floor_vec(v, active, None)
    rgb = _apply_palette(ctx, v)
    ctx.to_pixels_and_show(rgb)