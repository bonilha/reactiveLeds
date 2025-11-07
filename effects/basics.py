# effects/basics.py
import numpy as np

# Estado para rainbow
_rainbow_wave_pos = 0

# ------------------------- efeitos existentes -------------------------
def effect_line_spectrum(ctx, bands_u8, beat_flag, active):
    arr = np.asarray(bands_u8, dtype=np.float32)
    v_raw = ctx.segment_mean_from_cumsum(arr, ctx.SEG_STARTS_FULL, ctx.SEG_ENDS_FULL)
    v = ctx.amplify_quad(v_raw.astype(np.uint16))
    v = ctx.apply_floor_vec(v, active, None)
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
    rgb[ctx.CENTER + ctx.I_LEFT] = left_rgb
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

# ------------------------- VU Meter (novo) -------------------------
# Estado do VU
_vu_level_ema = 0.0
_vu_peak_left = 0
_vu_peak_right = 0
_vu_weights_cache = {}  # por número de bandas

def _vu_weights(n):
    """Pesos (low > mid > high), cacheado por n."""
    w = _vu_weights_cache.get(n)
    if w is None:
        # Curva exponencial suave (mais peso nos graves)
        # evita custo alto normalizando na criação
        w = np.exp(np.linspace(np.log(1.6), np.log(0.9), n)).astype(np.float32)
        w /= np.sum(w) if np.sum(w) > 0 else 1.0
        _vu_weights_cache[n] = w
    return w

def effect_vu_meter(ctx, bands_u8, beat_flag, active):
    """
    Barra central espelhada + 'peak hold'.
    - ataque rápido, release lento
    - gradiente verde->vermelho conforme nível
    - usa fita inteira (duas metades espelhadas)
    """
    global _vu_level_ema, _vu_peak_left, _vu_peak_right

    half = ctx.CENTER
    if half <= 0:
        # fita muito curta? zera e sai
        ctx.to_pixels_and_show(np.zeros((ctx.LED_COUNT, 3), dtype=np.uint8))
        return

    # -------- nível de entrada (0..1) ponderado (low > mid > high) --------
    x = np.asarray(bands_u8, dtype=np.float32)
    if x.size == 0:
        level_in = 0.0
    else:
        w = _vu_weights(x.size)
        level_in = float(np.dot(x, w)) / 255.0
        # leve realce para baixos níveis (sensação de "acordado")
        level_in = min(1.0, level_in ** 0.8)

    # -------- envelope (ataque/release) --------
    ATTACK = 0.65   # sobe rápido
    RELEASE = 0.25  # desce devagar
    if level_in >= _vu_level_ema:
        _vu_level_ema = (1.0 - ATTACK) * _vu_level_ema + ATTACK * level_in
    else:
        _vu_level_ema = (1.0 - RELEASE) * _vu_level_ema + RELEASE * level_in

    # acelera um pouco em beats (dá “punch”)
    if beat_flag:
        _vu_level_ema = min(1.0, _vu_level_ema + 0.08)

    # -------- comprimento aceso (meia-fita) com curva não linear --------
    gamma = 0.75  # mais sensível em níveis baixos
    lit = int((max(0.0, min(1.0, _vu_level_ema)) ** gamma) * half)

    # -------- pico com hold/decay --------
    DECAY = 1  # LEDs por frame
    _vu_peak_left = max(_vu_peak_left - DECAY, lit)
    _vu_peak_right = max(_vu_peak_right - DECAY, lit)

    # -------- cor/valor --------
    # Gradiente de HUE por posição: 100 (verde) -> 0 (vermelho)
    if lit > 0:
        hue_line = np.linspace(100, 0, lit, dtype=np.float32)
        hue_u8 = ((ctx.base_hue_offset + (ctx.hue_seed >> 1) + hue_line) % 256).astype(np.uint8)
        # Valor cresce com o nível + pequeno flash em beat
        val_base = int(96 + 140 * _vu_level_ema + (24 if beat_flag else 0))
        val_u8 = np.full(lit, min(255, val_base), dtype=np.uint8)
        sat_u8 = np.full(lit, ctx.base_saturation, dtype=np.uint8)

        # Gera RGB para um lado e reutiliza espelhando
        rgb_half = ctx.hsv_to_rgb_bytes_vec(hue_u8, sat_u8, val_u8)

        rgb = np.zeros((ctx.LED_COUNT, 3), dtype=np.uint8)
        # índices das metades
        idx_l = ctx.CENTER - 1 - np.arange(lit, dtype=np.int32)
        idx_r = ctx.CENTER + np.arange(lit, dtype=np.int32)

        # escreve (lado esquerdo espelhado para manter gradiente "igual")
        rgb[idx_l] = rgb_half[::-1]
        rgb[idx_r] = rgb_half

        # marca pico (branco) em cada lado, se >0
        if _vu_peak_left > 0:
            p = ctx.CENTER - 1 - (_vu_peak_left - 1)
            if 0 <= p < ctx.LED_COUNT:
                rgb[p] = np.array([255, 255, 255], dtype=np.uint8)
        if _vu_peak_right > 0:
            p = ctx.CENTER + (_vu_peak_right - 1)
            if 0 <= p < ctx.LED_COUNT:
                rgb[p] = np.array([255, 255, 255], dtype=np.uint8)
    else:
        # nada aceso — só deixa os picos (se houver) para desaparecerem
        rgb = np.zeros((ctx.LED_COUNT, 3), dtype=np.uint8)
        if _vu_peak_left > 0:
            p = ctx.CENTER - 1 - (_vu_peak_left - 1)
            if 0 <= p < ctx.LED_COUNT:
                rgb[p] = np.array([200, 200, 200], dtype=np.uint8)
        if _vu_peak_right > 0:
            p = ctx.CENTER + (_vu_peak_right - 1)
            if 0 <= p < ctx.LED_COUNT:
                rgb[p] = np.array([200, 200, 200], dtype=np.uint8)

    # Se não estiver ativo, forçamos o envelope a cair mais rápido
    if not active:
        _vu_level_ema *= 0.80
        _vu_peak_left = max(0, _vu_peak_left - 2)
        _vu_peak_right = max(0, _vu_peak_right - 2)

    ctx.to_pixels_and_show(rgb)
    