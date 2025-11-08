# -- Bass Center Bloom (robusto, usa paleta, anti-overflow) --
_bass_radius_ema = 0.0
_beat_flash = 0.0

def effect_bass_center_bloom(ctx, bands_u8, beat_flag, active):
    """
    Bloom central reativo aos graves.
    Implementação segura (float32) com caps em todas as etapas para evitar OverflowError,
    usando a fita toda, e com colorização por paleta (se ctx.current_palette existir).
    """
    import numpy as np, time

    global _bass_radius_ema, _beat_flash

    # --- estado de flash no beat, limitado a [0,1] ---
    if beat_flag:
        _beat_flash = 1.0
    else:
        _beat_flash *= 0.85
    _beat_flash = float(np.clip(_beat_flash, 0.0, 1.0))

    # --- guarda para pacotes vazios ---
    n = len(bands_u8)
    if n == 0 or ctx.LED_COUNT <= 0:
        ctx.to_pixels_and_show(np.zeros((ctx.LED_COUNT, 3), dtype=np.uint8))
        return

    # --- energia de graves -> raio alvo ---
    low_n = max(8, n // 8)
    low = np.asarray(bands_u8[:low_n], dtype=np.float32)
    low_mean = float(np.mean(low))  # 0..255

    # raio alvo (0..CENTER*0.95) e suavização por EMA
    target_radius = float(np.clip((low_mean / 255.0) * (ctx.CENTER * 0.95), 0.0, float(ctx.CENTER)))
    _bass_radius_ema = 0.80 * float(_bass_radius_ema) + 0.20 * target_radius
    radius = float(np.clip(_bass_radius_ema, 0.0, float(ctx.CENTER)))

    # --- perfil radial: gaussiano centrado + triângulo para “corpo” ---
    idx = ctx.I_ALL.astype(np.float32)
    d = np.abs(idx - float(ctx.CENTER))  # distância ao centro, float32

    # corpo: triângulo normalizado 0..1 com base ~2*radius (evita divisão por zero)
    base_w = max(1.0, radius)
    body = np.clip(1.0 - (d / base_w), 0.0, 1.0).astype(np.float32)

    # halo: gaussiano suave que cresce com o raio/energia
    sigma = 0.35 * max(6.0, radius)          # largura cresce com o “tamanho” do bass
    halo = np.exp(-0.5 * (d / max(1e-3, sigma))**2).astype(np.float32)

    # mistura corpo + halo (peso muda com energia para estabilidade)
    e01 = float(np.clip(low_mean / 255.0, 0.0, 1.0))
    shape = np.clip(0.65 * body + 0.35 * halo, 0.0, 1.0).astype(np.float32)

    # ganho do efeito (cap suave): 80..220 + pulso no beat
    base_amp = 80.0 + 140.0 * e01
    amp = float(np.clip(base_amp * (1.0 + 0.25 * _beat_flash), 0.0, 235.0))

    # brilho final por LED (float32 0..255), com cap antecipado
    v_f = np.clip(shape * amp, 0.0, 235.0).astype(np.float32)

    # aplica piso dinâmico (canal-agnóstico)
    floor = float(getattr(ctx, "dynamic_floor", 0))
    if floor > 0:
        v_f = np.maximum(v_f, floor).astype(np.float32)

    # ---------- COR ----------
    # 1) Preferir paleta, 2) fallback para HSV com variação temporal e espacial
    pal = getattr(ctx, "current_palette", None)
    use_pal = isinstance(pal, (list, tuple)) and len(pal) >= 2

    if use_pal:
        pal_arr = np.asarray(pal, dtype=np.uint8)
        m = pal_arr.shape[0]

        # varrer paleta ao longo do strip + leve animação no tempo
        t = time.time()
        phase = (t * 0.08) * m
        pos = ((idx / max(1.0, float(ctx.LED_COUNT))) * m + phase) % m

        i0 = np.floor(pos).astype(np.int32) % m
        i1 = (i0 + 1) % m
        frac = (pos - np.floor(pos)).astype(np.float32)[:, None]

        c0 = pal_arr[i0].astype(np.float32)
        c1 = pal_arr[i1].astype(np.float32)
        base_rgb = (c0 * (1.0 - frac) + c1 * frac)        # 0..255 float32

        # respeitar saturação base: mistura com cinza do próprio V
        sat_base = int(np.clip(getattr(ctx, "base_saturation", 220), 0, 255))
        s = sat_base / 255.0
        v_col = (v_f[:, None] / 255.0) * base_rgb         # escala por V
        gray  = v_f[:, None]                              # mesmo V nos 3 canais
        out   = gray * (1.0 - s) + v_col * s
        rgb   = np.clip(out, 0, 255).astype(np.uint8)

    else:
        # HSV: hue varia no tempo e no espaço para não ficar monocromático
        t = time.time()
        h_time = int((t * 35.0) % 256)
        h_spatial = ((ctx.I_ALL.astype(np.int32) * 256) // max(1, ctx.LED_COUNT))
        hue = (int(ctx.base_hue_offset) + (int(ctx.hue_seed) >> 2) + h_time + h_spatial) % 256

        sat = np.full(ctx.LED_COUNT, int(np.clip(getattr(ctx, "base_saturation", 220), 0, 255)),
                      dtype=np.uint8)
        v_u8 = np.clip(v_f, 0.0, 255.0).astype(np.uint8)
        rgb = ctx.hsv_to_rgb_bytes_vec(hue.astype(np.uint8), sat, v_u8)

    # ---------- Envio (com leve cap final anti-clipping) ----------
    rgb = np.clip(rgb, 0, 250).astype(np.uint8)
    ctx.to_pixels_and_show(rgb)


# Alias para compatibilidade com registradores que esperam "effect_bass_center"
def effect_bass_center(ctx, bands_u8, beat_flag, active):
    return effect_bass_center_bloom(ctx, bands_u8, beat_flag, active)