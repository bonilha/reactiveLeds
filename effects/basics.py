# effects/basics.py
import time
import numpy as np

# ------------------------- efeitos existentes -------------------------
_rainbow_wave_pos = 0

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

# ------------------------- Energy Comets (novo) -------------------------
# Estado persistente do efeito (anexado ao ctx)
def _get_comet_state(ctx):
    if not hasattr(ctx, "_comets_state"):
        ctx._comets_state = {
            "comets": [],   # lista de [pos(float), vel(float), life(float 0..1), hue(uint8), width(float)]
            "last_t": time.time()
        }
    return ctx._comets_state

def effect_energy_comets(ctx, bands_u8, beat_flag, active):
    """
    Cometas que nascem no centro (nos beats) e correm para as bordas
    com rastro suave. Brilho moderado (respeita brightness/power-cap).
    """
    st = _get_comet_state(ctx)
    comets = st["comets"]

    # ---- tempo delta seguro ----
    now = time.time()
    dt = now - st["last_t"]
    st["last_t"] = now
    dt = min(0.05, max(0.0, dt))  # limita a 50 ms para estabilidade

    # ---- energia média (0..1) e cor base ----
    x = np.asarray(bands_u8, dtype=np.float32)
    energy = float(np.mean(x) / 255.0) if x.size else 0.0
    energy = min(1.0, energy ** 0.9)  # ligeiro realce em níveis baixos

    hue_base = int((ctx.base_hue_offset + (ctx.hue_seed >> 2)) % 256)

    # ---- nascimentos ----
    max_comets = 10
    # Sempre que houver beat, nascem dois cometas (centro, direções opostas)
    if beat_flag and len(comets) <= max_comets - 2:
        v0 = 120.0 + 180.0 * energy   # velocidade inicial em LEDs/seg
        w0 = 1.6 + 1.8 * energy       # largura inicial do glow
        comets.append([ctx.CENTER - 1, -v0, 1.0, hue_base, w0])  # esquerda
        comets.append([ctx.CENTER,      +v0, 1.0, hue_base, w0]) # direita
    # Sem beat, nascimento probabilístico quando energia alta
    if (not beat_flag) and energy > 0.35 and len(comets) < max_comets:
        # chance cresce com a energia
        if np.random.random() < (0.03 + 0.12 * (energy - 0.35)):
            v0 = 100.0 + 160.0 * energy
            w0 = 1.6 + 1.6 * energy
            dir_right = (np.random.random() < 0.5)
            comets.append([ctx.CENTER if dir_right else ctx.CENTER - 1,
                           (+v0 if dir_right else -v0),
                           0.9, (hue_base + np.random.randint(-8, 9)) % 256, w0])

    # ---- atualizar cometas ----
    # atrito/fricção e decaimento de vida dependentes de dt
    friction = 0.86  # por segundo aproximado
    friction_dt = friction ** (dt * 40.0)  # aproximação (ajuste fino)
    decay = 0.88
    decay_dt = decay ** (dt * 40.0)

    for c in comets:
        c[0] += c[1] * dt                 # pos
        c[1] *= friction_dt               # vel
        c[2] *= decay_dt                  # life
        c[4] *= (1.0 + 0.35 * dt)         # width aumenta um pouco com o tempo

    # remover cometas fracos ou fora do range
    comets[:] = [c for c in comets if (0 <= c[0] < ctx.LED_COUNT) and (c[2] > 0.05)]

    # ---- renderização vetorizada ----
    rgb = np.zeros((ctx.LED_COUNT, 3), dtype=np.uint8)
    if comets:
        idxs = np.arange(ctx.LED_COUNT, dtype=np.float32)
        base_sat = min(230, ctx.base_saturation)
        for pos, vel, life, hue, width in comets:
            # Gaussiano centrado no cometa — rastro suave
            # sigma proporcional à largura acumulada
            sigma = max(0.8, width)
            dist = np.abs(idxs - float(pos))
            glow = np.exp(-(dist * dist) / (2.0 * sigma * sigma)).astype(np.float32)

            # brilho moderado, escalado pela energia e "vida"
            # teto reduzido para não estourar cap/brightness
            val = np.clip(220.0 * life * (0.35 + 0.65 * energy) * glow, 0, 200).astype(np.uint8)
            if not active:
                # cai rápido em idle
                val = (val.astype(np.float32) * 0.85).astype(np.uint8)

            sat = np.full_like(val, base_sat, dtype=np.uint8)
            hue_arr = np.full_like(val, int(hue) % 256, dtype=np.uint8)

            # compõe por 'max' (aditivo limitado) — evita saturar branco fácil
            rgb = np.maximum(rgb, ctx.hsv_to_rgb_bytes_vec(hue_arr, sat, val))

    ctx.to_pixels_and_show(rgb)
    