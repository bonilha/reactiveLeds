# effects/basics.py
import time
import numpy as np

# ------------------------- Efeitos já aprovados -------------------------
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

# ------------------------- MultiBand Comets (novo) -------------------------
# Estado persistente do efeito (anexado ao ctx)
def _mbc_state(ctx):
    if not hasattr(ctx, "_mbc"):
        ctx._mbc = {
            "comets": [],         # cada item: [pos(float), vel(float), life(float 0..1), hue(uint8), sigma(float)]
            "last_t": time.time(),
            "idxs": np.arange(ctx.LED_COUNT, dtype=np.float32),
            "cooldown": {},       # por índice de banda -> último spawn ts
        }
    return ctx._mbc

def _band_pos_led(k, n, led_count):
    # posição central daquela banda ao longo da fita inteira
    return (k + 0.5) * (led_count / max(1, n))

def _spawn_comet(comets, pos, dir_right, level, hue, energy, base_speed):
    # velocidade e largura ajustadas pela energia e nível local
    v0 = base_speed * (0.7 + 0.6 * energy) * (0.75 + 0.5 * level)
    v0 = max(60.0, min(320.0, v0))
    sigma0 = 1.4 + 2.2 * energy
    life0 = 0.9 + 0.1 * level
    comets.append([pos, (+v0 if dir_right else -v0), life0, int(hue) & 255, sigma0])


# Alias amigável (caso queira referenciar por outro nome)
effect_multicolor_comets = effect_multiband_comets

# ------------------------- Energy Comets Melhorado (Distribuído) -------------------------
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
    Cometas distribuídos: nascem em posições de picos espectrais (esquerda=graves, direita=agudos)
    e correm para bordas com rastro suave. Brilho moderado, reativo ao espectro completo.
    """
    st = _get_comet_state(ctx)
    comets = st["comets"]

    # ---- tempo delta seguro ----
    now = time.time()
    dt = now - st["last_t"]
    st["last_t"] = now
    dt = min(0.05, max(0.0, dt))  # limita a 50 ms para estabilidade

    # ---- energia por banda (0..1) e detecção de picos ----
    n_bands = len(bands_u8)
    if n_bands == 0:
        return  # sem dados, pula
    x = np.asarray(bands_u8, dtype=np.float32) / 255.0  # normaliza 0..1
    energy_per_band = x  # usa diretamente para reatividade espectral

    # Encontra top 3 picos (threshold dinâmico baseado em média + std)
    mean_energy = float(np.mean(energy_per_band))
    std_energy = float(np.std(energy_per_band))
    thresh = max(0.2, mean_energy + 0.8 * std_energy)  # ajusta sensibilidade
    high_energy_bands = np.where(energy_per_band > thresh)[0]  # índices de bandas ativas
    if len(high_energy_bands) > 0:
        # Top 3 por energia (evita todos os frames spawnando)
        top_indices = np.argsort(energy_per_band[high_energy_bands])[-min(3, len(high_energy_bands)):]
        top_bands = high_energy_bands[top_indices]
    else:
        top_bands = []

    hue_base = int((ctx.base_hue_offset + (ctx.hue_seed >> 2)) % 256)

    # ---- nascimentos (distribuídos) ----
    max_comets = 10
    spawn_count = 0
    for band_idx in top_bands:
        if len(comets) + spawn_count >= max_comets:
            break
        # Posição mapeada: esquerda (graves) → centro → direita (agudos)
        pos = (band_idx / max(1.0, n_bands - 1.0)) * (ctx.LED_COUNT - 1)
        # Direção: para bordas (esquerda se pos < centro, direita se >)
        dir_left = pos < ctx.CENTER
        v0 = (80.0 + 140.0 * energy_per_band[band_idx]) * (1.0 + 0.3 * (1 if dir_left else -1))  # vel mais alta em agudos
        w0 = 1.2 + 2.0 * energy_per_band[band_idx]  # largura escala com energia
        hue = (hue_base + (band_idx * 3) % 256) % 256  # cor varia por freq (graves: +0, agudos: +alta)
        life0 = 1.0 if beat_flag else 0.85  # boost em beat
        comets.append([float(pos), -v0 if dir_left else +v0, life0, int(hue), w0])
        spawn_count += 1

    # Nascimento extra probabilístico em energia global alta (fallback se poucos picos)
    global_energy = mean_energy
    if (not beat_flag) and global_energy > 0.3 and len(comets) < max_comets and np.random.random() < (0.02 + 0.1 * global_energy):
        pos = np.random.uniform(0, ctx.LED_COUNT - 1)
        dir_right = np.random.random() < 0.5
        v0 = 90.0 + 150.0 * global_energy
        w0 = 1.5 + 1.5 * global_energy
        hue = (hue_base + np.random.randint(-10, 11)) % 256
        comets.append([pos, (+v0 if dir_right else -v0), 0.85, hue, w0])

    # ---- atualizar cometas ----
    friction = 0.85  # por segundo (mais suave para rastros longos)
    friction_dt = friction ** (dt * 40.0)
    decay = 0.87
    decay_dt = decay ** (dt * 40.0)

    for c in comets:
        c[0] += c[1] * dt                 # pos
        c[1] *= friction_dt               # vel (decai gradualmente)
        c[2] *= decay_dt                  # life
        c[4] *= (1.0 + 0.4 * dt)          # width cresce levemente

    # remover cometas fracos ou fora do range
    comets[:] = [c for c in comets if (0 <= c[0] < ctx.LED_COUNT) and (c[2] > 0.04)]

    # ---- renderização vetorizada ----
    rgb = np.zeros((ctx.LED_COUNT, 3), dtype=np.uint8)
    if comets:
        idxs = np.arange(ctx.LED_COUNT, dtype=np.float32)
        base_sat = min(220, ctx.base_saturation)
        for pos, vel, life, hue, width in comets:
            # Gaussiano centrado — rastro suave
            sigma = max(0.7, width)
            dist = np.abs(idxs - float(pos))
            glow = np.exp(-(dist * dist) / (2.0 * sigma * sigma)).astype(np.float32)

            # brilho: escala com life/energia, cap moderado (respeita power no RPi)
            val = np.clip(200.0 * life * (0.4 + 0.6 * global_energy) * glow, 0, 190).astype(np.uint8)
            if not active:
                val = (val.astype(np.float32) * 0.8).astype(np.uint8)  # decai em idle

            sat = np.full_like(val, base_sat, dtype=np.uint8)
            hue_arr = np.full_like(val, int(hue) % 256, dtype=np.uint8)

            # Compõe por 'max' (aditivo limitado, evita branco excessivo)
            rgb = np.maximum(rgb, ctx.hsv_to_rgb_bytes_vec(hue_arr, sat, val))

    ctx.to_pixels_and_show(rgb)
