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


def effect_multiband_comets(ctx, bands_u8, beat_flag, active):
    """
    MultiBand Comets — v2 (full-strip)
    - Spawns dirigidos pelas bandas mais fortes + "cobertura por zonas" para ocupar toda a fita.
    - Vida/fricção ajustados para permitir percursos longos (cruzar o strip).
    - Direções alternadas para distribuir o fluxo.
    - Se ctx.current_palette existir: usa as cores da paleta por cometa; senão, HSV por banda.

    Estrutura dos cometas (mantida):
        [pos(float), vel(float), life(0..1), hue(uint8), sigma(float)]
    """
    st = _mbc_state(ctx)
    comets = st["comets"]
    idxs = st["idxs"]
    cooldown = st["cooldown"]

    # --- dt estável ---
    now = time.time()
    dt = now - st["last_t"]
    st["last_t"] = now
    dt = min(0.05, max(0.0, dt))

    # --- normalização e estatística das bandas ---
    x = np.asarray(bands_u8, dtype=np.float32)
    n = int(x.size) if x.size > 0 else 1
    if n == 0:
        x = np.zeros(1, dtype=np.float32); n = 1
    x01 = (x / 255.0).clip(0.0, 1.0)
    mu = float(np.mean(x01))
    sd = float(np.std(x01))
    thr = mu + 0.75 * sd
    energy = min(1.0, mu ** 0.85)

    # --- seleção de bandas candidatas (picos) ---
    mask = x01 >= thr
    cand_idx = np.nonzero(mask)[0]
    if cand_idx.size == 0:
        K = max(3, n // 12)  # top-K fallback
        cand_idx = np.argsort(x01)[-K:]

    # --- parâmetros (ajustados para strip longo) ---
    MAX_COMETS = 24
    BASE_SPEED = 160.0  # um pouco maior que antes
    COOLDOWN_S = 0.10
    base_hue = (ctx.base_hue_offset + (ctx.hue_seed >> 3)) & 255
    max_to_spawn = 8 if beat_flag else 4

    # --- spawns por picos (prioriza os mais fortes) ---
    spawned = 0
    flip_dir_seed = (int(now * 10) ^ ctx.hue_seed) & 1

    for rank_k in np.flip(cand_idx):
        if len(comets) >= MAX_COMETS or spawned >= max_to_spawn:
            break
        last_k = cooldown.get(int(rank_k), 0.0)
        if (now - last_k) < COOLDOWN_S:
            continue

        level = float(x01[rank_k])
        pos = _band_pos_led(rank_k, n, ctx.LED_COUNT)

        # Direção alternada para espalhar (em vez de sempre "lado oposto")
        dir_right = ((rank_k + flip_dir_seed) % 2) == 0

        # Hue baseado na banda (fallback HSV) — se houver paleta, vamos usá-la no render
        hue = (base_hue + int(255.0 * (rank_k / max(1, n - 1)))) & 255

        _spawn_comet(comets, pos, dir_right, level, hue, energy, base_speed=BASE_SPEED)
        cooldown[int(rank_k)] = now
        spawned += 1

    # --- cobertura por zonas: garante cometas “espalhados” pela fita inteira ---
    ZONES = max(6, ctx.LED_COUNT // 60)  # 300 LEDs => 6+ zonas
    seg = ctx.LED_COUNT / float(ZONES)
    if len(comets) < (ZONES // 2):  # só se estiver ralo
        # posições atuais para checar lacunas
        if comets:
            cur_pos = np.array([c[0] for c in comets], dtype=np.float32)
        else:
            cur_pos = np.array([], dtype=np.float32)

        for z in range(ZONES):
            if len(comets) >= MAX_COMETS or spawned >= max_to_spawn + 2:
                break
            z0 = z * seg
            z1 = (z + 1) * seg
            zc = 0.5 * (z0 + z1)

            # se não houver cometa na zona, nasce um “coverage comet”
            if cur_pos.size == 0 or np.all(np.abs(cur_pos - zc) > (0.35 * seg)):
                dir_right = (z % 2 == 0)
                # energia baixa/moderada -> evita estourar consumo
                level = 0.35 + 0.45 * energy
                hue = (base_hue + int(12 * z)) & 255  # varia levemente por zona
                _spawn_comet(comets, zc, dir_right, level, hue, energy, base_speed=BASE_SPEED * (0.9 + 0.2 * energy))
                spawned += 1

    # --- atualização dos cometas (mais vida para cruzar a fita) ---
    # fricção/decay suavizados para percursos longos; sigma cresce lento
    friction = 0.93
    decay = 0.94
    fr_dt = friction ** (dt * 40.0)
    dc_dt = decay ** (dt * 40.0)

    # wrap-around opcional para sempre “usar a fita toda”
    WRAP_AROUND = True

    for c in comets:
        c[0] += c[1] * dt     # pos
        c[1] *= fr_dt         # vel
        c[2] *= dc_dt         # life
        c[4] *= (1.0 + 0.28 * dt)  # sigma (rastro) abre suave

        if WRAP_AROUND:
            # volta pelo outro lado; mantém life/vel — cria fluxo contínuo
            if c[0] < 0.0:
                c[0] += ctx.LED_COUNT
            elif c[0] >= ctx.LED_COUNT:
                c[0] -= ctx.LED_COUNT

    # remove apenas quando "fracos" (com wrap, não usamos corte por borda)
    comets[:] = [c for c in comets if c[2] > 0.04]

    # --- render vetorizado (com paleta opcional) ---
    rgb = np.zeros((ctx.LED_COUNT, 3), dtype=np.uint8)
    if comets:
        base_sat = min(235, ctx.base_saturation)
        val_cap = 220
        pal = getattr(ctx, "current_palette", None)
        use_palette = isinstance(pal, (list, tuple)) and len(pal) >= 2
        pal_arr = None
        if use_palette:
            pal_arr = np.asarray(pal, dtype=np.uint8)
            m = pal_arr.shape[0]

        for pos, vel, life, hue, sigma in comets:
            dist = np.abs(idxs - float(pos))
            # se wrap ativo, o "outro lado" pode estar mais perto
            if WRAP_AROUND:
                dist = np.minimum(dist, ctx.LED_COUNT - dist)

            sig = max(0.9, sigma)
            glow = np.exp(-(dist * dist) / (2.0 * sig * sig)).astype(np.float32)

            # intensidade do rastro
            val = np.clip(val_cap * life * (0.40 + 0.60 * energy) * glow, 0, val_cap).astype(np.uint8)
            if not active:
                val = (val.astype(np.float32) * 0.85).astype(np.uint8)

            if use_palette:
                # cor da paleta em função da posição (mapeia 0..LED_COUNT -> 0..m)
                pos_norm = (pos / max(1.0, float(ctx.LED_COUNT))) * m
                i0 = int(np.floor(pos_norm)) % m
                i1 = (i0 + 1) % m
                t = float(pos_norm - np.floor(pos_norm))
                rgb_peak = (pal_arr[i0].astype(np.float32) * (1.0 - t) + pal_arr[i1].astype(np.float32) * t)
                rgb_peak = np.clip(rgb_peak, 0, 255).astype(np.uint8)
            else:
                # fallback HSV (mesma lógica anterior)
                rgb_peak = ctx.hsv_to_rgb_bytes_vec(
                    np.array([hue], dtype=np.uint8),
                    np.array([base_sat], dtype=np.uint8),
                    np.array([255], dtype=np.uint8)
                )[0]

            # compõe por canal com 'max' (estável, evita branco/clip)
            for ch in range(3):
                comp = (val.astype(np.uint16) * int(rgb_peak[ch]) // 255).astype(np.uint8)
                rgb[:, ch] = np.maximum(rgb[:, ch], comp)

    ctx.to_pixels_and_show(rgb)


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
