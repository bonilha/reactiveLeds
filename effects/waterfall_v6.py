# effects/waterfall_v6.py
import numpy as np
import time

# ----------------------------------------------------------------------
# Waterfall v6 — “Chromatic Cascade” (criativo, impactante e paleta-aware)
# ----------------------------------------------------------------------
# Características principais:
# 1. **Fluxo descendente com “gotas” individuais** – cada banda gera uma
#    partícula que cai com velocidade proporcional à sua energia.
# 2. **Trail gaussiano suave + brilho de borda** – evita flickering e dá
#    sensação de líquido luminoso.
# 3. **Interpolação de paleta ao longo da fita** – usa a paleta automática
#    (`ctx.current_palette`) para colorir cada gota; fallback HSV mantém
#    compatibilidade.
# 4. **Efeito “splash” no fundo** – quando a gota atinge o final, gera
#    ondas radiais curtas e coloridas.
# 5. **Decay adaptativo por saturação** – regiões mais saturadas decaem
#    mais rápido, evitando acúmulo de brilho.
# 6. **Limiter local + global** – garante consumo controlado e headroom.
# 7. **EMA leve no buffer final** – suaviza transições entre frames.
# ----------------------------------------------------------------------
_waterfall_state = None

def _init_state(ctx):
    """Inicializa ou reinicializa o estado persistente."""
    global _waterfall_state
    L = ctx.LED_COUNT
    _waterfall_state = {
        "buf": np.zeros((L, 3), dtype=np.float32),   # buffer de saída (float)
        "drops": [],                                 # lista de gotas ativas
        "last_t": time.time(),
        "ema": np.zeros((L, 3), dtype=np.float32),   # EMA de saída
        "pal_arr": None,                             # cache da paleta
        "pal_len": 0,
    }

def effect_chromatic_cascade(ctx, bands_u8, beat_flag, active):
    """
    Waterfall v6 – Chromatic Cascade
    Parâmetros:
        ctx        – FXContext (acesso a paleta, floor, etc.)
        bands_u8   – np.ndarray uint8 de bandas de áudio
        beat_flag  – bool (kick detectado)
        active     – bool (sinal ainda dentro do hold)
    """
    global _waterfall_state
    if _waterfall_state is None or _waterfall_state["buf"].shape[0] != ctx.LED_COUNT:
        _init_state(ctx)

    st = _waterfall_state
    L = ctx.LED_COUNT
    now = time.time()
    dt = min(now - st["last_t"], 0.05)
    st["last_t"] = now

    # -------------------------------------------------- 1) Paleta --------------------------------------------------
    pal = getattr(ctx, "current_palette", None)
    if isinstance(pal, (list, tuple)) and len(pal) >= 2:
        pal_arr = np.asarray(pal, dtype=np.uint8)
        if st["pal_arr"] is None or not np.array_equal(st["pal_arr"], pal_arr):
            st["pal_arr"] = pal_arr.copy()
            st["pal_len"] = pal_arr.shape[0]
    else:
        st["pal_arr"] = None
        st["pal_len"] = 0

    # -------------------------------------------------- 2) Spawn de gotas -------------------------------------------
    n_bands = len(bands_u8)
    if n_bands > 0 and active:
        # Energia média das 8 primeiras bandas (graves) → probabilidade de spawn
        low_energy = float(np.mean(bands_u8[:min(8, n_bands)]))
        spawn_prob = np.clip(low_energy / 255.0, 0.0, 1.0) ** 1.5

        # Spawn máximo por frame (evita explosão)
        max_spawns = 3
        if beat_flag:
            max_spawns += 2
            spawn_prob = min(1.0, spawn_prob * 1.4)

        # Distribuição uniforme ao longo da fita (band → LED)
        band_to_led = np.linspace(0, L - 1, n_bands, dtype=np.int32)

        for i in range(n_bands):
            if len(st["drops"]) >= 48:               # limite global de gotas
                break
            if np.random.random() < spawn_prob / max(1, n_bands // max_spawns):
                led = band_to_led[i]
                intensity = bands_u8[i]

                # Velocidade proporcional à intensidade (0-255 → 30-180 px/s)
                speed = 30.0 + 150.0 * (intensity / 255.0)
                if beat_flag:
                    speed *= 1.25

                # Cor da gota (paleta ou HSV)
                if st["pal_arr"] is not None:
                    p = (i / max(1, n_bands - 1)) * (st["pal_len"] - 1)
                    i0 = int(p)
                    i1 = (i0 + 1) % st["pal_len"]
                    t = p - i0
                    rgb = (st["pal_arr"][i0].astype(np.float32) * (1.0 - t) +
                           st["pal_arr"][i1].astype(np.float32) * t)
                else:
                    hue = (ctx.base_hue_offset + (i * 200 // max(1, n_bands)) +
                           (ctx.hue_seed >> 2)) % 256
                    rgb = ctx.hsv_to_rgb_bytes_vec(
                        np.array([hue], dtype=np.uint8),
                        np.array([min(220, ctx.base_saturation)], dtype=np.uint8),
                        np.array([255], dtype=np.uint8)
                    )[0].astype(np.float32)

                st["drops"].append({
                    "pos": float(led),
                    "speed": speed,
                    "rgb": rgb,
                    "life": 1.0,
                    "size": 2.0 + 4.0 * (intensity / 255.0)   # espessura gaussiana
                })

    # -------------------------------------------------- 3) Atualiza gotas --------------------------------------------
    new_drops = []
    for d in st["drops"]:
        d["pos"] += d["speed"] * dt
        d["life"] *= 0.96 ** (dt * 50.0)          # decaimento rápido
        d["size"] *= 1.0 + 0.12 * dt             # leve expansão

        if d["pos"] < L + 20 and d["life"] > 0.03:
            new_drops.append(d)
        else:
            # Splash no fundo
            if d["pos"] >= L - 5:
                splash_center = min(L - 1, int(d["pos"]))
                splash_rgb = d["rgb"] * (d["life"] * 1.3)
                splash_radius = 8.0 + 12.0 * d["life"]
                splash_life = 0.6
                new_drops.append({
                    "pos": float(splash_center),
                    "speed": 0.0,
                    "rgb": splash_rgb,
                    "life": splash_life,
                    "size": splash_radius,
                    "is_splash": True
                })
    st["drops"] = new_drops

    # -------------------------------------------------- 4) Render (acumula contribuições) ---------------------------
    acc = np.zeros((L, 3), dtype=np.float32)
    idx = np.arange(L, dtype=np.float32)

    for d in st["drops"]:
        dist = np.abs(idx - d["pos"])
        if "is_splash" in d:
            # Splash radial (gaussiano)
            sigma = d["size"] / 2.0
            contrib = np.exp(-0.5 * (dist / sigma) ** 2)
            contrib *= 180.0 * d["life"]
        else:
            # Gota com borda brilhante
            sigma = d["size"]
            core = np.exp(-0.5 * (dist / sigma) ** 2)
            edge = np.exp(-2.0 * (dist / sigma) ** 2)
            contrib = core * 140.0 + edge * 60.0
            contrib *= d["life"]

        contrib = np.clip(contrib, 0.0, 255.0)
        acc += contrib[:, None] * (d["rgb"] / 255.0)

    # Decay adaptativo por saturação
    sat = ctx.base_saturation / 255.0
    decay = 0.62 - 0.18 * sat
    st["buf"] *= np.clip(decay, 0.45, 0.92)

    # Mistura nova contribuição
    st["buf"] = np.clip(st["buf"] + acc, 0.0, 255.0)

    # -------------------------------------------------- 5) Limiter local (janelas deslizantes) -----------------------
    win = max(1, L // 12)
    step = max(1, int(win * 0.65))
    i = 0
    while i < L:
        s = i
        e = min(L, i + win)
        region = st["buf"][s:e]
        scale = 0.38 + 0.58 * (i / max(1, L - 1)) ** 0.75
        region *= scale
        i += step

    # -------------------------------------------------- 6) Piso dinâmico + clamp global -----------------------------
    floor = float(getattr(ctx, "dynamic_floor", 0))
    if floor > 0:
        st["buf"] = np.maximum(st["buf"], floor)

    # Clamp por pixel (evita branco puro)
    per_pix = np.sum(st["buf"], axis=1)
    mask = per_pix > 680
    if np.any(mask):
        scale = 680.0 / per_pix[mask]
        st["buf"][mask] *= scale[:, None]

    # -------------------------------------------------- 7) EMA final (suaviza flicker) -------------------------------
    beta = 0.38 if active else 0.28
    st["ema"] = st["ema"] * (1.0 - beta) + st["buf"] * beta
    out = np.clip(st["ema"], 0.0, 255.0).astype(np.uint8)

    # -------------------------------------------------- 8) Envio ----------------------------------------------------
    ctx.to_pixels_and_show(out)