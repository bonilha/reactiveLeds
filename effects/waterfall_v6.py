# effects/waterfall_v6.py
import numpy as np
import time

# ----------------------------------------------------------------------
# Waterfall v6 — “Chromatic Cascade” (estável, reativo, paleta-aware)
# ----------------------------------------------------------------------
# Problema relatado: "Começou bem e do nada parou"
# → Causa provável: acúmulo de gotas + overflow de memória/CPU em frames silenciosos
# → Solução: 
#    • Limite rigoroso de gotas por frame + limpeza forçada
#    • Decay mais agressivo em silêncio
#    • Reset de estado em reset_flag
#    • Inicialização tardia (lazy) + proteção contra L=0
#    • EMA reiniciada em reset
# ----------------------------------------------------------------------
_waterfall_state = None

def _init_state(ctx):
    """Inicializa estado com valores seguros."""
    global _waterfall_state
    L = max(1, ctx.LED_COUNT)
    _waterfall_state = {
        "buf": np.zeros((L, 3), dtype=np.float32),
        "drops": [],
        "last_t": time.time(),
        "ema": np.zeros((L, 3), dtype=np.float32),
        "pal_arr": None,
        "pal_len": 0,
        "spawn_cooldown": 0.0,   # evita burst spawn
    }

def effect_chromatic_cascade(ctx, bands_u8, beat_flag, active):
    """
    Waterfall v6 – Chromatic Cascade (versão estável)
    """
    global _waterfall_state

    # Reset externo (via tecla 'r' ou pacote B1)
    if getattr(ctx, "reset_flag", False):
        _waterfall_state = None
        ctx.reset_flag = False

    if _waterfall_state is None or _waterfall_state["buf"].shape[0] != ctx.LED_COUNT:
        _init_state(ctx)

    st = _waterfall_state
    L = max(1, ctx.LED_COUNT)
    now = time.time()
    dt = min(now - st["last_t"], 0.05)
    st["last_t"] = now

    # Atualiza cooldown de spawn
    st["spawn_cooldown"] = max(0.0, st["spawn_cooldown"] - dt)

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

    # -------------------------------------------------- 2) Spawn controlado ----------------------------------------
    n_bands = len(bands_u8)
    max_drops = 32  # limite rígido
    spawn_this_frame = 0
    max_per_frame = 4 if beat_flag else 2

    if n_bands > 0 and active and st["spawn_cooldown"] <= 0.0:
        low_energy = float(np.mean(bands_u8[:min(8, n_bands)]))
        spawn_prob = np.clip(low_energy / 255.0, 0.0, 1.0) ** 1.8

        if beat_flag:
            spawn_prob = min(1.0, spawn_prob * 1.5)
            max_per_frame += 1

        band_to_led = np.linspace(0, L - 1, max(1, n_bands), dtype=np.int32)

        for i in range(n_bands):
            if spawn_this_frame >= max_per_frame or len(st["drops"]) >= max_drops:
                break
            if np.random.random() < spawn_prob:
                led = band_to_led[i]
                intensity = float(bands_u8[i])

                speed = 40.0 + 140.0 * (intensity / 255.0)
                if beat_flag:
                    speed *= 1.3

                # Cor
                if st["pal_arr"] is not None:
                    p = (i / max(1, n_bands - 1)) * (st["pal_len"] - 1)
                    i0, i1 = int(p), (int(p) + 1) % st["pal_len"]
                    t = p - i0
                    rgb = (st["pal_arr"][i0].astype(np.float32) * (1.0 - t) +
                           st["pal_arr"][i1].astype(np.float32) * t)
                else:
                    hue = (ctx.base_hue_offset + (i * 220 // max(1, n_bands)) + (ctx.hue_seed >> 2)) % 256
                    rgb = ctx.hsv_to_rgb_bytes_vec(
                        np.array([hue], dtype=np.uint8),
                        np.array([min(215, ctx.base_saturation)], dtype=np.uint8),
                        np.array([255], dtype=np.uint8)
                    )[0].astype(np.float32)

                st["drops"].append({
                    "pos": float(led),
                    "speed": speed,
                    "rgb": rgb,
                    "life": 1.0,
                    "size": 2.5 + 3.5 * (intensity / 255.0)
                })
                spawn_this_frame += 1

        if spawn_this_frame > 0:
            st["spawn_cooldown"] = 0.06  # cooldown entre bursts

    # -------------------------------------------------- 3) Atualiza gotas --------------------------------------------
    new_drops = []
    for d in st["drops"]:
        d["pos"] += d["speed"] * dt
        d["life"] *= 0.95 ** (dt * 60.0)
        d["size"] *= 1.0 + 0.10 * dt

        if d["pos"] < L + 15 and d["life"] > 0.04:
            new_drops.append(d)
        else:
            # Splash controlado (só 1 por gota)
            if d["pos"] >= L - 8 and "is_splash" not in d:
                splash_center = min(L - 1, int(d["pos"]))
                splash_rgb = np.clip(d["rgb"] * 1.4, 0, 255)
                new_drops.append({
                    "pos": float(splash_center),
                    "speed": 0.0,
                    "rgb": splash_rgb,
                    "life": 0.7,
                    "size": 10.0 + 8.0 * d["life"],
                    "is_splash": True
                })
    st["drops"] = new_drops[:max_drops]  # força limite

    # -------------------------------------------------- 4) Render ----------------------------------------------------
    acc = np.zeros((L, 3), dtype=np.float32)
    idx = np.arange(L, dtype=np.float32)

    for d in st["drops"]:
        dist = np.abs(idx - d["pos"])
        if d.get("is_splash"):
            sigma = d["size"] / 2.2
            contrib = np.exp(-0.5 * (dist / sigma) ** 2) * 170.0 * d["life"]
        else:
            sigma = d["size"]
            core = np.exp(-0.5 * (dist / sigma) ** 2)
            edge = np.exp(-1.8 * (dist / sigma) ** 2)
            contrib = (core * 135.0 + edge * 55.0) * d["life"]

        contrib = np.clip(contrib, 0.0, 255.0)
        acc += contrib[:, None] * (d["rgb"] / 255.0)

    # Decay mais forte em silêncio
    decay = 0.68 if active else 0.52
    sat = ctx.base_saturation / 255.0
    decay = decay - 0.15 * sat
    st["buf"] *= np.clip(decay, 0.40, 0.90)

    st["buf"] = np.clip(st["buf"] + acc, 0.0, 255.0)

    # -------------------------------------------------- 5) Limiter local (suave) -------------------------------------
    win = max(1, L // 10)
    step = max(1, int(win * 0.7))
    i = 0
    while i < L:
        s = i
        e = min(L, i + win)
        scale = 0.42 + 0.55 * (i / max(1, L - 1)) ** 0.8
        st["buf"][s:e] *= scale
        i += step

    # -------------------------------------------------- 6) Piso + clamp ----------------------------------------------
    floor = float(getattr(ctx, "dynamic_floor", 0))
    if floor > 0:
        st["buf"] = np.maximum(st["buf"], floor)

    per_pix = np.sum(st["buf"], axis=1)
    mask = per_pix > 690
    if np.any(mask):
        scale = 690.0 / np.maximum(per_pix[mask], 1.0)
        st["buf"][mask] *= scale[:, None]

    # -------------------------------------------------- 7) EMA (anti-flicker) ----------------------------------------
    beta = 0.40 if active else 0.30
    st["ema"] = st["ema"] * (1.0 - beta) + st["buf"] * beta
    out = np.clip(st["ema"], 0.0, 255.0).astype(np.uint8)

    # -------------------------------------------------- 8) Envio ----------------------------------------------------
    ctx.to_pixels_and_show(out)