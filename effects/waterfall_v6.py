# effects/waterfall_v6.py
import numpy as np
import time

# ----------------------------------------------------------------------
# Waterfall v6 — “Chromatic Cascade” (CORRIGIDO: uso do terço final + diversidade de cores)
# ----------------------------------------------------------------------
# Problemas relatados:
# 1. **Terço final subutilizado** → gotas "morrem" antes do fim da fita.
# 2. **Monocromático** → cores muito parecidas (paleta não explorada).
#
# Soluções aplicadas:
# - **Velocidade ajustada por banda**: graves (lentos) → gotas caem até o fim.
# - **Spawn distribuído por frequência**: bandas altas → gotas nascem mais à direita.
# - **Paleta usada com alta saturação + hue offset por banda** → cores distintas.
# - **Splash mais forte no final** → ativa visualmente o terço final.
# - **Decay por região** → início decai rápido, fim mantém brilho.
# - **Hue jitter por gota** → evita repetição cromática.
# ----------------------------------------------------------------------
_waterfall_state = None

def _init_state(ctx):
    global _waterfall_state
    L = max(1, ctx.LED_COUNT)
    _waterfall_state = {
        "buf": np.zeros((L, 3), dtype=np.float32),
        "drops": [],
        "last_t": time.time(),
        "ema": np.zeros((L, 3), dtype=np.float32),
        "pal_arr": None,
        "pal_len": 0,
        "spawn_cooldown": 0.0,
    }

def effect_chromatic_cascade(ctx, bands_u8, beat_flag, active):
    global _waterfall_state

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

    st["spawn_cooldown"] = max(0.0, st["spawn_cooldown"] - dt)

    # -------------------------------------------------- 1) Paleta (alta saturação + cache) -------------------------
    pal = getattr(ctx, "current_palette", None)
    if isinstance(pal, (list, tuple)) and len(pal) >= 2:
        pal_arr = np.asarray(pal, dtype=np.uint8)
        # Força saturação alta para diversidade
        hsv = np.zeros((len(pal), 3), dtype=np.float32)
        for i, (r, g, b) in enumerate(pal_arr):
            mx = max(r, g, b) / 255.0
            mn = min(r, g, b) / 255.0
            df = mx - mn
            if mx == mn:
                h = 0
            elif mx == r/255.0:
                h = (60 * ((g-b)/df) + 360) % 360
            elif mx == g/255.0:
                h = (60 * ((b-r)/df) + 120) % 360
            else:
                h = (60 * ((r-g)/df) + 240) % 360
            h /= 360.0
            s = 0.0 if mx == 0 else df / mx
            v = mx
            # Aumenta saturação para 80-95%
            s = min(0.95, max(0.80, s))
            hsv[i] = [h, s, v]
        # Reconverte com saturação forçada
        pal_arr = ctx.hsv_to_rgb_bytes_vec(
            (hsv[:, 0] * 255).astype(np.uint8),
            (hsv[:, 1] * 255).astype(np.uint8),
            (hsv[:, 2] * 255).astype(np.uint8)
        )
        if st["pal_arr"] is None or not np.array_equal(st["pal_arr"], pal_arr):
            st["pal_arr"] = pal_arr.copy()
            st["pal_len"] = pal_arr.shape[0]
    else:
        st["pal_arr"] = None
        st["pal_len"] = 0

    # -------------------------------------------------- 2) Spawn por frequência (graves → esquerda, agudos → direita) -----
    n_bands = len(bands_u8)
    max_drops = 36
    spawn_this_frame = 0
    max_per_frame = 5 if beat_flag else 3

    if n_bands > 0 and active and st["spawn_cooldown"] <= 0.0:
        # Energia por terço de frequência
        low = bands_u8[:n_bands//3]
        mid = bands_u8[n_bands//3:2*n_bands//3]
        high = bands_u8[2*n_bands//3:]

        energies = [
            float(np.mean(low)) if len(low) > 0 else 0.0,
            float(np.mean(mid)) if len(mid) > 0 else 0.0,
            float(np.mean(high)) if len(high) > 0 else 0.0
        ]
        probs = np.array(energies) / 255.0
        probs = probs ** 1.6
        probs /= max(1e-6, np.sum(probs))

        # Posição de nascimento: graves (esquerda), agudos (direita)
        spawn_zones = [0.15, 0.5, 0.85]  # 15%, 50%, 85% da fita

        for zone_idx, prob in enumerate(probs):
            if spawn_this_frame >= max_per_frame or len(st["drops"]) >= max_drops:
                break
            if np.random.random() < prob * (2.0 if beat_flag else 1.0):
                # Escolhe banda dentro da zona
                start = (zone_idx * n_bands) // 3
                end = min(n_bands, ((zone_idx + 1) * n_bands) // 3)
                if start >= end:
                    continue
                i = np.random.randint(start, end)
                intensity = float(bands_u8[i])

                # Posição inicial na fita (proporcional à frequência)
                base_pos = spawn_zones[zone_idx] * (L - 1)
                jitter = np.random.uniform(-0.08, 0.08) * L
                led = int(np.clip(base_pos + jitter, 0, L - 1))

                # Velocidade: graves (lenta, chega ao fim), agudos (rápida)
                speed = 60.0 + 80.0 * (1.0 - zone_idx / 2.0)  # 140 → 100 → 60
                if beat_flag:
                    speed *= 1.35

                # Cor com hue jitter por gota
                if st["pal_arr"] is not None:
                    p = (zone_idx / 2.0) * (st["pal_len"] - 1)
                    i0 = int(p) % st["pal_len"]
                    i1 = (i0 + 1) % st["pal_len"]
                    t = p - i0
                    base_rgb = (st["pal_arr"][i0].astype(np.float32) * (1.0 - t) +
                                st["pal_arr"][i1].astype(np.float32) * t)
                    # Jitter de matiz
                    hue_jitter = np.random.randint(-25, 26)
                    hsv_jitter = ctx.hsv_to_rgb_bytes_vec(
                        np.array([(np.random.randint(0, 256) + hue_jitter) % 256], dtype=np.uint8),
                        np.array([min(240, ctx.base_saturation)], dtype=np.uint8),
                        np.array([255], dtype=np.uint8)
                    )[0].astype(np.float32)
                    rgb = base_rgb * 0.7 + hsv_jitter * 0.3
                else:
                    hue = (ctx.base_hue_offset + (i * 240 // max(1, n_bands)) +
                           (ctx.hue_seed >> 1) + np.random.randint(-30, 31)) % 256
                    rgb = ctx.hsv_to_rgb_bytes_vec(
                        np.array([hue], dtype=np.uint8),
                        np.array([min(235, ctx.base_saturation)], dtype=np.uint8),
                        np.array([255], dtype=np.uint8)
                    )[0].astype(np.float32)

                st["drops"].append({
                    "pos": float(led),
                    "speed": speed,
                    "rgb": np.clip(rgb, 0, 255),
                    "life": 1.0,
                    "size": 2.8 + 4.2 * (intensity / 255.0)
                })
                spawn_this_frame += 1

        if spawn_this_frame > 0:
            st["spawn_cooldown"] = 0.07

    # -------------------------------------------------- 3) Atualiza gotas (splash mais forte) ------------------------
    new_drops = []
    for d in st["drops"]:
        d["pos"] += d["speed"] * dt
        d["life"] *= 0.94 ** (dt * 55.0)
        d["size"] *= 1.0 + 0.09 * dt

        if d["pos"] < L + 25 and d["life"] > 0.03:
            new_drops.append(d)
        else:
            if d["pos"] >= L * 0.7 and "is_splash" not in d:  # splash a partir de 70%
                splash_center = min(L - 1, int(d["pos"]))
                splash_rgb = np.clip(d["rgb"] * 1.6, 0, 255)
                new_drops.append({
                    "pos": float(splash_center),
                    "speed": 0.0,
                    "rgb": splash_rgb,
                    "life": 0.8,
                    "size": 14.0 + 10.0 * d["life"],
                    "is_splash": True
                })
    st["drops"] = new_drops[:max_drops]

    # -------------------------------------------------- 4) Render ----------------------------------------------------
    acc = np.zeros((L, 3), dtype=np.float32)
    idx = np.arange(L, dtype=np.float32)

    for d in st["drops"]:
        dist = np.abs(idx - d["pos"])
        if d.get("is_splash"):
            sigma = d["size"] / 2.0
            contrib = np.exp(-0.5 * (dist / sigma) ** 2) * 200.0 * d["life"]
        else:
            sigma = d["size"]
            core = np.exp(-0.5 * (dist / sigma) ** 2)
            edge = np.exp(-2.2 * (dist / sigma) ** 2)
            contrib = (core * 145.0 + edge * 65.0) * d["life"]

        contrib = np.clip(contrib, 0.0, 255.0)
        acc += contrib[:, None] * (d["rgb"] / 255.0)

    # Decay por região (início decai mais, fim mantém)
    x = idx / max(1, L - 1)
    decay_map = 0.70 - 0.25 * x  # 0.70 → 0.45
    decay_map = np.clip(decay_map, 0.42, 0.90)
    if not active:
        decay_map *= 0.78
    st["buf"] *= decay_map[:, None]

    st["buf"] = np.clip(st["buf"] + acc, 0.0, 255.0)

    # -------------------------------------------------- 5) Limiter local ---------------------------------------------
    win = max(1, L // 9)
    step = max(1, int(win * 0.68))
    i = 0
    while i < L:
        s = i
        e = min(L, i + win)
        scale = 0.45 + 0.52 * (i / max(1, L - 1)) ** 0.78
        st["buf"][s:e] *= scale
        i += step

    # -------------------------------------------------- 6) Piso + clamp ----------------------------------------------
    floor = float(getattr(ctx, "dynamic_floor", 0))
    if floor > 0:
        st["buf"] = np.maximum(st["buf"], floor)

    per_pix = np.sum(st["buf"], axis=1)
    mask = per_pix > 695
    if np.any(mask):
        scale = 695.0 / np.maximum(per_pix[mask], 1.0)
        st["buf"][mask] *= scale[:, None]

    # -------------------------------------------------- 7) EMA --------------------------------------------------------
    beta = 0.42 if active else 0.32
    st["ema"] = st["ema"] * (1.0 - beta) + st["buf"] * beta
    out = np.clip(st["ema"], 0.0, 255.0).astype(np.uint8)

    # -------------------------------------------------- 8) Envio ----------------------------------------------------
    ctx.to_pixels_and_show(out)