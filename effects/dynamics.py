# effects/dynamics.py
import numpy as np
import time

# ===== Full Strip Pulse (com paleta) — como no seu arquivo =====
def effect_full_strip_pulse(ctx, bands_u8, beat_flag, active):
    """
    Full Strip Pulse — colorido
    - Se ctx.current_palette existir: usa a paleta (interpolação ao longo da fita + animação no tempo).
    - Senão: fallback para HSV com sweep de hue (temporal + espacial).
    Mantém a lógica de V (pulse de graves) do efeito original.
    """
    import numpy as np, time
    # ---- 1) Nível de pulse pelo grave (igual ao original) ----
    n = len(bands_u8)
    limit = min(8, n) if n > 0 else 1
    raw = int(np.sum(np.asarray(bands_u8[:limit], dtype=np.uint32)) // max(1, limit)) if n > 0 else 0
    lvl = ctx.amplify_quad(np.array([raw], dtype=np.uint16))[0]
    if beat_flag:
        lvl = int(min(255, lvl * 1.35))  # leve boost no beat
    v = np.full(ctx.LED_COUNT, min(255, lvl), dtype=np.uint8)
    v = ctx.apply_floor_vec(v, active, None)  # aplica piso dinâmico

    # ---- 2) Caminho A: usar paleta (se houver) ----
    pal = getattr(ctx, "current_palette", None)
    if isinstance(pal, (list, tuple)) and len(pal) >= 2:
        pal_arr = np.asarray(pal, dtype=np.uint8)
        m = pal_arr.shape[0]
        # anima a paleta ao longo do strip (movimento suave no tempo)
        t = time.time()
        base_phase = (t * 0.10) * m  # ~0.10 ciclos/seg
        if beat_flag:
            base_phase += 0.35 * m  # pequeno salto no beat
        pos = ((ctx.I_ALL.astype(np.float32) / max(1.0, float(ctx.LED_COUNT))) * m + base_phase) % m
        idx0 = np.floor(pos).astype(np.int32)
        idx1 = (idx0 + 1) % m
        frac = (pos - np.floor(pos)).astype(np.float32)[:, None]
        c0 = pal_arr[idx0].astype(np.float32)
        c1 = pal_arr[idx1].astype(np.float32)
        base_rgb = c0 * (1.0 - frac) + c1 * frac  # 0..255 float32
        # escala pelo pulse (V) e mistura com "cinza" conforme saturação
        sat_base = int(np.clip(getattr(ctx, "base_saturation", 220), 0, 255))
        s = sat_base / 255.0
        v_fac = (v.astype(np.float32) / 255.0)[:, None]
        colored = base_rgb * v_fac   # cor pela paleta * V
        gray = (v.astype(np.float32))[:, None]  # V em cinza
        out = gray * (1.0 - s) + colored * s
        rgb = np.clip(out, 0, 255).astype(np.uint8)
        ctx.to_pixels_and_show(rgb)
        return

    # ---- 3) Caminho B (fallback): HSV com sweep de hue ----
    # Sweep temporal + gradiente espacial para evitar "monocromia"
    t = time.time()
    h_time = int((t * 40.0) % 256)  # velocidade do sweep de hue
    h_spatial = ((ctx.I_ALL.astype(np.int32) * 256) // max(1, ctx.LED_COUNT))  # 0..255 ao longo do strip
    hue = (ctx.base_hue_offset + (ctx.hue_seed >> 2) + h_time + h_spatial) % 256
    if beat_flag:
        hue = (hue + 16) % 256
    sat = np.full(ctx.LED_COUNT, ctx.base_saturation, dtype=np.uint8)
    rgb = ctx.hsv_to_rgb_bytes_vec(hue.astype(np.uint8), sat, v.astype(np.uint8))
    ctx.to_pixels_and_show(rgb)

# effects/dynamics.py - WATERFALL V4 (substituir a função existente)

_water = None
def effect_waterfall(ctx, bands_u8, beat_flag, active):
    """
    Waterfall v4 — corrige saturação do terço inicial:
      - Atenuação progressiva (taper) do início → fim da fita
      - Reduz mapeamento log (menos concentração em graves)
      - Limiter por região (evita saturação localizada)
      - Decay mais agressivo para dar headroom dinâmico
    """
    import numpy as np
    global _water
    L = ctx.LED_COUNT
    if _water is None or _water.shape[0] != L:
        _water = np.zeros((L, 3), dtype=np.uint8)

    n = len(bands_u8)
    if n == 0:
        _water = (_water.astype(np.float32) * 0.60).astype(np.uint8)
        ctx.to_pixels_and_show(_water)
        return

    # ---------- 1) Mapear bandas → LEDs (menos log = mais uniforme) ----------
    arr = np.asarray(bands_u8, dtype=np.float32)
    
    x_led = np.linspace(0.0, 1.0, L, dtype=np.float32)
    alfa = 0.05  # REDUZIDO (era 0.12) - menos concentração em graves
    pos_bands = (np.log1p(alfa * x_led) / np.log1p(alfa))
    idx_bands = pos_bands * max(1, n - 1)
    band_pos = np.arange(n, dtype=np.float32)
    v_row = np.interp(idx_bands, band_pos, arr).astype(np.float32)

    if beat_flag:
        v_row *= 1.08  # boost menor

    # ---------- 2) TAPER: atenua início da fita progressivamente ----------
    # Curva suave: início em ~0.45, cresce até 1.0 no meio/fim
    # taper = np.clip(0.45 + 0.55 * (x_led ** 0.8), 0.45, 1.0).astype(np.float32)
    taper = np.clip(0.30 + 0.70 * (x_led ** 0.8), 0.30, 1.0).astype(np.float32)
    v_row *= taper  # <-- CHAVE: graves ficam ~45% do valor original

    # ---------- 3) Soft-knee (igual antes) ----------
    v01 = np.clip(v_row / 255.0, 0.0, 1.0)
    e = 0.60
    v01 = (v01 * (1.0 + e)) / (1.0 + e * v01)
    v_u8 = np.clip(v01 * 255.0, 0, 255).astype(np.uint8)

    # ---------- 4) Linha de cores ----------
    pal = getattr(ctx, "current_palette", None)
    if isinstance(pal, (list, tuple)) and len(pal) >= 2:
        pal_arr = np.asarray(pal, dtype=np.uint8)
        m = pal_arr.shape[0]
        pos = (np.arange(L, dtype=np.float32) / max(1.0, float(L))) * m
        if beat_flag:
            pos = (pos + 0.15) % m
        idx0 = np.floor(pos).astype(np.int32)
        idx1 = (idx0 + 1) % m
        t = (pos - idx0).astype(np.float32)[:, None]

        c0 = pal_arr[idx0].astype(np.float32)
        c1 = pal_arr[idx1].astype(np.float32)
        base_rgb = c0 * (1.0 - t) + c1 * t

        sat_base = int(np.clip(getattr(ctx, "base_saturation", 220), 0, 255))
        s = sat_base / 255.0
        v_fac = (v_u8.astype(np.float32) / 255.0)[:, None]
        colored = base_rgb * v_fac
        gray_rgb = (v_u8[:, None]).astype(np.float32)
        new_row = gray_rgb * (1.0 - s) + colored * s
        new_row = np.clip(new_row, 0, 255).astype(np.uint8)
    else:
        hue_base = (np.arange(L, dtype=np.float32) / max(1, L)) * 200.0
        hue_row = (ctx.base_hue_offset + hue_base.astype(np.int32) + (ctx.hue_seed >> 2)) % 256
        if beat_flag:
            hue_row = (hue_row + 10) % 256
        hue_row = hue_row.astype(np.uint8)
        sat_base = int(np.clip(getattr(ctx, "base_saturation", 220), 0, 255))
        sat_row = np.full(L, sat_base if active else max(100, sat_base - 60), dtype=np.uint8)
        new_row = ctx.hsv_to_rgb_bytes_vec(hue_row, sat_row, v_u8)

    # ---------- 5) Decay MAIS AGRESSIVO (dá headroom) ----------
    decay = 0.62 if active else 0.55  # era 0.70/0.60 — agora mais rápido
    if beat_flag:
        decay = min(0.68, decay + 0.03)
    _water = (_water.astype(np.float32) * decay).astype(np.uint8)

    # ---------- 6) Limiter POR REGIÃO (evita saturação local) ----------
    # Divide a fita em 3 regiões e limita cada uma individualmente
    third = L // 3
    regions = [
        (0, third, 0.40),           # início: 50% do normal
        (third, 2*third, 0.60),     # meio: 72%
        (2*third, L, 0.85)           # fim: 100%
    ]
    
    for start, end, scale in regions:
        region_old = _water[start:end].astype(np.uint16)
        region_new = new_row[start:end].astype(np.uint16)
        region_sum = np.clip(region_old + region_new, 0, 255).astype(np.uint8)
        
        # Aplica escala regional
        region_sum = np.clip(region_sum.astype(np.float32) * scale, 0, 255).astype(np.uint8)
        _water[start:end] = region_sum

    # ---------- 7) Piso dinâmico ----------
    out = ctx.apply_floor_vec(_water.astype(np.uint16), active, None).astype(np.uint8)

    # ---------- 8) Limiter global (55% do budget) ----------
    ma_per_channel = float(ctx.WS2812B_MA_PER_CHANNEL)
    idle_mA = float(ctx.WS2812B_IDLE_MA_PER_LED) * float(L)
    budget_mA = float(ctx.CURRENT_BUDGET_A) * 1000.0
    target_util = 0.45
    max_color_mA = max(0.0, budget_mA * target_util - idle_mA)

    sum_rgb = float(np.sum(out, dtype=np.uint64))
    if sum_rgb > 0.0 and max_color_mA > 0.0:
        need_mA = (ma_per_channel / 255.0) * sum_rgb
        scale = min(1.0, max_color_mA / need_mA) if need_mA > 0.0 else 1.0
        if scale < 0.999:
            out = np.clip(out.astype(np.float32) * scale, 0, 255).astype(np.uint8)

    ctx.to_pixels_and_show(out)


# ===== Bass Ripple Pulse v2 (anel gaussiano, anti-flick) =====
class _Ripple:
    __slots__ = ("r", "v", "spd", "thick", "hue_shift")
    def __init__(self, r, v, spd, thick, hue_shift):
        self.r = float(r)
        self.v = float(v)
        self.spd = float(spd)
        self.thick = float(thick)
        self.hue_shift = int(hue_shift)

_brp_active = []
_brp_env = 0.0
_brp_prev_low = 0.0

def effect_bass_ripple_pulse_v2(ctx, bands_u8, beat_flag, active):
    """
    Bass Ripple Pulse — v2 (calm/anti-flick, full-strip + palette)
    - Ondas em múltiplos centros distribuídos ao longo do strip.
    - Parâmetros mais estáveis (velocidade/pico/decay) para reduzir flick.
    - Limita spawns por frame + cooldown global curto.
    - Suavização temporal (EMA) antes do envio ao strip.
    - Usa ctx.current_palette se existir; fallback HSV caso contrário.
    """
    import numpy as np, time
    # ---------- Estado interno ----------
    st = getattr(ctx, "_brp3", None)
    if st is None:
        emitters = np.linspace(0, max(0, ctx.LED_COUNT - 1),
                               num=max(6, ctx.LED_COUNT // 60), dtype=np.float32)
        st = ctx._brp3 = {
            "last_t": time.time(),
            "emitters": emitters,  # centros distribuídos
            "ripples": [],         # dict(c,r,spd,thick,amp,life,rgb)
            "env": 0.0,
            "prev_low": 0.0,
            "flip": 0,             # alternância na escolha de centros
            "last_spawn_ts": 0.0,  # cooldown global anti-burst
            "ema": None,           # suavização temporal (float32 [L,3])
        }
    now = time.time()
    dt = now - st["last_t"]
    st["last_t"] = now
    dt = float(np.clip(dt, 0.0, 0.04))  # passo estável (<= 40 ms)

    # ---------- Entrada/energia (graves) ----------
    n = len(bands_u8)
    low_n = max(8, n // 8) if n > 0 else 8
    low_mean = float(np.mean(np.asarray(bands_u8[:low_n], dtype=np.float32))) if n > 0 else 0.0
    # envelope (attack/release)
    attack, release = 0.42, 0.20
    env_prev = st["env"]
    coef = attack if low_mean > env_prev else release
    env = coef * low_mean + (1.0 - coef) * env_prev
    st["env"] = env
    e01 = np.clip(env / 255.0, 0.0, 1.0)

    # ---------- Spawns ----------
    spawn = bool(beat_flag) or ((low_mean - st["prev_low"]) > 7.0)
    st["prev_low"] = low_mean
    rip = st["ripples"]
    MAX_RIPPLES = 12
    GLOBAL_COOLDOWN = 0.085
    can_spawn = (now - st["last_spawn_ts"]) >= GLOBAL_COOLDOWN
    if spawn and active and len(rip) < MAX_RIPPLES and can_spawn:
        base_count = 1 + int(3 * e01)  # 1..4
        if beat_flag:
            base_count = min(base_count + 1, 5)
        em = st["emitters"]
        st["flip"] ^= 1
        start = st["flip"] % 2
        step = max(1, em.size // max(1, base_count))
        picked_idx = np.arange(start, em.size, step, dtype=np.int32)[:base_count]
        centers = em[picked_idx]
        pal = getattr(ctx, "current_palette", None)
        use_pal = isinstance(pal, (list, tuple)) and len(pal) >= 2
        pal_arr = np.asarray(pal, dtype=np.uint8) if use_pal else None
        m = pal_arr.shape[0] if use_pal else 0
        for j, c in enumerate(centers):
            spd = 70.0 + 180.0 * e01
            thick = 3.0 + 4.0 * e01
            amp = 70.0 + 120.0 * e01
            if beat_flag: amp *= 1.10
            life = 0.96 if beat_flag else 0.88
            if use_pal:
                p = (j / max(1, base_count - 1)) * (m - 1) if base_count > 1 else 0.0
                i0 = int(np.floor(p)) % m
                i1 = (i0 + 1) % m
                tcol = float(p - np.floor(p))
                rgb_peak = (pal_arr[i0].astype(np.float32) * (1.0 - tcol) +
                            pal_arr[i1].astype(np.float32) * tcol)
                rgb_peak = np.clip(rgb_peak, 0, 255).astype(np.uint8)
            else:
                hue = (int(ctx.base_hue_offset) + int(ctx.hue_seed >> 2) +
                       int(240.0 * (c / max(1.0, float(ctx.LED_COUNT - 1))))) % 256
                rgb_peak = ctx.hsv_to_rgb_bytes_vec(
                    np.array([hue], dtype=np.uint8),
                    np.array([min(230, ctx.base_saturation)], dtype=np.uint8),
                    np.array([255], dtype=np.uint8)
                )[0]
            rip.append({"c": float(c), "r": 0.0, "spd": spd, "thick": thick,
                        "amp": amp, "life": life, "rgb": rgb_peak})
        if len(rip) > MAX_RIPPLES:
            rip[:] = rip[-MAX_RIPPLES:]
        st["last_spawn_ts"] = now

    # ---------- Atualização ----------
    life_decay = 0.96 ** (dt * 40.0)
    amp_decay  = 0.93 ** (dt * 40.0)
    width_grow = 1.0 + 0.18 * dt
    for rp in rip:
        rp["r"] += rp["spd"] * dt
        rp["life"] *= life_decay
        rp["amp"] *= amp_decay
        rp["thick"] *= width_grow
    max_reach = lambda c: max(c, (ctx.LED_COUNT - 1) - c) + 6.0
    rip[:] = [rp for rp in rip if (rp["life"] > 0.05 and rp["r"] < max_reach(rp["c"]))]

    # ---------- Render ----------
    L = ctx.LED_COUNT
    if L <= 0:
        return
    idx = ctx.I_ALL.astype(np.float32)
    rgb_acc = np.zeros((L, 3), dtype=np.float32)
    if rip:
        PER_COMET_CAP = 210.0
        for rp in rip:
            d = np.abs(idx - float(rp["c"]))
            diff = np.abs(d - float(rp["r"]))
            sig = max(0.9, float(rp["thick"]))
            shape = np.exp(-0.5 * (diff / sig) ** 2).astype(np.float32)  # [L]
            val = shape * float(rp["amp"]) * float(rp["life"])
            val = np.minimum(val, PER_COMET_CAP)
            rgb_acc += (val[:, None] / 255.0) * rp["rgb"].astype(np.float32)
    if not active:
        rgb_acc *= 0.85

    # Suavização temporal (EMA)
    st_ema = st.get("ema")
    if st_ema is None or st_ema.shape != (L, 3):
        st["ema"] = rgb_acc.copy()
    else:
        beta = 0.55 if active else 0.45
        st["ema"] = st_ema * (1.0 - beta) + rgb_acc * beta
    rgb_smooth = st["ema"]
    rgb_smooth = np.clip(rgb_smooth, 0, 235).astype(np.uint8)

    f = float(getattr(ctx, "dynamic_floor", 0))
    if f > 0:
        rgb_smooth = np.maximum(rgb_smooth, f).astype(np.uint8)

    ctx.to_pixels_and_show(rgb_smooth)