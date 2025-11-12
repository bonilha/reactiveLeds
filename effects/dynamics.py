# effects/dynamics.py
import numpy as np

_peaks = None
_dec_tick = 0

# Estado global do peak meter
_pm_state = {
    "peaks": None,     # float32 por banda, 0..LED_COUNT (em "altura de LED")
    "last_t": None,    # timestamp do último update
    "band_count": 0,   # n anterior para reinit se mudar
}

def effect_peak_hold_columns(ctx, bands_u8, beat_flag, active):
    """
    Peak Hold Columns (colunas com marcador de pico por banda).
    - Divide a fita em n colunas (n = len(bands_u8)).
    - Altura da coluna baseada na energia da banda (amplify_quad).
    - Marcador de pico por banda com hold + decay temporal estável.
    - Cores em HSV: hue varia por banda; beat dá leve shift.
    """
    global _pm_state
    L = int(getattr(ctx, "LED_COUNT", 0))
    if L <= 0:
        return

    n = int(len(bands_u8))
    if n <= 0:
        # Nada de áudio -> decay suave do buffer inteiro (escurece)
        ctx.to_pixels_and_show(np.zeros((L, 3), dtype=np.uint8))
        return

    now = time.time()
    if (_pm_state["last_t"] is None) or (_pm_state["band_count"] != n):
        _pm_state["peaks"] = np.zeros(n, dtype=np.float32)
        _pm_state["last_t"] = now
        _pm_state["band_count"] = n

    dt = float(now - _pm_state["last_t"])
    # Clampa dt para estabilidade em pausas/lag
    dt = max(0.0, min(dt, 0.04))  # ~25 FPS mínimo
    _pm_state["last_t"] = now

    # ----- 1) Normaliza energia por banda -> altura da coluna (0..LEDs_por_coluna)
    bands_u8 = np.asarray(bands_u8, dtype=np.uint8)
    # Curva de ganho do seu ambiente
    v16 = ctx.amplify_quad(bands_u8.astype(np.uint16))  # 0..~255 u16
    v = np.clip(v16, 0, 255).astype(np.uint8)

    # Colunas distribuem LEDs (aprox. uniforme)
    # start[i]: índice inicial da coluna i
    # end[i]: índice final exclusivo
    base = np.arange(n + 1, dtype=np.int32) * L // n
    starts = base[:-1]           # shape (n,)
    ends = np.maximum(starts + 1, base[1:])  # garante >= 1 LED por coluna
    col_len = np.maximum(1, ends - starts)   # LEDs por coluna (pode variar ±1)

    # Altura alvo (em "LEDs") por banda:
    # mapeia v (0..255) -> [0 .. col_len[i]]
    height_target = (v.astype(np.float32) / 255.0) * col_len.astype(np.float32)

    # ----- 2) Atualiza pico por banda (hold + decay temporal)
    peaks = _pm_state["peaks"]  # em "LEDs", float32 (0..col_len[i])
    # Se coluna diminuiu, clamp
    peaks = np.minimum(peaks, col_len.astype(np.float32))

    # Hit/raise imediato: pico sobe ao >= alvo
    peaks = np.maximum(peaks, height_target)

    # Decay temporal: cai a uma taxa por segundo, independente do FPS
    # Mais rápido quando não há beat; um pouco mais lento com beat
    # (ajuste fino: sinta-se à vontade para mexer)
    fall_per_sec = 12.0 if not beat_flag else 8.0  # LEDs/seg
    peaks = np.maximum(0.0, peaks - fall_per_sec * dt)

    _pm_state["peaks"] = peaks

    # ----- 3) Desenho: coluna preenchida + marcador de pico
    # Buffer RGB
    rgb = np.zeros((L, 3), dtype=np.uint8)

    # Hue base por banda (gradiente leve)
    band_ids = np.arange(n, dtype=np.int32)
    hue_band = (int(ctx.base_hue_offset) +
                ((band_ids * 7) % 256) +
                (int(ctx.hue_seed) & 0x3F)) % 256
    if beat_flag:
        hue_band = (hue_band + 18) % 256

    # Saturação por coluna (pode reduzir no topo para “glow” mais suave)
    sat_base = int(np.clip(getattr(ctx, "base_saturation", 220), 0, 255))

    # Para cada coluna, preenche e desenha o peak marker
    for i in range(n):
        s = int(starts[i])
        e = int(ends[i])
        length = int(col_len[i])

        if length <= 0:
            continue

        # Alturas discretizadas
        h_col = float(height_target[i])
        h_int = int(np.clip(np.floor(h_col + 1e-3), 0, length))  # LEDs cheios

        # Peak marker (1 LED acima do nível cheio)
        p_led_pos = int(np.clip(int(np.floor(peaks[i])), 0, length - 1))

        # Brilho base da coluna (V): escala do pé ao topo
        # Curva leve para “peso” visual
        if h_int > 0:
            # Gradiente de V dentro da coluna: do 40% até 100%
            grad = np.linspace(0.40, 1.00, h_int, dtype=np.float32)
            vals = (grad * 255.0).astype(np.uint8)
            # Converte HSV -> RGB para esses h_int LEDs
            hue_vec = np.full(h_int, hue_band[i], dtype=np.uint8)
            sat_vec = np.full(h_int, sat_base, dtype=np.uint8)
            col_rgb = ctx.hsv_to_rgb_bytes_vec(hue_vec, sat_vec, vals)
            # Escreve de baixo (s) para cima (s + h_int)
            rgb[s:s + h_int] = col_rgb

        # Se houver “meio LED” (fração) acima de h_int, dá um glow sutil
        frac = h_col - np.floor(h_col)
        if frac > 0 and h_int < length:
            frac_v = int(np.clip(80 + frac * 120, 0, 255))
            hue_v = np.uint8(hue_band[i])
            sat_v = np.uint8(min(255, sat_base))
            glow = ctx.hsv_to_rgb_bytes_vec(
                np.array([hue_v], dtype=np.uint8),
                np.array([sat_v], dtype=np.uint8),
                np.array([frac_v], dtype=np.uint8)
            )[0]
            rgb[s + h_int] = np.maximum(rgb[s + h_int], glow)

        # Peak marker: cor mais clara (alto V), saturação reduzida p/ destacar
        if length > 0:
            hue_p = np.uint8(hue_band[i])
            sat_p = np.uint8(max(80, sat_base - 90))
            val_p = np.uint8(240 if active else 200)
            peak_px = ctx.hsv_to_rgb_bytes_vec(
                np.array([hue_p], dtype=np.uint8),
                np.array([sat_p], dtype=np.uint8),
                np.array([val_p], dtype=np.uint8)
            )[0]
            rgb[s + p_led_pos] = np.maximum(rgb[s + p_led_pos], peak_px)

    # ----- 4) Piso dinâmico + envio
    # Aplica o floor por canal se existir; aqui usamos V ~ max canal
    # Convertemos para "V" aproximado, aplicamos floor em V e re-escalamos.
    # (Se preferir, pode aplicar floor direto nos canais.)
    v_approx = np.max(rgb, axis=1).astype(np.uint16)
    v_approx = ctx.apply_floor_vec(v_approx, active, None)  # 0..255 u16
    # Reescala por V aplicado para manter o “shape” sem crush
    scale = (v_approx.clip(1, 255) / np.maximum(np.max(rgb, axis=1).clip(1, 255), 1)).astype(np.float32)
    rgb = np.clip(rgb.astype(np.float32) * scale[:, None], 0, 255).astype(np.uint8)



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
        base_phase = (t * 0.10) * m               # ~0.10 ciclos/seg
        if beat_flag:
            base_phase += 0.35 * m               # pequeno salto no beat

        pos = ((ctx.I_ALL.astype(np.float32) / max(1.0, float(ctx.LED_COUNT))) * m + base_phase) % m
        idx0 = np.floor(pos).astype(np.int32)
        idx1 = (idx0 + 1) % m
        frac = (pos - np.floor(pos)).astype(np.float32)[:, None]

        c0 = pal_arr[idx0].astype(np.float32)
        c1 = pal_arr[idx1].astype(np.float32)
        base_rgb = c0 * (1.0 - frac) + c1 * frac   # 0..255 float32

        # escala pelo pulse (V) e mistura com "cinza" conforme saturação
        sat_base = int(np.clip(getattr(ctx, "base_saturation", 220), 0, 255))
        s = sat_base / 255.0
        v_fac = (v.astype(np.float32) / 255.0)[:, None]

        # cor "pura" pela paleta escalada por V
        colored = base_rgb * v_fac
        # cinza correspondente ao V (mesmo V em RGB)
        gray = (v.astype(np.float32))[:, None]
        # mistura para respeitar saturação base
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


# Waterfall clássico reativo: espectro full mapeado por LED, shift down com decay longo (enche strip)
_water = None

def effect_waterfall(ctx, bands_u8, beat_flag, active):
    """
    Waterfall que usa a current_palette (lista de 3-tuplas RGB em 0..255) se presente em ctx.current_palette.
    - Interpola suavemente as cores da paleta ao longo do strip.
    - Modula o brilho por banda (v_row) como antes.
    - Em beat, aplica um pequeno deslocamento de fase na paleta para "respirar".
    - Se ctx.current_palette não existir, cai no gradiente HSV clássico (fallback).

    Requer:
      - numpy como np
      - buffer global _water (np.uint8 [LED_COUNT x 3])
      - utilitários do ctx: amplify_quad, apply_floor_vec, to_pixels_and_show
    """
    import numpy as np

    global _water
    if _water is None or _water.shape[0] != ctx.LED_COUNT:
        _water = np.zeros((ctx.LED_COUNT, 3), dtype=np.uint8)
        print("[INFO] Waterfall init: buffers para {} LEDs WS2812B".format(ctx.LED_COUNT))

    n = len(bands_u8)
    if n == 0:
        # decay suave se não há sinal
        _water = (_water.astype(np.float32) * 0.92).astype(np.uint8)
        ctx.to_pixels_and_show(_water)
        return

    # ===== 1) Mapear bandas -> LEDs (valor/brilho por posição) =====
    arr = np.asarray(bands_u8, dtype=np.float32)
    if n < 2:
        v_row = np.full(ctx.LED_COUNT, arr[0] if n else 0, dtype=np.float32)
    else:
        band_pos = np.linspace(0, ctx.LED_COUNT - 1, n, dtype=np.float32)
        led_pos = np.arange(ctx.LED_COUNT, dtype=np.float32)
        v_row = np.interp(led_pos, band_pos, arr)

    # Ganhos/boosts (mantidos do seu efeito atual)
    v_row *= 2.0                               # base
    if beat_flag:
        v_row *= 1.8                           # beat boost
    v_row = np.clip(v_row, 0, 255).astype(np.uint16)
    v_row = ctx.amplify_quad(v_row)            # curva não-linear
    low_boost = np.linspace(1.4, 1.0, ctx.LED_COUNT)  # graves reforçados na esquerda
    v_row = np.clip(v_row.astype(np.float32) * low_boost, 0, 255).astype(np.uint16)
    v_u8 = v_row.astype(np.uint8)

    # ===== 2) Gerar "linha" de cores a partir da paleta (ou fallback HSV) =====
    pal = getattr(ctx, "current_palette", None)
    new_row = None

    if isinstance(pal, (list, tuple)) and len(pal) >= 2:
        # ---- 2A) USANDO PALETA (interp linear entre cores) ----
        pal_arr = np.asarray(pal, dtype=np.uint8)
        m = pal_arr.shape[0]

        # posição ao longo da paleta (0..m), com leve deslocamento em beat
        pos = (np.arange(ctx.LED_COUNT, dtype=np.float32) / max(1, float(ctx.LED_COUNT))) * m
        if beat_flag:
            pos = (pos + 0.35) % m  # "respira" a paleta no beat

        idx0 = np.floor(pos).astype(np.int32)
        idx1 = (idx0 + 1) % m
        t = (pos - idx0).astype(np.float32)        # fração 0..1
        t_col = t[:, None]                         # broadcast para canais RGB

        # interpola RGB
        c0 = pal_arr[idx0].astype(np.float32)
        c1 = pal_arr[idx1].astype(np.float32)
        base_rgb = (c0 * (1.0 - t_col) + c1 * t_col)  # float32 [LED x 3] em 0..255

        # aplica brilho v (0..255) como fator
        v_fac = (v_u8.astype(np.float32) / 255.0)[:, None]
        colored = base_rgb * v_fac  # ainda float32

        # opcional: respeitar saturação base da paleta (mistura com branco/cinza)
        sat_base = int(np.clip(getattr(ctx, "base_saturation", 220), 0, 255))
        s = sat_base / 255.0
        gray_rgb = (v_u8[:, None]).astype(np.float32)  # mesma V em RGB (cinza)
        colored = gray_rgb * (1.0 - s) + colored * s

        new_row = np.clip(colored, 0, 255).astype(np.uint8)

    if new_row is None:
        # ---- 2B) FALLBACK HSV clássico (gradiente procedural) ----
        hue_base = (np.arange(ctx.LED_COUNT, dtype=np.float32) / ctx.LED_COUNT) * 200.0
        hue_row = (ctx.base_hue_offset + hue_base.astype(np.int32) + (ctx.hue_seed >> 2)) % 256
        if beat_flag:
            hue_row = (hue_row + 20) % 256
        hue_row = hue_row.astype(np.uint8)

        sat_base = int(np.clip(getattr(ctx, "base_saturation", 220), 0, 255))
        sat_row = np.full(
            ctx.LED_COUNT,
            sat_base if active else max(100, sat_base - 60),
            dtype=np.uint8
        )
        new_row = ctx.hsv_to_rgb_bytes_vec(hue_row, sat_row, v_u8)

    # ===== 3) Composição no buffer waterfall com decay + soma =====
    _water = (_water.astype(np.float32) * 0.75).astype(np.uint8)  # decay rápido
    _water = np.clip(_water.astype(np.uint16) + new_row.astype(np.uint16), 0, 255).astype(np.uint8)

    # ===== 4) Floor dinâmico + render =====
    _water_out = ctx.apply_floor_vec(_water.astype(np.uint16), active, None).astype(np.uint8)
    ctx.to_pixels_and_show(_water_out)

# Bass Ripple Pulse v2 (anel gaussiano)
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
            "emitters": emitters,          # centros distribuídos
            "ripples": [],                 # dict(c,r,spd,thick,amp,life,rgb)
            "env": 0.0,
            "prev_low": 0.0,
            "flip": 0,                     # alternância na escolha de centros
            "last_spawn_ts": 0.0,          # cooldown global anti-burst
            "ema": None,                   # suavização temporal (float32 [L,3])
        }

    now = time.time()
    dt = now - st["last_t"]
    st["last_t"] = now
    dt = float(np.clip(dt, 0.0, 0.04))  # passo estável (<= 40 ms)

    # ---------- Entrada/energia (graves) ----------
    n = len(bands_u8)
    low_n = max(8, n // 8) if n > 0 else 8
    low_mean = float(np.mean(np.asarray(bands_u8[:low_n], dtype=np.float32))) if n > 0 else 0.0

    # envelope (attack/release mais "lento" para suavizar)
    attack, release = 0.42, 0.20
    env_prev = st["env"]
    coef = attack if low_mean > env_prev else release
    env = coef * low_mean + (1.0 - coef) * env_prev
    st["env"] = env
    e01 = np.clip(env / 255.0, 0.0, 1.0)

    # ---------- Spawns (múltiplos centros, mais contidos) ----------
    spawn = bool(beat_flag) or ((low_mean - st["prev_low"]) > 7.0)
    st["prev_low"] = low_mean

    rip = st["ripples"]
    MAX_RIPPLES = 12         # antes 18 — ajuda a evitar somas muito altas
    GLOBAL_COOLDOWN = 0.085  # ~85 ms entre bursts
    can_spawn = (now - st["last_spawn_ts"]) >= GLOBAL_COOLDOWN

    if spawn and active and len(rip) < MAX_RIPPLES and can_spawn:
        # menos ondas por frame; ainda usa a energia para escalar
        base_count = 1 + int(3 * e01)       # 1..4
        if beat_flag:
            base_count = min(base_count + 1, 5)

        em = st["emitters"]
        st["flip"] ^= 1
        start = st["flip"] % 2
        # distribuição "intercalada" e limitada ao que cabe
        step = max(1, em.size // max(1, base_count))
        picked_idx = np.arange(start, em.size, step, dtype=np.int32)[:base_count]
        centers = em[picked_idx]

        pal = getattr(ctx, "current_palette", None)
        use_pal = isinstance(pal, (list, tuple)) and len(pal) >= 2
        pal_arr = np.asarray(pal, dtype=np.uint8) if use_pal else None
        m = pal_arr.shape[0] if use_pal else 0

        for j, c in enumerate(centers):
            # **Parâmetros mais calmos**
            spd   = 70.0 + 180.0 * e01                 # px/s (antes 90..330)
            thick = 3.0  + 4.0   * e01                 # largura
            amp   = 70.0 + 120.0 * e01                # pico (0..255)
            if beat_flag: amp *= 1.10                  # leve boost em beat
            life  = 0.96 if beat_flag else 0.88       # vida um pouco maior em beat

            if use_pal:
                # cor distribuída na paleta
                p = (j / max(1, base_count - 1)) * (m - 1) if base_count > 1 else 0.0
                i0 = int(np.floor(p)) % m
                i1 = (i0 + 1) % m
                tcol = float(p - np.floor(p))
                rgb_peak = (pal_arr[i0].astype(np.float32) * (1.0 - tcol) +
                            pal_arr[i1].astype(np.float32) * tcol)
                rgb_peak = np.clip(rgb_peak, 0, 255).astype(np.uint8)
            else:
                # fallback HSV (varia com a posição do centro)
                hue = (int(ctx.base_hue_offset) + int(ctx.hue_seed >> 2) +
                       int(240.0 * (c / max(1.0, float(ctx.LED_COUNT - 1))))) % 256
                rgb_peak = ctx.hsv_to_rgb_bytes_vec(
                    np.array([hue], dtype=np.uint8),
                    np.array([min(230, ctx.base_saturation)], dtype=np.uint8),
                    np.array([255], dtype=np.uint8)
                )[0]

            rip.append({"c": float(c), "r": 0.0, "spd": spd, "thick": thick,
                        "amp": amp, "life": life, "rgb": rgb_peak})

        # respeita o teto global
        if len(rip) > MAX_RIPPLES:
            rip[:] = rip[-MAX_RIPPLES:]
        st["last_spawn_ts"] = now

    # ---------- Atualização das ondas ----------
    # decaimentos um pouco mais lentos (menos "estalo" e mais continuidade)
    life_decay  = 0.96 ** (dt * 40.0)   # antes 0.92
    amp_decay   = 0.93 ** (dt * 40.0)   # antes 0.90
    width_grow  = 1.0 + 0.18 * dt       # antes 0.25

    for rp in rip:
        rp["r"]     += rp["spd"] * dt
        rp["life"]  *= life_decay
        rp["amp"]   *= amp_decay
        rp["thick"] *= width_grow

    # limite geométrico (chegou na borda + folga)
    max_reach = lambda c: max(c, (ctx.LED_COUNT - 1) - c) + 6.0
    rip[:] = [rp for rp in rip if (rp["life"] > 0.05 and rp["r"] < max_reach(rp["c"]))]

    # ---------- Render vetorizado ----------
    L = ctx.LED_COUNT
    if L <= 0:
        return
    idx = ctx.I_ALL.astype(np.float32)

    rgb_acc = np.zeros((L, 3), dtype=np.float32)
    if rip:
        # cap suave por cometa -> reduz chance de "cap" duro no FXContext
        PER_COMET_CAP = 210.0

        for rp in rip:
            d = np.abs(idx - float(rp["c"]))
            diff = np.abs(d - float(rp["r"]))
            sig = max(0.9, float(rp["thick"]))
            # anel gaussiano
            shape = np.exp(-0.5 * (diff / sig) ** 2).astype(np.float32)   # [L]
            val   = shape * float(rp["amp"]) * float(rp["life"])           # 0..~255
            # cap suave (tanh-like via compressão)
            val = np.minimum(val, PER_COMET_CAP)
            rgb_acc += (val[:, None] / 255.0) * rp["rgb"].astype(np.float32)

    # em idle: reduz um pouco
    if not active:
        rgb_acc *= 0.85

    # --------- Suavização temporal (EMA) ----------
    # Mistura do frame atual com o anterior para reduzir "flick" visual
    if st["ema"] is None or st["ema"].shape != (L, 3):
        st["ema"] = rgb_acc.copy()
    else:
        # beta mais alto = mais responsivo; mais baixo = mais suave
        beta = 0.55 if active else 0.45
        st["ema"] = st["ema"] * (1.0 - beta) + rgb_acc * beta

    rgb_smooth = st["ema"]

    # cap global moderado antes do envio (mais uma camada anti-cap duro)
    rgb_smooth = np.clip(rgb_smooth, 0, 235).astype(np.uint8)

    # aplica piso dinâmico por canal (se existir)
    f = float(getattr(ctx, "dynamic_floor", 0))
    if f > 0:
        rgb_smooth = np.maximum(rgb_smooth, f).astype(np.uint8)

    ctx.to_pixels_and_show(rgb_smooth)

