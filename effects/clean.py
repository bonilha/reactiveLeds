# effects/clean.py
import numpy as np

__all__ = [
    "effect_spectral_blade",
    "effect_bass_center_bloom",
    "effect_bass_center",
    "effect_peak_dots",
    "effect_centroid_comet",
    "effect_beat_outward_burst",
    "effect_quantized_sections",
]

# ---- Spectral Blade ----
def effect_spectral_blade(ctx, bands_u8, beat_flag, active):
    arr = np.asarray(bands_u8, dtype=np.float32)
    v_raw = ctx.segment_mean_from_cumsum(arr, ctx.SEG_STARTS_HALF, ctx.SEG_ENDS_HALF)
    v = ctx.amplify_quad(v_raw.astype(np.uint16))
    v = ctx.apply_floor_vec(v, active, None)
    hue = (ctx.base_hue_offset + (ctx.I_LEFT >> 1) + (ctx.hue_seed >> 2) + (v >> 3)) % 256
    sat = np.maximum(0, ctx.base_saturation - (v >> 2)).astype(np.uint8)
    val = np.clip(v, 0, 255).astype(np.uint8)
    left_rgb = ctx.hsv_to_rgb_bytes_vec(hue.astype(np.uint8), sat, val)
    rgb = np.zeros((ctx.LED_COUNT, 3), dtype=np.uint8)
    rgb[ctx.CENTER - 1 - ctx.I_LEFT] = left_rgb
    rgb[ctx.CENTER + ctx.I_LEFT] = left_rgb
    ctx.to_pixels_and_show(rgb)

# ---- Bass Center Bloom (SIMPLE, REATIVO, POWER-AWARE) ----
_bass_radius_ema = 0.0
_beat_flash = 0.0

def effect_bass_center_bloom(ctx, bands_u8, beat_flag, active):
    """
    Centro forte + expansão radial + reatividade imediata.
    - Baseado na versão antiga (funcional)
    - Com power cap interno
    - Bloom cobre 2/3+ da fita
    """
    global _bass_radius_ema, _beat_flash

    # ----- Beat Flash (curto e forte) -----
    _beat_flash = 1.0 if beat_flag else _beat_flash * 0.82

    # ----- Energia dos graves -----
    n = len(bands_u8)
    low_n = max(8, n // 8)
    low_mean = float(np.mean(np.asarray(bands_u8[:low_n], dtype=np.float32)))

    # ----- Raio dinâmico (rápido, cobre até 90% do centro) -----
    target_radius = (low_mean / 255.0) * (ctx.CENTER * 0.95)
    _bass_radius_ema = 0.7 * _bass_radius_ema + 0.3 * target_radius
    radius = max(1.0, _bass_radius_ema)

    # ----- Perfil radial: triângulo largo + suavização -----
    dist = np.abs(ctx.I_ALL.astype(np.float32) - ctx.CENTER)
    # Triângulo largo: cai lentamente até o raio
    core = np.clip((radius * 2.2 - dist * 1.8), 0.0, 255.0)  # Ajuste: mais largo, mais suave
    # Adiciona halo gaussiano leve para glow
    sigma = radius * 0.6
    halo = np.exp(-0.5 * (dist / max(1.0, sigma))**2) * 80.0

    v_raw = core + halo
    v_raw *= (1.0 + 0.45 * _beat_flash)  # Boost forte no beat

    # =========================================================
    # POWER CAP INTERNO (respeita orçamento ANTES do RGB)
    # =========================================================
    sum_v = float(np.sum(v_raw))
    i_color_mA = (ctx.WS2812B_MA_PER_CHANNEL / 255.0) * sum_v * 3
    i_idle_mA = ctx.WS2812B_IDLE_MA_PER_LED * ctx.LED_COUNT
    i_budget_mA = ctx.CURRENT_BUDGET_A * 1000.0

    scale = 1.0
    if i_color_mA > 0 and (i_color_mA + i_idle_mA) > i_budget_mA:
        scale = (i_budget_mA - i_idle_mA) / i_color_mA
        scale = max(0.0, scale)

    v = np.clip(v_raw * scale, 0, 255).astype(np.uint8)
    v = ctx.apply_floor_vec(v, active, None)

    # ----- Cores: Hue varia com distância (graves = quente, bordas = frio) -----
    hue = (ctx.base_hue_offset + (dist.astype(np.int32) >> 2) + (ctx.hue_seed >> 2)) % 256
    sat = np.full(ctx.LED_COUNT, max(170, ctx.base_saturation), dtype=np.uint8)

    rgb = ctx.hsv_to_rgb_bytes_vec(hue.astype(np.uint8), sat, v)
    ctx.to_pixels_and_show(rgb)

# Alias para compatibilidade
def effect_bass_center(ctx, bands_u8, beat_flag, active):
    return effect_bass_center_bloom(ctx, bands_u8, beat_flag, active)

# ---- Peak Dots ----
_peak_levels = None
def effect_peak_dots(ctx, bands_u8, beat_flag, active, k=6):
    global _peak_levels
    n = len(bands_u8)
    if n == 0:
        ctx.to_pixels_and_show(np.zeros((ctx.LED_COUNT, 3), dtype=np.uint8))
        return
    if _peak_levels is None or len(_peak_levels) != n:
        _peak_levels = np.zeros(n, dtype=np.float32)
    x = np.asarray(bands_u8, dtype=np.float32)
    _peak_levels = np.maximum(_peak_levels * (0.90 if active else 0.80), x)
    k = min(k, n)
    idx = np.argpartition(_peak_levels, -k)[-k:]
    idx = idx[np.argsort(-_peak_levels[idx])]
    pos = (idx.astype(np.int32) * ctx.LED_COUNT) // n
    pos = np.clip(pos, 0, ctx.LED_COUNT - 1)
    rgb = np.zeros((ctx.LED_COUNT, 3), dtype=np.uint8)
    val = np.clip((_peak_levels[idx] * 1.3), 0, 255).astype(np.uint8)
    val = np.maximum(val, ctx.dynamic_floor).astype(np.uint8)
    hue = (ctx.base_hue_offset + (idx * 5) + (ctx.hue_seed & 0x3F)) % 256
    sat = np.full(k, ctx.base_saturation, dtype=np.uint8)
    dots = ctx.hsv_to_rgb_bytes_vec(hue.astype(np.uint8), sat, val)
    rgb[pos] = dots
    ctx.to_pixels_and_show(rgb)

# ---- Centroid Comet ----
_centroid_pos = None
_trail_buf = None
def effect_centroid_comet(ctx, bands_u8, beat_flag, active):
    global _centroid_pos, _trail_buf
    n = len(bands_u8)
    if _trail_buf is None or _trail_buf.shape[0] != ctx.LED_COUNT:
        _trail_buf = np.zeros(ctx.LED_COUNT, dtype=np.float32)
    if _centroid_pos is None:
        _centroid_pos = float(ctx.CENTER)
    if n == 0:
        ctx.to_pixels_and_show(np.zeros((ctx.LED_COUNT, 3), dtype=np.uint8)); return
    w = np.asarray(bands_u8, dtype=np.float32)
    s = float(np.sum(w)) + 1e-6
    idx = np.arange(n, dtype=np.float32)
    centroid_band = float(np.sum(idx * w) / s)
    target_pos = (centroid_band / max(1.0, n - 1.0)) * (ctx.LED_COUNT - 1)
    alpha_pos = 0.25 if active else 0.15
    _centroid_pos = (1.0 - alpha_pos) * _centroid_pos + alpha_pos * target_pos
    head_pos = int(round(_centroid_pos))
    mean_energy = float(np.mean(w))
    inj = ctx.amplify_quad(np.array([int(mean_energy)], dtype=np.uint16))[0]
    if beat_flag:
        inj = int(min(255, inj * 1.25))
    width = 6
    dist = np.abs(ctx.I_ALL - head_pos)
    head_shape = np.clip((width - dist), 0, width).astype(np.float32) / float(width)
    head_val = np.clip(head_shape * float(inj), 0, 255).astype(np.float32)
    decay = 0.90 if active else 0.80
    _trail_buf = _trail_buf * decay + head_val
    v = np.clip(_trail_buf, 0, 255).astype(np.uint8)
    v = ctx.apply_floor_vec(v, active, None)
    hue = (ctx.base_hue_offset + (head_pos >> 1) + (ctx.hue_seed >> 2)) % 256
    hue_arr = np.full(ctx.LED_COUNT, hue, dtype=np.uint8)
    sat = np.full(ctx.LED_COUNT, max(96, ctx.base_saturation - 32), dtype=np.uint8)
    rgb = ctx.hsv_to_rgb_bytes_vec(hue_arr, sat, v)
    ctx.to_pixels_and_show(rgb)

# ---- Quantized Sections ----
_QS_SECTIONS = 10
_QS_LEVELS = 8
def effect_quantized_sections(ctx, bands_u8, beat_flag, active):
    arr = np.asarray(bands_u8, dtype=np.float32)
    v_full = ctx.segment_mean_from_cumsum(arr, ctx.SEG_STARTS_FULL, ctx.SEG_ENDS_FULL)
    block = max(1, ctx.LED_COUNT // _QS_SECTIONS)
    v_led = v_full.astype(np.uint16)
    sec_idx = ctx.I_ALL // block
    sec_idx = np.clip(sec_idx, 0, _QS_SECTIONS - 1)
    sums = np.zeros(_QS_SECTIONS, dtype=np.float32)
    cnts = np.zeros(_QS_SECTIONS, dtype=np.int32)
    np.add.at(sums, sec_idx, v_led.astype(np.float32))
    np.add.at(cnts, sec_idx, 1)
    means = np.divide(sums, np.maximum(1, cnts), dtype=np.float32)
    step = 255.0 / float(max(1, _QS_LEVELS - 1))
    qvals = np.clip((np.rint(means / step) * step), 0, 255).astype(np.uint8)
    v = qvals[sec_idx]
    v = ctx.apply_floor_vec(v, active, None)
    hue = (ctx.base_hue_offset + (sec_idx * 7) + (ctx.hue_seed >> 1)) % 256
    sat = np.full(ctx.LED_COUNT, max(80, ctx.base_saturation - 20), dtype=np.uint8)
    rgb = ctx.hsv_to_rgb_bytes_vec(hue.astype(np.uint8), sat, v.astype(np.uint8))
    ctx.to_pixels_and_show(rgb)

# ==== BASS PULSE CORE — NOVO EFEITO REATIVO ====
_pulse_env = 0.0
_pulse_trans = 0.0
_ripple_buf = None
_beat_punch = 0.0

def effect_bass_pulse_core(ctx, bands_u8, beat_flag, active):
    """
    Núcleo pulsante + ondas radiais.
    - Centro: reatividade imediata ao grave
    - Ondas: expansão suave até as pontas
    - Power-aware, sem saturação
    """
    global _pulse_env, _pulse_trans, _ripple_buf, _beat_punch
    import numpy as np

    L = ctx.LED_COUNT
    if _ripple_buf is None or _ripple_buf.shape[0] != L:
        _ripple_buf = np.zeros(L, dtype=np.float32)

    # === 1. ENERGIA DOS GRAVES (rápida) ===
    n = len(bands_u8)
    low_n = max(8, n // 8)
    low = np.asarray(bands_u8[:low_n], dtype=np.float32)
    low_mean = float(np.mean(low))

    # EMA rápida + transiente
    _pulse_env = 0.55 * _pulse_env + 0.45 * low_mean
    _pulse_trans = max(0.0, low_mean - _pulse_env) * 1.4  # ataque forte

    # Beat punch (curto)
    _beat_punch = 1.0 if beat_flag else _beat_punch * 0.75

    # === 2. VALOR DO CENTRO (máximo reativo) ===
    center_val = _pulse_env * 0.8 + _pulse_trans * 1.2 + _beat_punch * 180
    center_val = np.clip(center_val, 0.0, 255.0)

    # === 3. ONDAS RADIAIS (3 camadas) ===
    d = np.abs(ctx.I_ALL.astype(np.float32) - ctx.CENTER)

    # Core: triângulo largo, escala com center_val
    core_w = ctx.CENTER * 0.4 * (1.0 + 0.6 * (center_val / 255.0))
    core = np.clip(1.0 - d / max(1.0, core_w), 0.0, 1.0) * center_val

    # Halo: gaussiano, expande com energia
    sigma = max(8.0, ctx.CENTER * 0.35 * (center_val / 255.0))
    halo = np.exp(-0.5 * (d / max(1.0, sigma))**2) * (center_val * 0.7)

    # Ripple: onda em movimento (decay lento)
    speed = 1.8 + 3.0 * (center_val / 255.0)
    phase = ctx.I_ALL * 0.05
    ripple = np.sin(phase - (center_val / 50.0)) * 30.0
    ripple = np.clip(ripple, 0.0, 80.0) * (center_val / 255.0)

    # === 4. COMPOSIÇÃO BRUTA ===
    v_raw = core + halo + ripple
    v_raw = np.clip(v_raw, 0.0, 255.0)

    # === 5. POWER CAP INTERNO (respeita orçamento) ===
    sum_v = float(np.sum(v_raw))
    i_color_mA = (ctx.WS2812B_MA_PER_CHANNEL / 255.0) * sum_v * 3
    i_idle_mA = ctx.WS2812B_IDLE_MA_PER_LED * L
    i_budget_mA = ctx.CURRENT_BUDGET_A * 1000.0

    scale = 1.0
    if i_color_mA > 0 and (i_color_mA + i_idle_mA) > i_budget_mA:
        scale = max(0.0, (i_budget_mA - i_idle_mA) / i_color_mA)

    v = np.clip(v_raw * scale, 0, 255).astype(np.uint8)
    v = ctx.apply_floor_vec(v, active, None)

    # === 6. CORES DINÂMICAS ===
    hue_center = (ctx.base_hue_offset + ctx.hue_seed) % 256
    hue_outer = (hue_center + 80) % 256
    hue = (hue_center + (d.astype(np.int32) * 80) // ctx.CENTER) % 256
    sat = np.full(L, max(190, ctx.base_saturation), dtype=np.uint8)

    rgb = ctx.hsv_to_rgb_bytes_vec(hue.astype(np.uint8), sat, v)
    ctx.to_pixels_and_show(rgb)

# ==== PEAK DOTS EXPANDED — USA TODA A FITA ====
_pdex_levels = None  # picos por banda (EMA)

def effect_peak_dots_expanded(ctx, bands_u8, beat_flag, active):
    """
    V3:
    - Mais reatividade em graves (peso espectral + componente transiente).
    - AGC pré-cap: reduz leito/ganho quando a corrente estimada se aproxima do orçamento.
    - Soft-clip nos acúmulos para reduzir brilho médio sem perder impacto.
    - Respeita paletas, dynamic_floor e cap global do FXContext.
    """
    import numpy as np

    L = ctx.LED_COUNT
    n = int(len(bands_u8))
    if L == 0 or n == 0:
        ctx.to_pixels_and_show(np.zeros((L, 3), dtype=np.uint8))
        return

    # -----------------------------
    # 1) Entrada e ênfase de graves
    # -----------------------------
    x = np.asarray(bands_u8, dtype=np.float32)  # 0..255
    # Curva de peso 0..1 (mais peso nos graves, menos nos agudos)
    f = np.arange(n, dtype=np.float32) / max(1, n - 1)
    w = 1.50 - 0.65 * (f ** 0.7)            # ~1.50 nos graves -> ~0.85 nos agudos
    x_w = x * w
    # Normaliza para não inflar energia total
    mean_x = float(np.mean(x)) + 1e-6
    mean_w = float(np.mean(x_w)) + 1e-6
    x_w *= (mean_x / mean_w)

    # ----------------------------------
    # 2) EMA de picos + componente "trans"
    # ----------------------------------
    global _pdex_levels
    if _pdex_levels is None or _pdex_levels.shape[0] != n:
        _pdex_levels = np.zeros(n, dtype=np.float32)

    prev = _pdex_levels.copy()
    # Decay mais rápido para responder melhor (especialmente samba)
    decay = 0.80 if active else 0.70
    _pdex_levels = np.maximum(_pdex_levels * decay, x_w)

    # Transiente: ataque imediato quando x_w sobe acima do nível anterior
    trans = np.maximum(0.0, x_w - prev)   # 0..255

    # Métricas de energia (global e graves)
    low_n = max(8, n // 6)                # pega um pouco mais de "baixo"
    low_mean = float(np.mean(x_w[:low_n]))
    low_energy = np.clip(low_mean / 255.0, 0.0, 1.0)
    energy = np.clip(float(np.mean(_pdex_levels)) / 255.0, 0.0, 1.0)

    # -----------------------------
    # 3) Seleção de picos (dinâmica)
    # -----------------------------
    k_base, k_max = 8, 22
    k = int(k_base + (k_max - k_base) * (energy ** 0.7))
    if low_energy > 0.35:
        k += 1
    k = max(8, min(k, n, max(6, L // 6)))

    if k >= n:
        idx = np.arange(n, dtype=np.int32)
    else:
        idx = np.argpartition(_pdex_levels, -k)[-k:]
        idx = idx[np.argsort(-_pdex_levels[idx])]

    # Valores de pico + transiente com boost de grave
    peak_vals = _pdex_levels[idx]
    trans_vals = trans[idx]
    bass_boost = 1.0 + 0.75 * (low_energy ** 0.9)       # mais presença no grave
    if beat_flag:
        bass_boost *= 1.12
    gain = (1.20 + 0.30 * energy) * bass_boost
    # mistura pico + trans, o trans reage MUITO rápido
    val_peaks = np.clip(peak_vals * gain + trans_vals * (0.9 + 0.6 * low_energy), 0.0, 255.0)

    # -----------------------------
    # 4) Mapeamento + espelhamento
    # -----------------------------
    band_norm = idx.astype(np.float32) / max(1, n - 1)
    led_norm = 0.10 + 0.80 * np.sqrt(band_norm)         # empurra mais p/ longe
    pos = np.clip((led_norm * (L - 1)).astype(np.int32), 0, L - 1)
    pos_mirror = (L - 1 - pos).astype(np.int32)
    pos_all = np.concatenate([pos, pos_mirror], axis=0)
    val_all = np.concatenate([val_peaks, val_peaks], axis=0)

    # ------------------------------------
    # 5) Glow gaussiano (com soft clip)
    # ------------------------------------
    # sigma fecha com grave alto (mais contundente)
    sigma = (1.8 + 3.0 * (val_all / 255.0) + 4.0 * energy) * (1.0 - 0.25 * low_energy)
    sigma = np.maximum(sigma, 1.0)

    I = ctx.I_ALL.astype(np.float32)[None, :]  # (1,L)
    P = pos_all[:, None].astype(np.float32)    # (p,1)
    S = sigma[:, None].astype(np.float32)
    V = val_all[:, None].astype(np.float32)
    glow = np.exp(-0.5 * ((I - P) / S) ** 2) * V
    glow_sum = np.sum(glow, axis=0)  # (L,)

    # Soft-clip do somatório de glows para reduzir picos de brilho
    # (curva ~exponencial; 180 controla a "dureza")
    glow_sum = 255.0 * (1.0 - np.exp(-glow_sum / 180.0))

    # ------------------------------------
    # 6) Rastro + "leito" (bed) mais contido
    # ------------------------------------
    if not hasattr(ctx, "_pr_buf") or ctx._pr_buf.shape[0] != L:
        ctx._pr_buf = np.zeros(L, dtype=np.float32)
    if not hasattr(ctx, "_pr_phase"):
        ctx._pr_phase = 0.0

    # Rastro: ligeiro viés para persistir quando grave está alto
    base_decay = 0.88 if active else 0.80
    ctx._pr_buf *= (base_decay + 0.02 * low_energy)
    ctx._pr_buf = np.maximum(ctx._pr_buf, glow_sum)

    # Leito animado: menor que antes, mas ainda cobre fita
    ctx._pr_phase += (0.10 + 0.22 * energy + 0.22 * low_energy + (0.10 if beat_flag else 0.0))
    tpos = (ctx.I_ALL.astype(np.float32) / max(1.0, float(L - 1)))
    bed_level = (energy ** 0.7) * 45.0 + (low_energy ** 0.9) * 35.0  # ↓ valores base
    bed = (0.55 + 0.45 * np.sin(2.0 * np.pi * (tpos + (ctx._pr_phase * 0.05)))) * bed_level

    # Pré-combinação (antes de AGC)
    v_pre = ctx._pr_buf + bed  # float

    # ------------------------------------
    # 7) AGC PRÉ-CAP (cap-aware de verdade)
    #    - estima corrente como no FXContext e cria folga (~85%)
    # ------------------------------------
    ma_per_ch = float(ctx.WS2812B_MA_PER_CHANNEL)  # ~20 mA
    idle_ma    = float(ctx.WS2812B_IDLE_MA_PER_LED) * float(L)
    budget_ma  = float(ctx.CURRENT_BUDGET_A) * 1000.0
    # estima consumo colorido (3 canais), antes de converter para RGB de fato
    sum_v      = float(np.sum(np.clip(v_pre, 0.0, 255.0)))
    i_color_ma = (ma_per_ch / 255.0) * sum_v * 3.0
    target_ma  = max(0.0, (budget_ma - idle_ma) * 0.85)  # 15% de folga

    pre_scale_raw = 1.0 if i_color_ma <= 1e-6 else min(1.0, target_ma / i_color_ma)

    # EMA do ganho para estabilidade entre frames
    if not hasattr(ctx, "_agc_ema"):
        ctx._agc_ema = 1.0
    ctx._agc_ema = 0.70 * float(ctx._agc_ema) + 0.30 * float(pre_scale_raw)
    pre_scale = min(pre_scale_raw, ctx._agc_ema)

    # Reduz o bed agressivamente quando falta folga; mantém picos
    bed_agc = bed * (0.55 * pre_scale + 0.25)  # cai rápido se pre_scale<1
    trail_agc = ctx._pr_buf * (0.92 * pre_scale + 0.08)
    v_raw = np.clip(trail_agc + bed_agc, 0.0, 255.0).astype(np.uint8)

    # Piso dinâmico
    v = ctx.apply_floor_vec(v_raw, active, None)

    # ------------------------------------
    # 8) Cores (paleta → HSV fallback)
    # ------------------------------------
    pal = getattr(ctx, "current_palette", None)
    if isinstance(pal, (list, tuple)) and len(pal) >= 2:
        pal_arr = np.asarray(pal, dtype=np.float32)
        m = pal_arr.shape[0]
        tpal = (tpos * (m - 1)).astype(np.float32)
        j = np.clip(np.floor(tpal).astype(np.int32), 0, m - 1)
        j2 = np.clip(j + 1, 0, m - 1)
        frac = (tpal - j).reshape(-1, 1)
        base_rgb = (pal_arr[j] * (1.0 - frac) + pal_arr[j2] * frac)
        rgb = np.clip(base_rgb * (v[:, None].astype(np.float32) / 255.0), 0, 255)
        if beat_flag:
            # brilhozinho proporcional ao grave, mas já sob AGC
            rgb = np.clip(rgb + (v[:, None].astype(np.float32) * (0.08 + 0.16 * low_energy)), 0, 255)
        rgb = rgb.astype(np.uint8)
    else:
        hue_base = (int(ctx.base_hue_offset) + (int(ctx.hue_seed) >> 1)) % 256
        hue = (hue_base + ((ctx.I_ALL * 96) // max(1, L)) + int((ctx._pr_phase * 10.0))) % 256
        sat = np.full(L, max(150, int(ctx.base_saturation)), dtype=np.uint8)  # sat um pouco menor
        rgb = ctx.hsv_to_rgb_bytes_vec(hue.astype(np.uint8), sat, v)

    # Envia (cap global do FXContext ainda atua se necessário)
    ctx.to_pixels_and_show(rgb)

# ==== CENTROID COMET EXPANDED — CORRIGIDO (sem NameError) ====
_comet_state = {
    "pos_ema": 0.0,
    "vel_ema": 0.0,
    "trail_buf": None,
    "last_energy": 0.0
}

def effect_centroid_comet_expanded(ctx, bands_u8, beat_flag, active):
    """
    Cometa que segue o centroide do espectro com trail expansivo.
    - CORRIGIDO: NameError em WS281B_IDLE_MA_PER_LED
    - Reativo, cobre toda a fita, power-aware
    """
    global _comet_state
    import numpy as np

    st = _comet_state
    L = ctx.LED_COUNT
    if st["trail_buf"] is None or st["trail_buf"].shape[0] != L:
        st["trail_buf"] = np.zeros(L, dtype=np.float32)

    n = len(bands_u8)
    if n == 0 or L == 0:
        ctx.to_pixels_and_show(np.zeros((L, 3), dtype=np.uint8))
        return

    x = np.asarray(bands_u8, dtype=np.float32)
    if not active:
        x[:] = 0.0

    # --- Energia total ---
    energy = float(np.mean(x)) / 255.0
    energy = np.clip(energy, 0.0, 1.0)

    # --- Centroide (posição do cometa) ---
    total = float(np.sum(x))
    if total <= 0:
        target_pos = float(L // 2)
    else:
        weights = x / total
        band_positions = np.arange(n, dtype=np.float32)
        centroid_band = float(np.sum(weights * band_positions))
        target_pos = (centroid_band / max(1, n - 1)) * (L - 1)

    # EMA suave para posição
    st["pos_ema"] = 0.6 * st["pos_ema"] + 0.4 * target_pos
    pos = float(np.clip(st["pos_ema"], 0.0, L - 1))

    # --- Velocidade (para trail) ---
    vel = abs(target_pos - pos) * 10.0
    st["vel_ema"] = 0.7 * st["vel_ema"] + 0.3 * vel

    # --- Beat punch ---
    beat_punch = 1.0 if beat_flag else 0.0
    if beat_punch > 0:
        st["vel_ema"] *= 1.5

    # --- Trail gaussiano expansivo ---
    trail = st["trail_buf"]
    idxs = ctx.I_ALL.astype(np.float32)

    sigma_base = 8.0 + 20.0 * energy
    sigma_vel = 2.0 + 8.0 * (st["vel_ema"] / 100.0)
    sigma = sigma_base + sigma_vel
    if beat_punch > 0:
        sigma *= 1.4

    dist = np.abs(idxs - pos)
    glow = np.exp(-0.5 * (dist / max(1.0, sigma))**2)

    comet_val = 180.0 + 75.0 * energy
    if beat_punch > 0:
        comet_val = min(255.0, comet_val * 1.6)

    trail[:] = glow * comet_val
    trail = np.clip(trail, 0.0, 255.0)

    # --- POWER CAP CORRIGIDO ---
    sum_v = float(np.sum(trail))
    i_color_mA = (ctx.WS2812B_MA_PER_CHANNEL / 255.0) * sum_v * 3
    i_idle_mA = ctx.WS2812B_IDLE_MA_PER_LED * L  # <<< CORRIGIDO
    i_budget_mA = ctx.CURRENT_BUDGET_A * 1000.0

    scale = 1.0
    if i_color_mA > 0 and (i_color_mA + i_idle_mA) > i_budget_mA:
        scale = max(0.0, (i_budget_mA - i_idle_mA) / i_color_mA)

    v = np.clip(trail * scale, 0, 255).astype(np.uint8)
    v = ctx.apply_floor_vec(v, active, None)

    # --- Cor: varia com posição ---
    hue_center = (ctx.base_hue_offset + ctx.hue_seed) % 256
    hue = (hue_center + (dist.astype(np.int32) * 60) // max(1, int(sigma))) % 256
    sat = np.full(L, max(200, ctx.base_saturation), dtype=np.uint8)

    rgb = ctx.hsv_to_rgb_bytes_vec(hue.astype(np.uint8), sat, v)
    ctx.to_pixels_and_show(rgb)


# ==== BEAT OUTWARD BURST — MELHORADO COM PALETAS ====
_burst_state = {
    "phase": 0.0,
    "speed_ema": 0.0,
    "energy_ema": 0.0,
    "wave_buf": None
}

# ==== BEAT OUTWARD BURST — BRUTAL, RÁPIDO, EXPLOSIVO ====
_burst_state = {
    "radius": 0.0,
    "last_beat": 0.0,
    "flash": 0.0
}

def effect_beat_outward_burst(ctx, bands_u8, beat_flag, active):
    """
    EXPLOSÃO NUCLEAR NO BEAT.
    - Onda de choque varre a fita em <0.3s
    - Cores da paleta + flash branco
    - Brilho máximo, power-aware
    """
    global _burst_state
    import numpy as np, time

    st = _burst_state
    L = ctx.LED_COUNT
    now = time.time()

    # --- Energia dos graves ---
    n = len(bands_u8)
    low_n = max(8, n // 8)
    low_mean = float(np.mean(np.asarray(bands_u8[:low_n], dtype=np.float32))) / 255.0

    # --- BEAT TRIGGER ---
    if beat_flag:
        st["radius"] = 0.0
        st["flash"] = 1.0
        st["last_beat"] = now

    # --- Flash decay (curto) ---
    if st["flash"] > 0:
        st["flash"] *= 0.7
        if st["flash"] < 0.01:
            st["flash"] = 0.0

    # --- Velocidade explosiva ---
    speed = 800.0 + 600.0 * low_mean  # 800 a 1400 px/s
    dt = min(0.05, now - getattr(st, "last_t", now))
    st["last_t"] = now
    st["radius"] += speed * dt

    # --- Onda: quadrada + gaussiana ---
    d = np.abs(ctx.I_ALL.astype(np.float32) - ctx.CENTER)
    radius = st["radius"]

    # Onda principal (quadrada)
    ring = np.where(np.abs(d - radius) < (20 + 40 * low_mean), 1.0, 0.0)

    # Halo gaussiano
    sigma = 15 + 30 * low_mean
    halo = np.exp(-0.5 * ((d - radius) / max(1.0, sigma))**2)

    # Intensidade
    intensity = (ring * 1.0 + halo * 0.6) * (200 + 55 * low_mean)
    intensity *= (1.0 + st["flash"] * 2.0)  # FLASH BRANCO NO BEAT

    v_raw = np.clip(intensity, 0, 255)

    # --- POWER CAP ---
    sum_v = float(np.sum(v_raw))
    i_color_mA = (ctx.WS2812B_MA_PER_CHANNEL / 255.0) * sum_v * 3
    i_idle_mA = ctx.WS2812B_IDLE_MA_PER_LED * L
    i_budget_mA = ctx.CURRENT_BUDGET_A * 1000.0

    scale = 1.0
    if i_color_mA > 0 and (i_color_mA + i_idle_mA) > i_budget_mA:
        scale = max(0.0, (i_budget_mA - i_idle_mA) / i_color_mA)

    v = np.clip(v_raw * scale, 0, 255).astype(np.uint8)
    v = ctx.apply_floor_vec(v, active, None)

    # --- CORES: PALETA + FLASH BRANCO ---
    pal = getattr(ctx, "current_palette", None)
    if isinstance(pal, (list, tuple)) and len(pal) >= 2:
        pal_arr = np.asarray(pal, dtype=np.uint8)
        m = pal_arr.shape[0]
        idx = int((radius * 0.05) % m)
        base_rgb = pal_arr[idx % m].astype(np.float32)
        colored = base_rgb * (v.astype(np.float32) / 255.0)[:, None]
        white = v.astype(np.float32)[:, None]
        rgb = np.clip(colored + white * st["flash"], 0, 255).astype(np.uint8)
    else:
        hue = (ctx.base_hue_offset + int(radius * 0.5)) % 256
        sat = np.full(L, 255, dtype=np.uint8)
        rgb = ctx.hsv_to_rgb_bytes_vec(
            np.full(L, hue, dtype=np.uint8),
            sat, v
        )

    ctx.to_pixels_and_show(rgb)


# ==== BASS IMPACT WAVE — REATIVO, SÓ NO BEAT, CENTRO APAGADO ====
_impact_state = {
    "radius": 0.0,
    "active": False,
    "last_t": 0.0
}

def effect_bass_impact_wave(ctx, bands_u8, beat_flag, active):
    """
    BEAT IMPACT WAVE (robusto, sem apagões, sem KeyError):
      - Estado armazenado em atributos do ctx (sem dicionários)
      - Bed mínimo + rastro com decaimento (nunca fica 100% preto)
      - Reativo: no beat injeta anel + halo e expande rápido
      - Sem cap local (deixe o cap global do FXContext limitar corretamente)
    """
    import numpy as np, time

    L = ctx.LED_COUNT

    # -------- Inicialização de estado no ctx (sem dict, sem KeyError) --------
    if not hasattr(ctx, "_impact_buf") or ctx._impact_buf.shape[0] != L:
        ctx._impact_buf = np.zeros(L, dtype=np.float32)
        ctx._impact_radius = float(L) * 2.0   # começa "fora" da fita
        ctx._impact_last_t = 0.0

    buf = ctx._impact_buf
    now = time.time()
    dt = 0.0 if ctx._impact_last_t == 0.0 else min(0.05, now - ctx._impact_last_t)
    ctx._impact_last_t = now

    # -------- Rastro + bed mínimo (evita apagões) --------
    decay = 0.90 if active else 0.88
    buf *= decay

    bed_val = max(int(ctx.dynamic_floor), 6)  # ajuste se quiser mais escuro/claro

    # -------- Energia nos graves => controla velocidade/largura --------
    n = len(bands_u8)
    if n > 0:
        low_n = max(8, n // 8)
        low = np.asarray(bands_u8[:low_n], dtype=np.float32)
        e = float(np.mean(low)) / 255.0
    else:
        e = 0.0
    e = np.clip(e, 0.0, 1.0)

    # -------- Trigger no beat --------
    if beat_flag:
        ctx._impact_radius = 0.0

    # Avanço do raio mesmo sem beat (onda contínua)
    speed = 1200.0 + 600.0 * e  # px/s
    ctx._impact_radius += speed * dt
    radius = ctx._impact_radius

    # -------- Perfil da onda: anel fino + halo --------
    d = np.abs(ctx.I_ALL.astype(np.float32) - ctx.CENTER)
    ring_width = 22.0 + 18.0 * e
    sigma = 14.0 + 24.0 * e

    ring = np.clip(1.0 - np.abs(d - radius) / max(1.0, ring_width), 0.0, 1.0)
    halo = np.exp(-0.5 * ((d - radius) / max(1.0, sigma))**2)

    impact = (ring + 0.6 * halo) * (190.0 + 65.0 * e)

    # Acumula no buffer de rastro (sem criar arrays novos)
    np.maximum(buf, impact, out=buf)

    # -------- V final: rastro + bed, sem cap local (cap global no FXContext) --------
    v = np.clip(buf, 0.0, 255.0).astype(np.uint8)
    if bed_val > 0:
        v = np.maximum(v, bed_val).astype(np.uint8)
    v = ctx.apply_floor_vec(v, active, None)

    # -------- Cores: usa paleta se houver; senão HSV --------
    pal = getattr(ctx, "current_palette", None)
    if isinstance(pal, (list, tuple)):
        pal_arr = np.asarray(pal, dtype=np.uint8)
        if pal_arr.ndim == 2 and pal_arr.shape[0] > 0:
            m = pal_arr.shape[0]
            idx = int(radius * 0.08) % m if m > 0 else 0
            base_rgb = pal_arr[idx].astype(np.float32)  # (R,G,B)
            rgb = np.clip((v.astype(np.float32)[:, None] / 255.0) * base_rgb[None, :], 0, 255).astype(np.uint8)
        else:
            # fallback seguro se paleta vier vazia/irregular
            hue = (ctx.base_hue_offset + (ctx.hue_seed >> 2) + (ctx.I_ALL >> 1) + int(radius * 0.3)) % 256
            sat = np.full(L, max(180, ctx.base_saturation), dtype=np.uint8)
            rgb = ctx.hsv_to_rgb_bytes_vec(hue.astype(np.uint8), sat, v)
    else:
        hue = (ctx.base_hue_offset + (ctx.hue_seed >> 2) + (ctx.I_ALL >> 1) + int(radius * 0.3)) % 256
        sat = np.full(L, max(180, ctx.base_saturation), dtype=np.uint8)
        rgb = ctx.hsv_to_rgb_bytes_vec(hue.astype(np.uint8), sat, v)

    # Envia (o FXContext aplicará o cap global já considerando o brightness)
    ctx.to_pixels_and_show(rgb)


# ==== WAVELENGTH REACTIVE (1D) — compatível com paletas, floor e cap global ====
def effect_wavelength_reactive(ctx, bands_u8, beat_flag, active):
    """
    Gradiente que percorre a fita com velocidade/fase guiadas pelo áudio.
    - Velocidade: energia total (com ênfase nos graves) + kick no beat
    - Fase/Hue: centroide espectral e seed de cor
    - Paleta se ctx.current_palette existir; senão HSV
    - AGC pré-cap leve para reduzir consumo médio (folga antes do cap global)
    - Zero imports extras; tudo vetorizado
    """
    import numpy as np

    L = ctx.LED_COUNT
    n = int(len(bands_u8))
    if L == 0 or n == 0:
        ctx.to_pixels_and_show(np.zeros((L, 3), dtype=np.uint8))
        return

    # ---------- 1) Medidas de áudio (ênfase em graves + centroide) ----------
    x = np.asarray(bands_u8, dtype=np.float32)  # 0..255
    # pesos pró-grave (de 1.35 no grave até ~0.9 no agudo)
    f = np.arange(n, dtype=np.float32) / max(1, n - 1)
    w = 1.35 - 0.45 * (f ** 0.8)
    xw = x * w
    # normaliza para não inflar energia média
    xw *= (np.mean(x) + 1e-6) / (np.mean(xw) + 1e-6)

    energy = float(np.mean(xw)) / 255.0  # 0..1
    energy = np.clip(energy, 0.0, 1.0)
    low_n = max(8, n // 6)
    low_energy = float(np.mean(xw[:low_n])) / 255.0
    low_energy = np.clip(low_energy, 0.0, 1.0)

    # centroide (0..1)
    s = float(np.sum(xw)) + 1e-6
    centroid_band = float(np.sum(np.arange(n, dtype=np.float32) * xw)) / s
    centroid = centroid_band / max(1.0, n - 1.0)  # 0..1

    # ---------- 2) Estado persistente (fase, agc) ----------
    if not hasattr(ctx, "_wl_phase"):
        ctx._wl_phase = 0.0
    if not hasattr(ctx, "_wl_agc_ema"):
        ctx._wl_agc_ema = 1.0

    # ---------- 3) Velocidade da onda (frames são uniformes no sync) ----------
    # base + ganho por energia + ganho por graves + kick no beat
    speed = 0.35 + 0.70 * energy + 0.55 * low_energy
    if beat_flag:
        speed += 0.35
    # direcionalidade sutil pelo centroide (mais agudo → um pouco mais rápido)
    speed *= (0.90 + 0.20 * centroid)
    ctx._wl_phase = (ctx._wl_phase + speed) % (L * 4.0)

    # ---------- 4) Envelope de brilho (Value), antes de cor ----------
    # leito baixo para manter fita cheia + ondulação leve pela fase
    t = ctx.I_ALL.astype(np.float32) / max(1.0, float(L - 1))
    bed_level = (energy ** 0.7) * 70.0 + (low_energy ** 0.9) * 55.0  # mais grave → mais bed
    ripple = 0.25 + 0.75 * np.sin(2.0 * np.pi * (t + ctx._wl_phase * 0.004 + centroid * 0.15))
    v_pre = np.clip(bed_level * (0.45 + 0.55 * ripple), 0.0, 255.0)  # float 0..255

    # ---------- 5) AGC PRÉ-CAP (leve) ----------
    # estima corrente = (mA/canal / 255) * sum(v) * 3 + idle
    ma_per_ch = float(ctx.WS2812B_MA_PER_CHANNEL)  # ~20
    idle_ma   = float(ctx.WS2812B_IDLE_MA_PER_LED) * float(L)
    budget_ma = float(ctx.CURRENT_BUDGET_A) * 1000.0
    target_ma = max(0.0, (budget_ma - idle_ma) * 0.84)  # ~16% de folga
    i_color_ma = (ma_per_ch / 255.0) * float(np.sum(v_pre)) * 3.0
    pre_scale_raw = 1.0 if i_color_ma <= 1e-6 else min(1.0, target_ma / i_color_ma)
    # estabiliza entre frames
    ctx._wl_agc_ema = 0.70 * ctx._wl_agc_ema + 0.30 * pre_scale_raw
    pre_scale = min(pre_scale_raw, ctx._wl_agc_ema)
    # reduz um pouco mais se o cap global já vinha atuando
    if ctx.last_cap_scale is not None and ctx.last_cap_scale < 0.92:
        pre_scale *= (0.92 + 0.08 * ctx.last_cap_scale)  # 0.85→~0.93

    v_raw = np.clip(v_pre * pre_scale, 0.0, 255.0).astype(np.uint8)
    v = ctx.apply_floor_vec(v_raw, active, None)

    # ---------- 6) Cor: paleta (se houver) → HSV fallback ----------
    pal = getattr(ctx, "current_palette", None)
    if isinstance(pal, (list, tuple)) and len(pal) >= 2:
        pal_arr = np.asarray(pal, dtype=np.float32)   # (m,3)
        m = pal_arr.shape[0]
        # mapeia gradiente da paleta ao longo da fita com fase móvel
        # índice pal = (posição + fase normalizada + leve “tilt” pelo centroide)
        pal_phase = (ctx._wl_phase * 0.003 + centroid * 0.35) % 1.0
        g = (t + pal_phase) % 1.0
        g *= (m - 1)
        j = np.clip(np.floor(g).astype(np.int32), 0, m - 1)
        j2 = np.clip(j + 1, 0, m - 1)
        frac = (g - j).reshape(-1, 1)
        base_rgb = pal_arr[j] * (1.0 - frac) + pal_arr[j2] * frac  # (L,3)
        # aplica envelope v (Value)
        rgb = np.clip(base_rgb * (v[:, None].astype(np.float32) / 255.0), 0, 255).astype(np.uint8)
    else:
        # HSV contínuo, hue avança com a fase e levemente com o centroide
        hue_base = (int(ctx.base_hue_offset) + (int(ctx.hue_seed) >> 2)) % 256
        hue_shift = ((ctx._wl_phase * 0.6) + centroid * 96.0) % 256
        hue = (hue_base + ((t * 160.0).astype(np.int32)) + int(hue_shift)) % 256
        sat = np.full(L, max(170, int(ctx.base_saturation)), dtype=np.uint8)
        rgb = ctx.hsv_to_rgb_bytes_vec(hue.astype(np.uint8), sat, v)

    # ---------- 7) Envio (cap global do FXContext ainda protege) ----------
    ctx.to_pixels_and_show(rgb)

# ==== FIM DO ARQUIVO effects/clean.py ====