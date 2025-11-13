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

# ==== WAVELENGTH REACTIVE — FAST/AGGRESSIVE (alta reatividade em graves) ====
def effect_wavelength_reactive(ctx, bands_u8, beat_flag, active):
    """
    Wavelength 1D super reativa:
      - Velocidade e brilho guiados por energia (com ênfase nos graves) + transiente (ataque)
      - Kick forte no beat
      - Paleta (se ctx.current_palette) ou HSV
      - AGC pré-cap rápido para manter folga antes do cap global

    Knobs principais (ajuste fino rápido):
      S0..S3, S_BEAT  -> aceleram/desaceleram a onda
      BED_E, BED_L    -> brilho médio do "leito" (energia e graves)
      SPARK_GAIN      -> brilho do flash por transiente
      F_TARGET        -> folga do AGC pré-cap (menor = mais folga, menor brilho médio)
    """
    import numpy as np

    L = ctx.LED_COUNT
    n = int(len(bands_u8))
    if L == 0 or n == 0:
        ctx.to_pixels_and_show(np.zeros((L, 3), dtype=np.uint8))
        return

    # ---------------- Knobs (pode ajustar sem medo) ----------------
    # Velocidade (quanto maior, mais rápida a onda)
    S0, S1, S2, S3, S_BEAT = 0.60, 1.10, 0.90, 1.60, 0.70
    # Brilho base do leito (proporcional à energia e aos graves)
    BED_E, BED_L = 58.0, 48.0
    # Brilho do "spark" por transiente (quanto maior, mais flash rápido)
    SPARK_GAIN = 220.0
    # Folga alvo do AGC pré-cap (ex.: 0.82 = usa 82% do orçamento de cor)
    F_TARGET = 0.82

    # ---------------- 1) Medidas de áudio (ênfase + centroide) ----------------
    x = np.asarray(bands_u8, dtype=np.float32)  # 0..255
    f = np.arange(n, dtype=np.float32) / max(1, n - 1)
    # ênfase pró-grave (1.40 no grave → ~0.9 no agudo)
    w = 1.40 - 0.50 * (f ** 0.8)
    xw = x * w
    xw *= (np.mean(x) + 1e-6) / (np.mean(xw) + 1e-6)

    energy = np.clip(float(np.mean(xw)) / 255.0, 0.0, 1.0)
    low_n = max(8, n // 6)
    low_energy = np.clip(float(np.mean(xw[:low_n])) / 255.0, 0.0, 1.0)

    s = float(np.sum(xw)) + 1e-6
    centroid_band = float(np.sum(np.arange(n, dtype=np.float32) * xw)) / s
    centroid = centroid_band / max(1.0, n - 1.0)  # 0..1

    # ---------------- 2) Estado persistente (fase + envelopes p/ transiente) ----------------
    if not hasattr(ctx, "_wl_phase"):
        ctx._wl_phase = 0.0
    if not hasattr(ctx, "_wl_e_fast"):
        ctx._wl_e_fast = energy
    if not hasattr(ctx, "_wl_e_slow"):
        ctx._wl_e_slow = energy
    if not hasattr(ctx, "_wl_agc_ema"):
        ctx._wl_agc_ema = 1.0

    # Envelopes: rápido e lento (transiente = ataque)
    ctx._wl_e_fast = 0.55 * ctx._wl_e_fast + 0.45 * energy
    ctx._wl_e_slow = 0.85 * ctx._wl_e_slow + 0.15 * energy
    trans = max(0.0, ctx._wl_e_fast - ctx._wl_e_slow)  # 0..1

    # ---------------- 3) Velocidade MUITO mais responsiva ----------------
    speed = (
        S0
        + S1 * energy
        + S2 * low_energy
        + S3 * trans
        + (S_BEAT if beat_flag else 0.0)
    )
    # pequeno viés por centroide (mais agudo → um pouco mais rápido)
    speed *= (0.92 + 0.18 * centroid)
    # avança fase; fase é usada como deslocamento de gradiente e ondulação
    ctx._wl_phase = (ctx._wl_phase + speed) % (L * 8.0)

    # ---------------- 4) Envelope de brilho (Value) ----------------
    t = ctx.I_ALL.astype(np.float32) / max(1.0, float(L - 1))
    # leito: depende de energia e graves (menos que versões antigas → menos consumo)
    bed_level = (energy ** 0.7) * BED_E + (low_energy ** 0.9) * BED_L
    # ondulação do leito (1 ciclo) para não ficar “chapado”
    bed = (0.50 + 0.50 * np.sin(2.0 * np.pi * (t + ctx._wl_phase * 0.004))) * bed_level

    # "spark" por transiente: padrão de alta frequência (3 ciclos) + forte com beat
    spark_wave = 0.5 + 0.5 * np.sin(2.0 * np.pi * (t * 3.0 + ctx._wl_phase * 0.010 + centroid * 0.20))
    spark = SPARK_GAIN * (trans + (0.25 if beat_flag else 0.0)) * spark_wave

    v_pre = np.clip(bed + spark, 0.0, 255.0)  # float 0..255

    # ---------------- 5) AGC PRÉ-CAP (mais rápido) ----------------
    ma_per_ch = float(ctx.WS2812B_MA_PER_CHANNEL)
    idle_ma   = float(ctx.WS2812B_IDLE_MA_PER_LED) * float(L)
    budget_ma = float(ctx.CURRENT_BUDGET_A) * 1000.0
    target_ma = max(0.0, (budget_ma - idle_ma) * F_TARGET)

    i_color_ma = (ma_per_ch / 255.0) * float(np.sum(v_pre)) * 3.0
    pre_scale_raw = 1.0 if i_color_ma <= 1e-6 else min(1.0, target_ma / i_color_ma)

    # EMA mais rápido para reagir em poucos frames
    ctx._wl_agc_ema = 0.50 * ctx._wl_agc_ema + 0.50 * pre_scale_raw
    pre_scale = min(pre_scale_raw, ctx._wl_agc_ema)

    # Se o cap global já vinha atuando, diminuímos um pouco mais o leito (não o spark)
    last_cap = float(ctx.last_cap_scale or 1.0) if hasattr(ctx, "last_cap_scale") else 1.0
    bed_scale_extra = 1.0 if last_cap >= 0.95 else (0.90 + 0.10 * last_cap)  # 0.85..1.0

    # Aplica AGC: pesa mais no leito (consumo médio) e menos no spark (impacto visual)
    v_scaled = np.clip(bed * pre_scale * bed_scale_extra + spark * (0.85 * pre_scale + 0.15), 0.0, 255.0)
    v = ctx.apply_floor_vec(v_scaled.astype(np.uint8), active, None)

    # ---------------- 6) Cor: paleta (se houver) → HSV fallback ----------------
    pal = getattr(ctx, "current_palette", None)
    if isinstance(pal, (list, tuple)) and len(pal) >= 2:
        pal_arr = np.asarray(pal, dtype=np.float32)
        m = pal_arr.shape[0]
        # índice no gradiente avança rápido com a fase; leve viés pelo centroide
        pal_phase = (ctx._wl_phase * 0.004 + centroid * 0.35) % 1.0
        g = (t + pal_phase) % 1.0
        g *= (m - 1)
        j = np.clip(np.floor(g).astype(np.int32), 0, m - 1)
        j2 = np.clip(j + 1, 0, m - 1)
        frac = (g - j).reshape(-1, 1)
        base_rgb = pal_arr[j] * (1.0 - frac) + pal_arr[j2] * frac
        rgb = np.clip(base_rgb * (v[:, None].astype(np.float32) / 255.0), 0, 255).astype(np.uint8)
    else:
        hue_base = (int(ctx.base_hue_offset) + (int(ctx.hue_seed) >> 2)) % 256
        # Hue corre bastante com a fase (sensação de rapidez) + centroide
        hue_shift = int((ctx._wl_phase * 1.1) + centroid * 120.0) % 256
        hue = (hue_base + ((t * 160.0).astype(np.int32)) + hue_shift) % 256
        sat = np.full(L, max(170, int(ctx.base_saturation)), dtype=np.uint8)
        rgb = ctx.hsv_to_rgb_bytes_vec(hue.astype(np.uint8), sat, v)

    # ---------------- 7) Envio (cap global ainda protege) ----------------
    ctx.to_pixels_and_show(rgb)

# ==== STROBE REACTIVE — FAST/AGGRESSIVE (reatividade alta com consumo controlado) ====
def effect_strobe_reactive(ctx, bands_u8, beat_flag, active):
    """
    Strobe 1D super reativo para música:
      - Dispara no beat e em transientes (ataque entre envelopes rápido/lento)
      - Listras alternadas (par/ímpar) por flash para reduzir corrente instantânea
      - Leito (bed) baixo e tintado pela paleta para ocupar a fita sempre
      - AGC pré-cap rápido para manter folga antes do cap global

    Knobs principais (ajuste fino rápido):
      ON_BASE, ON_E, ON_L, ON_T, ON_BEAT  -> duração do flash (em frames)
      V_BASE, V_E, V_L, V_T, V_BEAT       -> brilho do flash
      BED_E, BED_L                        -> brilho médio do leito
      WHITE_MIX_BASE, WHITE_MIX_T, WHITE_MIX_BEAT -> quanto "branqueia" o flash
      STRIPES                             -> 2 = par/ímpar; 3 = 1/3 dos LEDs por flash
      F_TARGET                            -> fração de orçamento alvo do AGC pré-cap (0.0–1.0)
    """
    import numpy as np

    L = ctx.LED_COUNT
    n = int(len(bands_u8))
    if L == 0 or n == 0:
        ctx.to_pixels_and_show(np.zeros((L, 3), dtype=np.uint8))
        return

    # -------------------- Knobs --------------------
    # Duração do flash (frames on)
    ON_BASE, ON_E, ON_L, ON_T, ON_BEAT = 1.0, 1.4, 2.0, 2.8, 2.5
    # Brilho do flash
    V_BASE, V_E, V_L, V_T, V_BEAT = 90.0, 120.0, 140.0, 180.0, 110.0
    # Brilho do "leito" (bed) — mantém a fita ocupada
    BED_E, BED_L = 12.0, 10.0
    # Mistura de branco no flash (0..1)
    WHITE_MIX_BASE, WHITE_MIX_T, WHITE_MIX_BEAT = 0.28, 0.40, 0.22
    # Quantas faixas por flash (2 = par/ímpar; 3 = 1/3)
    STRIPES = 2
    # Folga alvo do AGC pré-cap: menor = mais folga (menos brilho médio)
    F_TARGET = 0.80

    # -------------------- 1) Métricas de áudio --------------------
    x = np.asarray(bands_u8, dtype=np.float32)  # 0..255
    f = np.arange(n, dtype=np.float32) / max(1, n - 1)
    w = 1.40 - 0.50 * (f ** 0.8)      # ênfase pró-grave
    xw = x * w
    xw *= (np.mean(x) + 1e-6) / (np.mean(xw) + 1e-6)

    energy = np.clip(float(np.mean(xw)) / 255.0, 0.0, 1.0)
    low_n = max(8, n // 6)
    low_energy = np.clip(float(np.mean(xw[:low_n])) / 255.0, 0.0, 1.0)

    # Envelopes p/ transiente
    if not hasattr(ctx, "_stb_e_fast"):
        ctx._stb_e_fast = energy
    if not hasattr(ctx, "_stb_e_slow"):
        ctx._stb_e_slow = energy
    ctx._stb_e_fast = 0.55 * ctx._stb_e_fast + 0.45 * energy
    ctx._stb_e_slow = 0.88 * ctx._stb_e_slow + 0.12 * energy
    trans = max(0.0, ctx._stb_e_fast - ctx._stb_e_slow)  # 0..1

    # Centroide (opcional para cor/variação de fase da paleta)
    s = float(np.sum(xw)) + 1e-6
    centroid = (float(np.sum(np.arange(n, dtype=np.float32) * xw)) / s) / max(1.0, n - 1.0)  # 0..1

    # -------------------- 2) Estado do strobe --------------------
    if not hasattr(ctx, "_stb_on_left"):
        ctx._stb_on_left = 0.0
    if not hasattr(ctx, "_stb_cool"):
        ctx._stb_cool = 0.0
    if not hasattr(ctx, "_stb_on_total"):
        ctx._stb_on_total = 1.0
    if not hasattr(ctx, "_stb_phase"):
        ctx._stb_phase = 0  # alterna as listras
    if not hasattr(ctx, "_stb_agc_ema"):
        ctx._stb_agc_ema = 1.0

    # Critérios de disparo: beat OU transiente forte OU grave alto
    trans_thr = 0.08
    grave_thr = 0.35
    trigger = False
    if ctx._stb_cool <= 0.0:
        if beat_flag or trans > trans_thr or (low_energy > grave_thr and trans > 0.03):
            trigger = True

    # Quando dispara, define frames ON e cooldown
    if trigger:
        on_frames = (
            ON_BASE
            + ON_E * energy
            + ON_L * low_energy
            + ON_T * trans
            + (ON_BEAT if beat_flag else 0.0)
        )
        on_frames = max(1.0, min(on_frames, 8.0))  # guarda-chuva
        ctx._stb_on_left = on_frames
        ctx._stb_on_total = on_frames
        # cooldown mínimo para evitar flicker excessivo
        ctx._stb_cool = max(2.0, 2.0 + 6.0 * energy)
        # alterna padrão de listras para dividir corrente entre frames
        ctx._stb_phase = (ctx._stb_phase + 1) % STRIPES

    # -------------------- 3) Composição de Value (v_pre) --------------------
    # Leito (sempre presente, baixo)
    t = ctx.I_ALL.astype(np.float32) / max(1.0, float(L - 1))
    bed_level = BED_E * (energy ** 0.7) + BED_L * (low_energy ** 0.9)
    bed = np.full(L, bed_level, dtype=np.float32)

    # Flash (apenas se on_left > 0)
    v_pre = bed.copy()
    if ctx._stb_on_left > 0.0:
        # Decaimento ao longo do flash: mais forte no início
        prog = 1.0 - (ctx._stb_on_left - 1.0) / max(1.0, ctx._stb_on_total)
        decay = 0.85 + 0.15 * prog  # 0.85→1.0
        v_on = (
            V_BASE
            + V_E * energy
            + V_L * low_energy
            + V_T * trans
            + (V_BEAT if beat_flag else 0.0)
        ) * decay
        v_on = np.clip(v_on, 0.0, 255.0)

        # Máscara de listras: reduz corrente instantânea
        if STRIPES <= 1:
            mask = np.ones(L, dtype=bool)
        else:
            mask = (ctx.I_ALL % STRIPES) == ctx._stb_phase

        # Eleva somente nos LEDs da máscara
        v_pre[mask] = np.maximum(v_pre[mask], v_on)

        # Anda o relógio do flash
        ctx._stb_on_left = max(0.0, ctx._stb_on_left - 1.0)
    else:
        # nenhum flash neste frame
        pass

    # Cooldown desce
    ctx._stb_cool = max(0.0, ctx._stb_cool - 1.0)

    # -------------------- 4) AGC PRÉ-CAP (rápido) --------------------
    ma_per_ch = float(ctx.WS2812B_MA_PER_CHANNEL)
    idle_ma   = float(ctx.WS2812B_IDLE_MA_PER_LED) * float(L)
    budget_ma = float(ctx.CURRENT_BUDGET_A) * 1000.0
    target_ma = max(0.0, (budget_ma - idle_ma) * float(F_TARGET))

    i_color_ma = (ma_per_ch / 255.0) * float(np.sum(np.clip(v_pre, 0.0, 255.0))) * 3.0
    pre_scale_raw = 1.0 if i_color_ma <= 1e-6 else min(1.0, target_ma / i_color_ma)
    # EMA mais agressivo (50/50) para seguir flashes
    ctx._stb_agc_ema = 0.50 * ctx._stb_agc_ema + 0.50 * pre_scale_raw
    pre_scale = min(pre_scale_raw, ctx._stb_agc_ema)

    # Se o cap global já está atuando, reduz mais o leito (não o pico do flash)
    last_cap = float(ctx.last_cap_scale or 1.0) if hasattr(ctx, "last_cap_scale") else 1.0
    bed_scale_extra = 1.0 if last_cap >= 0.95 else (0.88 + 0.12 * last_cap)  # 0.88..1.0

    # Aplica pré-escala diferenciada: bed mais cortado, flash menos
    if ctx._stb_on_left > 0.0:
        # como não sabemos exatamente quais LEDs estão em flash agora, aproximamos:
        # priorizamos preservar ~85% do nível acima do bed
        v = bed * (pre_scale * bed_scale_extra) + np.maximum(0.0, v_pre - bed) * (0.85 * pre_scale + 0.15)
    else:
        v = v_pre * (pre_scale * bed_scale_extra)

    v = np.clip(v, 0.0, 255.0).astype(np.uint8)
    v = ctx.apply_floor_vec(v, active, None)

    # -------------------- 5) Cor: paleta → HSV fallback + branco no flash --------------------
    pal = getattr(ctx, "current_palette", None)
    if isinstance(pal, (list, tuple)) and len(pal) >= 2:
        pal_arr = np.asarray(pal, dtype=np.float32)
        m = pal_arr.shape[0]
        pal_phase = (centroid * 0.35) % 1.0
        g = (t + pal_phase) % 1.0
        g *= (m - 1)
        j = np.clip(np.floor(g).astype(np.int32), 0, m - 1)
        j2 = np.clip(j + 1, 0, m - 1)
        frac = (g - j).reshape(-1, 1)
        base_rgb = pal_arr[j] * (1.0 - frac) + pal_arr[j2] * frac  # (L,3)
        rgb = np.clip(base_rgb * (v[:, None].astype(np.float32) / 255.0), 0, 255)

        # Branco adicional nos LEDs em flash (mistura controlada)
        if STRIPES <= 1:
            in_flash_mask = v > bed.astype(np.uint8)
        else:
            in_flash_mask = (v > bed.astype(np.uint8))  # aproximação suficiente

        white_mix = (
            WHITE_MIX_BASE
            + WHITE_MIX_T * trans
            + (WHITE_MIX_BEAT if beat_flag else 0.0)
        )
        white_mix = np.clip(white_mix, 0.0, 1.0)
        if np.any(in_flash_mask):
            add = (v[in_flash_mask].astype(np.float32) * white_mix).reshape(-1, 1)
            rgb[in_flash_mask] = np.clip(rgb[in_flash_mask] + add, 0, 255)

        rgb = rgb.astype(np.uint8)
    else:
        hue_base = (int(ctx.base_hue_offset) + (int(ctx.hue_seed) >> 2)) % 256
        hue = (hue_base + (centroid * 96.0)) % 256
        hue_arr = np.full(L, int(hue) & 0xFF, dtype=np.uint8)
        sat = np.full(L, max(170, int(ctx.base_saturation)), dtype=np.uint8)
        rgb = ctx.hsv_to_rgb_bytes_vec(hue_arr, sat, v)
        # toque de branco no flash (mesma lógica)
        if STRIPES <= 1:
            in_flash_mask = v > bed.astype(np.uint8)
        else:
            in_flash_mask = (v > bed.astype(np.uint8))
        if np.any(in_flash_mask):
            white_mix = np.clip(WHITE_MIX_BASE + WHITE_MIX_T * trans + (WHITE_MIX_BEAT if beat_flag else 0.0), 0.0, 1.0)
            add = (v[in_flash_mask].astype(np.float32) * white_mix).reshape(-1, 1)
            rgb[in_flash_mask] = np.clip(rgb[in_flash_mask].astype(np.float32) + add, 0, 255).astype(np.uint8)

    # -------------------- 6) Envio (cap global ainda protege) --------------------
    ctx.to_pixels_and_show(rgb)



# ==== FIM DO ARQUIVO effects/clean.py ====