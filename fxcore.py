# fxcore.py
import numpy as np
import time

class FXContext:
    """
    Contexto para efeitos:
    - Vetores e índices (I_ALL, CENTER, I_LEFT/I_RIGHT)
    - Conversão HSV vetorizada
    - Utilitários de amplitude e piso dinâmico
    - Segmentações FULL / HALF
    - Render (to_pixels_and_show) com limitador de corrente + métricas
    """
    def __init__(self, pixels, led_count, base_hue_offset, hue_seed, base_saturation,
                 current_budget_a=18.0, ma_per_channel=20.0, idle_ma_per_led=1.0):
        self.pixels = pixels
        self.LED_COUNT = int(led_count)
        self.CENTER = self.LED_COUNT // 2
        self.I_ALL = np.arange(self.LED_COUNT, dtype=np.int32)
        self.I_LEFT = np.arange(self.CENTER, dtype=np.int32)
        self.I_RIGHT = self.LED_COUNT - 1 - self.I_LEFT

        # Paleta / seeds (o principal pode atualizar ao trocar paleta)
        self.base_hue_offset = int(base_hue_offset)
        self.hue_seed = int(hue_seed)
        self.base_saturation = int(base_saturation)

        # Piso dinâmico (atualizado pelo principal a cada frame)
        self.dynamic_floor = 0

        # Limite de corrente (orçamento) e modelo de consumo WS2812B
        self.CURRENT_BUDGET_A = float(current_budget_a)
        self.WS2812B_MA_PER_CHANNEL = float(ma_per_channel)   # ~20 mA em 255 por canal
        self.WS2812B_IDLE_MA_PER_LED = float(idle_ma_per_led) # ~1 mA/LED idle

        # Métricas (injetadas pelo principal)
        self.metrics = None

        # EMAs para exibir no status
        self._current_a_ema = None
        self._power_w_ema = None
        self._last_cap_scale = 1.0

        # Segmentações FULL e HALF (índices de bandas por LED)
        self.SEG_STARTS_FULL, self.SEG_ENDS_FULL = self._precompute_segment_starts_ends(self.LED_COUNT, self.LED_COUNT)
        self.SEG_STARTS_HALF, self.SEG_ENDS_HALF = self._precompute_segment_starts_ends(self.LED_COUNT, self.CENTER)

    # ---------- getters para o status ----------
    @property
    def current_a_ema(self):
        return self._current_a_ema
    @property
    def power_w_ema(self):
        return self._power_w_ema
    @property
    def last_cap_scale(self):
        return self._last_cap_scale

    # ---------- utilitários ----------
    @staticmethod
    def amplify_quad(v):
        v = np.asarray(v, dtype=np.uint16)
        return np.clip((v * v * 12) // (255 * 10), 0, 255).astype(np.uint8)  # ganho ajustado

    def apply_floor_vec(self, v, active, floor_val=None):
        if not active:
            return v
        f = self.dynamic_floor if floor_val is None else int(floor_val)
        return np.maximum(v, f).astype(v.dtype)

    @staticmethod
    def _precompute_segment_starts_ends(n_bands, n_leds):
        """
        Gera os vetores starts/ends para fazer média de bandas em n_leds segmentos.
        """
        i = np.arange(n_leds, dtype=np.int32)
        starts = (i * n_bands) // n_leds
        ends = ((i + 1) * n_bands) // n_leds
        ends = np.maximum(ends, starts + 1)
        return starts, ends

    @staticmethod
    def hsv_to_rgb_bytes_vec(h, s, v):
        """
        Conversão HSV vetorizada robusta.
        Aceita:
        - h: uint8 (0..255) OU float (0..1) -> internamente normalizado para 0..1
        - s: uint8 (0..255)
        - v: uint8 (0..255)
        Retorna: np.uint8[*,3] em formato RGB.
        """
        # --- Normalizar H para 0..1 com segurança ---
        h = np.asarray(h)
        if np.issubdtype(h.dtype, np.integer):
            hf = (h.astype(np.float32) / 255.0)
        else:
            hf = h.astype(np.float32)
        hf = np.mod(hf, 1.0)

        # --- Normalizar S e V para 0..1 ---
        s = np.asarray(s).astype(np.float32) / 255.0
        v = np.asarray(v).astype(np.float32) / 255.0

        # --- Conversão vetorizada ---
        i = np.floor(hf * 6.0).astype(np.int32)
        f = hf * 6.0 - i.astype(np.float32)
        i = i % 6

        p = v * (1.0 - s)
        q = v * (1.0 - f * s)
        t = v * (1.0 - (1.0 - f) * s)

        r = np.empty_like(v)
        g = np.empty_like(v)
        b = np.empty_like(v)

        m0 = (i == 0); r[m0], g[m0], b[m0] = v[m0], t[m0], p[m0]
        m1 = (i == 1); r[m1], g[m1], b[m1] = q[m1], v[m1], p[m1]
        m2 = (i == 2); r[m2], g[m2], b[m2] = p[m2], v[m2], t[m2]
        m3 = (i == 3); r[m3], g[m3], b[m3] = p[m3], q[m3], v[m3]
        m4 = (i == 4); r[m4], g[m4], b[m4] = t[m4], p[m4], v[m4]
        m5 = (i == 5); r[m5], g[m5], b[m5] = v[m5], p[m5], q[m5]

        rgb = np.stack([r, g, b], axis=-1)
        return np.clip((rgb * 255.0).round(), 0, 255).astype(np.uint8)

    @staticmethod
    def _rgb_to_rgbw(arr_rgb_u8):
        """
        Converte RGB -> RGBW por extração de branco:
        W = min(R,G,B); R,G,B = R-W, G-W, B-W.
        """
        r = arr_rgb_u8[:, 0].astype(np.int16)
        g = arr_rgb_u8[:, 1].astype(np.int16)
        b = arr_rgb_u8[:, 2].astype(np.int16)
        w = np.minimum(np.minimum(r, g), b)
        r2 = (r - w).clip(0, 255).astype(np.uint8)
        g2 = (g - w).clip(0, 255).astype(np.uint8)
        b2 = (b - w).clip(0, 255).astype(np.uint8)
        w2 = w.clip(0, 255).astype(np.uint8)
        return np.stack([r2, g2, b2, w2], axis=-1)

    # ---------- mapeamento de bandas -> LEDs ----------
    def segment_mean_from_cumsum(self, bands_float, starts, ends):
        """
        Média das bandas mapeadas para N LEDs (N = len(starts) = len(ends)).
        • Se starts/ends estiverem desalinhados ao tamanho real de bands, recalcula.
        """
        bands_float = np.asarray(bands_float, dtype=np.float32)
        n_bands = int(bands_float.shape[0])
        cs = np.concatenate(([0.0], np.cumsum(bands_float, dtype=np.float32)))
        starts = np.asarray(starts, dtype=np.int32)
        ends = np.asarray(ends, dtype=np.int32)
        if ends.size == 0 or starts.size == 0 or np.max(ends) >= cs.shape[0]:
            n_leds_target = int(max(len(starts), len(ends)))
            s2, e2 = self._precompute_segment_starts_ends(n_bands, n_leds_target)
            starts, ends = s2.astype(np.int32), e2.astype(np.int32)
        sums = cs[ends] - cs[starts]
        lens = (ends - starts).astype(np.float32)
        return sums / np.maximum(lens, 1.0)

    # ---------- render com 'power cap' + métricas ----------
    def to_pixels_and_show(self, rgb_array_u8):
        """
        Define os pixels e apresenta o frame, com:
        - Estimativa de corrente/potência
        - Auto-cap de brilho por frame para respeitar CURRENT_BUDGET_A
        - Alimenta coletor de métricas (se existir)
        """
        if not isinstance(rgb_array_u8, np.ndarray):
            arr = np.asarray(rgb_array_u8, dtype=np.uint8)
        else:
            arr = rgb_array_u8
        arr = arr.reshape(self.LED_COUNT, 3).astype(np.uint8)

        # Consumo de cor (antes do cap)
        sum_rgb_unscaled = float(np.sum(arr, dtype=np.uint64))
        i_color_mA_unscaled = (self.WS2812B_MA_PER_CHANNEL / 255.0) * sum_rgb_unscaled
        i_idle_mA = self.WS2812B_IDLE_MA_PER_LED * float(self.LED_COUNT)
        i_budget_mA = self.CURRENT_BUDGET_A * 1000.0

        # Power-cap
        scale = 1.0
        if i_color_mA_unscaled > 0.0 and (i_color_mA_unscaled + i_idle_mA) > i_budget_mA:
            scale = max(0.0, (i_budget_mA - i_idle_mA) / i_color_mA_unscaled)
        self._last_cap_scale = float(scale)
        if scale < 0.999:
            arr = np.clip(arr.astype(np.float32) * scale, 0, 255).astype(np.uint8)

        # Consumo pós-cap (sem novo sum: reutiliza unscaled * scale)
        i_color_mA = i_color_mA_unscaled * scale
        i_total_A = (i_color_mA + i_idle_mA) / 1000.0
        p_total_W = 5.0 * i_total_A

        # EMAs para status
        if self._current_a_ema is None:
            self._current_a_ema = i_total_A
            self._power_w_ema = p_total_W
        else:
            self._current_a_ema = 0.80 * self._current_a_ema + 0.20 * i_total_A
            self._power_w_ema = 0.80 * self._power_w_ema + 0.20 * p_total_W

        # métricas por segundo (se houver)
        if self.metrics is not None:
            try:
                self.metrics.on_frame(time.time(), i_total_A, p_total_W, self._last_cap_scale)
            except Exception:
                pass

        # ---- Suporte RGBW ----
        bpp = getattr(self.pixels, "bpp", 3)
        if bpp == 4:
            arr_to_send = self._rgb_to_rgbw(arr)
        else:
            arr_to_send = arr

        # Envia (manter caminho compatível com a lib neopixel)
        # Observação: o caminho seguro e compatível continua usando lista.
        self.pixels[:] = arr_to_send.tolist()
        self.pixels.show()