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

        self.base_hue_offset = int(base_hue_offset)
        self.hue_seed = int(hue_seed)
        self.base_saturation = int(base_saturation)
        self.dynamic_floor = 0

        self.CURRENT_BUDGET_A = float(current_budget_a)
        self.WS2812B_MA_PER_CHANNEL = float(ma_per_channel)
        self.WS2812B_IDLE_MA_PER_LED = float(idle_ma_per_led)

        self.metrics = None
        self._current_a_ema = None
        self._power_w_ema = None
        self._last_cap_scale = 1.0

        self.SEG_STARTS_FULL, self.SEG_ENDS_FULL = self._precompute_segment_starts_ends(self.LED_COUNT, self.LED_COUNT)
        self.SEG_STARTS_HALF, self.SEG_ENDS_HALF = self._precompute_segment_starts_ends(self.LED_COUNT, self.CENTER)

    # status getters
    @property
    def current_a_ema(self):
        return self._current_a_ema
    @property
    def power_w_ema(self):
        return self._power_w_ema
    @property
    def last_cap_scale(self):
        return self._last_cap_scale

    # utils
    @staticmethod
    def amplify_quad(v):
        v = np.asarray(v, dtype=np.uint16)
        return (v * v) // 255

    def apply_floor_vec(self, v, active, floor_val=None):
        if not active:
            return v
        f = self.dynamic_floor if floor_val is None else int(floor_val)
        return np.maximum(v, f).astype(v.dtype)

    @staticmethod
    def _precompute_segment_starts_ends(n_bands, n_leds):
        i = np.arange(n_leds, dtype=np.int32)
        starts = (i * n_bands) // n_leds
        ends = ((i + 1) * n_bands) // n_leds
        ends = np.maximum(ends, starts + 1)
        return starts, ends

    @staticmethod
    def hsv_to_rgb_bytes_vec(h, s, v):
        h = h.astype(np.float32) / 255.0
        s = s.astype(np.float32) / 255.0
        v = v.astype(np.float32) / 255.0
        i = np.floor(h * 6.0).astype(np.int32) % 6
        f = h * 6.0 - i.astype(np.float32)
        p = v * (1.0 - s)
        q = v * (1.0 - f * s)
        t = v * (1.0 - (1.0 - f) * s)
        r = np.empty_like(v); g = np.empty_like(v); b = np.empty_like(v)
        m0 = (i == 0); r[m0], g[m0], b[m0] = v[m0], t[m0], p[m0]
        m1 = (i == 1); r[m1], g[m1], b[m1] = q[m1], v[m1], p[m1]
        m2 = (i == 2); r[m2], g[m2], b[m2] = p[m2], v[m2], t[m2]
        m3 = (i == 3); r[m3], g[m3], b[m3] = p[m3], q[m3], v[m3]
        m4 = (i == 4); r[m4], g[m4], b[m4] = t[m4], p[m4], v[m4]
        m5 = (i == 5); r[m5], g[m5], b[m5] = v[m5], p[m5], q[m5]
        rgb = np.stack([r, g, b], axis=-1)
        return np.clip((rgb * 255.0).round(), 0, 255).astype(np.uint8)

def segment_mean_from_cumsum(self, bands_float, starts, ends):
    # Calcula a média das bandas mapeando para N LEDs (len(starts)/len(ends)).
    # Torna-se robusto ao tamanho de bands (recalcula starts/ends se preciso).

    bands_float = np.asarray(bands_float, dtype=np.float32)
    n_bands = int(bands_float.shape[0])

    # Prefix-sum das bandas (len == n_bands + 1)
    cs = np.concatenate(([0.0], np.cumsum(bands_float, dtype=np.float32)))

    # Garantir que starts/ends são arrays inteiros
    starts = np.asarray(starts, dtype=np.int32)
    ends   = np.asarray(ends,   dtype=np.int32)

    # Se as tabelas não combinarem com o tamanho real de bands, recalcula
    # alvo = número de LEDs (comprimento de starts/ends)
    if ends.size == 0 or starts.size == 0 or np.max(ends) >= cs.shape[0]:
        n_leds_target = int(max(len(starts), len(ends)))
        s2, e2 = self._precompute_segment_starts_ends(n_bands, n_leds_target)
        starts, ends = s2.astype(np.int32), e2.astype(np.int32)

    # Média por segmento: soma parcial / largura
    sums = cs[ends] - cs[starts]
    lens = (ends - starts).astype(np.float32)
    return sums / np.maximum(lens, 1.0)


    def to_pixels_and_show(self, rgb_array_u8):
        """Render com limitador de corrente e métricas."""
        if not isinstance(rgb_array_u8, np.ndarray):
            arr = np.asarray(rgb_array_u8, dtype=np.uint8)
        else:
            arr = rgb_array_u8
        arr = arr.reshape(self.LED_COUNT, 3).astype(np.uint8)

        sum_rgb = float(np.sum(arr, dtype=np.uint64))
        i_color_mA = (self.WS2812B_MA_PER_CHANNEL / 255.0) * sum_rgb
        i_idle_mA  = self.WS2812B_IDLE_MA_PER_LED * float(self.LED_COUNT)
        i_budget_mA = self.CURRENT_BUDGET_A * 1000.0

        scale = 1.0
        if i_color_mA > 0.0 and (i_color_mA + i_idle_mA) > i_budget_mA:
            scale = max(0.0, (i_budget_mA - i_idle_mA) / i_color_mA)
        self._last_cap_scale = float(scale)
        if scale < 0.999:
            arr = np.clip(arr.astype(np.float32) * scale, 0, 255).astype(np.uint8)
            sum_rgb = float(np.sum(arr, dtype=np.uint64))
            i_color_mA = (self.WS2812B_MA_PER_CHANNEL / 255.0) * sum_rgb

        i_total_A = (i_color_mA + i_idle_mA) / 1000.0
        p_total_W = 5.0 * i_total_A

        if self._current_a_ema is None:
            self._current_a_ema = i_total_A
            self._power_w_ema   = p_total_W
        else:
            self._current_a_ema = 0.80 * self._current_a_ema + 0.20 * i_total_A
            self._power_w_ema   = 0.80 * self._power_w_ema   + 0.20 * p_total_W

        if self.metrics is not None:
            try:
                self.metrics.on_frame(time.time(), i_total_A, p_total_W, self._last_cap_scale)
            except Exception:
                pass

        self.pixels[:] = arr.tolist()
        self.pixels.show()
