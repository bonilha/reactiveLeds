import colorsys

def update_mirror_spectrum(strip, num_leds, spectrum):
    half = num_leds // 2
    left = [(0,0,0)] * half
    right = [(0,0,0)] * (num_leds - half)

    # esquerda
    for i in range(half):
        if i < len(spectrum):
            h = (i / half) % 1.0
            v = min(1.0, spectrum[i] * 3.0)
            v = round(v * 8) / 8
            if v < 0.1: v = 0
            r, g, b = colorsys.hsv_to_rgb(h, 1.0, v)
            left[i] = (int(r*255), int(g*255), int(b*255))

    # espelha
    for i in range(half):
        if i < len(spectrum):
            r, g, b = left[half - 1 - i] if i < half else (0,0,0)
            right[num_leds - half + i] = (r, g, b)

    strip[:half] = left
    strip[half:] = right[:num_leds-half]
    return strip
