import colorsys

def update_line_spectrum(strip, num_leds, spectrum):
    for i in range(num_leds):
        h = (i / num_leds) % 1.0
        s = 1.0
        v = min(1.0, spectrum[i] * 3.0)
        v = round(v * 8) / 8
        if v < 0.1:
            v = 0

        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        r, g, b = int(r*255), int(g*255), int(b*255)

        if i < len(strip):
            cr, cg, cb = strip[i]
            if abs(cr - r) > 10 or abs(cg - g) > 10 or abs(cb - b) > 10:
                strip[i] = (r, g, b)
    return strip
