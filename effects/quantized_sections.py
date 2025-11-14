from utils import hsv_to_rgb, lerp_color

def update_quantized_sections(strip, num_leds, bands):
    if len(bands) < 8:
        bands += [0] * (8 - len(bands))

    palette = [0.0, 0.08, 0.16, 0.33, 0.5, 0.66, 0.83, 0.91]
    section_size = num_leds // 8

    for s in range(8):
        energy = bands[s]
        base_hue = palette[s]
        hue = (base_hue + energy * 0.15) % 1.0

        start = s * section_size
        end = min(start + section_size, num_leds)
        for i in range(start, end):
            r, g, b = hsv_to_rgb(hue, 1.0, energy)
            strip[i] = (int(r*255), int(g*255), int(b*255))

        if s > 0 and start > 0:
            prev = strip[start - 1]
            curr = strip[start]
            strip[start - 1] = lerp_color(prev, curr, 0.5)

    for i in range(8 * section_size, num_leds):
        strip[i] = strip[i % (8 * section_size)]

    return strip
