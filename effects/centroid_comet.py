from utils import hsv_to_rgb

def update_centroid_comet(strip, num_leds, freq_bands):
    if not freq_bands:
        return strip

    max_idx = freq_bands.index(max(freq_bands))
    head = int(max_idx / len(freq_bands) * num_leds)
    tail_len = int(max(freq_bands) * 20) + 5
    max_energy = max(freq_bands)

    for i in range(num_leds):
        dist = abs(i - head)
        if dist < tail_len:
            brightness = 1.0 - (dist / tail_len)
            brightness *= (freq_bands[max_idx] / max_energy if max_energy > 0 else 0)
            r, g, b = hsv_to_rgb(0.6, 1.0, brightness)
            strip[i] = (int(r*255), int(g*255), int(b*255))
        else:
            strip[i] = (0, 0, 0)
    return strip
