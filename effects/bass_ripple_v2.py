import math
from utils import hsv_to_rgb

def update_bass_ripple_v2(strip, num_leds, bass_energy):
    center = num_leds // 2
    intensity = min(255, bass_energy * 400)

    for i in range(num_leds):
        dist = abs(i - center) / center if center > 0 else 0
        pulse = intensity * (1 - math.pow(dist, 1.5))
        pulse = max(0, pulse)

        r, g, b = hsv_to_rgb(0.0, 1.0, pulse / 255.0)
        strip[i] = (int(r*255), int(g*255), int(b*255))

    return strip
