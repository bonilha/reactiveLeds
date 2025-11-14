import colorsys

class VUMeterHSV:
    def __init__(self, num_leds):
        self.num_leds = num_leds
        self.current_level = 0.0

    def update(self, strip, bass_energy):
        target = int(bass_energy * self.num_leds)
        self.current_level += (target - self.current_level) * 0.35
        level = int(self.current_level)

        for i in range(self.num_leds):
            if i < level:
                h = 0.3 - (i / self.num_leds) * 0.3
                r, g, b = colorsys.hsv_to_rgb(h, 1.0, 1.0)
                strip[i] = (int(r*255), int(g*255), int(b*255))
            else:
                strip[i] = (0, 0, 0)

        self.current_level *= 0.92
        return strip
