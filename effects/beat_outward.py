from utils import hsv_to_rgb

class BeatOutward:
    def __init__(self, num_leds):
        self.num_leds = num_leds
        self.wave_speed = 0
        self.amplitude = 0.0
        self.center = num_leds // 2

    def trigger(self):
        self.wave_speed = 3
        self.amplitude = 1.0

    def update(self, strip, beat_detected):
        if beat_detected:
            self.trigger()

        for i in range(self.num_leds):
            dist = abs(i - self.center)
            if dist < self.wave_speed:
                amp = self.amplitude * pow(1 - dist / self.num_leds, 0.7)
                r, g, b = hsv_to_rgb(0.0, 1.0, amp)
                strip[i] = (int(r*255), int(g*255), int(b*255))
            else:
                strip[i] = (0, 0, 0)

        self.wave_speed += 3
        self.amplitude *= 0.9
        return strip
