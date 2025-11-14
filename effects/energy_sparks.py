import random
from utils import hsv_to_rgb

class EnergySparks:
    def __init__(self, num_leds):
        self.num_leds = num_leds
        self.sparks = []
        self.max_sparks = 8

    def update(self, strip, bass_beat_detected, bass_energy):
        if bass_beat_detected and len(self.sparks) < self.max_sparks:
            pos = random.randint(0, self.num_leds - 1)
            direction = random.choice([-1, 1])
            self.sparks.append({'pos': pos, 'life': 1.0, 'dir': direction})

        new_sparks = []
        for s in self.sparks:
            s['pos'] += s['dir'] * 2
            s['life'] *= 0.88

            if 0 <= s['pos'] < self.num_leds and s['life'] >= 0.1:
                r, g, b = hsv_to_rgb(0.15, 1.0, s['life'])
                strip[int(s['pos'])] = (int(r*255), int(g*255), int(b*255))
                new_sparks.append(s)

        self.sparks = new_sparks
        return strip

    def clear(self):
        self.sparks = []
