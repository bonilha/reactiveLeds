import random

class Waterfall:
    def __init__(self, num_leds):
        self.num_leds = num_leds
        self.filling = True
        self.fill_pos = 0
        self.drop_speed = 0.8

    def update(self, strip, trigger=False):
        if trigger:
            self.filling = True
            self.fill_pos = 0
            self.drop_speed = 0.8

        if self.filling:
            self.fill_pos += 3
            for i in range(min(self.fill_pos, self.num_leds)):
                strip[i] = (255, 255, 255)
            if self.fill_pos >= self.num_leds:
                self.filling = False
        else:
            self.drop_speed = min(2.5, self.drop_speed + 0.05)
            for i in range(self.num_leds - 1, -1, -1):
                if random.random() < self.drop_speed / 100:
                    strip[i] = strip[i-1] if i > 0 else (0, 0, 0)

        return strip
