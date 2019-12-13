import random

ACTIVATION_LEVEL = 0.6
# MIRRORING_ACTIVATION_LEVEL = 0.5
NOISE_ACTIVATION_LEVEL = 0.7


def position_displacement(positions, is_copy=False):
    activate = random.random() > ACTIVATION_LEVEL

    if activate or is_copy:
        d_x = random.randrange(0, 10)
        d_y = random.randrange(0, 10)
        for pos in positions:
            pos['x'] += d_x
            pos['y'] += d_y
    return positions


def add_noise(positions, is_copy=False):
    activate = random.random() > NOISE_ACTIVATION_LEVEL

    if activate or is_copy:
        for coords in positions:
            coords['x'] += random.randrange(0, 5) * 0.001 - 0.0025
            coords['y'] += random.randrange(0, 5) * 0.001 - 0.0025
            coords['w'] += random.randrange(0, 5) * 0.001 - 0.0025
            coords['h'] += random.randrange(0, 5) * 0.001 - 0.0025
    return positions
