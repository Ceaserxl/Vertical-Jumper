# config.py
import pygame

pygame.init()  # Ensures all pygame modules are initialized

# Screen
WIDTH, HEIGHT = 512, 768
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Vertical Jumper - NEAT")
clock = pygame.time.Clock()

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
BLUE = (50, 100, 255)
GREEN = (0, 200, 0)
RED = (255, 0, 0)

SCROLL_SPEED     = 10         # pixels/frame for smooth scroll
SCROLL_THRESHOLD = 3 * HEIGHT // 4