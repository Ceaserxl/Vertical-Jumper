import pygame
import sys
from env import JumperEnv
from agent_neat import run_neat

WIDTH, HEIGHT = 600, 800
BLACK, WHITE = (0,0,0), (255,255,255)

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Vertical Jumper")
clock = pygame.time.Clock()

def draw_text(text, size, color, x, y, center=True):
    font = pygame.font.SysFont(None, size)
    surf = font.render(text, True, color)
    rect = surf.get_rect(center=(x, y) if center else (x, y))
    screen.blit(surf, rect)

def play_manual():
    env = JumperEnv()
    env.reset()
    done = False
    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return  # Go back to main menu
        env.step(None)
        env.render()
        pygame.time.Clock().tick(60)

def main_menu():
    while True:
        screen.fill(BLACK)
        draw_text("VERTICAL JUMPER", 60, WHITE, WIDTH // 2, HEIGHT // 3)
        draw_text("Press N for NEAT", 30, WHITE, WIDTH // 2, HEIGHT // 2)
        draw_text("Press SPACE for Manual Play", 30, WHITE, WIDTH // 2, HEIGHT // 2 + 40)
        pygame.display.flip()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_n:
                    run_neat("config-feedforward.txt")
                elif event.key == pygame.K_SPACE:
                    play_manual()

if __name__ == "__main__":
    main_menu()
