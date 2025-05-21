import pygame
import sys
from env import JumperEnv
from agent_neat import run_neat

WIDTH, HEIGHT = 600, 800
BLACK, WHITE = (0, 0, 0), (255, 255, 255)

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Vertical Jumper")
clock = pygame.time.Clock()

def draw_text(surf, text, size, color, x, y, center=True):
    font = pygame.font.SysFont(None, size)
    txt = font.render(text, True, color)
    rect = txt.get_rect(center=(x, y) if center else (x, y))
    surf.blit(txt, rect)

def draw_hud(score):
    surf = pygame.display.get_surface()
    draw_text(surf, f"Score: {score}", 28, WHITE, 20, 20, center=False)

def play_manual():
    env = JumperEnv(agent_count=1)
    env.reset()
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                return  # Exit to menu

        obs, rewards, dones, _ = env.step([None])  # Manual input

        if dones[0]:
            print("You died!")
            return

        screen.fill(BLACK)
        env.render(screen)
        draw_hud(env.score[0])
        pygame.display.flip()
        clock.tick(60)

def main_menu():
    while True:
        screen.fill(BLACK)
        draw_text(screen, "VERTICAL JUMPER", 60, WHITE, WIDTH // 2, HEIGHT // 3)
        draw_text(screen, "Press N for NEAT", 30, WHITE, WIDTH // 2, HEIGHT // 2)
        draw_text(screen, "Press SPACE for Manual Play", 30, WHITE, WIDTH // 2, HEIGHT // 2 + 40)
        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_n:
                    run_neat("config-feedforward.txt")
                elif event.key == pygame.K_SPACE:
                    play_manual()

if __name__ == "__main__":
    main_menu()
