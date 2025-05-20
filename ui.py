# ui.py
import pygame
import os
from config import screen, WIDTH, HEIGHT, BLACK, WHITE
from utils import draw_text
from neat_runner import run_neat
from trunner import evolve
from game import run_game_with_genome

def show_game_over():
    while True:
        screen.fill(BLACK)
        draw_text("GAME OVER", 60, WHITE, WIDTH // 2, HEIGHT // 3)
        draw_text("[ENTER] Watch NEAT Learn", 30, WHITE, WIDTH // 2, HEIGHT // 2)
        draw_text("[SPACE] Play Again Yourself", 30, WHITE, WIDTH // 2, HEIGHT // 2 + 40)
        draw_text("[T]     Watch TF Learn", 30, WHITE, WIDTH // 2, HEIGHT // 2 + 80)
        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN:
                    local_dir = os.path.dirname(__file__)
                    config_path = os.path.join(local_dir, "config-feedforward.txt")
                    run_neat(config_path)
                    return
                elif event.key == pygame.K_SPACE:
                    while True:
                        result = run_game_with_genome(manual=True)
                        if result == "game_over":
                            show_game_over()
                        else:
                            return
                elif event.key == pygame.K_t:
                    evolve()
                    return

def main_menu():
    while True:
        screen.fill(BLACK)
        draw_text("VERTICAL JUMPER", 60, WHITE, WIDTH // 2, HEIGHT // 3)
        draw_text("[ENTER] Watch NEAT Learn",       30, WHITE, WIDTH // 2, HEIGHT // 2)
        draw_text("[SPACE] Play Yourself",          30, WHITE, WIDTH // 2, HEIGHT // 2 + 40)
        draw_text("[T]     Watch TF Learn",        30, WHITE, WIDTH // 2, HEIGHT // 2 + 80)
        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN:
                    local_dir = os.path.dirname(__file__)
                    config_path = os.path.join(local_dir, "config-feedforward.txt")
                    run_neat(config_path)
                    return
                elif event.key == pygame.K_SPACE:
                    while True:
                        result = run_game_with_genome(manual=True)
                        if result == "game_over":
                            show_game_over()
                        else:
                            return
                elif event.key == pygame.K_t:
                    evolve()
                    return
