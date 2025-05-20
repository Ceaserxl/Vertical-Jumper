# game.py
import pygame
import time
import random
from config import WIDTH, HEIGHT, screen, clock, BLACK, WHITE, BLUE, GREEN
from utils import draw_text, create_platforms
import neat

def run_game_with_genome(genome=None, config=None, generation=0, evolution=0, manual=False):
    net = neat.nn.FeedForwardNetwork.create(genome, config) if genome else None
    platforms = create_platforms()
    spawn_plat = platforms[-1]
    player = pygame.Rect(spawn_plat.x + 15, spawn_plat.y - 30, 30, 30)
    velocity_y = 0
    can_jump = False
    score = 0
    max_score = 0
    jumps = 0
    last_score = 0
    no_progress = 0
    run = True
    fitness = 0
    min_y = player.y
    min_y_timer = time.time()

    while run:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        keys = pygame.key.get_pressed()
        if manual:
            if keys[pygame.K_a]:
                player.x -= 5
            if keys[pygame.K_d]:
                player.x += 5
            if keys[pygame.K_SPACE] and can_jump:
                velocity_y = -16
                can_jump = False
        elif net:
            closest_platforms = sorted([p for p in platforms if p.y > player.y], key=lambda p: p.y)[:3]
            inputs = [player.x / WIDTH, player.y / HEIGHT, velocity_y / 10, float(can_jump)]
            for plat in closest_platforms:
                inputs.append((plat.x - player.x) / WIDTH)
                inputs.append((plat.y - player.y) / HEIGHT)
            while len(inputs) < 10:
                inputs.append(0.0)
            output = net.activate(inputs)
            move_left, move_right, jump = output
            if move_left > 0.5:
                player.x -= 5
            if move_right > 0.5:
                player.x += 5
            if jump > 0.5 and can_jump:
                velocity_y = -16
                can_jump = False
                jumps += 1
                if score <= last_score:
                    no_progress += 1
                else:
                    no_progress = 0
                last_score = score

        player.x = max(0, min(WIDTH - player.width, player.x))

        velocity_y += 0.5
        player.y += velocity_y

        for plat in platforms:
            if player.colliderect(plat) and velocity_y > 0 and player.bottom <= plat.bottom + 10:
                velocity_y = 0
                player.bottom = plat.top
                can_jump = True
                break

        if player.y < HEIGHT // 3:
            scroll = HEIGHT // 3 - player.y
            player.y = HEIGHT // 3
            for plat in platforms:
                plat.y += scroll
            score += scroll

        while len(platforms) < 10:
            top_y = min(p.y for p in platforms)
            new = pygame.Rect(random.randint(0, WIDTH - 60), top_y - 80, 60, 10)
            platforms.append(new)
        platforms = [p for p in platforms if p.y < HEIGHT]

        if not manual:
            if player.y < min_y:
                min_y = player.y
                min_y_timer = time.time()
            elif time.time() - min_y_timer > 5:
                fitness -= 10
                run = False

        if player.y > HEIGHT:
            run = False
            if not manual:
                fitness -= 20
            else:
                return "game_over"

        if score > max_score:
            max_score = score
            if not manual:
                fitness += 10
        if not manual:
            fitness += 0.1

        screen.fill(BLACK)
        pygame.draw.rect(screen, BLUE, player)
        for plat in platforms:
            pygame.draw.rect(screen, GREEN, plat)
        draw_text(f"Score: {score}", 30, WHITE, WIDTH // 2, 20)
        pygame.display.flip()
        clock.tick(60)

    return fitness if genome else 0