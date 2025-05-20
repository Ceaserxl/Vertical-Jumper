# neat_runner.py
import pygame
import os
import neat
import math
import time
import random
from config import WIDTH, HEIGHT, screen, clock, BLACK, WHITE, BLUE, GREEN, RED
from utils import draw_text, create_platforms

def run_neat(config_path):
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path
    )

    config.pop_size = 100
    generation_counter = 0

    def eval_genomes(genomes, config):
        nonlocal generation_counter
        generation_counter += 1
        evolution = (generation_counter - 1) // 25 + 1
        print(f"--- Generation {generation_counter} (Evolution {evolution}) ---")

        platforms = create_platforms()
        spawn_plat = platforms[-1]

        players, nets, ge, colors = [], [], [], []
        for genome_id, genome in genomes:
            mutation_factor = len([cg for cg in genome.connections.values() if cg.enabled]) / 10
            mutation_factor = min(max(mutation_factor, 0), 1)
            red = int(255 * (1 - mutation_factor))
            blue = int(255 * mutation_factor)
            color = (red, 0, blue)
            player = pygame.Rect(spawn_plat.x + random.randint(-30, 30), spawn_plat.y - 30, 30, 30)
            genome.fitness = 0
            net = neat.nn.FeedForwardNetwork.create(genome, config)
            players.append(player)
            nets.append(net)
            ge.append(genome)
            colors.append(color)

        velocity_y = [0 for _ in players]
        can_jump = [False for _ in players]
        min_ys = [p.y for p in players]
        platform_history = [[] for _ in players]
        scroll_offset = 0
        frame_count = 0
        best_data_logged = False

        run = True
        while run:
            frame_count += 1
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()

            screen.fill(BLACK)
            best_index = min(range(len(players)), key=lambda i: min_ys[i]) if players else None
            best_y = min(min_ys) if min_ys else HEIGHT

            i = 0
            while i < len(players):
                player = players[i]

                if frame_count > 10 and player.y - scroll_offset > HEIGHT:
                    ge[i].fitness -= 20
                    players.pop(i)
                    nets.pop(i)
                    ge.pop(i)
                    velocity_y.pop(i)
                    can_jump.pop(i)
                    min_ys.pop(i)
                    platform_history.pop(i)
                    colors.pop(i)
                    continue

                sensor_angles = [0, 45, 90, 135, 180, 225, 270, 315]
                sensor_data = []
                for angle in sensor_angles:
                    radians = math.radians(angle)
                    dx = math.cos(radians)
                    dy = -math.sin(radians)
                    hit_point = None
                    for dist in range(10, WIDTH, 10):
                        sx = player.centerx + dx * dist
                        sy = player.centery + dy * dist
                        sensor_rect = pygame.Rect(sx, sy, 2, 2)
                        if any(sensor_rect.colliderect(p) for p in platforms):
                            sensor_data.append(dist / 300)
                            hit_point = (sx, sy)
                            break
                    else:
                        sensor_data.append(1.0)
                        hit_point = (player.centerx + dx * WIDTH, player.centery + dy * WIDTH)
                    # Draw the sensor line
                    pygame.draw.line(screen, RED, player.center, hit_point, 1)

                inputs = [player.x / WIDTH, player.y / HEIGHT, velocity_y[i] / 10, float(can_jump[i])] + sensor_data

                expected_inputs = config.genome_config.num_inputs
                if len(inputs) != expected_inputs:
                    print(f"[Warning] Update your NEAT config: num_inputs should be {len(inputs)}")
                    raise RuntimeError(f"Expected {expected_inputs} inputs, got {len(inputs)}")

                output = nets[i].activate(inputs)
                previous_x = player.x
                move_left, move_right, jump = output

                if move_left > 0.5:
                    player.x -= 5
                if move_right > 0.5:
                    player.x += 5
                if jump > 0.5 and can_jump[i]:
                    velocity_y[i] = -16
                    can_jump[i] = False
                    # Removed reward for jumping; reward only for climbing

                player.x = max(0, min(WIDTH - player.width, player.x))
                velocity_y[i] += 0.5
                player.y += velocity_y[i]

                for plat in platforms:
                    if abs(player.y - plat.y) < 20 and abs(player.x - plat.x) < 30:
                        ge[i].fitness += 0.5

                    if player.colliderect(plat) and velocity_y[i] > 0 and player.bottom <= plat.bottom + 10:
                        velocity_y[i] = 0
                        player.bottom = plat.top
                        can_jump[i] = True

                        if plat.y < min_ys[i]:
                            climb_bonus = (min_ys[i] - plat.y) / HEIGHT * 20
                            ge[i].fitness += climb_bonus
                            if ge[i].fitness > 100 and not best_data_logged:
                                with open("best_genome_log.txt", "a") as f:
                                    f.write(f"Inputs: {inputs} | Outputs: {output}\n")
                                best_data_logged = True

                        platform_history[i].append(plat.y)
                        if len(platform_history[i]) > 10:
                            platform_history[i].pop(0)

                        if len(platform_history[i]) == 10 and len(set(platform_history[i])) <= 2:
                            ge[i].fitness -= 15
                            players.pop(i)
                            nets.pop(i)
                            ge.pop(i)
                            velocity_y.pop(i)
                            can_jump.pop(i)
                            min_ys.pop(i)
                            platform_history.pop(i)
                            colors.pop(i)
                            continue
                        break

                if i < len(min_ys) and players[i].y < min_ys[i]:
                    min_ys[i] = players[i].y

                if i < len(ge) and i < len(velocity_y):
                    scroll_score = scroll_offset
                    ge[i].fitness += scroll_score * 0.001 + (-velocity_y[i] * 0.01)
                    if player.x == previous_x:
                        ge[i].fitness -= 0.5
                i += 1

            if best_index is not None and best_index < len(players):
                if best_y < HEIGHT // 3:
                    scroll = HEIGHT // 3 - best_y
                    scroll_offset += scroll
                    for i in range(len(players)):
                        players[i].y += scroll
                        min_ys[i] += scroll
                        min_ys[i] = min(min_ys[i], players[i].y)
                    for plat in platforms:
                        plat.y += scroll

            while len(platforms) < 10:
                top_y = min(p.y for p in platforms)
                new = pygame.Rect(random.randint(0, WIDTH - 60), top_y - 80, 60, 10)
                platforms.append(new)

            platforms = [plat for plat in platforms if plat.y < HEIGHT]

            for plat in platforms:
                pygame.draw.rect(screen, GREEN, plat)
            for idx, player in enumerate(players):
                pygame.draw.rect(screen, colors[idx], player)

            draw_text(f"Best Score: {int(scroll_offset)}", 24, WHITE, WIDTH // 2, 10)
            draw_text(f"Generation: {generation_counter}", 24, WHITE, WIDTH // 2, 30)
            draw_text(f"Evolution: {evolution}", 24, WHITE, WIDTH // 2, 50)
            draw_text(f"Alive: {len(players)}", 24, WHITE, WIDTH // 2, 70)  # unchanged

            pygame.display.flip()
            clock.tick(60)

            if len(players) == 0:
                pygame.time.wait(20)
                return

    pop = neat.Population(config)
    pop.add_reporter(neat.StdOutReporter(True))
    pop.add_reporter(neat.StatisticsReporter())
    pop.run(eval_genomes, 999999999)
