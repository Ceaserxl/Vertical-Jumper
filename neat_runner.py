# neat_runner.py
import pygame
import neat
import math
import random
from config import (
    WIDTH, HEIGHT,
    BLACK, WHITE, GREEN, RED,
    SCROLL_SPEED, SCROLL_THRESHOLD,
    screen, clock
)
from utils import (
    draw_text,
    create_platforms,
    refill_platforms,
    draw_hud,
    smooth_scroll,
    draw_network,
)

# Tuning constants
GRACE_FRAMES        = 10
MAX_FALL_SPEED      = 10
STALL_SECONDS       = 3
FPS                 = 60
STALL_FRAMES        = STALL_SECONDS * FPS
TARGET_SCORE        = 50000
CLOSE_INCENTIVE     = 0.07
MUTATION_THRESHOLD  = 0.1

# Jump punishment & ascension reward
JUMP_PENALTY_FRAMES = 30      # frames window to penalize rapid repeat jumps
JUMP_PENALTY        = 1.0
ASCENSION_REWARD    = 5.0

# Mutation visualization thresholds
LOW_MUT_THR         = 0.2
HIGH_MUT_THR        = 0.5

# Toggles
SHOW_SENSOR_RAYS    = True
SHOW_NETWORK        = True
SHOW_GENOME_SHAPES  = True

# Store best-gen weights for mutation comparison
prev_best_weights   = {}

def run_neat(config_path):
    global prev_best_weights

    # Load NEAT config
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path
    )
    config.fitness_threshold = TARGET_SCORE

    generation_counter = 0
    parent_ids = set()

    def eval_genomes(genomes, config):
        nonlocal generation_counter, parent_ids
        generation_counter += 1
        evolution = (generation_counter - 1) // 25 + 1
        print(f"--- Generation {generation_counter} (Evolution {evolution}) ---")

        # Platforms setup
        platforms = create_platforms()
        spawn_plat = max(platforms, key=lambda p: p.y)

        # Per-agent state
        players, nets, ge, colors, genome_ids = [], [], [], [], []
        velocity_y, can_jump, min_ys = [], [], []
        history, last_improve, last_jump_frame = [], [], []
        mutation_scores = []

        # Initialize genomes
        for idx, (gid, genome) in enumerate(genomes):
            genome_ids.append(gid)
            color = RED if not parent_ids or gid in parent_ids else (
                255,
                int(255 * idx / max(len(genomes)-1,1)),
                int(255 * (1 - idx / max(len(genomes)-1,1)))
            )
            rect = pygame.Rect(
                spawn_plat.x + random.randint(-30,30),
                spawn_plat.y - 30,
                30, 30
            )
            genome.fitness = 0
            net = neat.nn.FeedForwardNetwork.create(genome, config)

            # Compute max weight delta vs last-gen best
            deltas = []
            for node in net.node_evals:
                nid, inputs = node[0], node[-1]
                for in_id, w in inputs:
                    prev = prev_best_weights.get((in_id, nid))
                    if prev is not None:
                        deltas.append(abs(w - prev))
            mut_score = max(deltas) if deltas else 0.0

            players.append(rect)
            nets.append(net)
            ge.append(genome)
            colors.append(color)
            velocity_y.append(0)
            can_jump.append(False)
            min_ys.append(rect.y)
            history.append([])
            last_improve.append(0)
            last_jump_frame.append(0)
            mutation_scores.append(mut_score)

        scroll_offset = 0
        frame_count   = 0
        best_logged   = False

        # Main loop
        while True:
            frame_count += 1
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    pygame.quit()
                    return

            screen.fill(BLACK)
            if players:
                best_idx = min(range(len(players)), key=lambda i: min_ys[i])
                best_y   = min(min_ys)
            else:
                best_idx, best_y = None, HEIGHT

            # Update each genome
            i = 0
            while i < len(players):
                p = players[i]

                # Off-screen death
                if frame_count > GRACE_FRAMES and p.y - scroll_offset > HEIGHT:
                    ge[i].fitness -= 20
                    for lst in (players, nets, ge, colors,
                                velocity_y, can_jump, min_ys,
                                history, genome_ids, last_improve, last_jump_frame, mutation_scores):
                        lst.pop(i)
                    continue

                # Stall death
                if frame_count - last_improve[i] > STALL_FRAMES:
                    ge[i].fitness -= 10
                    for lst in (players, nets, ge, colors,
                                velocity_y, can_jump, min_ys,
                                history, genome_ids, last_improve, last_jump_frame, mutation_scores):
                        lst.pop(i)
                    continue

                # Sense platforms
                close = sorted(platforms, key=lambda pl: abs(pl.y - p.y))[:5]
                if close:
                    cp = close[0]
                    dist = math.hypot(
                        (cp.x + cp.width/2) - (p.x + p.width/2),
                        cp.y - p.y
                    )
                    norm = dist / math.hypot(WIDTH, HEIGHT)
                    ge[i].fitness += (1 - norm) * CLOSE_INCENTIVE

                # Build neural inputs
                inputs = [
                    p.x / WIDTH,
                    p.y / HEIGHT,
                    velocity_y[i] / MAX_FALL_SPEED,
                    float(can_jump[i])
                ]
                for pl in close:
                    flag = 1 if pl.x+pl.width < p.x else -1 if pl.x > p.x else 0
                    d = math.hypot(pl.x - p.x, pl.y - p.y)
                    inputs += [flag, d / math.hypot(WIDTH, HEIGHT)]
                inputs += [0, 1.0] * (5 - len(close))

                if len(inputs) != config.genome_config.num_inputs:
                    raise RuntimeError(
                        f"Expected {config.genome_config.num_inputs}, got {len(inputs)}"
                    )

                ml, mr, jump = nets[i].activate(inputs)
                prev_x = p.x

                # Punish rapid consecutive jumps
                if jump > 0.5 and can_jump[i]:
                    if frame_count - last_jump_frame[i] < JUMP_PENALTY_FRAMES:
                        ge[i].fitness -= JUMP_PENALTY
                    last_jump_frame[i] = frame_count

                # Move and jump
                if ml > 0.5: p.x -= 5
                if mr > 0.5: p.x += 5
                if jump > 0.5 and can_jump[i]:
                    velocity_y[i], can_jump[i] = -16, False

                # Physics
                p.x = max(0, min(WIDTH - p.width, p.x))
                velocity_y[i] = min(velocity_y[i] + 0.5, MAX_FALL_SPEED)
                p.y += velocity_y[i]

                # Landing logic
                for plat in platforms:
                    if p.colliderect(plat) and velocity_y[i] > 0:
                        p.bottom, velocity_y[i], can_jump[i] = plat.top, 0, True
                        # Reward for ascension
                        if plat.y < min_ys[i]:
                            ge[i].fitness += ASCENSION_REWARD
                            bonus = (min_ys[i] - plat.y) / HEIGHT * 20
                            ge[i].fitness += bonus
                            min_ys[i], last_improve[i] = plat.y, frame_count
                            if ge[i].fitness > 100 and not best_logged:
                                with open("best_genome_log.txt", "a") as f:
                                    f.write(f"Inputs:{inputs} Outputs:{[ml,mr,jump]}\n")
                                best_logged = True
                        history[i].append(plat.y)
                        if len(history[i]) > 10:
                            history[i].pop(0)
                        break

                # Fitness tweaks
                ge[i].fitness += scroll_offset * 0.001 - velocity_y[i] * 0.01
                if p.x == prev_x:
                    ge[i].fitness -= 0.5
                i += 1

            # World management
            scroll_offset = smooth_scroll(
                players, platforms, min_ys, last_improve, scroll_offset, best_y
            )
            refill_platforms(platforms)

            # Draw platforms
            for plat in platforms:
                pygame.draw.rect(screen, GREEN, plat)

            # Optional sensor rays
            if SHOW_SENSOR_RAYS:
                for idx, p in enumerate(players):
                    for plat in sorted(platforms, key=lambda pl: abs(pl.y - p.y))[:5]:
                        pygame.draw.line(screen, colors[idx], p.center, plat.center, 1)
                        pygame.draw.circle(screen, colors[idx], plat.center, 4, 1)

            # Draw genomes based on mutation score
            for idx, p in enumerate(players):
                score = mutation_scores[idx]
                if score > HIGH_MUT_THR:
                    # big mutation: blue triangle
                    pts = [(p.centerx, p.top), (p.left, p.bottom), (p.right, p.bottom)]
                    pygame.draw.polygon(screen, (0,0,255), pts)
                elif score > LOW_MUT_THR:
                    # medium mutation: yellow circle
                    pygame.draw.ellipse(screen, (255,255,0), p)
                else:
                    # small/no mutation: normal rect
                    pygame.draw.rect(screen, colors[idx], p)

            # Draw network
            if SHOW_NETWORK and best_idx is not None and best_idx < len(nets):
                draw_network(screen, nets[best_idx], position=(10,10))

            # HUD
            draw_hud(scroll_offset, generation_counter, evolution, len(players))

            pygame.display.flip()
            clock.tick(FPS)

            # End of generation
            if not players:
                if best_idx is not None and best_idx < len(nets):
                    prev_best_weights.clear()
                    for node in nets[best_idx].node_evals:
                        nid, inputs = node[0], node[-1]
                        for in_id, w in inputs:
                            if abs(w) > 1e-6:
                                prev_best_weights[(in_id, nid)] = w
                survivors = sorted(
                    zip(genome_ids, ge), key=lambda x: x[1].fitness, reverse=True
                )
                parent_ids = {pid for pid, _ in survivors[:2]}
                pygame.time.wait(10)
                return

    pop = neat.Population(config)
    pop.add_reporter(neat.StdOutReporter(True))
    pop.add_reporter(neat.StatisticsReporter())
    pop.run(eval_genomes)