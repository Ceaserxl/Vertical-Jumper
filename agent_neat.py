import os
import math
import pickle
import neat
import pygame
import env

from env import WIDTH, HEIGHT, GREEN, RED, BLACK, WHITE, draw_text, JumperEnv, SENSOR_LABELS

MAX_FRAMES = 1200
STAGNATE_LIMIT = 240
FPS = 60
RESUME = False
CHECKPOINT_FILE = "neat-checkpoint.pkl"

generation_counter = 0
best_score_ever = 0
network_mode = False

class EarlyExitException(Exception):
    pass

def color_from_sensor(norm_dist):
    r = int(255 * (1 - norm_dist))
    g = int(255 * norm_dist)
    return (r, g, 0)

def draw_network_sideways(surface, agent, cell_rect, node_alpha=160, edge_alpha=80):
    """Draws the neural network sideways with see-through (alpha-blended) nodes and edges."""
    genome = agent['genome']
    obs = agent['obs']
    net = agent['net']
    config = agent['config']
    env_obj = agent['env']

    left, top, width, height = cell_rect

    num_inputs = len(obs)
    num_outputs = 3
    node_inputs = list(config.genome_config.input_keys)
    node_outputs = list(config.genome_config.output_keys)
    node_hidden = [k for k in genome.nodes.keys() if k not in node_inputs and k not in node_outputs]
    node_pos = {}

    # Arrange inputs vertically on left, outputs on right, hidden in center
    in_y_gap = height // (num_inputs + 1)
    out_y_gap = height // (num_outputs + 1)
    x_input = left + 35
    x_output = left + width - 35
    x_hidden = left + width // 2

    # Node positions
    for i, k in enumerate(node_inputs):
        y = top + in_y_gap * (i+1)
        node_pos[k] = (x_input, y)
    for i, k in enumerate(node_outputs):
        y = top + out_y_gap * (i+1)
        node_pos[k] = (x_output, y)
    if node_hidden:
        h_gap = height // (len(node_hidden) + 1)
        for i, k in enumerate(sorted(node_hidden)):
            y = top + h_gap * (i+1)
            node_pos[k] = (x_hidden, y)

    # Transparent surface
    net_surface = pygame.Surface((width, height), pygame.SRCALPHA)

    # Draw connections
    for cg in genome.connections.values():
        if not cg.enabled:
            continue
        src, dst = cg.key
        if src in node_pos and dst in node_pos:
            sx, sy = node_pos[src][0] - left, node_pos[src][1] - top
            dx, dy = node_pos[dst][0] - left, node_pos[dst][1] - top
            weight = cg.weight
            col = (0,200,0, edge_alpha) if weight > 0 else (200,30,30, edge_alpha)
            w = min(4, max(1, int(abs(weight)*2)))
            pygame.draw.line(net_surface, col, (sx, sy), (dx, dy), w)

    # Draw input nodes, color sensors even if sensors aren't visible
    sensor_count = 8  # Last 8 inputs are sensors
    for i, k in enumerate(node_inputs):
        x, y = node_pos[k][0] - left, node_pos[k][1] - top
        color = (90, 130, 255, node_alpha)  # default for non-sensor
        # Color the sensor nodes using live obs (always updates)
        if i >= num_inputs - sensor_count:
            norm_dist = obs[i]
            color = (*color_from_sensor(norm_dist), node_alpha)
        pygame.draw.circle(net_surface, color, (int(x), int(y)), 16)
        txt = pygame.font.SysFont(None, 14).render(str(i+1), True, WHITE)
        net_surface.blit(txt, (x-7, y-8))
    # Outputs
    for i, k in enumerate(node_outputs):
        x, y = node_pos[k][0] - left, node_pos[k][1] - top
        pygame.draw.circle(net_surface, (100, 220, 220, node_alpha), (int(x), int(y)), 16)
        label = ["←", "→", "⎵"][i]
        txt = pygame.font.SysFont(None, 20).render(label, True, WHITE)
        net_surface.blit(txt, (x-8, y-12))
    # Hidden
    for k in node_hidden:
        x, y = node_pos[k][0] - left, node_pos[k][1] - top
        pygame.draw.circle(net_surface, (180,180,180, node_alpha), (int(x), int(y)), 10)

    surface.blit(net_surface, (left, top))

def render_agent_env(surface, agent, col, row, cell_width, cell_height, show_network=False):
    x_offset = col * cell_width
    y_offset = row * cell_height

    env_obj = agent['env']

    # Prepare a subsurface for this cell (prevents drawing outside the agent's window)
    cell_surface = surface.subsurface((x_offset, y_offset, cell_width, cell_height))
    cell_surface.fill((30, 30, 30) if not agent['alive'] else BLACK)

    # --- World to cell mapping ---
    # Draw platforms (relative to agent's own scroll)
    for plat in env_obj.platforms:
        plat_y = plat.y - env_obj.scroll
        if 0 <= plat_y < HEIGHT:
            px = int(plat.x * cell_width / WIDTH)
            py = int(plat_y * cell_height / HEIGHT)
            pw = int(plat.width * cell_width / WIDTH)
            ph = max(2, int(plat.height * cell_height / HEIGHT))
            pygame.draw.rect(cell_surface, GREEN, (px, py, pw, ph))

    # Draw player
    player_y = env_obj.player.y - env_obj.scroll
    px = int(env_obj.player.x * cell_width / WIDTH)
    py = int(player_y * cell_height / HEIGHT)
    pw = int(env_obj.player.width * cell_width / WIDTH)
    ph = int(env_obj.player.height * cell_height / HEIGHT)
    color = (255, 64, 64) if agent['alive'] else (80, 80, 80)
    pygame.draw.rect(cell_surface, color, (px, py, pw, ph))

    # Draw sensors (relative)
    if env.SHOW_SENSORS and hasattr(env_obj, "_sensor_rays"):
        for sx, sy, ex, ey, norm_dist in env_obj._sensor_rays:
            sx_cell = int((sx) * cell_width / WIDTH)
            sy_cell = int((sy - env_obj.scroll) * cell_height / HEIGHT)
            ex_cell = int((ex) * cell_width / WIDTH)
            ey_cell = int((ey - env_obj.scroll) * cell_height / HEIGHT)
            color = color_from_sensor(norm_dist)
            pygame.draw.line(cell_surface, color, (sx_cell, sy_cell), (ex_cell, ey_cell), 2)
            pygame.draw.circle(cell_surface, color, (int(ex_cell), int(ey_cell)), 3)

    # Overlay network (see-through)
    if show_network:
        draw_network_sideways(cell_surface, agent, (0, 0, cell_width, cell_height))

    # UI overlays
    pygame.draw.rect(cell_surface, WHITE, (0, 0, cell_width, cell_height), 3)
    draw_text(cell_surface, f"S:{agent['score']}", 16, WHITE, 36, 12)
    draw_text(cell_surface, f"ID:{agent['genome_id']}", 12, WHITE, 36, 28)


def save_checkpoint(gen, agents, best_score, config):
    with open(CHECKPOINT_FILE, "wb") as f:
        pickle.dump((gen, [a['score'] for a in agents], best_score, config), f)
    print(f"Checkpoint saved at generation {gen}.")

def load_checkpoint(config):
    global generation_counter, best_score_ever
    if os.path.exists(CHECKPOINT_FILE):
        try:
            with open(CHECKPOINT_FILE, "rb") as f:
                generation_counter, agent_scores, best_score_ever, loaded_config = pickle.load(f)
                print("Loaded checkpoint. Generation:", generation_counter)
                return True
        except Exception as e:
            print("Failed to load checkpoint:", e)
    generation_counter = 0
    best_score_ever = 0
    return False

def eval_genomes(genomes, config):
    global generation_counter, best_score_ever, network_mode

    pop_size = config.pop_size
    grid_cols = math.ceil(math.sqrt(pop_size))
    grid_rows = math.ceil(pop_size / grid_cols)
    cell_width = WIDTH // grid_cols
    cell_height = HEIGHT // grid_rows

    generation_counter += 1

    if generation_counter % 10 == 0:
        dummy_agents = [{"score": 0} for _ in genomes]
        save_checkpoint(generation_counter, dummy_agents, best_score_ever, config)

    agents = []
    for genome_id, genome in genomes:
        env_obj = JumperEnv()
        obs = env_obj.reset()
        agent = {
            'env': env_obj,
            'obs': obs,
            'score': 0,
            'genome': genome,
            'net': neat.nn.FeedForwardNetwork.create(genome, config),
            'alive': True,
            'genome_id': genome_id,
            'min_y': env_obj.player.y,
            'stagnate': 0,
            'config': config,
        }
        genome.fitness = 0.0
        agents.append(agent)

    frames = 0
    clock = pygame.time.Clock()

    while any(agent['alive'] for agent in agents) and frames < MAX_FRAMES:
        frames += 1

        # Handle key events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_s:
                    save_checkpoint(generation_counter, agents, best_score_ever, config)
                    raise EarlyExitException()
                if event.key == pygame.K_v:
                    env.SHOW_SENSORS = not env.SHOW_SENSORS
                    print(f"SHOW_SENSORS set to {env.SHOW_SENSORS}")
                if event.key == pygame.K_n:
                    network_mode = not network_mode

        for agent in agents:
            if not agent['alive']:
                continue
            # Always update obs to get current sensor values (even if sensors hidden)
            obs = agent['env']._get_obs()
            agent['obs'] = obs
            output = agent['net'].activate(agent['obs'])
            action = None
            if output[0] > 0.5:
                action = 0
            elif output[1] > 0.5:
                action = 1
            elif output[2] > 0.5:
                action = 2

            obs, reward, done_env, info = agent['env'].step(action)
            agent['obs'] = obs
            agent['score'] += max(0, reward)
            agent['genome'].fitness += reward
            if reward < 0:
                agent['stagnate'] += 1
            else:
                agent['stagnate'] = 0
            if done_env or agent['stagnate'] > STAGNATE_LIMIT:
                agent['alive'] = False

        screen = pygame.display.get_surface()
        screen.fill(BLACK)
        alive_count = sum(1 for a in agents if a['alive'])

        for idx, agent in enumerate(agents):
            row = idx // grid_cols
            col = idx % grid_cols
            render_agent_env(
                screen, agent, col, row, cell_width, cell_height, show_network=network_mode
            )

        all_scores = [a['score'] for a in agents]
        current_best_score = max(all_scores)
        if current_best_score > best_score_ever:
            best_score_ever = current_best_score

        pygame.display.set_caption(
            f"NEAT Gen:{generation_counter}  Pop:{alive_count}/{pop_size}  Best:{best_score_ever}"
        )

        pygame.display.flip()
        clock.tick(FPS)

def run_neat(config_path):
    config = neat.Config(
        neat.DefaultGenome, neat.DefaultReproduction,
        neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path
    )
    load_checkpoint(config)

    if RESUME and os.path.exists("winner.pkl"):
        with open("winner.pkl", "rb") as f:
            winner = pickle.load(f)
        print("Loaded winner genome. Resuming training.")
        p = neat.Population(config)
        for gid in p.population:
            p.population[gid] = winner
    else:
        p = neat.Population(config)

    p.add_reporter(neat.StdOutReporter(True))
    p.add_reporter(neat.StatisticsReporter())

    try:
        winner = p.run(eval_genomes, 5000)
    except EarlyExitException:
        print("Training stopped by user, returning to main menu.")
        return
    except neat.CompleteExtinctionException:
        print("Training completed! Solution found.")
        winner = p.best_genome if hasattr(p, "best_genome") else None

    if winner:
        with open("winner.pkl", "wb") as f:
            pickle.dump(winner, f)
        print("Winner saved to winner.pkl")

if __name__ == "__main__":
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config-feedforward.txt")
    run_neat(config_path)
