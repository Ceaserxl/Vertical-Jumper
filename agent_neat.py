import os
import math
import pickle
import glob
import neat
import pygame
from neat.checkpoint import Checkpointer

from env import WIDTH, HEIGHT, BLACK, WHITE, JumperEnv

MAX_FRAMES = 1200
STAGNATE_LIMIT = 600
FPS = 60
RESUME = True
CHECKPOINT_DIR = "neat-checkpoints"
CHECKPOINT_PREFIX = os.path.join(CHECKPOINT_DIR, "neat-checkpoint-")

if not os.path.exists(CHECKPOINT_DIR):
    os.makedirs(CHECKPOINT_DIR)

generation_counter = 0
best_score_ever = 0
network_mode = False

class EarlyExitException(Exception):
    pass

def find_latest_checkpoint(prefix):
    files = glob.glob(f"{prefix}*")
    return max(files, key=os.path.getctime) if files else None

def color_from_sensor(norm_dist):
    r = int(255 * (1 - norm_dist))
    g = int(255 * norm_dist)
    return (r, g, 0)

def draw_network_sideways(surface, agent, cell_rect, node_alpha=160, edge_alpha=80):
    genome = agent['genome']
    obs = agent['obs']
    config = agent['config']

    left, top, width, height = cell_rect
    num_inputs = len(obs)
    num_outputs = 3
    node_inputs = list(config.genome_config.input_keys)
    node_outputs = list(config.genome_config.output_keys)
    node_hidden = [k for k in genome.nodes if k not in node_inputs + node_outputs]
    node_pos = {}

    in_y_gap = height // (num_inputs + 1)
    out_y_gap = height // (num_outputs + 1)
    x_input = left + 35
    x_output = left + width - 35
    x_hidden = left + width // 2

    for i, k in enumerate(node_inputs):
        y = top + in_y_gap * (i + 1)
        node_pos[k] = (x_input, y)
    for i, k in enumerate(node_outputs):
        y = top + out_y_gap * (i + 1)
        node_pos[k] = (x_output, y)
    if node_hidden:
        h_gap = height // (len(node_hidden) + 1)
        for i, k in enumerate(sorted(node_hidden)):
            y = top + h_gap * (i + 1)
            node_pos[k] = (x_hidden, y)

    net_surface = pygame.Surface((width, height), pygame.SRCALPHA)

    for cg in genome.connections.values():
        if not cg.enabled:
            continue
        src, dst = cg.key
        if src in node_pos and dst in node_pos:
            sx, sy = node_pos[src][0] - left, node_pos[src][1] - top
            dx, dy = node_pos[dst][0] - left, node_pos[dst][1] - top
            weight = cg.weight
            col = (0, 200, 0, edge_alpha) if weight > 0 else (200, 30, 30, edge_alpha)
            w = min(4, max(1, int(abs(weight) * 2)))
            pygame.draw.line(net_surface, col, (sx, sy), (dx, dy), w)

    for i, k in enumerate(node_inputs):
        x, y = node_pos[k][0] - left, node_pos[k][1] - top
        color = (90, 130, 255, node_alpha)
        if i >= num_inputs - 8:  # last 8 inputs are sensors
            norm_dist = obs[i]
            color = (*color_from_sensor(norm_dist), node_alpha)
        pygame.draw.circle(net_surface, color, (int(x), int(y)), 16)

    for i, k in enumerate(node_outputs):
        x, y = node_pos[k][0] - left, node_pos[k][1] - top
        pygame.draw.circle(net_surface, (100, 220, 220, node_alpha), (int(x), int(y)), 16)

    for k in node_hidden:
        x, y = node_pos[k][0] - left, node_pos[k][1] - top
        pygame.draw.circle(net_surface, (180, 180, 180, node_alpha), (int(x), int(y)), 10)

    surface.blit(net_surface, (left, top))

def eval_genomes(genomes, config):
    global generation_counter, best_score_ever, network_mode

    generation_counter += 1
    env_obj = JumperEnv(agent_count=len(genomes))
    obs_list = env_obj.reset()

    agents = []
    for i, (genome_id, genome) in enumerate(genomes):
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        agents.append({
            'genome_id': genome_id,
            'genome': genome,
            'net': net,
            'score': 0,
            'stagnate': 0,
            'alive': True,
            'obs': obs_list[i],
            'config': config,
            'env': env_obj,
        })
        genome.fitness = 0.0

    frames = 0
    clock = pygame.time.Clock()
    best_alive_agent = None

    while any(agent['alive'] for agent in agents) and frames < MAX_FRAMES:
        frames += 1

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_s:
                    raise EarlyExitException()
                if event.key == pygame.K_v:
                    env_obj.SHOW_SENSORS = not env_obj.SHOW_SENSORS
                if event.key == pygame.K_n:
                    network_mode = not network_mode

        actions = []
        for agent in agents:
            if not agent['alive']:
                actions.append(None)
                continue
            output = agent['net'].activate(agent['obs'])
            action = 0 if output[0] > 0.5 else 1 if output[1] > 0.5 else 2 if output[2] > 0.5 else None
            actions.append(action)

        obs_list, reward_list, done_list, _ = env_obj.step(actions)

        for i, agent in enumerate(agents):
            if not agent['alive']:
                continue
            agent['obs'] = obs_list[i]
            agent['score'] += max(0, reward_list[i])
            agent['genome'].fitness += reward_list[i]
            agent['stagnate'] = 0 if reward_list[i] > 0 else agent['stagnate'] + 1

            if done_list[i] or agent['stagnate'] > STAGNATE_LIMIT:
                agent['alive'] = False

        alive_agents = [a for a in agents if a['alive']]
        best_alive_agent = max(alive_agents, key=lambda a: a['score']) if alive_agents else None

        screen = pygame.display.get_surface()
        screen.fill(BLACK)
        env_obj.render(screen)

        if network_mode and best_alive_agent:
            draw_network_sideways(screen, best_alive_agent, (20, 20, WIDTH - 40, HEIGHT - 40))

        current_best_score = max(agent['score'] for agent in agents)
        if current_best_score > best_score_ever:
            best_score_ever = current_best_score

        pygame.display.set_caption(
            f"NEAT Gen:{generation_counter} Alive:{sum(a['alive'] for a in agents)}/{len(agents)} Best:{int(best_score_ever)}"
        )
        pygame.display.flip()
        clock.tick(FPS)

def run_neat(config_path):
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path
    )

    checkpoint_file = find_latest_checkpoint(CHECKPOINT_PREFIX)
    if RESUME and checkpoint_file:
        print(f"Resuming from {checkpoint_file}")
        p = Checkpointer.restore_checkpoint(checkpoint_file)
    else:
        p = neat.Population(config)

    p.add_reporter(neat.StdOutReporter(True))
    p.add_reporter(neat.StatisticsReporter())
    p.add_reporter(Checkpointer(generation_interval=5, filename_prefix=CHECKPOINT_PREFIX))

    try:
        winner = p.run(eval_genomes, 5000)
    except EarlyExitException:
        print("Training stopped by user.")
        return
    except neat.CompleteExtinctionException:
        print("All genomes extinct.")
        winner = p.best_genome if hasattr(p, "best_genome") else None

    if winner:
        with open("winner.pkl", "wb") as f:
            pickle.dump(winner, f)
        print("Winner saved to winner.pkl")

if __name__ == "__main__":
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    config_path = os.path.join(os.path.dirname(__file__), "config-feedforward.txt")
    run_neat(config_path)
