# utils.py
import pygame
import random
from config import * 

# Physics‐derived maxima for reachable gaps
MAX_JUMP_VELOCITY = 16      # initial jump impulse
GRAVITY           = 0.5     # per‐frame gravity
# max vertical gap ≈ v²/(2g), with safety factor
MAX_VERT_GAP      = (MAX_JUMP_VELOCITY**2) / (2 * GRAVITY) * 0.8
# max horizontal speed per frame
MAX_HORIZ_SPEED   = 5
# max horizontal gap covered during jump arc (≈ airtime × speed)
MAX_HORIZ_GAP     = (2 * (MAX_JUMP_VELOCITY / GRAVITY)) * MAX_HORIZ_SPEED * 0.8

def draw_text(text, size, color, x, y, center=True):
    font = pygame.font.SysFont(None, size)
    surf = font.render(text, True, color)
    rect = surf.get_rect(center=(x, y) if center else (x, y))
    pygame.display.get_surface().blit(surf, rect)

def create_platforms():
    """
    Generate 10 platforms with gaps that a human (or agent) can actually reach.
    """
    platforms = []
    y = HEIGHT
    # start with a floor platform
    platforms.append(pygame.Rect(WIDTH//2 - 50, y - 20, 150, 5))
    for _ in range(9):
        # enforce reachable vertical gap
        gap = random.uniform(60, min(120, MAX_VERT_GAP))
        y -= gap
        # enforce reachable horizontal offset
        last_x = platforms[-1].x
        dx = random.uniform(-MAX_HORIZ_GAP, MAX_HORIZ_GAP)
        x = max(0, min(WIDTH - 100, last_x + dx))
        platforms.append(pygame.Rect(int(x), int(y), 150, 5))
    return platforms

def refill_platforms(platforms):
    """
    Keep exactly 10 platforms in view, using the same reachable‐gap rules.
    """
    while len(platforms) < 10:
        top_y = min(p.y for p in platforms)
        gap   = random.uniform(60, min(120, MAX_VERT_GAP))
        y     = top_y - gap
        last_x = max(platforms, key=lambda p: p.y).x
        dx     = random.uniform(-MAX_HORIZ_GAP, MAX_HORIZ_GAP)
        x      = max(0, min(WIDTH - 100, last_x + dx))
        platforms.append(pygame.Rect(int(x), int(y), 150, 5))
    # remove any that fell off bottom
    platforms[:] = [p for p in platforms if p.y < HEIGHT]

def draw_hud(scroll_offset, generation, evolution, alive):
    """
    Draw the score, generation, evolution phase, and alive count.
    """
    hx = WIDTH - 200
    draw_text(f"Best Score: {int(scroll_offset)}", 24, WHITE, hx, 10, center=False)
    draw_text(f"Gen: {generation}",            24, WHITE, hx, 40, center=False)
    draw_text(f"Evol: {evolution}",            24, WHITE, hx, 70, center=False)
    draw_text(f"Alive: {alive}",               24, WHITE, hx,100, center=False)

def smooth_scroll(players, platforms, min_ys, last_improve, scroll_offset, best_y):
    """
    Smoothly scroll the world to keep the best agent in the bottom quarter.
    """
    if best_y < SCROLL_THRESHOLD:
        shift = min(SCROLL_SPEED, SCROLL_THRESHOLD - best_y)
        scroll_offset += shift
        for p in players:
            p.y += shift
        for i in range(len(min_ys)):
            min_ys[i]       += shift
            last_improve[i] += shift
        for plat in platforms:
            plat.y += shift
    return scroll_offset

def draw_network(surface, network, position=(0, 0), node_radius=8):
    """
    Visualize the NEAT network: green for positive weights, red for negative.
    """
    x0, y0 = position
    layer_spacing, node_gap = 80, 20
    used, conns = set(), []
    # gather connections
    for node_eval in network.node_evals:
        nid    = node_eval[0]
        inputs = node_eval[-1]
        for in_id, w in inputs:
            if abs(w) > 1e-6:
                used.add(in_id); used.add(nid)
                conns.append((in_id, nid, w))
    inp_nodes = [n for n in network.input_nodes  if n in used]
    hid_nodes = [n for n in used if n not in network.input_nodes and n not in network.output_nodes]
    out_nodes = [n for n in network.output_nodes if n in used]

    def place(nodes, layer, center=False):
        pos = {}
        total_h = max(len(inp_nodes), len(hid_nodes)) * node_gap
        offset  = (total_h - len(nodes)*node_gap)//2 if center else 0
        px = x0 + layer * layer_spacing
        for i, nid in enumerate(nodes):
            pos[nid] = (px, y0 + offset + i*node_gap)
        return pos

    node_pos = {}
    node_pos.update(place(inp_nodes,  0))
    node_pos.update(place(hid_nodes,  1))
    node_pos.update(place(out_nodes,  2, center=True))

    # draw connections
    for in_id, out_id, w in conns:
        if in_id in node_pos and out_id in node_pos:
            color = (0,255,0) if w > 0 else (255,0,0)
            pygame.draw.line(surface, color, node_pos[in_id], node_pos[out_id], 1)
    # draw nodes
    for nid, (nx, ny) in node_pos.items():
        pygame.draw.circle(surface, (255,255,255), (nx, ny), node_radius)
