import pygame
import random
import math

WIDTH, HEIGHT = 600, 800
PLATFORM_WIDTH = 100
PLATFORM_HEIGHT = 5
PLAYER_SIZE = 30
GREEN = (50, 200, 50)
RED = (220, 50, 50)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
SHOW_SENSORS = False
SENSOR_ANGLES = [0, 45, 90, 135, 180, 225, 270, 315]
SENSOR_LABELS = ["E", "NE", "N", "NW", "W", "SW", "S", "SE"]

def draw_text(surf, text, size, color, x, y, center=True):
    font = pygame.font.SysFont(None, size)
    txt = font.render(text, True, color)
    rect = txt.get_rect(center=(x, y) if center else (x, y))
    surf.blit(txt, rect)

def draw_hud(score):
    surf = pygame.display.get_surface()
    draw_text(surf, f"Score: {score}", 28, WHITE, WIDTH - 100, 20, center=False)

def wall_intersect(start, dx, dy, left, top, right, bottom, max_dist):
    x0, y0 = start
    t_vals = []
    if dx != 0:
        t_left = (left - x0) / dx
        if t_left > 0: t_vals.append(t_left)
        t_right = (right - x0) / dx
        if t_right > 0: t_vals.append(t_right)
    if dy != 0:
        t_top = (top - y0) / dy
        if t_top > 0: t_vals.append(t_top)
        t_bottom = (bottom - y0) / dy
        if t_bottom > 0: t_vals.append(t_bottom)
    min_t = max_dist
    for t in t_vals:
        if 0 < t < min_t:
            min_t = t
    wx = x0 + dx * min_t
    wy = y0 + dy * min_t
    return wx, wy, min_t

class JumperEnv:
    def __init__(self, agent_count=1):
        self.agent_count = agent_count
        self.players = []
        self.vy = []
        self.can_jump = []
        self.score = []
        self.history = []
        self.highest_platform_y = []
        self.scroll = 0
        self.platforms = []
        self.frame = 0
        self._sensor_rays = [[] for _ in range(agent_count)]

    def create_platforms(self):
        plats = []
        y = HEIGHT
        for _ in range(10):
            gap = random.randint(60, 120)
            y -= gap
            x = random.randint(0, WIDTH - PLATFORM_WIDTH)
            plats.append(pygame.Rect(x, y, PLATFORM_WIDTH, PLATFORM_HEIGHT))
        return plats

    def refill_platforms(self):
        self.platforms = [p for p in self.platforms if p.y - self.scroll < HEIGHT + 50]
        while len(self.platforms) < 10:
            top_y = min(p.y for p in self.platforms) if self.platforms else HEIGHT
            gap = random.randint(60, 120)
            y = top_y - gap
            x = random.randint(0, WIDTH - PLATFORM_WIDTH)
            self.platforms.append(pygame.Rect(x, y, PLATFORM_WIDTH, PLATFORM_HEIGHT))

    def reset(self):
        self.platforms = self.create_platforms()
        spawn = max(self.platforms, key=lambda p: p.y)
        px = spawn.x + (spawn.width - PLAYER_SIZE) // 2
        py = spawn.y - PLAYER_SIZE

        self.players = [pygame.Rect(px, py, PLAYER_SIZE, PLAYER_SIZE) for _ in range(self.agent_count)]
        self.vy = [0.0] * self.agent_count
        self.can_jump = [False] * self.agent_count
        self.score = [0] * self.agent_count
        self.highest_platform_y = [spawn.y] * self.agent_count
        self.history = [[] for _ in range(self.agent_count)]
        self.scroll = 0
        self.frame = 0
        self._sensor_rays = [[] for _ in range(self.agent_count)]
        return [self._get_obs(i) for i in range(self.agent_count)]

    def step(self, actions):
        self.frame += 1
        rewards = [0.0] * self.agent_count
        dones = [False] * self.agent_count
        observations = [None] * self.agent_count

        for i in range(self.agent_count):
            player = self.players[i]
            action = actions[i]

            if action is None:
                keys = pygame.key.get_pressed()
                left = keys[pygame.K_LEFT]
                right = keys[pygame.K_RIGHT]
                jump = keys[pygame.K_SPACE]
            else:
                left = (action == 0)
                right = (action == 1)
                jump = (action == 2)

            if left:
                player.x -= 5
            if right:
                player.x += 5
            if jump and self.can_jump[i]:
                self.vy[i] = -16
                self.can_jump[i] = False

            player.x = max(0, min(WIDTH - player.width, player.x))
            self.vy[i] = min(self.vy[i] + 0.5, 10)
            player.y += self.vy[i]

            if player.y - self.scroll > HEIGHT:
                rewards[i] -= 20.0
                dones[i] = True
                continue

            for plat in self.platforms:
                if player.colliderect(plat) and self.vy[i] > 0:
                    player.bottom = plat.top
                    self.vy[i] = 0
                    self.can_jump[i] = True
                    if plat.y < self.highest_platform_y[i]:
                        self.score[i] += 100
                        rewards[i] += 100
                        self.highest_platform_y[i] = plat.y
                    if plat.y in self.history[i]:
                        rewards[i] -= 1.0
                    self.history[i].append(plat.y)
                    if len(self.history[i]) > 10:
                        self.history[i].pop(0)
                    break

            observations[i] = self._get_obs(i)

        self.refill_platforms()
        self.scroll_logic()
        return observations, rewards, dones, {}

    def _get_obs(self, i):
        player = self.players[i]
        cx = player.x + player.width // 2
        cy = player.y + player.height // 2

        visible_top = self.scroll
        visible_bottom = self.scroll + HEIGHT
        max_sensor_length = 1000
        sensor_distances = []
        if SHOW_SENSORS:
            self._sensor_rays[i] = []

        for ang in SENSOR_ANGLES:
            dx = math.cos(math.radians(ang))
            dy = -math.sin(math.radians(ang))
            wall_x, wall_y, wall_dist = wall_intersect((cx, cy), dx, dy, 0, visible_top, WIDTH, visible_bottom, max_sensor_length)
            min_dist = wall_dist
            hit_x, hit_y = wall_x, wall_y

            for step in range(10, int(max_sensor_length), 10):
                x = cx + dx * step
                y = cy + dy * step
                if x < 0 or x > WIDTH or y < visible_top or y > visible_bottom:
                    break
                pt = pygame.Rect(x - 1, y - 1, 2, 2)
                if any(p.colliderect(pt) for p in self.platforms):
                    dist = math.hypot(x - cx, y - cy)
                    if dist < min_dist:
                        min_dist = dist
                        hit_x, hit_y = x, y
                    break

            norm = min(min_dist / max_sensor_length, 1.0)
            sensor_distances.append(norm)
            if SHOW_SENSORS:
                self._sensor_rays[i].append((cx, cy, hit_x, hit_y, norm))

        obs = [player.x / WIDTH, player.y / HEIGHT, self.vy[i] / 10.0, float(self.can_jump[i]), player.y / HEIGHT]
        obs.extend(sensor_distances)
        return obs

    def scroll_logic(self):
        min_highest = min(self.highest_platform_y)
        target_scroll = min_highest - HEIGHT * 0.75
        delta = target_scroll - self.scroll
        self.scroll += delta * 0.1

    def render(self, surface=None):
        surf = surface or pygame.display.get_surface()
        surf.fill(BLACK)

        pygame.draw.rect(surf, WHITE, (0, 0, WIDTH, HEIGHT), 3)

        for plat in self.platforms:
            y = plat.y - self.scroll
            pygame.draw.rect(surf, GREEN, (plat.x, y, plat.width, plat.height))

        for player in self.players:
            py = player.y - self.scroll
            pygame.draw.rect(surf, RED, (player.x, py, player.width, player.height))

        if SHOW_SENSORS:
            for rays in self._sensor_rays:
                for sx, sy, ex, ey, norm_dist in rays:
                    r = int(255 * (1 - norm_dist))
                    g = int(255 * norm_dist)
                    color = (r, g, 0)
                    sy_disp = sy - self.scroll
                    ey_disp = ey - self.scroll
                    pygame.draw.line(surf, color, (sx, sy_disp), (ex, ey_disp), 2)
                    pygame.draw.circle(surf, color, (int(ex), int(ey_disp)), 4)

