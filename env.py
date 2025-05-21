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
SENSOR_LABELS = [
    "E", "NE", "N", "NW", "W", "SW", "S", "SE"
]

def draw_text(surf, text, size, color, x, y, center=True):
    font = pygame.font.SysFont(None, size)
    surf = pygame.display.get_surface() if surf is None else surf
    txt = font.render(text, True, color)
    rect = txt.get_rect(center=(x, y) if center else (x, y))
    surf.blit(txt, rect)

def draw_hud(score):
    surf = pygame.display.get_surface()
    draw_text(surf, f"Score: {score}", 28, WHITE, WIDTH - 100, 20, center=False)

def wall_intersect(start, dx, dy, left, top, right, bottom, max_dist):
    """Returns (x, y, dist) where the ray from start in (dx,dy) first hits the visible wall."""
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

def platform_or_wall_intersect(start, dx, dy, platforms, left, top, right, bottom, max_dist):
    wx, wy, wall_dist = wall_intersect(start, dx, dy, left, top, right, bottom, max_dist)
    min_dist = wall_dist
    hit_x, hit_y = wx, wy
    # Check for intersection with all platforms
    for plat in platforms:
        steps = int(max_dist)
        for step in range(1, steps):
            x = start[0] + dx * step
            y = start[1] + dy * step
            if x < left or x > right or y < top or y > bottom:
                break
            point_rect = pygame.Rect(x-1, y-1, 2, 2)
            if plat.colliderect(point_rect):
                dist = math.hypot(x - start[0], y - start[1])
                if dist < min_dist:
                    min_dist = dist
                    hit_x, hit_y = x, y
                break
    return hit_x, hit_y, min_dist

class JumperEnv:
    def __init__(self):
        self.platforms = []
        self.player = None
        self.vy = 0
        self.can_jump = False
        self.frame = 0
        self.score = 0
        self.highest_platform_y = None
        self.scroll = 0
        self.history = []
        self._auto_reset = False

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
        self.player = pygame.Rect(px, py, PLAYER_SIZE, PLAYER_SIZE)
        self.vy = 0
        self.can_jump = False
        self.frame = 0
        self.score = 0
        self.highest_platform_y = spawn.y
        self.scroll = 0
        self.history = []
        self._auto_reset = True
        return self._get_obs()

    def step(self, action=None):
        self.frame += 1

        left = right = jump = False
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
            self.player.x -= 5
        if right:
            self.player.x += 5
        if jump and self.can_jump:
            self.vy = -16
            self.can_jump = False

        self.player.x = max(0, min(WIDTH - self.player.width, self.player.x))
        self.vy = min(self.vy + 0.5, 10)
        self.player.y += self.vy

        reward = 0
        done = False

        # Death if fall below screen
        if self.player.y - self.scroll > HEIGHT:
            reward -= 20.0
            done = True
            obs = self.reset()
            return obs, reward, done, {"reset": True}

        # Platform collisions
        for plat in self.platforms:
            if self.player.colliderect(plat) and self.vy > 0:
                self.player.bottom, self.vy, self.can_jump = plat.top, 0, True
                if plat.y < self.highest_platform_y:
                    self.score += 100
                    reward += 100
                    self.highest_platform_y = plat.y
                if plat.y in self.history:
                    reward -= 1.0
                self.history.append(plat.y)
                if len(self.history) > 10:
                    self.history.pop(0)
                break

        # Scrolling logic
        wanted = self.highest_platform_y - HEIGHT * 0.75
        delta = wanted - self.scroll
        self.scroll += delta * 0.1

        self.refill_platforms()

        # Always update obs and sensor rays every step!
        obs = self._get_obs()
        return obs, reward, done, {"reset": False}

    def _get_obs(self):
        player_cx = self.player.x + self.player.width // 2
        player_cy = self.player.y + self.player.height // 2

        visible_top = self.scroll
        visible_bottom = self.scroll + HEIGHT
        visible_left = 0
        visible_right = WIDTH

        max_sensor_length = 1000
        sensor_distances = []
        # Only create _sensor_rays list if rendering sensors
        if SHOW_SENSORS:
            self._sensor_rays = []

        for idx, ang in enumerate(SENSOR_ANGLES):
            dx = math.cos(math.radians(ang))
            dy = -math.sin(math.radians(ang))

            # Find intersection with wall first
            wall_x, wall_y, wall_dist = wall_intersect(
                (player_cx, player_cy), dx, dy,
                visible_left, visible_top, visible_right, visible_bottom, max_sensor_length
            )
            min_dist = wall_dist
            hit_x, hit_y = wall_x, wall_y

            # Platform check with coarser step (faster!)
            step_size = 10
            for step in range(step_size, int(max_sensor_length), step_size):
                x = player_cx + dx * step
                y = player_cy + dy * step
                if x < visible_left or x > visible_right or y < visible_top or y > visible_bottom:
                    break
                pt_rect = pygame.Rect(x-1, y-1, 2, 2)
                if any(plat.colliderect(pt_rect) for plat in self.platforms):
                    dist = math.hypot(x - player_cx, y - player_cy)
                    if dist < min_dist:
                        min_dist = dist
                        hit_x, hit_y = x, y
                    break

            norm_dist = min(min_dist / max_sensor_length, 1.0)
            sensor_distances.append(norm_dist)
            if SHOW_SENSORS:
                self._sensor_rays.append((player_cx, player_cy, hit_x, hit_y, norm_dist))

        vertical_height = self.player.y / HEIGHT
        obs = [
            self.player.x / WIDTH,
            self.player.y / HEIGHT,
            self.vy / 10,
            float(self.can_jump),
            vertical_height,
        ]
        obs.extend(sensor_distances)
        return obs


    def render(self):
        surf = pygame.display.get_surface()
        surf.fill(BLACK)
        # Draw border for visible area
        pygame.draw.rect(surf, WHITE, (0, 0, WIDTH, HEIGHT), 3)
        # Draw platforms
        for plat in self.platforms:
            y = plat.y - self.scroll
            pygame.draw.rect(surf, GREEN, (plat.x, y, plat.width, plat.height))
        # Draw player
        py = self.player.y - self.scroll
        pygame.draw.rect(surf, RED, (self.player.x, py, self.player.width, self.player.height))
        # Draw sensors
        if SHOW_SENSORS and hasattr(self, '_sensor_rays'):
            for i, (sx, sy, ex, ey, norm_dist) in enumerate(self._sensor_rays):
                r = int(255 * (1 - norm_dist))
                g = int(255 * norm_dist)
                color = (r, g, 0)
                sy_disp = sy - self.scroll
                ey_disp = ey - self.scroll
                pygame.draw.line(surf, color, (sx, sy_disp), (ex, ey_disp), 2)
                pygame.draw.circle(surf, color, (int(ex), int(ey_disp)), 4)
        draw_hud(self.score)
        # Draw sensor values/labels at left
        if SHOW_SENSORS and hasattr(self, '_sensor_rays'):
            for i, (_, _, _, _, val) in enumerate(self._sensor_rays):
                label = SENSOR_LABELS[i]
                draw_text(surf, f"{label}: {val:.2f}", 22, WHITE, 50, 30 + 26 * i, center=False)
        pygame.display.flip()
