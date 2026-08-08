import random
import time
from typing import List, Dict, Optional, Tuple

import numpy as np
import pygame

from .config import game_cfg, reward_cfg, ppo_cfg
from .utils import choose_weighted


class Platform:
    __slots__ = ("rect", "type", "broken", "move_dir", "rng", "platform_id", "bounce_factor", "used")

    def __init__(self, x, y, ptype, rng, platform_id=None):
        self.rect = pygame.FRect(x, y, game_cfg.PLATFORM_WIDTH, game_cfg.PLATFORM_HEIGHT)
        self.type = ptype
        self.broken = False
        self.used = False
        self.move_dir = 1.0 if rng.random() < 0.5 else -1.0
        self.rng = rng
        self.platform_id = platform_id if platform_id else f"{int(x)}_{int(y)}_{ptype}"
        self.bounce_factor = game_cfg.SPRING_BOOST if ptype == "spring" else 1.0

    def update(self, dt):
        if self.type == "moving" and not self.broken:
            self.rect.x += self.move_dir * game_cfg.MOVING_SPEED * dt
            if self.rect.left <= 0:
                self.rect.left = 0
                self.move_dir *= -1.0
            elif self.rect.right >= game_cfg.WINDOW_WIDTH:
                self.rect.right = game_cfg.WINDOW_WIDTH
                self.move_dir *= -1.0

    def get_color(self):
        if self.broken:
            return game_cfg.BLACK
        colors = {
            "normal": game_cfg.GREEN,
            "moving": game_cfg.BLUE,
            "breakable": game_cfg.RED,
            "spring": game_cfg.PURPLE,
        }
        return colors.get(self.type, game_cfg.GREEN)

    def draw(self, screen, camera_y):
        if self.broken:
            return
        y = self.rect.y - camera_y
        if y > game_cfg.WINDOW_HEIGHT + 50 or y < -50:
            return
        color = self.get_color()
        rect = pygame.FRect(self.rect.x, y, self.rect.w, self.rect.h)
        pygame.draw.rect(screen, color, rect)
        pygame.draw.rect(screen, game_cfg.BLACK, rect, 1)
        if self.type == "spring":
            center_x = self.rect.x + self.rect.w // 2
            pygame.draw.circle(screen, game_cfg.ORANGE, (int(center_x), int(y + self.rect.h // 2)), 3)


class Player:
    __slots__ = ("rect", "vx", "vy", "on_ground", "last_platform_id", "jumps_count")

    def __init__(self, x, y):
        self.rect = pygame.FRect(x, y, game_cfg.PLAYER_SIZE, game_cfg.PLAYER_SIZE)
        self.vx = 0.0
        self.vy = 0.0
        self.on_ground = False
        self.last_platform_id = None
        self.jumps_count = 0

    def update_input(self, dt, action=None):
        self.vx = 0.0
        if action is not None:
            if action == 1:
                self.vx -= game_cfg.MOVE_SPEED
            elif action == 2:
                self.vx += game_cfg.MOVE_SPEED
        else:
            keys = pygame.key.get_pressed()
            if keys[pygame.K_LEFT] or keys[pygame.K_a]:
                self.vx -= game_cfg.MOVE_SPEED
            if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
                self.vx += game_cfg.MOVE_SPEED

        self.rect.x += self.vx * dt
        # горизонтальный оборот
        if self.rect.right < 0:
            self.rect.left = game_cfg.WINDOW_WIDTH - 1
        elif self.rect.left > game_cfg.WINDOW_WIDTH:
            self.rect.right = 1

    def apply_physics(self, dt):
        self.vy += game_cfg.GRAVITY * dt
        if self.vy > game_cfg.MAX_FALL_SPEED:
            self.vy = game_cfg.MAX_FALL_SPEED
        self.rect.y += self.vy * dt
        self.on_ground = False

    def jump(self, boost_factor=1.0):
        self.vy = game_cfg.JUMP_VELOCITY * boost_factor
        self.on_ground = True
        self.jumps_count += 1

    def draw(self, screen, camera_y):
        y = self.rect.y - camera_y
        rect = pygame.FRect(self.rect.x, y, self.rect.w, self.rect.h)
        pygame.draw.rect(screen, game_cfg.YELLOW, rect)
        pygame.draw.rect(screen, game_cfg.BLACK, rect, 2)
        eye_size = 3
        eye_y = y + 6
        pygame.draw.circle(screen, game_cfg.BLACK, (int(self.rect.x + 6), int(eye_y)), eye_size)
        pygame.draw.circle(screen, game_cfg.BLACK, (int(self.rect.x + 16), int(eye_y)), eye_size)


class Game:
    def __init__(self, seed=game_cfg.DEFAULT_SEED, headless=True):
        self.seed = seed
        self.rng = random.Random(self.seed)
        self.headless = headless

        if not headless:
            pygame.init()
            self.screen = pygame.display.set_mode((game_cfg.WINDOW_WIDTH, game_cfg.WINDOW_HEIGHT))
            pygame.display.set_caption("Boodle Jump PPO")
            self.font = pygame.font.Font(None, 32)
            self.small_font = pygame.font.Font(None, 20)
            self.tiny_font = pygame.font.Font(None, 16)
            self.clock = pygame.time.Clock()
        else:
            self.screen = None
            self.font = None
            self.small_font = None
            self.tiny_font = None
            self.clock = None

        self.platform_id_counter = 0
        self.total_jumps = 0
        self.consecutive_jumps = 0
        self.max_consecutive_jumps = 0
        self.platform_types_hit = {"normal": 0, "moving": 0, "breakable": 0, "spring": 0}
        self.reset_game()

    def reset_game(self):
        self.platform_id_counter = 0

        self.player = Player(game_cfg.WINDOW_WIDTH * 0.5 - game_cfg.PLAYER_SIZE * 0.5,
                             game_cfg.WINDOW_HEIGHT - 100)
        self.platforms: List[Platform] = []
        self.camera_y = 0.0
        self.score = 0

        self.highest_player_y = self.player.rect.y  # smallest y so far
        self.best_height = game_cfg.WINDOW_HEIGHT - 100 - self.highest_player_y
        self.game_over = False
        self.truncated = False
        self.steps = 0
        self.start_time = time.time()
        self.current_time = time.time()

        # Stagnation tracking
        self.steps_since_progress = 0
        self.no_progress_time = 0.0

        self.last_reward_components: Dict[str, float] = {}
        self.total_jumps = 0
        self.consecutive_jumps = 0
        self.max_consecutive_jumps = 0
        self.platform_types_hit = {"normal": 0, "moving": 0, "breakable": 0, "spring": 0}

        self._generate_initial_level()

    def _get_next_platform_id(self):
        self.platform_id_counter += 1
        return self.platform_id_counter

    def _generate_initial_level(self):
        base_x = game_cfg.WINDOW_WIDTH * 0.5 - game_cfg.PLATFORM_WIDTH * 0.5
        base_y = game_cfg.WINDOW_HEIGHT - 50
        platform_id = self._get_next_platform_id()
        self.platforms.append(Platform(base_x, base_y, "normal", self.rng, f"start_{platform_id}"))

        y = base_y - 120
        for _ in range(60):
            x = self.rng.randint(0, int(game_cfg.WINDOW_WIDTH - game_cfg.PLATFORM_WIDTH))
            ptype = choose_weighted(self.rng, game_cfg.PLATFORM_TYPES_DIST)
            platform_id = self._get_next_platform_id()
            self.platforms.append(Platform(x, y, ptype, self.rng, f"init_{platform_id}"))
            y -= self.rng.randint(game_cfg.PLATFORM_GAP_MIN, game_cfg.PLATFORM_GAP_MAX)

    def _generate_more(self):
        if not self.platforms:
            return
        min_y = min(p.rect.y for p in self.platforms)
        while min_y > self.camera_y - game_cfg.WINDOW_HEIGHT * 1.5:
            x = self.rng.randint(0, int(game_cfg.WINDOW_WIDTH - game_cfg.PLATFORM_WIDTH))

            height_factor = max(0.0, -min_y / 1000.0)
            adjusted = dict(game_cfg.PLATFORM_TYPES_DIST)
            adjusted["moving"] = max(0.0, adjusted["moving"] + height_factor * 0.1)
            adjusted["breakable"] = max(0.0, adjusted["breakable"] + height_factor * 0.05)
            adjusted["normal"] = max(0.0, adjusted["normal"] - height_factor * 0.15)
            s = sum(adjusted.values())

            if s <= 1e-6:
                adjusted = dict(game_cfg.PLATFORM_TYPES_DIST)
                s = sum(adjusted.values())

            if s <= 1e-6:
                raise ValueError(
                    "PLATFORM_TYPES_DIST должна содержать как минимум один положительный вес"
                )
            for k in adjusted:
                adjusted[k] /= s

            ptype = choose_weighted(self.rng, adjusted)
            min_y -= self.rng.randint(game_cfg.PLATFORM_GAP_MIN, game_cfg.PLATFORM_GAP_MAX)
            platform_id = self._get_next_platform_id()
            self.platforms.append(Platform(x, min_y, ptype, self.rng, f"gen_{platform_id}"))

        cutoff = self.camera_y + game_cfg.WINDOW_HEIGHT + 300
        self.platforms = [p for p in self.platforms if p.rect.y < cutoff]

    def _collide_platforms(self, prev_y) -> Tuple[bool, Optional[Platform], bool]:
        if self.player.vy <= 0:
            return False, None, False

        pr = self.player.rect
        for p in self.platforms:
            if p.broken:
                continue

            r = p.rect
            crossed = (prev_y + pr.h) <= r.top <= pr.bottom
            horizontal = (pr.right > r.left) and (pr.left < r.right)

            if crossed and horizontal:
                boost = p.bounce_factor
                self.player.jump(boost)

                was_new_platform = not p.used
                self.total_jumps += 1

                if was_new_platform:
                    self.consecutive_jumps += 1
                    self.max_consecutive_jumps = max(self.max_consecutive_jumps, self.consecutive_jumps)
                    self.platform_types_hit[p.type] += 1
                else:
                    self.consecutive_jumps = 0
                p.used = True
                if p.type == "breakable":
                    p.broken = True
                else:
                    pass
                return True, p, was_new_platform

        return False, None, False

    def _update_camera_and_progress(self, dt) -> float:
        if self.player.rect.y < self.highest_player_y:
            self.highest_player_y = self.player.rect.y

        prev_best_height = self.best_height
        self.best_height = game_cfg.WINDOW_HEIGHT - 100 - self.highest_player_y
        height_gain = max(0.0, self.best_height - prev_best_height)

        self.score = int(self.best_height / 10.0)

        target = self.highest_player_y - game_cfg.WINDOW_HEIGHT * (1.0 / 3.0)
        alpha = min(1.0, game_cfg.CAMERA_LERP * dt)
        self.camera_y += (target - self.camera_y) * alpha

        if height_gain > 0:
            self.steps_since_progress = 0
        else:
            self.steps_since_progress += 1
        self.no_progress_time = (self.steps_since_progress * (1.0 / game_cfg.FPS))
        return height_gain

    def _check_game_over(self):
        if self.player.rect.y > self.camera_y + game_cfg.WINDOW_HEIGHT + 150:
            self.game_over = True

    def get_state(self) -> np.ndarray:
        player = self.player

        norm_x = player.rect.x / game_cfg.WINDOW_WIDTH
        norm_y = (player.rect.y - self.camera_y) / game_cfg.WINDOW_HEIGHT
        death_margin = ((self.camera_y + game_cfg.WINDOW_HEIGHT + 150.0 - self.player.rect.y) / game_cfg.WINDOW_HEIGHT)
        death_margin = np.clip(death_margin, 0.0, 2.0)
        norm_vx = player.vx / game_cfg.MOVE_SPEED
        norm_vy = player.vy / game_cfg.MAX_FALL_SPEED
        is_falling = 1.0 if player.vy > 0 else 0.0
        # progress_score = min(self.score, 3000) / 3000.0
        progress_score = np.log1p(float(self.score)) / np.log1p(3000.0)

        stagnation_norm = min(self.no_progress_time / reward_cfg.TRUNCATE_IF_NO_PROGRESS_SEC, 1.0)

        # Narrow Y-window around player: a bit above and moderately below
        player_bottom_cam = player.rect.bottom - self.camera_y
        y_top = player_bottom_cam - 0.20 * game_cfg.WINDOW_HEIGHT
        y_bot = player_bottom_cam + 0.60 * game_cfg.WINDOW_HEIGHT

        candidates = []
        for p in self.platforms:
            if p.broken:
                continue
            py = p.rect.y - self.camera_y
            if not (y_top <= py <= y_bot):
                continue

            dx = (p.rect.centerx - player.rect.centerx)
            W = game_cfg.WINDOW_WIDTH
            if dx > W / 2:
                dx -= W
            if dx < -W / 2:
                dx += W

            rel_x = dx / game_cfg.WINDOW_WIDTH
            rel_y = (p.rect.y - player.rect.bottom) / game_cfg.WINDOW_HEIGHT

            candidates.append({
                "p": p,
                "rel_x": rel_x,
                "rel_y": rel_y,
                "is_moving": 1.0 if p.type == "moving" else 0.0,
                "move_dir": 0.0 if p.type != "moving" else (1.0 if p.move_dir > 0 else -1.0),
                "is_breakable": 1.0 if p.type == "breakable" else 0.0,
                "is_spring": 1.0 if p.type == "spring" else 0.0,
                "is_used": 1.0 if p.used else 0.0,
            })

        ahead = [c for c in candidates if c["rel_y"] < 0.0]
        behind = [c for c in candidates if c["rel_y"] >= 0.0]
        ahead.sort(key=lambda c: (c["rel_y"], abs(c["rel_x"])))
        behind.sort(key=lambda c: (c["rel_y"], abs(c["rel_x"])))
        ordered = ahead + behind

        platform_amount = ppo_cfg.PLATFORM_AMOUNT
        taken = ordered[:platform_amount]

        platforms_info: List[float] = []
        for c in taken:
            platforms_info.extend([
                c["rel_x"], c["rel_y"], c["is_moving"], c["move_dir"],
                c["is_breakable"], c["is_spring"], c["is_used"],
                1.0,  # флаг существования
            ])

        # Zero-padding
        slot_size = 8
        while len(platforms_info) < platform_amount * slot_size:
            platforms_info.extend([0.0] * slot_size)

        # Остатки логгера
        self.last_obs_debug = {
            "empty_slots": platform_amount - len(taken),
            "ahead_in_topK": sum(1 for c in taken if c["rel_y"] < 0.0),
            "first_slot_ahead": 1 if (taken and taken[0]["rel_y"] < 0.0) else 0,
        }

        state = np.array(
            [norm_x, norm_y, death_margin, norm_vx, norm_vy, is_falling, progress_score,
             stagnation_norm] + platforms_info,
            dtype=np.float32
        )
        return state

    def step(self, action=None) -> Tuple[float, bool]:
        if self.game_over:
            return 0.0, True

        dt = 1.0 / game_cfg.FPS
        prev_y = self.player.rect.y

        self.player.update_input(dt, action)
        self.player.apply_physics(dt)

        for p in self.platforms:
            p.update(dt)

        jumped, platform, was_new_platform = self._collide_platforms(prev_y)
        height_gain = self._update_camera_and_progress(dt)
        self._generate_more()

        self._check_game_over()
        self.steps += 1

        reward = 0.0
        components = {"score": 0.0, "jump": 0.0, "survival": 0.0, "death": 0.0, "stagnation": 0.0, "idle": 0.0}

        # 1 Награда за прогресс (по очкам)
        if height_gain > 0.0:
            progress_reward = reward_cfg.SCORE_PROGRESS_COEF * (height_gain / 10.0)
            reward += progress_reward
            components["score"] = progress_reward

        # 2 Награда за прыжок на новую платформу
        if jumped and platform is not None and was_new_platform:
            type_bonus = reward_cfg.PLATFORM_TYPE_BONUSES.get(platform.type, 1.0)
            jump_reward = reward_cfg.REWARD_BASE_JUMP * type_bonus
            consecutive_bonus = min(
                self.consecutive_jumps * reward_cfg.CONSECUTIVE_JUMP_BONUS_STEP,
                reward_cfg.CONSECUTIVE_JUMP_BONUS_MAX
            )
            jr = jump_reward + consecutive_bonus
            reward += jr
            components["jump"] = jr

        # 3 Награда за выживание
        survival = reward_cfg.SURVIVAL_REWARD_PER_SEC * dt
        reward += survival
        components["survival"] = survival

        # 4 Наказание за стагнацию
        if self.no_progress_time > reward_cfg.STAGNATION_GRACE_SEC:
            stagnation_penalty = reward_cfg.STAGNATION_PENALTY_PER_SEC * dt
            reward -= stagnation_penalty
            components["stagnation"] = -stagnation_penalty

        if (height_gain == 0.0) and (abs(self.player.vx) < 0.1 * game_cfg.MOVE_SPEED):
            idle_pen = reward_cfg.IDLE_PENALTY_PER_SEC * dt
            reward -= idle_pen
            components["idle"] = -idle_pen

        # 5 Раннее прерывание
        if (self.no_progress_time >= reward_cfg.TRUNCATE_IF_NO_PROGRESS_SEC) and not self.game_over:
            self.truncated = True
            self.game_over = True

        # 6 Наказание за смерть
        if self.game_over and not self.truncated:
            penalty = reward_cfg.DEATH_PENALTY_BASE - min(
                self.score * reward_cfg.DEATH_PENALTY_SCORE_COEF,
                reward_cfg.DEATH_PENALTY_SCORE_CAP
            )
            reward += penalty
            components["death"] = penalty

        # 7 Центрирование
        if reward_cfg.CENTERING_PENALTY_COEF > 0.0:
            center_penalty = (abs(self.player.rect.centerx - game_cfg.WINDOW_WIDTH / 2) / game_cfg.WINDOW_WIDTH)
            cpen = center_penalty * reward_cfg.CENTERING_PENALTY_COEF * dt
            reward -= cpen

        self.last_reward_components = components
        self.current_time = time.time()
        return reward, self.game_over

    def draw(self):
        if self.headless or self.screen is None:
            return
        self.screen.fill(game_cfg.WHITE)
        for p in self.platforms:
            p.draw(self.screen, self.camera_y)
        self.player.draw(self.screen, self.camera_y)
        self._draw_ui()
        pygame.display.flip()

    def _draw_ui(self):
        if self.headless or self.screen is None:
            return
        y_offset = 10
        score_text = self.font.render(f"Очки: {self.score}", True, game_cfg.BLACK)
        self.screen.blit(score_text, (10, y_offset))
        y_offset += 30
        stats_texts = [
            f"Прыжки: {self.total_jumps}",
            f"Комбо: {self.consecutive_jumps}",
            f"Лучшее комбо: {self.max_consecutive_jumps}",
            f"",
            f"Шаг: {self.steps}",
            f"Время: {int(self.current_time - self.start_time)}s",
            f"Простой: {self.no_progress_time:.1f}s",
        ]
        for text in stats_texts:
            surface = self.small_font.render(text, True, game_cfg.BLACK)
            self.screen.blit(surface, (10, y_offset))
            y_offset += 18

    def handle_events(self):
        if self.headless:
            return True
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                return False
            if e.type == pygame.KEYDOWN:
                if e.key == pygame.K_ESCAPE:
                    return False
                if e.key == pygame.K_r and self.game_over:
                    self.reset_game()
        return True
