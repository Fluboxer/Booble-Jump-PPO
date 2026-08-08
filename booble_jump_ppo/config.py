import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict

import torch

MODELS_ROOT = Path("models")
os.makedirs("models", exist_ok=True)
os.makedirs("logs", exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class DebugConfig:
    LOG_EPISODE: bool = True
    LOG_STEP: bool = True
    LOG_LR: bool = False


@dataclass
class PPOConfig:
    PLATFORM_AMOUNT: int = 4
    PLATFORM_STATES: int = 8

    STATE_DIM: int = (8 + (PLATFORM_AMOUNT * PLATFORM_STATES))
    HIDDEN_DIM: int = 256

    ACTION_DIM: int = 3

    LEARNING_RATE: float = 3.5e-4
    GAMMA: float = 0.99
    GAE_LAMBDA: float = 0.95
    CLIP_EPSILON: float = 0.21
    VALUE_LOSS_COEF: float = 0.5
    ENTROPY_COEF: float = 0.022
    MAX_GRAD_NORM: float = 0.5

    PPO_EPOCHS: int = 4
    BATCH_SIZE: int = 2048
    N_STEPS: int = 4096

    LEARNING_RATE_DECAY: float = 0.90  # LEARNING_RATE_DECAY * progress


@dataclass
class RewardsConfig:
    SCORE_PROGRESS_COEF: float = 1.0

    REWARD_BASE_JUMP: float = 2.5
    PLATFORM_TYPE_BONUSES: Dict[str, float] = field(default_factory=lambda: {
        "normal": 1.0,
        "moving": 1.5,
        "breakable": 1.2,
        "spring": 1.5,
    })
    CONSECUTIVE_JUMP_BONUS_STEP: float = 0.3
    CONSECUTIVE_JUMP_BONUS_MAX: float = 3.0

    SURVIVAL_REWARD_PER_SEC: float = 0.02

    DEATH_PENALTY_BASE: float = -55.0
    DEATH_PENALTY_SCORE_COEF: float = 0.10
    DEATH_PENALTY_SCORE_CAP: float = 30.0

    STAGNATION_GRACE_SEC: float = 8.0
    STAGNATION_PENALTY_PER_SEC: float = 1.5
    TRUNCATE_IF_NO_PROGRESS_SEC: float = 12.0

    IDLE_PENALTY_PER_SEC: float = 0.08

    CENTERING_PENALTY_COEF: float = 0.0


@dataclass
class GameConfig:
    WINDOW_WIDTH: int = 400
    WINDOW_HEIGHT: int = 600
    FPS: int = 120

    WHITE: tuple = (255, 255, 255)
    BLACK: tuple = (0, 0, 0)
    GREEN: tuple = (0, 200, 0)
    BLUE: tuple = (80, 140, 255)
    RED: tuple = (220, 50, 50)
    YELLOW: tuple = (245, 225, 50)
    PURPLE: tuple = (128, 0, 128)
    ORANGE: tuple = (255, 165, 0)

    PLATFORM_WIDTH: int = 80
    PLATFORM_HEIGHT: int = 14
    PLAYER_SIZE: int = 22
    GRAVITY: float = 1200.0
    JUMP_VELOCITY: float = -750.0
    MOVE_SPEED: float = 240.0
    MAX_FALL_SPEED: float = 900.0
    CAMERA_LERP: float = 10.0

    PLATFORM_GAP_MIN: int = 90
    PLATFORM_GAP_MAX: int = 130
    # сумма весов должна быть 1.0
    # они будут меняться во время работы, сводя normal до нуля
    PLATFORM_TYPES_DIST: dict = field(default_factory=lambda: {
        "normal": 0.60,
        "moving": 0.20,
        "breakable": 0.15,
        "spring": 0.05,
    })
    MOVING_SPEED: float = 60.0
    SPRING_BOOST: float = 1.5

    DEFAULT_SEED: int = 3456236


game_cfg = GameConfig()
ppo_cfg = PPOConfig()
reward_cfg = RewardsConfig()
debug_cfg = DebugConfig()
