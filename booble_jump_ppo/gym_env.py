import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .config import game_cfg, ppo_cfg
from .game_env import Game


def get_state(game: Game) -> np.ndarray:
    return game.get_state()



class BoobleJumpEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, seed=None):
        super().__init__()
        self.seed_val = seed if seed is not None else game_cfg.DEFAULT_SEED
        self.game = Game(seed=self.seed_val, headless=True)

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(ppo_cfg.STATE_DIM,), dtype=np.float32)
        self.action_space = spaces.Discrete(ppo_cfg.ACTION_DIM)

    def seed(self, seed=None):
        if seed is None:
            return
        self.seed_val = seed
        self.game.seed = seed

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.seed_val = int(seed)
            self.game.seed = self.seed_val
            self.game.rng.seed(self.seed_val)

        self.game.reset_game()
        obs = get_state(self.game)
        info = {}
        return obs, info

    def step(self, action):
        reward, done = self.game.step(int(action))
        obs = get_state(self.game)
        info = {
            "score": self.game.score,
            "steps": self.game.steps,
            "no_progress_time": self.game.no_progress_time,
            "reward_info": self.game.last_reward_components,
            "obs_debug": getattr(self.game, "last_obs_debug", None),
        }
        terminated = done and not self.game.truncated
        truncated = done and self.game.truncated
        return obs, reward, terminated, truncated, info

    def render(self):
        self.game.draw()

    def close(self):
        pass


def make_env_seed_offset(seed_offset: int, base_seed: int = game_cfg.DEFAULT_SEED):
    def _thunk():
        env_seed = base_seed + seed_offset
        env = BoobleJumpEnv(seed=env_seed)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env

    return _thunk
