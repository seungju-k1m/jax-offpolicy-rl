"""Rollout."""

from copy import deepcopy
from typing import Any, Type

from gymnasium.wrappers import RecordEpisodeStatistics
from gymnasium import spaces
import jax
import numpy as np

from jax_rl.replay_memory.abc import AbcReplayMemory


class RandomSampler:
    def __init__(self, action_space: spaces.Space, seed: int = 42):
        self.action_space = action_space
        self.action_space.seed(seed)

    def sample(self, *args, **kwargs):
        return self.action_space.sample()


class Rollout:
    """Rollout Worker."""

    def __init__(
        self,
        env: RecordEpisodeStatistics,
        replay_memory: Type[AbcReplayMemory],
        seed: int = 42,
    ):
        """Initialize."""
        self.env = env
        self.replay_memory = replay_memory
        self.count_replay_buffer: int = 0
        self.sampler = RandomSampler(self.env.action_space, seed)
        self.need_reset: bool = True
        self.n_episode: int = 0
        self.deterministic: bool = False
        self.obs: np.ndarray

    def set_sampler(self, sampler: Any) -> None:
        """Set sampler."""
        self.sampler = sampler

    def get_batch(self, batch_size: int) -> dict[str, jax.Array]:
        """Return batch for train ops."""
        return self.replay_memory.sample(batch_size)

    def sample(self) -> bool:
        """Sample action using policy."""
        if self.need_reset:
            self.need_reset = False
            self.obs = self.env.reset()[0]
        action = self.sampler.sample(self.obs, self.deterministic)
        next_obs, reward, terminated, truncated, info = self.env.step(action)
        done = truncated or terminated
        self.replay_memory.append(
            deepcopy([self.obs, action, reward, next_obs, 1.0 - float(terminated)])
        )
        self.obs = next_obs
        self.need_reset = done
        return done
