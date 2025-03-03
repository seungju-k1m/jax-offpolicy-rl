"""Replay Buffer."""

from typing import Any

import jax
import numpy as np
from gymnasium import spaces

from jax_rl.replay_memory.abc import AbcReplayMemory
from jax_rl.utils import get_action_bias_scale


class SimpleReplayMemory(AbcReplayMemory):
    """Simple Replay Memory."""

    def __init__(
        self,
        replay_buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        **kwargs,
    ) -> None:
        """Initialize."""
        super().__init__(replay_buffer_size, observation_space, action_space, **kwargs)
        self.ptr = 0
        self.size = 0
        self.ind: list[int]
        obs_dim, action_dim = (
            self.observation_space.shape[-1],
            self.action_space.shape[-1],
        )
        self.action_bias, self.action_scale = get_action_bias_scale(self.action_space)
        self.alls = np.zeros(
            (replay_buffer_size, obs_dim + action_dim + obs_dim + 1 + 1),
            dtype=np.float32,
        )
        self.obs_dim, self.action_dim = obs_dim, action_dim

    def append(self, transition: list[Any]) -> None:
        """Append transition."""
        assert len(transition) == 5
        obs, action, reward, next_obs, float_done = transition
        action = (action - self.action_bias) / self.action_scale
        reward = np.array([reward])
        float_done = np.array([float_done])
        zz = np.concatenate((obs, action, reward, next_obs, float_done), 0)
        self.alls[self.ptr] = zz
        self.ptr = (self.ptr + 1) % self.replay_buffer_size
        self.size = min(self.size + 1, self.replay_buffer_size)

    def sample(self, batch_size: int, **kwargs) -> dict[str, jax.Array]:
        """Sample."""
        ind = np.random.randint(low=0, high=self.size, size=(batch_size))
        all = self.alls[ind]
        obs, action, reward, next_obs, done = np.split(
            all,
            [
                self.obs_dim,
                self.obs_dim + self.action_dim,
                self.obs_dim + self.action_dim + 1,
                self.obs_dim + self.action_dim + 1 + self.obs_dim,
            ],
            axis=-1,
        )
        return dict(obs=obs, action=action, reward=reward, next_obs=next_obs, done=done)

    def __len__(self) -> int:
        return self.ptr
