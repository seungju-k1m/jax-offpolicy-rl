"""Replay Buffer."""

from functools import partial
from typing import Any
import jax
import numpy as np
from gymnasium import spaces

import jax.numpy as jnp

from jax_rl.replay_memory.simple import SimpleReplayMemory


class LAPReplayMemory(SimpleReplayMemory):
    """LAP."""

    def __init__(
        self,
        replay_buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        seed: int = 42,
        **kwargs,
    ) -> None:
        """Init."""
        super().__init__(replay_buffer_size, observation_space, action_space, **kwargs)
        self.priority = np.zeros(replay_buffer_size, dtype=np.float32)
        self.key = jax.random.PRNGKey(seed)
        self.max_priority = 1.0

    def append(self, transition: list[Any]) -> None:
        """Append transition."""
        assert len(transition) == 5
        obs, action, reward, next_obs, float_done = transition
        action = (action - self.action_bias) / self.action_scale
        reward = np.array([reward])
        float_done = np.array([float_done])
        zz = np.concatenate((obs, action, reward, next_obs, float_done), 0)
        self.alls[self.ptr] = zz
        self.priority[self.ptr] = self.max_priority
        self.ptr = (self.ptr + 1) % self.replay_buffer_size
        self.size = min(self.size + 1, self.replay_buffer_size)

    @staticmethod
    @partial(jax.jit, static_argnames=["batch_size"])
    def _sample(
        priority: np.ndarray, batch_size: int, key: jax.Array
    ) -> tuple[dict[str, jax.Array], jax.Array, jax.Array]:
        """Sample."""
        csum = jnp.cumsum(priority, axis=0)
        key, random_key = jax.random.split(key)
        value = jax.random.uniform(random_key, (batch_size,), maxval=csum[-1])
        ind = jnp.searchsorted(csum, value, method="scan_unrolled")
        return ind, key

    def sample(self, batch_size: int) -> dict[str, jax.Array]:
        """Sample."""
        ind, self.key = self._sample(self.priority, batch_size, self.key)
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
        self.ind = ind
        return dict(obs=obs, action=action, reward=reward, next_obs=next_obs, done=done)

    def update_priority(self, priority: np.ndarray) -> None:
        """Update priority."""
        self.priority[self.ind] = priority
        # self.max_priority = max(float(priority.max()), self.max_priority)

    def reset_max_priority(self):
        """Reset max proirity."""
        self.max_priority = float(self.priority[: self.size].max())
