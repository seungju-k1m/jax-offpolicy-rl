from abc import ABC
from typing import Any

from gymnasium import spaces


class AbcReplayMemory(ABC):
    """Abstract Class for Replay Memory."""

    def __init__(
        self,
        replay_buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        **kwargs,
    ) -> None:
        """Initialize."""
        self.replay_buffer_size = replay_buffer_size
        self.observation_space = observation_space
        self.action_space = action_space

    def sample(self, batch_size: int):
        """Sample Batch."""
        raise NotImplementedError("Sample method should be implemented.")

    def append(self, transition: list[Any]) -> None:
        """Append."""
        raise NotImplementedError("Append method should bed implemented.")

    def __len__(self) -> int:
        """Return length."""
        raise NotImplementedError("__len__ should be implemented.")
