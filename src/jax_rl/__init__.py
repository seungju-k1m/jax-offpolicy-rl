import flax
from flax.training.train_state import TrainState
import jax


class RLTrainState(TrainState):  # type: ignore[misc]
    target_params: flax.core.FrozenDict  # type: ignore[misc]
    target_fixed_params: flax.core.FrozenDict
    key: jax.Array


from jax_rl import utils
from jax_rl import replay_memory
from jax_rl import rollout


__all__ = ["replay_memory", "rollout", "utils"]
