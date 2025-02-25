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
from jax_rl.evaluate_agent import evaluate_agent
from jax_rl.auto_ent import create_ent_coef_state
from jax_rl.abc import BaseAgent


__all__ = [
    "replay_memory",
    "rollout",
    "utils",
    "evaluate_agent",
    "BaseAgent",
    "create_ent_coef_state",
]
