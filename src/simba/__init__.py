import jax
import flax
from flax.training.train_state import TrainState
import jax


class RLTrainState(TrainState):  # type: ignore[misc]
    target_params: flax.core.FrozenDict  # type: ignore[misc]
    key: jax.Array
