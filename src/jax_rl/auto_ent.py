from typing import Callable
from flax.training.train_state import TrainState
import jax
import optax
import jax.numpy as jnp
import flax.linen as nn


class EntropyCoef(nn.Module):
    ent_coef_init: float = 1.0

    @nn.compact
    def __call__(self) -> jnp.ndarray:
        log_ent_coef = self.param(
            "log_ent_coef",
            init_fn=lambda key: jnp.full((), jnp.log(self.ent_coef_init)),
        )
        return jnp.exp(log_ent_coef)


class ConstantEntropyCoef(nn.Module):
    ent_coef_init: float = 1.0

    @nn.compact
    def __call__(self) -> float:
        # Hack to not optimize the entropy coefficient while not having to use if/else for the jit
        # TODO: add parameter in train to remove that hack
        self.param("dummy_param", init_fn=lambda key: jnp.full((), self.ent_coef_init))
        return self.ent_coef_init


DEFAULT_OPTIMIZER_KWARGS = {"learning_rate": 3e-4}


def create_ent_coef_state(
    ent_coef: float,
    key: jax.Array,
    optimizer_class: Callable = optax.adam,
    optimizer_kwargs: dict | None = None,
) -> TrainState:
    """Create Entroy Coeffieicnt Train state."""
    if ent_coef < 0.0:
        # Default initial value of ent_coef when learned
        ent_coef_init = 1.0
        # Note: we optimize the log of the entropy coeff which is slightly different from the paper
        # as discussed in https://github.com/rail-berkeley/softlearning/issues/37
        ent_coef_ = EntropyCoef(ent_coef_init)
    else:
        # This will throw an error if a malformed string (different from 'auto') is passed
        ent_coef_ = ConstantEntropyCoef(ent_coef)  # type: ignore[assignment]

    optimizer_kwargs = optimizer_kwargs or DEFAULT_OPTIMIZER_KWARGS
    return TrainState.create(
        apply_fn=ent_coef_.apply,
        params=ent_coef_.init(key)["params"],
        tx=optimizer_class(**optimizer_kwargs),
    )
