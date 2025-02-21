import math
from typing import Any, Callable, Sequence
import flax.linen as nn

import jax
import jax.numpy as jnp


# Pytorch Version
def kaiming_uniform(key, shape, dtype=jnp.float32, a=math.sqrt(5), scale: float = 1.0):
    fan_in = shape[-2]
    gain = math.sqrt(2.0 / (1 + a**2.0))
    std = gain / math.sqrt(fan_in)
    bound = math.sqrt(3.0) * std
    return jax.random.uniform(key, shape, dtype, -bound, bound) * scale


def bias_initializer(key, shape, dtype=jnp.float32, scale: float = 1.0):
    fan_in = shape[-1]
    bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
    return jax.random.uniform(key, shape, dtype, -bound, bound) * scale


class Dense(nn.Module):
    features: int
    use_bias: bool = True
    kernel_init: Callable[[Any, tuple], jnp.ndarray] = kaiming_uniform
    bias_init: Callable[[Any, tuple], jnp.ndarray] = bias_initializer

    @nn.compact
    def __call__(self, inputs):
        kernel = self.param(
            "kernel", self.kernel_init, (inputs.shape[-1], self.features)
        )
        y = jnp.dot(inputs, kernel)
        bias = self.param("bias", self.bias_init, (self.features,))
        y = y + bias
        return y


class AvgL1Norm(nn.Module):
    """Average L1 Norm."""

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Forward."""
        y = jnp.abs(x)
        mean_y = jnp.mean(y, axis=-1, keepdims=True).clip(1e-8)
        return jnp.divide(x, mean_y)


class Encoder(nn.Module):
    """Encoder for state embedding."""

    net_arch: Sequence[int]
    activ: Callable = nn.elu

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Forward."""
        for n_units in self.net_arch[:-1]:
            x = Dense(n_units)(x)
            x = self.activ(x)
        x = Dense(self.net_arch[-1])(x)
        x = AvgL1Norm()(x)
        return x


class Dynamics(nn.Module):
    """Dynamics: projection state embedding and action into next embedding."""

    net_arch: Sequence[int]
    activ: Callable = nn.elu

    @nn.compact
    def __call__(self, zs: jnp.ndarray, action: jnp.ndarray) -> jnp.ndarray:
        """Forward."""
        zsa = jnp.concat([zs, action], axis=-1)
        for n_units in self.net_arch[:-1]:
            zsa = Dense(n_units)(zsa)
            zsa = self.activ(zsa)
        zsa = Dense(self.net_arch[-1])(zsa)
        return zsa


class Critic(nn.Module):
    net_arch: Sequence[int]
    activ: Callable = nn.elu

    @nn.compact
    def __call__(
        self, obs: jnp.ndarray, action: jnp.ndarray, zsa: jnp.ndarray, zs: jnp.ndarray
    ) -> jnp.ndarray:
        """Forward."""
        obs_action = jnp.concat([obs, action], axis=-1)
        embeddings = jnp.concat([zsa, zs], axis=-1)

        embed = Dense(self.net_arch[0])(obs_action)
        embed = AvgL1Norm()(embed)

        x = jnp.concat([embed, embeddings], axis=-1)
        for n_units in self.net_arch[1:-1]:
            x = Dense(n_units)(x)
            x = self.activ(x)
        q_fn = Dense(self.net_arch[-1])(x)
        return q_fn


class VectorCritic(nn.Module):
    net_arch: Sequence[int]
    activ: Callable = nn.elu
    n_critics: int = 2

    @nn.compact
    def __call__(
        self, obs: jnp.ndarray, action: jnp.ndarray, zsa: jnp.ndarray, zs: jnp.ndarray
    ):
        # Idea taken from https://github.com/perrin-isir/xpag
        # Similar to https://github.com/tinkoff-ai/CORL for PyTorch
        vmap_critic = nn.vmap(
            Critic,
            variable_axes={"params": 0},  # parameters not shared between the critics
            split_rngs={"params": True, "dropout": True},  # different initializations
            in_axes=None,
            out_axes=1,
            axis_size=self.n_critics,
        )
        q_values = vmap_critic(net_arch=self.net_arch, activ=self.activ)(
            obs, action, zsa, zs
        )
        return q_values


class Actor(nn.Module):
    net_arch: Sequence[int]
    activ: Callable = nn.relu
    n_bins: int = 101

    @nn.compact
    def __call__(self, obs: jnp.ndarray, zs: jnp.ndarray, train: bool) -> jnp.ndarray:
        """Forward."""
        z = Dense(self.net_arch[0])(obs)
        z = AvgL1Norm()(z)

        zs = jnp.concat([z, zs], axis=-1)

        for n_units in self.net_arch[1:-1]:
            zs = Dense(n_units)(zs)
            zs = self.activ(zs)
        zs = Dense(self.net_arch[-1])(zs)
        return zs
