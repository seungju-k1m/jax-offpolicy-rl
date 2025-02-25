import flax.linen as nn
import jax.numpy as jnp


def orthogonal_init(scale: float = jnp.sqrt(2)):
    return nn.initializers.orthogonal(scale)


def he_normal_init():
    return nn.initializers.he_normal()


def he_uniform_init():
    return nn.initializers.he_uniform()


class MLPBlock(nn.Module):
    hidden_dim: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # sqrt(2) is recommended when using with ReLU activation.
        x = nn.Dense(
            self.hidden_dim,
            kernel_init=orthogonal_init(jnp.sqrt(2)),
        )(x)
        x = nn.relu(x)
        x = nn.Dense(
            self.hidden_dim,
            kernel_init=orthogonal_init(jnp.sqrt(2)),
        )(x)
        x = nn.relu(x)
        return x


class ResidualBlock(nn.Module):
    hidden_dim: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        res = x
        x = nn.LayerNorm()(x)
        x = nn.Dense(self.hidden_dim * 4, kernel_init=he_normal_init())(x)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_dim, kernel_init=he_normal_init())(x)
        return res + x


class SACEncoder(nn.Module):
    num_blocks: int
    hidden_dim: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = nn.Dense(self.hidden_dim, kernel_init=orthogonal_init(1))(x)
        for _ in range(self.num_blocks):
            x = ResidualBlock(self.hidden_dim)(x)
        x = nn.LayerNorm()(x)
        return x


class Actor(nn.Module):
    num_blocks: int
    hidden_dim: int
    action_dim: int

    def setup(self):
        self.encoder = SACEncoder(
            num_blocks=self.num_blocks,
            hidden_dim=self.hidden_dim,
        )
        self.predictor = nn.Dense(self.action_dim, kernel_init=orthogonal_init(1))

    def __call__(
        self,
        observations: jnp.ndarray,
    ) -> jnp.ndarray:
        z = self.encoder(observations)
        zs = self.predictor(z)
        return zs


class Critic(nn.Module):
    num_blocks: int
    hidden_dim: int
    n_qunatile: int

    def setup(self):
        self.encoder = SACEncoder(
            num_blocks=self.num_blocks,
            hidden_dim=self.hidden_dim,
        )
        self.predictor = nn.Dense(self.n_qunatile, kernel_init=orthogonal_init(1.0))

    @nn.compact
    def __call__(self, obs: jnp.ndarray, action: jnp.ndarray) -> jnp.ndarray:
        """Forward."""
        obs_action = jnp.concat([obs, action], axis=-1)
        z = self.encoder(obs_action)
        q = self.predictor(z)
        return q


class VectorCritic(nn.Module):
    num_blocks: int
    hidden_dim: int
    n_qunatile: int
    n_critics: int = 2

    @nn.compact
    def __call__(self, obs: jnp.ndarray, action: jnp.ndarray):
        vmap_critic = nn.vmap(
            Critic,
            variable_axes={"params": 0},  # parameters not shared between the critics
            split_rngs={"params": True},  # different initializations
            in_axes=None,
            out_axes=1,
            axis_size=self.n_critics,
        )
        q_values = vmap_critic(
            num_blocks=self.num_blocks,
            hidden_dim=self.hidden_dim,
            n_qunatile=self.n_qunatile,
        )(obs, action)
        return q_values
