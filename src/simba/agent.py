from copy import deepcopy
from functools import partial
import os
from typing import Any
import flax
import flax.training
import flax.training.orbax_utils
import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
from sbx.common.distributions import TanhTransformedDistribution
import tensorflow_probability.substrates.jax as tfp

from simba.network import Actor, VectorCritic
from jax_rl.utils import (
    get_action_bias_scale,
    get_obs_action_dim,
)
from simba import RLTrainState

tfd = tfp.distributions


class RunningMeanStd:
    """Tracks the mean, variance and count of values."""

    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, epsilon=1e-4, shape=(), dtype=np.float64):
        """Tracks the mean, variance and count of values."""
        self.mean = np.zeros(shape, dtype=dtype)
        self.var = np.ones(shape, dtype=dtype)
        self.count = epsilon

    def update(self, x):
        """Updates the mean, var and count from a batch of samples."""
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        """Updates from batch mean, variance and count moments."""
        self.mean, self.var, self.count = update_mean_var_count_from_moments(
            self.mean, self.var, self.count, batch_mean, batch_var, batch_count
        )


def update_mean_var_count_from_moments(
    mean, var, count, batch_mean, batch_var, batch_count
):
    """Updates the mean, var and count using the previous mean, var, count and batch values."""
    delta = batch_mean - mean
    tot_count = count + batch_count

    new_mean = mean + delta * batch_count / tot_count
    m_a = var * count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + np.square(delta) * count * batch_count / tot_count
    new_var = M2 / tot_count
    new_count = tot_count

    return new_mean, new_var, new_count


class SimbaAgent:
    """Simba Agent."""

    def __init__(
        self,
        env_id: str,
        n_critics: int = 2,
        seed: int = 42,
        n_quantile: int = 25,
        n_quantile_drop: int = 5,
        *args,
        **kwargs,
    ) -> None:
        """Init."""
        obs_dim, action_dim = get_obs_action_dim(env_id)
        # Make Kwargs for Net
        self.n_quntile_target = (n_quantile - n_quantile_drop) * n_critics
        self.n_quntile_drop = n_quantile_drop
        self.n_critics = n_critics

        self.obs_dim, self.action_dim = obs_dim, action_dim
        self.d_critic_kwargs = {
            "num_blocks": 2,
            "hidden_dim": 512,
            "n_qunatile": n_quantile,
        }

        self.d_actor_kwargs = {
            "num_blocks": 1,
            "hidden_dim": 128,
            "action_dim": self.action_dim * 2,
        }

        self.key = jax.random.PRNGKey(seed)
        self.key, self.noise_key = jax.random.split(self.key, 2)
        self.optimizer_class = optax.adamw

        self.action_bias, self.action_scale = get_action_bias_scale(env_id=env_id)
        self.env_id = env_id
        self.train = True

        self.obs_rms = RunningMeanStd(shape=(self.obs_dim), dtype=np.float32)

        self.manager = None

    def save(self, checkpoint_dir: os.PathLike, step: int = 0) -> None:
        """Save Train State."""
        if self.manager is None:
            options = ocp.CheckpointManagerOptions(max_to_keep=2, create=True)
            checkpointer = ocp.PyTreeCheckpointer()
            self.manager = ocp.CheckpointManager(checkpoint_dir, checkpointer, options)
        combined_states = {
            "qfn_state": self.qfn_state,
            "actor_state": self.actor_state,
            "config": {
                "env_id": self.env_id,
            },
        }
        save_args = flax.training.orbax_utils.save_args_from_target(combined_states)
        self.manager.save(step, combined_states, save_kwargs={"save_args": save_args})

    @classmethod
    def load(cls, checkpoint_dir: os.PathLike, step: int = 0) -> "SimbaAgent":
        """Load Agent."""
        checkpointer = ocp.PyTreeCheckpointer()
        manager = ocp.CheckpointManager(checkpoint_dir, checkpointer)
        restored = manager.restore(step=step)
        config = restored["config"]
        agent = SimbaAgent(**config)
        key = jax.random.PRNGKey(42)
        agent.build(key)

        agent.load_all_components(restored)
        return agent

    def load_all_components(self, restored: dict[str, Any]) -> None:
        """Load All components from states."""

        def _replace(name: str) -> None:
            setattr(
                self,
                name,
                getattr(self, name).replace(
                    params=restored[name]["params"],
                    target_params=restored[name]["target_params"],
                ),
            )

        _replace("actor_state")
        _replace("qfn_state")

    def load_all_components_from_other(self, agent: "SimbaAgent") -> None:
        """Load all components."""
        self.encoder_state = deepcopy(agent.encoder_state)
        self.dynamics_state = deepcopy(agent.dynamics_state)
        self.noise_key = deepcopy(agent.noise_key)
        self.key = deepcopy(agent.key)

    def build(
        self,
        key: jax.Array,
        actor_lr: float = 1e-4,
        actor_weight_decay: float = 1e-2,
        critic_lr: float = 1e-4,
        critic_weight_decay: float = 1e-2,
    ) -> jax.Array:
        """Build neural network architecture and optimizer."""
        # Keep a key for the actor
        key, actor_key, qf_key = jax.random.split(key, 3)
        _, self.key = jax.random.split(key, 2)

        # Prepare input: observation and action
        obs = jnp.ones((1, self.obs_dim))
        action = jnp.ones((1, self.action_dim))

        # Make Critic.
        self.qfn = VectorCritic(
            n_critics=self.n_critics,
            **self.d_critic_kwargs,
        )

        self.qfn_state = RLTrainState.create(
            apply_fn=self.qfn.apply,
            params=self.qfn.init(qf_key, obs, action),
            target_params=self.qfn.init(qf_key, obs, action),
            tx=self.optimizer_class(
                learning_rate=critic_lr, eps=1e-6, weight_decay=critic_weight_decay
            ),
            key=qf_key,
        )

        # Make actor
        self.actor = Actor(**self.d_actor_kwargs)

        self.actor_state = RLTrainState.create(
            apply_fn=self.actor.apply,
            params=self.actor.init(actor_key, obs),
            target_params=self.actor.init(actor_key, obs),
            tx=self.optimizer_class(
                learning_rate=actor_lr, eps=1e-6, weight_decay=actor_weight_decay
            ),
            key=actor_key,
        )

        self.actor.apply = jax.jit(self.actor.apply)
        self.qfn.apply = jax.jit(self.qfn.apply)
        return key

    def normalize(self, observation: jnp.ndarray) -> jnp.ndarray:
        """Normalize."""
        return (observation - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + 1e-8)

    def sample(
        self, obs: np.ndarray, deterministic: bool = False, temperature: float = 1.0
    ) -> np.ndarray:
        """Sample."""
        if obs.ndim == 1:
            obs = np.expand_dims(obs, 0)
        if deterministic:
            action, self.key = self._get_deterministic_action(
                self.actor_state,
                self.actor_state.params,
                obs,
                self.key,
            )

        else:
            action, entropy, self.noise_key = self._rsample(
                self.actor_state,
                self.actor_state.params,
                obs,
                self.noise_key,
            )
        action = action * self.action_scale + self.action_bias
        if action.shape[0] == 1:
            action = action[0]
        return action

    @staticmethod
    @jax.jit
    def _get_deterministic_action(
        actor_state: RLTrainState,
        actor_params,
        obs: jax.Array,
        key: jax.Array,
    ) -> jax.Array:
        """Calculate deterministic action."""
        key, dropout_key = jax.random.split(key, 2)

        raw_action = actor_state.apply_fn(actor_params, obs)
        arctanh_action, _ = jnp.split(raw_action, 2, axis=-1)
        action = jnp.tanh(arctanh_action)
        return action, key

    @staticmethod
    @jax.jit
    def _rsample(
        actor_state: RLTrainState,
        actor_params,
        obs: jax.Array,
        key: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Re-parameterization trick."""
        key, noise_key, dropout_key = jax.random.split(key, 3)

        raw_action = actor_state.apply_fn(actor_params, obs)
        raw_mean, log_std = jnp.split(raw_action, 2, axis=-1)
        log_std = jnp.clip(log_std, -20.0, 2.0)

        # It is Enforcing action bound in SAC !!
        dist = TanhTransformedDistribution(
            tfd.MultivariateNormalDiag(loc=raw_mean, scale_diag=jnp.exp(log_std))
        )

        # action = tanh(u)
        action = dist.sample(seed=noise_key)
        batch_size = action.shape[0]
        # Calculate entropy of action
        entropy = compute_action_entropy(
            jnp.reshape(raw_mean, -1), jnp.reshape(jnp.exp(log_std), -1), key
        )
        entropy = jnp.reshape(entropy, (batch_size, -1)).sum(-1, keepdims=True)
        return action, entropy, key
