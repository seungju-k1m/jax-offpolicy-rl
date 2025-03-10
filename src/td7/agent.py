from copy import deepcopy
import os
from typing import Any, Callable, Sequence
import flax
import flax.training
import flax.training.orbax_utils
import jax
import jax.numpy as jnp
import numpy as np
import optax
import flax.linen as nn
import orbax.checkpoint as ocp

from td7.network import Actor, Dynamics, Encoder, VectorCritic
from jax_rl.utils import get_action_bias_scale, get_obs_action_dim
from jax_rl import RLTrainState


DEFAULT_NET_KWARGS: dict[str, Sequence[int]] = {
    "encoder": [256] * 3,
    "dynamics": [256] * 3,
    "critic": [256] * 3 + [1],
    # Add last dimension
    "actor": [256, 256, 256],
}

DEFAULT_NET_ACTIV_KWARGS: dict[str, Callable] = {
    "encoder": nn.elu,
    "dynamics": nn.elu,
    "critic": nn.elu,
    "actor": nn.relu,
}


class TD7Agent:
    """Policy for Critic-Gradient Algorithm."""

    def __init__(
        self,
        env_id: str,
        net_kwargs: dict[str, Sequence[int]] | None = None,
        net_activ_kwargs: dict[str, Callable] | None = None,
        n_critics: int = 2,
        exploration_noise: float = 0.1,
        seed: int = 42,
        *args,
        **kwargs,
    ) -> None:
        """Init."""
        obs_dim, action_dim = get_obs_action_dim(env_id)
        # Make Kwargs for Net
        net_kwargs = net_kwargs or DEFAULT_NET_KWARGS
        net_kwargs["actor"].append(action_dim)
        self.exploration_noise = exploration_noise
        self.n_critics = n_critics
        self.obs_dim, self.action_dim = obs_dim, action_dim
        net_activ_kwargs = net_activ_kwargs or DEFAULT_NET_ACTIV_KWARGS

        self.key = jax.random.PRNGKey(seed)
        self.key, self.noise_key = jax.random.split(self.key, 2)
        self.net_kwargs = net_kwargs
        self.net_activ_kwargs = net_activ_kwargs
        self.optimizer_class = optax.adam

        self.action_bias, self.action_scale = get_action_bias_scale(env_id=env_id)
        self.env_id = env_id
        self.train = True
        self.manager = None

    def save(self, checkpoint_dir: os.PathLike, step: int = 0) -> None:
        """Save Train State."""
        if self.manager is None:
            options = ocp.CheckpointManagerOptions(max_to_keep=2, create=True)
            checkpointer = ocp.PyTreeCheckpointer()
            self.manager = ocp.CheckpointManager(checkpoint_dir, checkpointer, options)
        combined_states = {
            "encoder_state": self.encoder_state,
            "dynamics_state": self.dynamics_state,
            "qfn_state": self.qfn_state,
            "actor_state": self.actor_state,
            "config": {
                "net_kwargs": self.net_kwargs,
                "env_id": self.env_id,
            },
        }
        save_args = flax.training.orbax_utils.save_args_from_target(combined_states)
        self.manager.save(step, combined_states, save_kwargs={"save_args": save_args})

    @classmethod
    def load(cls, checkpoint_dir: os.PathLike, step: int = 0) -> "TD7Agent":
        """Load Agent."""
        checkpointer = ocp.PyTreeCheckpointer()
        manager = ocp.CheckpointManager(checkpoint_dir, checkpointer)
        restored = manager.restore(step=step)
        config = restored["config"]
        config["net_kwargs"]["actor"].pop(-1)
        agent = TD7Agent(**config)
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
                    target_fixed_params=restored[name]["target_fixed_params"],
                ),
            )

        _replace("actor_state")
        _replace("qfn_state")
        _replace("dynamics_state")
        _replace("encoder_state")

    def load_all_components_from_other(self, agent: "TD7Agent") -> None:
        """Load all components."""
        self.encoder_state = deepcopy(agent.encoder_state)
        self.dynamics_state = deepcopy(agent.dynamics_state)
        self.qfn_state = deepcopy(agent.qfn_state)
        self.actor_state = deepcopy(agent.actor_state)
        self.noise_key = deepcopy(agent.noise_key)
        self.key = deepcopy(agent.key)

    def build(
        self,
        key: jax.Array,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        encoder_lr: float = 3e-4,
    ) -> jax.Array:
        """Build neural network architecture and optimizer."""
        # Keep a key for the actor
        key, actor_key, qf_key, encoder_key, dynamics_key = jax.random.split(key, 5)
        _, self.key = jax.random.split(key, 2)

        # Prepare input: observation and action
        obs = jnp.ones((1, self.obs_dim))
        action = jnp.ones((1, self.action_dim))

        # Make Encoder.
        self.encoder = Encoder(
            net_arch=self.net_kwargs["encoder"], activ=self.net_activ_kwargs["encoder"]
        )
        self.encoder_state = RLTrainState.create(
            apply_fn=self.encoder.apply,
            params=self.encoder.init(encoder_key, obs),
            target_params=self.encoder.init(encoder_key, obs),
            target_fixed_params=self.encoder.init(encoder_key, obs),
            tx=self.optimizer_class(learning_rate=encoder_lr),
            key=encoder_key,
        )
        zs = self.encoder_state.apply_fn(self.encoder_state.params, obs)

        # Make Dynamics.
        self.dynamics = Dynamics(
            net_arch=self.net_kwargs["dynamics"],
            activ=self.net_activ_kwargs["dynamics"],
        )
        self.dynamics_state = RLTrainState.create(
            apply_fn=self.dynamics.apply,
            params=self.dynamics.init(dynamics_key, zs, action),
            target_params=self.dynamics.init(dynamics_key, zs, action),
            target_fixed_params=self.dynamics.init(dynamics_key, zs, action),
            tx=self.optimizer_class(learning_rate=encoder_lr),
            key=dynamics_key,
        )
        zsa = self.dynamics_state.apply_fn(self.dynamics_state.params, zs, action)

        # Make Critic.
        self.qfn = VectorCritic(
            net_arch=self.net_kwargs["critic"],
            activ=self.net_activ_kwargs["critic"],
            n_critics=self.n_critics,
        )

        self.qfn_state = RLTrainState.create(
            apply_fn=self.qfn.apply,
            params=self.qfn.init(qf_key, obs, action, zsa, zs),
            target_params=self.qfn.init(qf_key, obs, action, zsa, zs),
            target_fixed_params=self.qfn.init(qf_key, obs, action, zsa, zs),
            tx=self.optimizer_class(learning_rate=critic_lr),
            key=qf_key,
        )

        # Make actor
        self.actor = Actor(
            net_arch=self.net_kwargs["actor"], activ=self.net_activ_kwargs["actor"]
        )

        self.actor_state = RLTrainState.create(
            apply_fn=self.actor.apply,
            params=self.actor.init(actor_key, obs, zs, train=False),
            target_params=self.actor.init(actor_key, obs, zs, train=False),
            target_fixed_params=self.actor.init(actor_key, obs, zs, train=False),
            tx=self.optimizer_class(learning_rate=actor_lr),
            key=actor_key,
        )

        self.actor.apply = jax.jit(self.actor.apply)
        self.qfn.apply = jax.jit(self.qfn.apply)
        self.encoder.apply = jax.jit(self.encoder.apply)
        self.dynamics.apply = jax.jit(self.dynamics.apply)
        return key

    def sample(
        self,
        obs: np.ndarray,
        deterministic: bool = False,
        exploration_scale: float | None = None,
    ) -> np.ndarray:
        """Sample."""
        if obs.ndim == 1:
            obs = np.expand_dims(obs, 0)
        action, self.key = self._get_deterministic_action(
            self.encoder_state,
            self.encoder_state.target_params,
            self.actor_state,
            self.actor_state.params,
            obs,
            self.key,
        )
        if not deterministic:
            exploration_scale = exploration_scale or self.exploration_noise
            self.key, exploration_key = jax.random.split(self.key)
            noise = (
                jax.random.normal(
                    exploration_key,
                    action.shape,
                )
                * exploration_scale
            )
            action = action + noise
            action = jnp.clip(action, -1.0, 1.0)
        action = action * self.action_scale + self.action_bias
        if action.shape[0] == 1:
            action = action[0]
        return action

    @staticmethod
    @jax.jit
    def _get_deterministic_action(
        encoder_state: RLTrainState,
        encoder_params,
        actor_state: RLTrainState,
        actor_params,
        obs: jax.Array,
        key: jax.Array,
    ) -> jax.Array:
        """Calculate deterministic action."""
        key, dropout_key = jax.random.split(key, 2)
        zs = encoder_state.apply_fn(encoder_params, obs)
        action_logits = actor_state.apply_fn(
            actor_params, obs, zs, True, rngs={"dropout": dropout_key}
        )
        action = jnp.tanh(action_logits)
        return action, key
