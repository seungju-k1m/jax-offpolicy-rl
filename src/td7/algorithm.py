from copy import deepcopy
from datetime import timedelta
from functools import partial
from logging import Logger
import os
from typing import Any, Type
import flax
import gymnasium
import jax
import jax.flatten_util

# from tqdm.rich import tqdm
from tqdm import tqdm
import jax.numpy as jnp
from gymnasium.wrappers import RecordEpisodeStatistics

from td7.agent import TD7Agent
from jax_rl import RLTrainState
from jax_rl.replay_memory import (
    LAPReplayMemory,
    SimpleReplayMemory,
    AbcReplayMemory,
)
from jax_rl.rollout import Rollout
from jax_rl.utils import log_train_infos


def evaluate_agent(
    agent: TD7Agent,
    env: RecordEpisodeStatistics,
    n_episodes: int = 10,
    deterministic: bool = True,
) -> dict[str, float]:
    """Evaluate Agent."""
    obs, _ = env.reset()
    env.return_queue.clear()
    while len(env.return_queue) < n_episodes:
        action = agent.sample(obs, deterministic)
        action = jax.device_get(action)
        next_obs, _, _, _, _ = env.step(action)
        obs = next_obs
    returns_list = list(env.return_queue)[:n_episodes]
    mean = sum(returns_list) / len(returns_list)
    min_return, max_return = min(returns_list), max(returns_list)
    # Calculate the median
    if n_episodes % 2 == 1:
        # If odd, return the middle element
        median = returns_list[n_episodes // 2]
    else:
        # If even, return the average of the two middle elements
        median = (returns_list[n_episodes // 2 - 1] + returns_list[n_episodes // 2]) / 2
    info = {
        "perf/mean": mean,
        "perf/min": min_return,
        "perf/max": max_return,
        "median": median,
    }
    returns = {str(idx): value for idx, value in enumerate(returns_list)}
    info.update(returns)
    return info


class TD7Algorithm:
    """TD7 Algorithm."""

    def __init__(
        self,
        agent: TD7Agent,
        discount_factor: float = 0.99,
        target_update_rate: int = 250,
        policy_freq: int = 2,
        target_noise: float = 0.2,
        *args,
        **kwargs,
    ) -> None:
        """Init."""
        agent.train = False
        self.agent = agent
        self.ckpt_agent = deepcopy(agent)
        self.best_agent = deepcopy(self.ckpt_agent)
        self.policy_freq = policy_freq
        self.discount_factor = discount_factor
        self.target_update_rate = target_update_rate
        self.target_noise = target_noise
        self.key = self.agent.key
        self.value_max = -1e8
        self.value_min = 1e8
        self.value_target_max = 0
        self.value_target_min = 0
        self.n_runs: int = 0

    def run(
        self,
        total_timesteps: int = 5_000_000,
        exploration_timesteps: int = 25_000,
        max_epsidoes_per_ckpt: int = 20,
        init_episodes_per_ckpt: int = 1,
        batch_size: int = 256,
        reset_weight: float = 0.9,
        update_timestep: int = int(75e4),
        eval_period: int = 5_000,
        replay_memory_class: Type[AbcReplayMemory] = LAPReplayMemory,
        replay_kwargs: dict[str, Any] | None = None,
        env: gymnasium.Env | None = None,
        train_logger: Logger | None = None,
        eval_logger: Logger | None = None,
        rollout_logger: Logger | None = None,
        seed: int = 42,
        use_progressbar: bool = False,
        deterministic: bool = True,
        *args,
        **kwargs,
    ) -> None:
        """Run the TD7 training algorithm."""
        # Initialize the environment if not provided
        if env is None:
            env = gymnasium.make(self.agent.env_id)
        env.reset(seed=seed)

        # Create a vectorized evaluation environment with 10 instances
        eval_env = gymnasium.make_vec(env.spec.id, 10)
        eval_env = RecordEpisodeStatistics(eval_env, 200)
        eval_env.reset(seed=seed)
        eval_env.is_vector_env = True
        eval_env.num_envs = 10

        # Ensure the training environment is wrapped with RecordEpisodeStatistics
        if not isinstance(env, RecordEpisodeStatistics):
            env = RecordEpisodeStatistics(env, 1)

        # Initialize replay memory settings if not provided
        if replay_kwargs is None:
            replay_kwargs = {
                "replay_buffer_size": 1_000_000,
            }
        replay_kwargs["observation_space"] = env.observation_space
        replay_kwargs["action_space"] = env.action_space

        # Create the replay memory and rollout manager
        replay_memory = replay_memory_class(**replay_kwargs)
        rollout = Rollout(env, replay_memory, seed)

        # Initialize the progress bar if enabled
        if use_progressbar:
            progress_bar = tqdm(total_timesteps, dynamic_ncols=True, leave=True)

        # Set initial checkpoint parameters
        current_max_episodes = init_episodes_per_ckpt
        timestep = 0
        best_min_return = -1e8
        best_return = -1e8

        n_updates = 0
        train_flag = False  # Flag to indicate when training starts
        episode_returns_list: list[float] = list()
        start_logging = True  # Whether to start logging
        root_dir = None  # Directory for saving checkpoints

        # Perform initial evaluation of the agent
        eval_info = evaluate_agent(self.ckpt_agent, eval_env)

        # Log evaluation headers if evaluation logger exists
        if eval_logger is not None:
            eval_logger.info(",".join(["timestep"] + list(eval_info.keys())))

        # Log rollout information and save the initial checkpoint
        if rollout_logger is not None:
            rollout_logger.info(",".join(["timestep", "return", "length"]))
            root_dir = "/".join(rollout_logger.name.split("/")[:-1])
            aa = os.getcwd()
            root_dir = f"{aa}/{root_dir}/ckpt"
            self.ckpt_agent.save(root_dir, timestep)

        # Main training loop
        while timestep < total_timesteps:
            train_infos: list[dict[str, float]] = list()
            agent_min_return = 1e8

            # Run episodes up to the current max episodes per checkpoint
            for idx in range(current_max_episodes):
                done = False

                # Run an episode
                while not done:
                    done = rollout.sample()
                    timestep += 1

                    # Start training after `exploration_timesteps`
                    if not train_flag:
                        if len(rollout.replay_memory) > exploration_timesteps:
                            train_flag = True
                            rollout.set_sampler(self.agent)

                    # Perform periodic evaluation
                    if train_flag and timestep % eval_period == 0:
                        eval_info = evaluate_agent(
                            self.ckpt_agent, eval_env, deterministic=deterministic
                        )

                        # Save the best-performing checkpoint
                        if eval_info["perf/mean"] > best_return:
                            best_return = eval_info["perf/mean"]
                            self.best_agent.load_all_components_from_other(
                                self.ckpt_agent
                            )
                            self.ckpt_agent.save(root_dir, timestep)

                        # Log evaluation results
                        stats_string = ",".join(
                            [f"{value:.4f}" for value in eval_info.values()]
                        )
                        eval_logger.info(f"{timestep},{stats_string}")

                # Store episode return and length
                episode_return: float = float(rollout.env.return_queue[-1][0])
                episode_length: int = int(rollout.env.length_queue[-1][0])

                # Increase update count based on episode length
                if train_flag:
                    n_updates += episode_length

                # Track the minimum return of the agent
                agent_min_return = min(episode_return, agent_min_return)

                # Log rollout data
                episode_returns_list.append(episode_return)

                if rollout_logger is not None:
                    rollout_logger.info(f"{timestep},{episode_return},{episode_length}")

                # Stop if agent's performance drops below the best minimum return
                if agent_min_return < best_min_return:
                    break

            # Update the checkpoint if agent performance improves
            if (
                agent_min_return > best_min_return
                and idx == current_max_episodes - 1
                and train_flag
            ):
                best_min_return = agent_min_return
                self.ckpt_agent.load_all_components_from_other(self.agent)

            # Perform training updates
            if train_flag:
                for _ in range(n_updates):
                    train_infos.append(self.train(batch_size, replay_memory))

                # Compute rollout statistics
                rollout_info = {
                    "returns": sum(episode_returns_list) / len(episode_returns_list),
                    "best_min_return": best_min_return,
                }

                # Update progress bar
                if use_progressbar:
                    progressbar_info = deepcopy(rollout_info)
                    progressbar_info["perf/mean"] = eval_info["perf/mean"]
                    progressbar_info["perf/median"] = eval_info["median"]
                    progress_bar.update(n_updates)
                    progress_bar.set_postfix(progressbar_info)

                # Log training information
                if train_logger is not None:
                    log_train_infos(
                        timestep,
                        train_logger,
                        train_infos,
                        eval_info,
                        rollout_info,
                        start_logging,
                    )
                    start_logging = False

                # Reset tracking variables
                episode_returns_list.clear()
                n_updates = 0

                # Adjust checkpoint update frequency based on timestep
                if timestep > update_timestep:
                    current_max_episodes = max_epsidoes_per_ckpt
                    best_min_return *= reset_weight
                    reset_weight = 1.0

    # Train Parts !!!!
    ###########################################################################################
    def train(
        self, batch_size: int, replay_memory: SimpleReplayMemory | LAPReplayMemory
    ) -> dict[str, float]:
        """Train."""
        self.n_runs += 1
        batch = replay_memory.sample(batch_size)
        carry = self._train(
            # Neural Network
            self.agent.encoder_state,
            self.agent.dynamics_state,
            self.agent.actor_state,
            self.agent.qfn_state,
            # Train
            self.discount_factor,
            self.target_noise,
            self.policy_freq,
            self.value_target_max,
            self.value_target_min,
            # Train.
            self.key,
            # Variable
            self.n_runs,
            **batch,
        )
        self.agent.encoder_state = carry["encoder_state"]
        self.agent.dynamics_state = carry["dynamics_state"]
        self.agent.actor_state = carry["actor_state"]
        self.agent.qfn_state = carry["qfn_state"]
        td_error: jax.Array = jax.device_get(carry["td_error"])
        self.key = carry["key"]
        value_max = carry["value_max"]
        value_min = carry["value_min"]

        info = carry["info"]
        info = {key: jax.device_get(value) for key, value in info.items()}
        priority = jnp.reshape(td_error.clip(1.0) ** 0.4, -1)
        info["priority/mean"] = priority.mean().item()
        info["priority/std"] = priority.std().item()

        self.value_max = max(value_max, self.value_max)
        self.value_min = min(value_min, self.value_min)
        if isinstance(replay_memory, LAPReplayMemory):
            replay_memory.update_priority(priority)
        if self.n_runs % self.target_update_rate == 0:
            (
                self.agent.encoder_state,
                self.agent.dynamics_state,
                self.agent.actor_state,
                self.agent.qfn_state,
            ) = self.hard_update(
                self.agent.encoder_state,
                self.agent.dynamics_state,
                self.agent.actor_state,
                self.agent.qfn_state,
            )
            self.value_target_max = self.value_max
            self.value_target_min = self.value_min
            if isinstance(replay_memory, LAPReplayMemory):
                replay_memory.reset_max_priority()
        self.agent.train = False
        return info

    @staticmethod
    @jax.jit
    def hard_update(
        encoder_state: RLTrainState,
        dynamics_state: RLTrainState,
        actor_state: RLTrainState,
        qf_state: RLTrainState,
    ) -> tuple[RLTrainState, RLTrainState, RLTrainState, RLTrainState]:
        actor_state = actor_state.replace(target_params=actor_state.params)
        qf_state = qf_state.replace(target_params=qf_state.params)
        encoder_state = encoder_state.replace(
            target_fixed_params=encoder_state.target_params
        )
        dynamics_state = dynamics_state.replace(
            target_fixed_params=dynamics_state.target_params
        )

        encoder_state = encoder_state.replace(target_params=encoder_state.params)
        dynamics_state = dynamics_state.replace(target_params=dynamics_state.params)
        return encoder_state, dynamics_state, actor_state, qf_state

    @classmethod
    @partial(
        jax.jit,
        static_argnames=[
            "cls",
            "discount_factor",
            "target_noise",
            "policy_freq",
        ],
    )
    def _train(
        cls,
        # neural network arch
        encoder_state: RLTrainState,
        dynamics_state: RLTrainState,
        actor_state: RLTrainState,
        qfn_state: RLTrainState,
        # Train
        discount_factor: float,
        target_noise: float,
        policy_freq: int,
        value_target_max: float,
        value_target_min: float,
        # optimizer
        key: jax.Array,
        # Variable
        n_runs: int,
        # batch
        obs: jax.Array,
        action: jax.Array,
        reward: jax.Array,
        next_obs: jax.Array,
        done: jax.Array,
    ) -> dict[str, Any]:
        """Train."""
        dummy_info = {
            key: jnp.array(0.0)
            for key in [
                "loss/critic",
                "loss/actor",
                "value/mean",
                "grad/encoder",
                "grad/actor",
                "grad/qfn",
                "weight/critic",
                "weight/actor",
                "loss/encoder",
                "weight/encoder",
                "norm/next_zs",
                "norm/zsa",
            ]
        }
        batch_size = obs.shape[0]
        carry = {
            "actor_state": actor_state,
            "qfn_state": qfn_state,
            "encoder_state": encoder_state,
            "dynamics_state": dynamics_state,
            "key": key,
            "info": dummy_info,
            "td_error": jnp.ones((batch_size, 1)),
            "value_max": jnp.array(0),
            "value_min": jnp.array(0),
        }

        def one_update(idx: int, carry: dict[str, Any]) -> dict[str, Any]:
            """Update."""
            actor_state = carry["actor_state"]
            qfn_state = carry["qfn_state"]
            encoder_state = carry["encoder_state"]
            dynamics_state = carry["dynamics_state"]
            key = carry["key"]
            info: dict[str, Any] = carry["info"]
            (
                encoder_state,
                dynamics_state,
                key,
                encoder_info,
            ) = cls._update_encoder(
                encoder_state,
                dynamics_state,
                obs,
                action,
                next_obs,
                key,
            )

            (
                qfn_state,
                value_max,
                value_min,
                td_error,
                key,
                critic_info,
            ) = cls._update_critic(
                encoder_state,
                dynamics_state,
                actor_state,
                qfn_state,
                obs,
                action,
                reward,
                next_obs,
                done,
                discount_factor,
                value_target_max,
                value_target_min,
                target_noise,
                key,
            )

            def update_actor():
                return cls._update_actor(
                    encoder_state,
                    dynamics_state,
                    actor_state,
                    qfn_state,
                    obs,
                    key,
                )

            def skip_update_actor():
                return (
                    actor_state,
                    key,
                    {
                        key: jnp.nan
                        for key in [
                            "loss/actor",
                            "value/mean",
                            "grad/actor",
                            "weight/actor",
                        ]
                    },
                )

            actor_state, key, actor_info = jax.lax.cond(
                n_runs % policy_freq == 0, update_actor, skip_update_actor
            )
            info.update(encoder_info)
            info.update(critic_info)
            info.update(actor_info)

            return {
                "encoder_state": encoder_state,
                "dynamics_state": dynamics_state,
                "actor_state": actor_state,
                "qfn_state": qfn_state,
                "key": key,
                "info": info,
                "value_max": value_max,
                "value_min": value_min,
                "td_error": td_error,
            }

        update_carry = jax.lax.fori_loop(0, 1, one_update, carry)
        return update_carry

    @staticmethod
    @jax.jit
    def _update_encoder(
        encoder_state: RLTrainState,
        dynamics_state: RLTrainState,
        observations: jax.Array,
        actions: jax.Array,
        next_observations: jax.Array,
        key: jax.Array,
    ) -> tuple[RLTrainState, RLTrainState, jax.Array, dict[str, float]]:
        """Update encoder and dynamics."""

        def mse_loss(
            encoder_params: flax.core.FrozenDict, dynamics_params: flax.core.FrozenDict
        ) -> jax.Array:
            next_zs = jax.lax.stop_gradient(
                encoder_state.apply_fn(encoder_params, next_observations)
            )
            zs = encoder_state.apply_fn(encoder_params, observations)
            zsa = dynamics_state.apply_fn(dynamics_params, zs, actions)
            loss = ((next_zs - zsa) ** 2.0).mean()
            return loss, (
                jnp.linalg.norm(zsa, axis=1).mean(),
                jnp.linalg.norm(next_zs, axis=1).mean(),
            )

        (
            (encoder_loss, (zsa_norm, next_zs_norm)),
            (
                encoder_grads,
                dynamics_grads,
            ),
        ) = jax.value_and_grad(mse_loss, argnums=(0, 1), has_aux=True)(
            encoder_state.params, dynamics_state.params
        )

        encoder_state = encoder_state.apply_gradients(grads=encoder_grads)
        dynamics_state = dynamics_state.apply_gradients(grads=dynamics_grads)
        encoder_norm, _ = jax.flatten_util.ravel_pytree(encoder_grads)
        dynamics_norm, _ = jax.flatten_util.ravel_pytree(dynamics_grads)

        encoder_params, _ = jax.flatten_util.ravel_pytree(
            encoder_state.params["params"]
        )
        dynamics_params, _ = jax.flatten_util.ravel_pytree(
            dynamics_state.params["params"]
        )
        weight_encoder_norm = jnp.linalg.norm(encoder_params) + jnp.linalg.norm(
            dynamics_params
        )
        info = {
            "loss/encoder": encoder_loss,
            "grad/encoder": jnp.linalg.norm(encoder_norm)
            + jnp.linalg.norm(dynamics_norm),
            "norm/zsa": zsa_norm,
            "norm/next_zs": next_zs_norm,
            "weight/encoder": weight_encoder_norm,
        }
        return encoder_state, dynamics_state, key, info

    @staticmethod
    @partial(jax.jit, static_argnames=["discount_factor", "target_noise"])
    def _update_critic(
        encoder_state: RLTrainState,
        dynamics_state: RLTrainState,
        actor_state: RLTrainState,
        qfn_state: RLTrainState,
        obs: jax.Array,
        action: jax.Array,
        reward: jax.Array,
        next_obs: jax.Array,
        done: jax.Array,
        discount_factor: float,
        value_target_max: float,
        value_target_min: float,
        target_noise: float,
        key: jax.Array,
    ) -> tuple[RLTrainState, jax.Array, dict[str, float]]:
        """Update critic."""
        key, noise_key = jax.random.split(key, 2)

        def critic_loss(params: flax.core.FrozenDict) -> tuple[jax.Array]:
            """Calculate Actor Loss."""
            next_action, _ = TD7Agent._get_deterministic_action(
                encoder_state,
                encoder_state.target_fixed_params,
                actor_state,
                actor_state.target_params,
                next_obs,
                noise_key,
            )
            noise = (jax.random.normal(noise_key, next_action.shape) * target_noise).clip(-0.5, 0.5)
            next_action = jnp.clip(next_action + noise, -1.0, 1.0)

            next_zs = encoder_state.apply_fn(
                encoder_state.target_fixed_params, next_obs
            )
            next_zsa = dynamics_state.apply_fn(
                dynamics_state.target_fixed_params, next_zs, next_action
            )
            # Batch, N_Critic, 1
            next_qvalue = qfn_state.apply_fn(
                qfn_state.target_params,
                next_obs,
                next_action,
                next_zsa,
                next_zs,
            )
            next_qvalue = jnp.min(next_qvalue, axis=1)
            next_qvalue = jnp.clip(next_qvalue, value_target_min, value_target_max)

            q_target = reward + discount_factor * next_qvalue * done
            zs = encoder_state.apply_fn(encoder_state.target_params, obs)
            zsa = dynamics_state.apply_fn(dynamics_state.target_params, zs, action)

            new_value_max = q_target.max()
            new_value_min = q_target.min()
            qvalue = qfn_state.apply_fn(params, obs, action, zsa, zs)
            q_target = jnp.expand_dims(q_target, axis=1)
            td_error = jax.lax.stop_gradient(q_target) - qvalue
            td_abs_error = jax.lax.abs(td_error)
            huber_coeff = 1.0
            loss = jnp.where(
                td_abs_error < huber_coeff,
                0.5 * jnp.power(td_error, 2),
                huber_coeff * td_abs_error - huber_coeff**2.0 / 2.0,
            )
            loss = loss.mean()
            return loss, (new_value_max, new_value_min, td_abs_error.max(axis=1))

        (loss, aux), grads = jax.value_and_grad(critic_loss, has_aux=True)(
            qfn_state.params
        )
        new_value_max, new_value_min, td_error = aux
        qfn_state = qfn_state.apply_gradients(grads=grads)
        flat_grads, _ = jax.flatten_util.ravel_pytree(grads)

        qfn_params, _ = jax.flatten_util.ravel_pytree(qfn_state.params["params"])
        weight_critic = jnp.linalg.norm(qfn_params)

        info = {
            "loss/critic": loss,
            "grad/qfn": jnp.linalg.norm(flat_grads),
            "weight/critic": weight_critic,
        }

        return qfn_state, new_value_max, new_value_min, td_error, key, info

    @staticmethod
    @jax.jit
    def _update_actor(
        encoder_state: RLTrainState,
        dynamics_state: RLTrainState,
        actor_state: RLTrainState,
        qfn_state: RLTrainState,
        obs: jax.Array,
        key: jax.Array,
    ) -> tuple[RLTrainState, jax.Array, dict[str, float]]:
        """Update actor."""
        key, _ = jax.random.split(key, 2)

        def actor_loss(params: flax.core.FrozenDict) -> jax.Array:
            """Calculate Actor Loss."""
            action, _ = TD7Agent._get_deterministic_action(
                encoder_state,
                encoder_state.target_params,
                actor_state,
                params,
                obs,
                key,
            )
            zs = encoder_state.apply_fn(encoder_state.target_params, obs)
            zsa = dynamics_state.apply_fn(dynamics_state.target_params, zs, action)
            qvalue = qfn_state.apply_fn(qfn_state.params, obs, action, zsa, zs)
            qvalue = qvalue.mean(axis=-1)
            actor_loss = -qvalue.mean()
            return actor_loss, qvalue

        (loss, aux), grads = jax.value_and_grad(actor_loss, has_aux=True)(
            actor_state.params
        )
        qvalue = aux
        actor_state = actor_state.apply_gradients(grads=grads)
        flat_grads, _ = jax.flatten_util.ravel_pytree(grads)
        actor_params, _ = jax.flatten_util.ravel_pytree(actor_state.params["params"])
        weight_actor = jnp.linalg.norm(actor_params)

        info = {
            "loss/actor": loss,
            "value/mean": qvalue.mean(),
            "grad/actor": jnp.linalg.norm(flat_grads),
            "weight/actor": weight_actor,
        }
        return actor_state, key, info
