from copy import deepcopy
from functools import partial
from logging import Logger
import os
from typing import Any, Type
import warnings
import flax
import gymnasium
import jax
import jax.flatten_util
from flax.training.train_state import TrainState

# from tqdm.rich import tqdm
import optax
from tqdm import tqdm
import jax.numpy as jnp
from gymnasium.wrappers import RecordEpisodeStatistics

from jax_rl.auto_ent import create_ent_coef_state
from jax_rl.replay_memory import (
    LAPReplayMemory,
    SimpleReplayMemory,
    AbcReplayMemory,
)
from jax_rl.rollout import Rollout
from jax_rl.utils import log_train_infos
from jax_rl import evaluate_agent
from simba import RLTrainState
from simba.agent import SimbaAgent


class SimbaAlgorithm:
    """Simba Algorithm."""

    def __init__(
        self,
        agent: SimbaAgent,
        discount_factor: float = 0.99,
        ent_coef: float = 0.1,
        policy_freq: int = 2,
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
        self.key = self.agent.key
        self.n_runs: int = 0
        _, ent_key = jax.random.split(self.key)
        self.ent_coef_state = create_ent_coef_state(ent_coef, ent_key)
        self.target_entropy = -float(self.agent.action_dim / 2)

    # Train Parts !!!!
    ###########################################################################################
    def train(
        self, batch_size: int, replay_memory: SimpleReplayMemory | LAPReplayMemory
    ) -> dict[str, float]:
        """Train."""
        self.agent.train = True
        batch = replay_memory.sample(batch_size)
        self.agent.obs_rms.update(batch["obs"])

        # Normalize.
        batch["obs"] = self.agent.normalize(batch["obs"])
        batch["next_obs"] = self.agent.normalize(batch["next_obs"])

        carry = self._train(
            # Neural Network
            self.agent.actor_state,
            self.agent.qfn_state,
            self.ent_coef_state,
            # Train
            self.discount_factor,
            self.policy_freq,
            self.agent.n_quntile_target,
            self.target_entropy,
            # Train.
            self.key,
            # Variable
            self.n_runs,
            **batch,
        )
        self.agent.actor_state = carry["actor_state"]
        self.agent.qfn_state = carry["qfn_state"]
        self.ent_coef_state = carry["ent_coef_state"]
        self.key = carry["key"]

        info = carry["info"]
        info = {key: jax.device_get(value) for key, value in info.items()}

        self.agent.qfn_state = self.soft_update(0.005, self.agent.qfn_state)
        self.agent.actor_state = self.soft_update(0.005, self.agent.actor_state)

        self.n_runs += 1
        self.agent.train = False
        return info

    @staticmethod
    @jax.jit
    def soft_update(tau: float, qf_state: RLTrainState) -> RLTrainState:
        qf_state = qf_state.replace(
            target_params=optax.incremental_update(
                qf_state.params, qf_state.target_params, tau
            )
        )
        return qf_state

    @classmethod
    @partial(
        jax.jit,
        static_argnames=[
            "cls",
            "discount_factor",
            "policy_freq",
            "n_quantile_target",
            "target_entropy",
        ],
    )
    def _train(
        cls,
        # neural network arch
        actor_state: RLTrainState,
        qfn_state: RLTrainState,
        ent_coef_state: TrainState,
        # Train
        discount_factor: float,
        policy_freq: int,
        n_quantile_target: int,
        target_entropy: float,
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
                "loss/temp",
                "entropy/mean",
                "value/mean",
                "grad/actor",
                "grad/qfn",
                "grad/temp",
                "weight/critic",
                "weight/actor",
            ]
        }
        carry = {
            "actor_state": actor_state,
            "qfn_state": qfn_state,
            "ent_coef_state": ent_coef_state,
            "key": key,
            "info": dummy_info,
        }

        def one_update(idx: int, carry: dict[str, Any]) -> dict[str, Any]:
            """Update."""
            actor_state = carry["actor_state"]
            qfn_state = carry["qfn_state"]
            key = carry["key"]
            info: dict[str, Any] = carry["info"]
            (
                qfn_state,
                key,
                critic_info,
            ) = cls._update_critic(
                actor_state,
                qfn_state,
                ent_coef_state,
                obs,
                action,
                reward,
                next_obs,
                done,
                discount_factor,
                n_quantile_target,
                key,
            )

            def update_actor():
                return cls._update_actor(
                    actor_state,
                    qfn_state,
                    ent_coef_state,
                    obs,
                    target_entropy,
                    key,
                )

            def skip_update_actor():
                return (
                    actor_state,
                    ent_coef_state,
                    key,
                    {
                        key: jnp.nan
                        for key in [
                            "loss/actor",
                            "loss/temp",
                            "entropy/mean",
                            "value/mean",
                            "grad/actor",
                            "grad/temp",
                            "weight/actor",
                        ]
                    },
                )

            actor_state, key, actor_info = jax.lax.cond(
                n_runs % policy_freq == 0, update_actor, skip_update_actor
            )

            info.update(critic_info)
            info.update(actor_info)

            return {
                "actor_state": actor_state,
                "qfn_state": qfn_state,
                "ent_coef_state": ent_coef_state,
                "key": key,
                "info": info,
            }

        update_carry = jax.lax.fori_loop(0, 1, one_update, carry)
        return update_carry

    @staticmethod
    @partial(
        jax.jit,
        static_argnames=["discount_factor", "n_quantile_target"],
    )
    def _update_critic(
        actor_state: RLTrainState,
        qfn_state: RLTrainState,
        ent_coef_state: TrainState,
        obs: jax.Array,
        action: jax.Array,
        reward: jax.Array,
        next_obs: jax.Array,
        done: jax.Array,
        discount_factor: float,
        n_quantile_target: int,
        key: jax.Array,
    ) -> tuple[RLTrainState, jax.Array, dict[str, float]]:
        """Update critic."""
        key, noise_key = jax.random.split(key, 2)

        def critic_loss(params: flax.core.FrozenDict) -> tuple[jax.Array]:
            """Calculate Actor Loss."""
            next_action, entropy, _ = SimbaAgent._rsample(
                actor_state,
                actor_state.params,
                next_obs,
                noise_key,
            )

            # Batch, N_Critic, N_Quantile
            next_qvalue = qfn_state.apply_fn(
                qfn_state.target_params,
                next_obs,
                next_action,
            )
            batch, n_critic, n_qunatile = next_qvalue.shape
            next_qvalue = jnp.sort(
                next_qvalue.reshape(batch, n_critic * n_qunatile), axis=-1
            )
            # Batch, N_Qunatile_Target
            ent_coef = ent_coef_state.apply_fn({"params": ent_coef_state.params})
            next_qvalue = next_qvalue[:, :n_quantile_target] + ent_coef * entropy
            q_target = reward + discount_factor * next_qvalue * done

            # Batch, N,Critic, N_Quantile
            qvalue = qfn_state.apply_fn(params, obs, action)

            # [Q Target] Batch, N_Qunatile_Target
            # -> Batch, N_Quantile_Target, N_Quantile
            q_target = jnp.repeat(q_target[:, :, None], n_qunatile, axis=-1)
            # [Q Target] Batch, N_Quantile_Target, N_Quantile -> Batch N_Critic NT NQ
            q_target = jnp.repeat(q_target[:, None], n_critic, axis=1)

            # [Q Value]: Batch N_Critic NT NQ
            qvalue = jnp.repeat(qvalue[:, :, None], n_quantile_target, axis=2)
            td_error = jax.lax.stop_gradient(q_target) - qvalue
            td_abs_error = jax.lax.abs(td_error)
            huber_coeff = 1.0
            # Batch N_Critic NT NQ
            loss = (
                jnp.where(
                    td_abs_error < huber_coeff,
                    0.5 * jnp.power(td_error, 2),
                    huber_coeff * td_abs_error - huber_coeff**2.0 / 2.0,
                )
                # .sum(axis=0)
                # .mean()
            )
            priority_loss = td_abs_error.reshape(batch, -1).mean(axis=-1, keepdims=True)

            # (NQ, )
            tau = jnp.expand_dims(
                (jnp.arange(0, n_qunatile, dtype=jnp.float32) + 0.5) / n_qunatile, 0
            )
            tau = jax.lax.stop_gradient(
                jnp.abs(tau - (q_target < qvalue).astype(jnp.float32))
            )
            loss = loss * tau
            loss = loss.reshape(batch, -1).mean(axis=-1).mean()
            return loss, priority_loss

        (loss, _), grads = jax.value_and_grad(critic_loss, has_aux=True)(
            qfn_state.params
        )
        qfn_state = qfn_state.apply_gradients(grads=grads)
        flat_grads, _ = jax.flatten_util.ravel_pytree(grads)

        qfn_params, _ = jax.flatten_util.ravel_pytree(qfn_state.params["params"])
        weight_critic = jnp.linalg.norm(qfn_params)

        info = {
            "loss/critic": loss,
            "grad/qfn": jnp.linalg.norm(flat_grads),
            "weight/critic": weight_critic,
        }

        return qfn_state, key, info

    @staticmethod
    @partial(jax.jit, static_argnames=["target_entropy"])
    def _update_actor(
        actor_state: RLTrainState,
        qfn_state: RLTrainState,
        ent_coef_state: TrainState,
        obs: jax.Array,
        target_entropy: float,
        key: jax.Array,
    ) -> tuple[RLTrainState, TrainState, jax.Array, dict[str, float]]:
        """Update actor."""
        key, _ = jax.random.split(key, 2)

        def actor_loss(params: flax.core.FrozenDict) -> jax.Array:
            """Calculate Actor Loss."""
            action, entropy, _ = SimbaAgent._rsample(
                actor_state,
                params,
                obs,
                key,
            )
            qvalue = qfn_state.apply_fn(qfn_state.params, obs, action)
            ent_coef = ent_coef_state.apply_fn({"params": ent_coef_state.params})

            batch, _, _ = qvalue.shape
            qvalue = qvalue.reshape(batch, -1).mean(axis=-1)

            actor_loss = (-qvalue).mean() - ent_coef * (entropy).mean()
            return actor_loss, (entropy, qvalue)

        (loss, aux), grads = jax.value_and_grad(actor_loss, has_aux=True)(
            actor_state.params
        )
        entropy, qvalue = aux
        actor_state = actor_state.apply_gradients(grads=grads)
        flat_grads, _ = jax.flatten_util.ravel_pytree(grads)
        actor_params, _ = jax.flatten_util.ravel_pytree(actor_state.params["params"])
        weight_actor = jnp.linalg.norm(actor_params)

        def temperature_loss(temp_params) -> jax.Array:
            """."""
            ent_coef_value = ent_coef_state.apply_fn({"params": temp_params})
            ent_coef_loss = ent_coef_value * (entropy - target_entropy)  # type: ignore[union-attr]
            return ent_coef_loss.mean()

        temp_loss, temp_grads = jax.value_and_grad(temperature_loss)(
            ent_coef_state.params
        )
        ent_coef_state = ent_coef_state.apply_gradients(grads=temp_grads)
        flat_temp_grads, _ = jax.flatten_util.ravel_pytree(temp_grads)

        info = {
            "loss/actor": loss,
            "loss/temp": temp_loss,
            "entropy/mean": entropy.mean(),
            "value/mean": qvalue.mean(),
            "grad/actor": jnp.linalg.norm(flat_grads),
            "grad/temp": jnp.linalg.norm(flat_temp_grads),
            "weight/actor": weight_actor,
        }
        return actor_state, ent_coef_state, key, info

    ###########################################################################################

    def run(
        self,
        total_timesteps: int = 5_000_000,
        exploration_timesteps: int = 25_000,
        batch_size: int = 256,
        eval_period: int = 10_000,
        replay_memory_class: Type[AbcReplayMemory] = LAPReplayMemory,
        replay_kwargs: dict[str, Any] | None = None,
        env: gymnasium.Env | None = None,
        train_logger: Logger | None = None,
        eval_logger: Logger | None = None,
        rollout_logger: Logger | None = None,
        seed: int = 42,
        use_progressbar: bool = False,
    ) -> None:
        """Run Algorithm."""
        warnings.filterwarnings("ignore")
        # Initialize the environment if not provided
        if env is None:
            env = gymnasium.make(self.agent.env_id)
        env.reset(seed=seed)
        if "dm_control" in self.agent.env_id:
            eval_env = gymnasium.make_vec(f"eval_{self.agent.env_id}", 10)
        else:
            eval_env = gymnasium.make_vec(self.agent.env_id, 10)
        eval_env = RecordEpisodeStatistics(eval_env, 10)
        eval_env.reset(seed=seed)
        eval_env.is_vector_env = True
        eval_env.num_envs = 10
        if not isinstance(env, RecordEpisodeStatistics):
            env = RecordEpisodeStatistics(env, 1)

        if replay_kwargs is None:
            replay_kwargs = {
                "replay_buffer_size": 1_000_000,
            }
        # Prepare ReplayMemory
        replay_kwargs["observation_space"] = env.observation_space
        replay_kwargs["action_space"] = env.action_space
        replay_memory = replay_memory_class(**replay_kwargs)
        rollout = Rollout(env, replay_memory, seed)

        if use_progressbar:
            progress_bar = tqdm(total_timesteps)
        timestep = 0

        train_flag = False
        # Logging
        epsiode_returns_list: list[float] = list()
        start_logging = True
        root_dir = None
        eval_info = evaluate_agent(self.ckpt_agent, eval_env)
        if eval_logger is not None:
            eval_logger.info(",".join(["timestep"] + list(eval_info.keys())))
        if rollout_logger is not None:
            rollout_logger.info(",".join(["timestep", "return", "length"]))
            root_dir = "/".join(rollout_logger.name.split("/")[:-1])
            aa = os.getcwd()
            root_dir = f"{aa}/{root_dir}/ckpt"
        logging_interval = 1_000
        train_infos = list()
        while timestep < total_timesteps:
            done = rollout.sample()
            timestep += 1
            if use_progressbar:
                progress_bar.update(1)
            if not train_flag:
                if len(rollout.replay_memory) > exploration_timesteps:
                    train_flag = True
                    rollout.set_sampler(self.agent)

            if train_flag:
                train_infos.append(self.train(batch_size, replay_memory))

            # After episode, decide whether to gather data without updating.
            if done:
                episode_return: float = float(rollout.env.return_queue[-1][0])
                episode_length: int = int(rollout.env.length_queue[-1][0])
                # Logging
                epsiode_returns_list.append(episode_return)

                if rollout_logger is not None:
                    rollout_logger.info(f"{timestep},{episode_return},{episode_length}")

            # Evaluation.
            if timestep % eval_period == 0:
                if train_flag:
                    eval_info = evaluate_agent(self.agent, eval_env)
                    stats_string = ",".join(
                        [f"{value:.4f}" for value in eval_info.values()]
                    )
                    eval_logger.info(f"{timestep},{stats_string}")

            # Run Train.
            if train_flag and timestep % logging_interval == 0:
                rollout_info = {
                    "returns": sum(epsiode_returns_list) / len(epsiode_returns_list),
                }
                if use_progressbar:
                    progressbar_info = deepcopy(rollout_info)
                    progressbar_info["perf/mean"] = eval_info["perf/mean"]
                    progressbar_info["perf/median"] = eval_info["median"]
                    progress_bar.set_postfix(progressbar_info)
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

                epsiode_returns_list.clear()
                train_infos.clear()
