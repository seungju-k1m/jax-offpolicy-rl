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

from tqdm import tqdm
import jax.numpy as jnp
from gymnasium.wrappers import RecordEpisodeStatistics

import flax.linen as nn
from jax_rl import RLTrainState
from flax.training.train_state import TrainState

from jax_rl.auto_ent import create_ent_coef_state
from sale_tqc.agent import SALETQCAgent
from jax_rl.replay_memory import LAPReplayMemory, SimpleReplayMemory, AbcReplayMemory
from jax_rl.rollout import Rollout
from jax_rl.utils import log_train_infos
from jax_rl import evaluate_agent


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


class SALETQCAlgorithm:
    """SALE-TQC Algorithm."""

    def __init__(
        self,
        agent: SALETQCAgent,
        discount_factor: float = 0.99,
        target_update_rate: int = 250,
        ent_coef: float = 0.1,
        policy_freq: int = 2,
    ) -> None:
        """Init."""
        agent.train = False
        self.agent = agent
        self.ckpt_agent = deepcopy(agent)
        self.best_agent = deepcopy(self.ckpt_agent)
        self.policy_freq = policy_freq
        self.discount_factor = discount_factor
        self.target_update_rate = target_update_rate
        self.key = self.agent.key
        self.value_max = -1e8
        self.value_min = 1e8
        self.value_target_max = 0
        self.value_target_min = 0
        self.n_runs: int = 0
        _, ent_key = jax.random.split(self.key)
        self.ent_coef_state = create_ent_coef_state(
            ent_coef,
            ent_key,
        )
        self.target_entropy = -float(self.agent.action_dim // 2)

    # Train Parts !!!!
    ###########################################################################################
    def train(
        self, batch_size: int, replay_memory: SimpleReplayMemory | LAPReplayMemory
    ) -> dict[str, float]:
        """Train."""
        self.agent.train = True
        batch = replay_memory.sample(batch_size)
        carry = self._train(
            # Neural Network
            self.agent.encoder_state,
            self.agent.dynamics_state,
            self.agent.actor_state,
            self.agent.qfn_state,
            self.ent_coef_state,
            # Train
            self.discount_factor,
            self.policy_freq,
            self.agent.n_quntile_target,
            self.value_target_max,
            self.value_target_min,
            self.target_entropy,
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
        self.ent_coef_state = carry["ent_coef_state"]
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
        self.n_runs += 1
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
            "policy_freq",
            "n_quantile_target",
        ],
    )
    def _train(
        cls,
        # neural network arch
        encoder_state: RLTrainState,
        dynamics_state: RLTrainState,
        actor_state: RLTrainState,
        qfn_state: RLTrainState,
        ent_coef_state: TrainState,
        # Train
        discount_factor: float,
        policy_freq: int,
        n_quantile_target: int,
        value_target_max: float,
        value_target_min: float,
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
                "entropy/mean",
                "loss/actor",
                "loss/temp",
                "value/mean",
                "grad/encoder",
                "grad/actor",
                "grad/temp",
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
            "ent_coef_state": ent_coef_state,
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
            ent_coef_state = carry["ent_coef_state"]
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
                ent_coef_state,
                obs,
                action,
                reward,
                next_obs,
                done,
                discount_factor,
                n_quantile_target,
                value_target_max,
                value_target_min,
                key,
            )

            def update_actor():
                return cls._update_actor(
                    encoder_state,
                    dynamics_state,
                    actor_state,
                    qfn_state,
                    ent_coef_state,
                    target_entropy,
                    obs,
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

            actor_state, ent_coef_state, key, actor_info = jax.lax.cond(
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
                "ent_coef_state": ent_coef_state,
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
            # loss = (jnp.abs(next_zs - zsa)).mean()
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
    @partial(
        jax.jit,
        static_argnames=["discount_factor", "n_quantile_target"],
    )
    def _update_critic(
        encoder_state: RLTrainState,
        dynamics_state: RLTrainState,
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
        value_target_max: float,
        value_target_min: float,
        key: jax.Array,
    ) -> tuple[RLTrainState, jax.Array, dict[str, float]]:
        """Update critic."""
        key, noise_key = jax.random.split(key, 2)

        def critic_loss(params: flax.core.FrozenDict) -> tuple[jax.Array]:
            """Calculate Actor Loss."""
            next_action, (entropy, log_prob), _ = SALETQCAgent._rsample(
                encoder_state,
                encoder_state.target_fixed_params,
                actor_state,
                actor_state.target_params,
                next_obs,
                noise_key,
            )
            next_zs = encoder_state.apply_fn(
                encoder_state.target_fixed_params, next_obs
            )
            next_zsa = dynamics_state.apply_fn(
                dynamics_state.target_fixed_params, next_zs, next_action
            )
            # Batch, N_Critic, N_Quantile
            next_qvalue = qfn_state.apply_fn(
                qfn_state.target_params,
                next_obs,
                next_action,
                next_zsa,
                next_zs,
            )
            batch, n_critic, n_qunatile = next_qvalue.shape
            next_qvalue = jnp.sort(
                next_qvalue.reshape(batch, n_critic * n_qunatile), axis=-1
            )
            # Batch, N_Qunatile_Target
            ent_coef_value = ent_coef_state.apply_fn({"params": ent_coef_state.params})
            next_qvalue = next_qvalue[:, :n_quantile_target] + ent_coef_value * entropy
            next_qvalue = next_qvalue.clip(value_target_min, value_target_max)

            q_target = reward + discount_factor * next_qvalue * done
            zs = encoder_state.apply_fn(encoder_state.target_params, obs)
            zsa = dynamics_state.apply_fn(dynamics_state.target_params, zs, action)

            new_value_max = q_target.max()
            new_value_min = q_target.min()

            # Batch, N,Critic, N_Quantile
            qvalue = qfn_state.apply_fn(params, obs, action, zsa, zs)

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
            return loss, (new_value_max, new_value_min, priority_loss)

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
    @partial(jax.jit, static_argnames=["target_entropy"])
    def _update_actor(
        encoder_state: RLTrainState,
        dynamics_state: RLTrainState,
        actor_state: RLTrainState,
        qfn_state: RLTrainState,
        ent_coef_state: TrainState,
        target_entropy: float,
        obs: jax.Array,
        key: jax.Array,
    ) -> tuple[RLTrainState, TrainState, jax.Array, dict[str, float]]:
        """Update actor."""
        key, _ = jax.random.split(key, 2)

        def actor_loss(params: flax.core.FrozenDict) -> jax.Array:
            """Calculate Actor Loss."""
            batch = obs.shape[0]

            action, (entropy, log_prob), _ = SALETQCAgent._rsample(
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
            qvalue = qvalue.reshape(batch, -1).mean(axis=-1, keepdims=True)
            log_prob = log_prob.reshape(batch, -1)

            ent_coef_value = ent_coef_state.apply_fn({"params": ent_coef_state.params})
            actor_loss = -jnp.abs(qvalue).mean() + (ent_coef_value * log_prob).mean()

            return actor_loss, (entropy, log_prob)

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
        # Prepare Env.
        warnings.filterwarnings("ignore")
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
