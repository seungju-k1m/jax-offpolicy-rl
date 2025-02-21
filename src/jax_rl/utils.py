from functools import partial
import logging
import os
import gymnasium as gym
import numpy as np
from gymnasium import spaces
import jax.numpy as jnp
import jax


def get_obs_action_dim(env_id: str) -> tuple[int, int]:
    """Return state and action dimension."""
    env = gym.make(env_id)
    return env.observation_space.shape[0], env.action_space.shape[0]


def get_action_bias_scale(
    action_space: spaces.Space | None = None, env_id: str | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Return action bias and scale."""
    if action_space is None:
        env = gym.make(env_id)
        action_space = env.action_space
    action_upper_bound: np.ndarray = action_space.high
    action_lower_bound: np.ndarray = action_space.low
    action_bias = (action_lower_bound + action_upper_bound) / 2.0
    action_scale = (action_upper_bound - action_lower_bound) / 2.0
    return action_bias, action_scale


@partial(jax.jit, static_argnames="clip_value")
def clip_gradient(gradients, clip_value):
    clipped_gradients = jax.tree_map(
        lambda g: jnp.clip(g, -clip_value, clip_value), gradients
    )
    return clipped_gradients


def setup_logger(path: str) -> logging.Logger:
    """Set up logger."""
    logger = logging.getLogger(path)
    logger.setLevel(logging.DEBUG)

    if logger.hasHandlers():
        logger.handlers.clear()

    # Create a console handler that only logs WARNING and above
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)  # Suppress INFO messages in console

    if os.path.isfile(path):
        os.remove(path)
    file_handler = logging.FileHandler(path)
    file_handler.setLevel(logging.INFO)
    # Add the file handler to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logger.propagate = False
    return logger


def calculate_mean_with_dirty_eles(eles: list[float | None]) -> float:
    """Get mean."""
    eles = [ele for ele in eles if not np.isnan(ele)]
    mean = sum(eles) / len(eles) if len(eles) != 0 else -1e6
    return mean


def log_train_infos(
    iteration: int,
    logger: logging.Logger,
    train_infos: list[dict],
    eval_info: dict,
    rollout_info: dict,
    start_logging: bool,
    **kwargs,
) -> None:
    """Logging."""
    train_keies = list(train_infos[0].keys())
    logging_info = {
        key: calculate_mean_with_dirty_eles(list(map(lambda x: x[key], train_infos)))
        for key in train_keies
    }
    if start_logging:
        logger.info(
            ",".join(
                ["timestep"]
                + sorted(
                    list(logging_info.keys())
                    + list(rollout_info.keys())
                    + list(eval_info.keys())
                )
            )
        )
    logging_info.update(rollout_info)
    logging_info.update(eval_info)
    logging_info = {key: value for key, value in sorted(logging_info.items())}
    stats_string = ",".join([f"{value:.4f}" for value in logging_info.values()])
    logger.info(f"{iteration},{stats_string}")
