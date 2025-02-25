from datetime import datetime
import os
from pathlib import Path
import random
import jax
import numpy as np
from jax_rl.replay_memory.lap import LAPReplayMemory
from jax_rl.utils import setup_logger
from td7 import TD7Agent, TD7Algorithm

from typing import Callable
import click


def args_for_td7(func: Callable) -> Callable:
    """Argument for TD7."""

    @click.option(
        "--env-id",
        type=str,
        required=True,
        help="Environment ID (e.g., 'HalfCheetah-v4')",
    )
    @click.option(
        "--total-timesteps",
        type=int,
        default=5_000_000,
        show_default=True,
        help="Total training timesteps",
    )
    @click.option("--seed", type=int, default=42, show_default=True, help="Random seed")
    @click.option(
        "--save-path",
        type=str,
        default="save/TD7/",
        show_default=True,
        help="Path to save checkpoints",
    )
    @click.option(
        "--discount-factor",
        type=float,
        default=0.99,
        show_default=True,
        help="Discount factor (gamma)",
    )
    @click.option(
        "--target-update-rate",
        type=int,
        default=250,
        show_default=True,
        help="Target network update frequency",
    )
    @click.option(
        "--policy-freq",
        type=int,
        default=2,
        show_default=True,
        help="Policy network update frequency",
    )
    @click.option(
        "--target-noise",
        type=float,
        default=0.2,
        show_default=True,
        help="Target policy noise",
    )
    @click.option(
        "--exploration-timesteps",
        type=int,
        default=25_000,
        show_default=True,
        help="Timesteps for exploration before training starts",
    )
    @click.option(
        "--max-episodes-per-ckpt",
        type=int,
        default=20,
        show_default=True,
        help="Max episodes per checkpoint update",
    )
    @click.option(
        "--init-episodes-per-ckpt",
        type=int,
        default=1,
        show_default=True,
        help="Initial episodes per checkpoint update",
    )
    @click.option(
        "--batch-size",
        type=int,
        default=256,
        show_default=True,
        help="Batch size for training",
    )
    @click.option(
        "--reset-weight",
        type=float,
        default=0.9,
        show_default=True,
        help="Weight for resetting best min return",
    )
    @click.option(
        "--update-timestep",
        type=int,
        default=int(75e4),
        show_default=True,
        help="Timesteps before increasing checkpoint episodes",
    )
    @click.option(
        "--eval-period",
        type=int,
        default=5_000,
        show_default=True,
        help="Evaluation frequency (in timesteps)",
    )
    @click.option("--use-progressbar", is_flag=True, help="Show training progress bar")
    @click.option("--deterministic", is_flag=True, help="Use deterministic evaluation")
    def wrapper(*args, **kwargs) -> Callable:
        """Wrapper."""
        return func(*args, **kwargs)

    return wrapper


def run_td7(
    env_id: str,
    total_timesteps: int = 5_000_000,
    seed: int = 42,
    save_path: str = "save/TD7/",
    *args,
    **kwargs,
):
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    agent = TD7Agent(env_id, seed=seed, *args, **kwargs)

    key = jax.random.PRNGKey(seed)
    agent.build(key)

    algorithm = TD7Algorithm(agent, *args, **kwargs)

    # SAVE_DIR
    save_dir = Path(save_path)
    timestamp = datetime.strftime(datetime.now(), "%Y-%m-%d-%H:%M:%S")
    save_dir = save_dir / str(seed) / f"{env_id}-{timestamp}"
    save_dir.mkdir(exist_ok=True, parents=True)

    train_logger = setup_logger(str(save_dir / "train.csv"))
    eval_logger = setup_logger(str(save_dir / "eval.csv"))
    rollout_logger = setup_logger(str(save_dir / "rollout.csv"))

    np.random.seed(seed=seed)
    random.seed(seed)

    algorithm.run(
        total_timesteps=total_timesteps,
        train_logger=train_logger,
        eval_logger=eval_logger,
        rollout_logger=rollout_logger,
        seed=seed,
        replay_memory_class=LAPReplayMemory,
        *args,
        **kwargs,
    )
