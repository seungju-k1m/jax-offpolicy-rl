from copy import deepcopy
from datetime import datetime
import os
from pathlib import Path
import random
from typing import Callable
import click
import jax
import numpy as np
import yaml
from simba.agent import SimbaAgent
from simba.algorithm import SimbaAlgorithm
from jax_rl.replay_memory import SimpleReplayMemory
from jax_rl.utils import setup_logger


def args_for_simba(func: Callable) -> Callable:
    """Argument for SIMBA."""

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
        default="./save/SIMBA/",
        show_default=True,
        help="Path to save checkpoints",
    )
    @click.option(
        "--output-name",
        type=str,
        default=None,
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
        "--policy-freq",
        type=int,
        default=1,
        show_default=True,
        help="Policy network update frequency",
    )
    @click.option(
        "--ent-coef",
        type=float,
        default=0.1,
        show_default=True,
        help="Entropy Coeffieicnt for Actor.",
    )
    @click.option(
        "--exploration-timesteps",
        type=int,
        default=25_000,
        show_default=True,
        help="Timesteps for exploration before training starts",
    )
    @click.option(
        "--batch-size",
        type=int,
        default=256,
        show_default=True,
        help="Batch size for training",
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
    @click.option(
        "--n-critics",
        type=click.INT,
        default=2,
        help="# of Critics",
        show_default=True,
    )
    @click.option(
        "--n-quantile",
        type=click.INT,
        default=25,
        help="# of Quantile.",
        show_default=True,
    )
    @click.option(
        "--n-quantile-drop",
        type=click.INT,
        default=2,
        help="# of Quantile Drop.",
        show_default=True,
    )
    def wrapper(*args, **kwargs) -> Callable:
        """Wrapper."""
        return func(*args, **kwargs)

    return wrapper


def run_simba(
    env_id: str,
    seed: int,
    use_progressbar: bool = False,
    n_critics: int = 2,
    n_quantile: int = 25,
    n_quantile_drop: int = 5,
    total_timesteps: int = 3_000_000,
    ent_coef: float = 0.1,
    save_path: str = "./save/SIMBA",
    output_name: str | None = None,
    **kwargs,
) -> None:
    params = deepcopy(locals())
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    agent = SimbaAgent(
        env_id,
        seed=seed,
        n_critics=n_critics,
        n_quantile=n_quantile,
        n_quantile_drop=n_quantile_drop,
        **kwargs,
    )
    key = jax.random.PRNGKey(seed)
    agent.build(key)
    algorithm = SimbaAlgorithm(
        agent,
        ent_coef=ent_coef,
        **kwargs,
    )

    # SAVE_DIR
    save_path = Path(save_path)
    timestamp = datetime.strftime(datetime.now(), "%Y-%m-%d-%H:%M:%S")
    output_name = output_name or env_id
    output_name = f"{output_name}-{timestamp}"
    save_path = save_path / str(seed) / output_name
    save_path.mkdir(exist_ok=True, parents=True)

    train_logger = setup_logger(str(save_path / "train.csv"))
    eval_logger = setup_logger(str(save_path / "eval.csv"))
    rollout_logger = setup_logger(str(save_path / "rollout.csv"))

    np.random.seed(seed=seed)
    random.seed(seed)
    with open(save_path / "config.yaml", "w") as file_handler:
        yaml.dump(params, file_handler)
    algorithm.run(
        total_timesteps=total_timesteps,
        train_logger=train_logger,
        eval_logger=eval_logger,
        rollout_logger=rollout_logger,
        seed=seed,
        use_progressbar=use_progressbar,
        eval_period=5000,
        replay_memory_class=SimpleReplayMemory,
    )
