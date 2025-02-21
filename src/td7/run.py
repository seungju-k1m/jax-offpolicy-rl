from datetime import datetime
import os
from pathlib import Path
import random
import jax
import numpy as np
from jax_rl.replay_memory.lap import LAPReplayMemory
from jax_rl.utils import setup_logger
from td7 import TD7Agent, TD7Algorithm


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
