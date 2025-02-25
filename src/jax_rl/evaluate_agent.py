import jax
from gymnasium.wrappers import RecordEpisodeStatistics


def evaluate_agent(
    agent,
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
