"""Random trajectory generator for Leduc Poker SFT training.

Plays games with uniform-random actions against the MCTS opponent and returns
(messages, final_reward) so that generate_trajectories.py can apply
score-based sampling to bias toward games with positive outcomes.

Why random instead of expert: Leduc Poker's tiny state space lets MCTS play
near-optimally, so even a strong heuristic player can rarely win. Generating
random games and filtering/sampling by final reward is more practical than
trying to craft a hand-coded expert that can beat MCTS.
"""

import random
import re

import requests

from envs.leduc_poker_env import _BASE_SYSTEM_PROMPT, _format_observation

_TIMEOUT = 2400


def _random_action(obs: str) -> str:
    """Pick a uniformly random legal action from the observation text."""
    actions = re.findall(r"^(\d+)\s*->", obs, re.MULTILINE)
    return random.choice(actions) if actions else "1"


def generate_random_episode(
    game_id: int,
    env_endpoint: str,
    max_turn: int = 10,
) -> "tuple[list[dict], float] | None":
    """
    Run one Leduc Poker game using a random policy against the MCTS opponent.

    Returns ``(messages, final_reward)`` on success, or ``None`` on failure.
    ``final_reward`` is the raw env reward from the terminal step (positive =
    win, negative = loss).  generate_trajectories.py clamps this to [0, 1]
    before using it as a sampling probability.
    """
    reset_payload = {
        "task_id": game_id,
        "seed": game_id,
        "opponent": "mcts",
        "mcts_max_simulations": 50,
        "mcts_num_rollouts": 1,
    }
    try:
        res = requests.post(f"{env_endpoint}/reset", json=reset_payload, timeout=_TIMEOUT)
        res.raise_for_status()
        block       = res.json()["result"]
        episode_id  = block.get("episode_id", "")
        observation = _format_observation(block.get("observation", ""))
    except Exception as exc:
        print(f"[leduc_poker_trajectories] Reset failed (game {game_id}): {exc}")
        return None

    messages: list[dict] = [
        {"role": "system", "content": _BASE_SYSTEM_PROMPT},
        {"role": "user",   "content": observation},
    ]

    final_reward = 0.0

    for _ in range(max_turn):
        action = _random_action(observation)
        messages.append({"role": "assistant", "content": action})

        try:
            step_res = requests.post(
                f"{env_endpoint}/step",
                json={"action": action, "episode_id": episode_id},
                timeout=_TIMEOUT,
            )
            step_res.raise_for_status()
            step_block  = step_res.json()["result"]
            observation = _format_observation(step_block.get("observation", ""))
            done        = step_block.get("done", False)
            if done:
                final_reward = float(step_block.get("reward", 0.0))
        except Exception as exc:
            print(f"[leduc_poker_trajectories] Step failed (game {game_id}): {exc}")
            return None

        if done:
            break
        messages.append({"role": "user", "content": observation})
    else:
        print(f"[leduc_poker_trajectories] max_turn={max_turn} reached (game {game_id})")

    return messages, final_reward
