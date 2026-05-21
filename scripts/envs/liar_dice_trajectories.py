"""Expert trajectory generator for Liar's Dice SFT training."""

import math
import random
import re

import requests

from envs.liar_dice_env import parse_game_state

_TIMEOUT = 2400
_SAMPLING_TEMPERATURE = 0.01

_SYSTEM_PROMPT = (
    "You are playing liars_dice.\n\n# Game Rules\nLIAR'S DICE RULES:\n\n"
    "Setup: Each player has N dice (1-5 depending on variant). All players roll their dice secretly.\n\n"
    "Goal: Make bids about total dice across ALL players, or call \"Liar\" on opponent's bid.\n\n"
    "Actions:\n- Bid (quantity, face): Claim there are at least 'quantity' dice showing 'face' among all dice.\n"
    "- Call Liar: Challenge the previous bid.\n\n"
    "Bidding rules: Each bid must be higher than the previous bid. \"Higher\" means:\n"
    "  - Same face value but higher quantity (e.g., \"2 fours\" beats \"1 four\")\n"
    "  - Same quantity but higher face value (e.g., \"2 fives\" beats \"2 fours\")\n\n"
    "Wild dice: 6s are WILD and count as ANY face value.\n"
    "- When counting dice for a bid, include 6s in the count\n"
    "- Example: Bid \"3 fours\" means at least 3 dice showing EITHER 4 OR 6\n\n"
    "Winning: If you call Liar and previous bid was false, opponent loses. "
    "If bid was true or exact, you lose.\n\n\n\n"
    "# Output Format\nYou must respond with ONLY the action ID (a single number).\n"
    "Do NOT include descriptions or explanations.\n\n"
    "Examples:\n- For action \"0 -> roll\": respond \"0\"\n"
    "- For action \"89 -> a3\": respond \"89\""
)


def _softmax_weights(probs: list[float], temperature: float) -> list[float]:
    """Convert raw probs to softmax weights."""
    if temperature <= 0:
        best = max(range(len(probs)), key=lambda i: probs[i])
        return [1.0 if i == best else 0.0 for i in range(len(probs))]
    scaled = [p / temperature for p in probs]
    m = max(scaled)
    exps = [math.exp(s - m) for s in scaled]
    total = sum(exps)
    return [e / total for e in exps]


def get_expert_action(messages: list[dict]) -> str:
    """Probability-based action selection."""
    try:
        gs = parse_game_state(messages)
    except Exception:
        last_user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        raw = re.findall(r"^(\d+)\s*->", last_user, re.MULTILINE)
        return min(raw, key=int) if raw else "0"

    if not gs.actions:
        return "0"

    probs = [a.prob for a in gs.actions]
    weights = _softmax_weights(probs, _SAMPLING_TEMPERATURE)
    chosen = random.choices(gs.actions, weights=weights, k=1)[0]
    return str(chosen.action_id)


def generate_expert_episode(
    game_id: int,
    env_endpoint: str,
    max_turn: int = 30,
) -> "list[dict] | None":
    """
    Run one Liar's Dice game against the env server using the expert policy.
    Returns the messages list (system/user/assistant) or None on failure.
    """
    reset_payload = {
        "task_id": game_id,
        "seed": game_id,
        "opponent": "mcts",
        "mcts_max_simulations": 225,
        "mcts_num_rollouts": 1,
    }
    try:
        res = requests.post(f"{env_endpoint}/reset", json=reset_payload, timeout=_TIMEOUT)
        res.raise_for_status()
        block = res.json()["result"]
        episode_id = block.get("episode_id", "")
        observation = block.get("observation", "")
    except Exception as exc:
        print(f"[liar_dice_trajectories] Reset failed (game {game_id}): {exc}")
        return None

    messages: list[dict] = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user",   "content": observation},
    ]

    for _ in range(max_turn):
        action = get_expert_action(messages)
        messages.append({"role": "assistant", "content": action})

        try:
            step_res = requests.post(
                f"{env_endpoint}/step",
                json={"action": action, "episode_id": episode_id},
                timeout=_TIMEOUT,
            )
            step_res.raise_for_status()
            step_block = step_res.json()["result"]
            observation = step_block.get("observation", "")
            done = step_block.get("done", False)
        except Exception as exc:
            print(f"[liar_dice_trajectories] Step failed (game {game_id}): {exc}")
            return None

        if done:
            break
        messages.append({"role": "user", "content": observation})
    else:
        print(f"[liar_dice_trajectories] max_turn={max_turn} reached (game {game_id})")

    return messages
