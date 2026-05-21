"""Lightweight registry mapping env names to their SFT trajectory generator.

Generators may return either ``list[dict]`` (messages only) or
``tuple[list[dict], float]`` (messages + final reward score).  The score is
used by generate_trajectories.py for optional score-based sampling.
"""

from typing import Callable

from envs.liar_dice_trajectories     import generate_expert_episode as _liar_gen
from envs.gin_rummy_trajectories     import generate_expert_episode as _gin_gen
from envs.leduc_poker_trajectories   import generate_random_episode as _leduc_gen

_SFT_REGISTRY: dict[str, Callable] = {
    "liars_dice":  _liar_gen,
    "gin_rummy":   _gin_gen,
    "leduc_poker": _leduc_gen,
}


def supports_sft(env_name: str) -> bool:
    return env_name in _SFT_REGISTRY


def get_sft_trajectory_generator(env_name: str) -> Callable:
    if env_name not in _SFT_REGISTRY:
        raise ValueError(f"No SFT trajectory generator for env: {env_name!r}")
    return _SFT_REGISTRY[env_name]
