"""
Generate game trajectories against env servers and save as an HF DatasetDict
(train / validation splits) ready for train_sft_env.py.

Analogous to tokenize_instruct.py but for environment SFT tasks.

Run from /workspace/scripts/:
  python -m envs.generate_trajectories --environment_name liars_dice \
      --output_path /path/to/dataset --num_games 50000

  python -m envs.generate_trajectories --environment_name gin_rummy \
      --output_path /path/to/dataset --num_games 5000 --max_turn 200

  python -m envs.generate_trajectories --environment_name leduc_poker \
      --output_path /path/to/dataset --num_games 200000 --max_turn 10 \
      --sample-by-score --score-power 2

Score-based sampling:
  Some generators (e.g. leduc_poker) return (messages, score) tuples.  When
  --sample-by-score is set, each game is kept with probability
  clamp(score, 0, 1) ** score_power.  --wins-only is a stricter filter that
  discards any game where score <= 0.  For generators that return only
  messages (no score), all games are kept regardless of these flags.
"""

import argparse
import random
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed

from datasets import Dataset, DatasetDict

from envs.shared_env import GAMES_TO_TASK_ID_RANGE, init_env_pool
from envs.sft_env_configs import get_sft_trajectory_generator


# ── Process-pool worker ───────────────────────────────────────────────────────
# Each worker process loads the expert generator once via _worker_init, then
# handles multiple games sequentially. Using processes (not threads) gives each
# worker its own GIL so CPU-bound expert computation runs truly in parallel
# without contention. --num_workers controls how many concurrent env server
# connections are open, letting you tune without overloading either side.

_GENERATE_FN = None


def _worker_init(env_name: str) -> None:
    global _GENERATE_FN
    _GENERATE_FN = get_sft_trajectory_generator(env_name)


def _worker_play(
    game_id: int, endpoint: str, max_turn: int
) -> "list[dict] | tuple[list[dict], float] | None":
    return _GENERATE_FN(game_id, endpoint, max_turn)

# ─────────────────────────────────────────────────────────────────────────────

MIN_ASSISTANT_TURNS = 1


def _sliding_windows(conv: list[dict], window_turns: int, window_step: int) -> list[list[dict]]:
    """
    Split a conversation into overlapping sub-conversations.
    Each window: [system] + window_turns × (user, assistant) pairs.
    Short games (fewer than window_turns pairs) are kept as one window.
    """
    system = [m for m in conv if m["role"] == "system"]
    turns  = [m for m in conv if m["role"] != "system"]

    pairs = []
    i = 0
    while i + 1 < len(turns):
        if turns[i]["role"] == "user" and turns[i + 1]["role"] == "assistant":
            pairs.append((turns[i], turns[i + 1]))
            i += 2
        else:
            i += 1

    if not pairs:
        return []

    windows = []
    for start in range(0, len(pairs), window_step):
        chunk = pairs[start : start + window_turns]
        if not chunk:
            break
        window_conv = system[:]
        for user_msg, asst_msg in chunk:
            window_conv.extend([user_msg, asst_msg])
        windows.append(window_conv)

    return windows


def _clean(messages: "list[dict] | None") -> "list[dict] | None":
    if not messages:
        return None
    messages = [{"role": m["role"], "content": str(m["content"])} for m in messages]
    while messages and messages[-1]["role"] != "assistant":
        messages.pop()
    if not messages:
        return None
    if sum(1 for m in messages if m["role"] == "assistant") < MIN_ASSISTANT_TURNS:
        return None
    return messages


def _stats(conversations: list[list[dict]]) -> dict:
    turn_counts = [sum(1 for m in c if m["role"] == "assistant") for c in conversations]
    return {
        "total": len(conversations),
        "avg_assistant_turns": round(sum(turn_counts) / len(turn_counts), 2),
        "turn_distribution": dict(sorted(Counter(turn_counts).items())),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--environment_name", required=True)
    p.add_argument("--output_path",      required=True)
    p.add_argument("--num_games",   type=int, default=50000)
    p.add_argument("--max_turn",    type=int, default=30)
    p.add_argument("--window_turns", type=int, default=10,
                   help="Split each game into sub-conversations of this many (user,assistant) "
                        "pairs. Games shorter than this are kept whole. Default 10.")
    p.add_argument("--window_step", type=int, default=0,
                   help="Slide window by this many pairs (default: window_turns // 2).")
    p.add_argument("--num_workers", type=int, default=0,
                   help="Number of worker processes. Default 0 = num_servers. "
                        "Each process holds one concurrent env server connection; "
                        "raise to increase throughput, lower to reduce env server load.")
    p.add_argument("--seed", type=int, default=42)
    # Score-based sampling (for generators that return (messages, score) tuples)
    p.add_argument("--wins-only", action="store_true",
                   help="Discard games where score <= 0. Only applies when the "
                        "generator returns a (messages, score) tuple.")
    p.add_argument("--sample-by-score", action="store_true",
                   help="Keep each game with probability clamp(score, 0, 1) ** score-power. "
                        "Only applies when the generator returns a (messages, score) tuple.")
    p.add_argument("--score-power", type=float, default=1.0,
                   help="Exponent applied to the clamped score when sampling (default: 1.0). "
                        "Higher values bias more strongly toward high-scoring games.")
    args = p.parse_args()
    if args.window_step == 0:
        args.window_step = args.window_turns // 2 or 1

    task_id_min, task_id_max = GAMES_TO_TASK_ID_RANGE[args.environment_name]

    reset_payload = {
        "task_id": task_id_min,
        "seed": 42,
        "opponent": "mcts",
        "mcts_max_simulations": 225,
        "mcts_num_rollouts": 1,
    }
    _, env_pool, num_servers, _, _ = init_env_pool(reset_payload)

    num_workers = args.num_workers or max(1, num_servers)

    print(f"Environment  : {args.environment_name}")
    print(f"Output       : {args.output_path}")
    print(f"Num games    : {args.num_games}")
    print(f"Window turns : {args.window_turns}  step {args.window_step}")
    print(f"Env servers  : {num_servers}   Workers: {num_workers}")

    random.seed(args.seed)
    game_ids = random.sample(range(task_id_min + 1, task_id_max), args.num_games)
    tasks = [
        (gid, env_pool[i % num_servers]["base_url"], args.max_turn)
        for i, gid in enumerate(game_ids)
    ]

    use_score_filter = args.wins_only or args.sample_by_score
    print(f"Playing {args.num_games} games...")
    if use_score_filter:
        print(f"Score filter: wins_only={args.wins_only}  sample_by_score={args.sample_by_score}"
              f"  score_power={args.score_power}")
    conversations: list[list[dict]] = []
    skipped = 0
    score_filtered = 0
    all_scores: list[float] = []
    with ProcessPoolExecutor(
        max_workers=num_workers,
        initializer=_worker_init,
        initargs=(args.environment_name,),
    ) as pool:
        futures = {pool.submit(_worker_play, gid, ep, mt): gid for gid, ep, mt in tasks}
        completed = 0
        for future in as_completed(futures):
            result = future.result()

            # Unpack score when the generator returns (messages, score)
            if isinstance(result, tuple):
                raw_messages, score = result
                all_scores.append(score)
            else:
                raw_messages, score = result, None

            # Apply score-based filters only when a score is available
            if score is not None and use_score_filter:
                if args.wins_only and score <= 0:
                    score_filtered += 1
                    completed += 1
                    continue
                if args.sample_by_score:
                    prob = max(0.0, min(1.0, score)) ** args.score_power
                    if random.random() >= prob:
                        score_filtered += 1
                        completed += 1
                        continue

            cleaned = _clean(raw_messages)
            if cleaned is None:
                skipped += 1
            else:
                conversations.append(cleaned)
            completed += 1
            if completed % 100 == 0:
                print(f"  {completed}/{args.num_games} games done", flush=True)

    score_summary = ""
    if all_scores:
        wins = sum(1 for s in all_scores if s > 0)
        score_summary = (
            f"   Score stats: min={min(all_scores):.3f}  max={max(all_scores):.3f}"
            f"  wins(>0)={wins}/{len(all_scores)} ({100*wins/len(all_scores):.1f}%)\n"
            f"   Score-filtered: {score_filtered}"
        )
    print(f"Valid : {len(conversations)}   Skipped : {skipped}{chr(10) + score_summary if score_summary else ''}")

    if not conversations:
        raise RuntimeError("No valid conversations generated. Check ENVIRONMENT_SERVER_URLS.")

    # Raw game length stats — helps diagnose max_turn being hit or unexpectedly long games.
    raw_lengths = [sum(1 for m in c if m["role"] == "assistant") for c in conversations]
    max_turn_hits = sum(1 for l in raw_lengths if l >= args.max_turn)
    length_buckets = Counter(l // 10 * 10 for l in raw_lengths)
    print(f"\n  ── Raw game stats (before windowing) ──────────────────")
    print(f"  Avg turns/game   : {sum(raw_lengths)/len(raw_lengths):.1f}")
    print(f"  Min/Max turns    : {min(raw_lengths)} / {max(raw_lengths)}")
    print(f"  Hit max_turn={args.max_turn}  : {max_turn_hits} / {len(conversations)} games"
          f"  ({100*max_turn_hits/len(conversations):.1f}%)")
    print(f"  Length buckets   : " +
          "  ".join(f"{k}-{k+9}:{v}" for k, v in sorted(length_buckets.items())))

    # Apply sliding window — expands long games into overlapping sub-conversations.
    # Short games (< window_turns pairs) are kept whole as a single window.
    windowed: list[list[dict]] = []
    for conv in conversations:
        windows = _sliding_windows(conv, args.window_turns, args.window_step)
        windowed.extend(windows if windows else [conv])
    conversations = windowed
    print(f"\n  ── After windowing (turns={args.window_turns} step={args.window_step}) ──")
    print(f"  Total examples   : {len(conversations)}")

    for k, v in _stats(conversations).items():
        print(f"  {k}: {v}")

    dataset = Dataset.from_list([{"messages": c} for c in conversations])
    dd = DatasetDict({"train": dataset})
    print(f"Train: {len(dd['train'])}")

    dd.save_to_disk(args.output_path)
    print(f"Dataset saved → {args.output_path}")


if __name__ == "__main__":
    main()
