from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym

try:
    import rware  # noqa: F401
    from rware.warehouse import Action, Direction
except Exception as exc:  # pragma: no cover - environment dependent
    raise RuntimeError(
        "RWARE is not installed. Install with `pip install rware` before running this script."
    ) from exc


DATA_DIR = Path(__file__).resolve().parent
PAPER_DIR = DATA_DIR / "paper_artifacts"
Z95 = 1.959963984540054


LEFT_TURN = {
    Direction.UP: Direction.LEFT,
    Direction.LEFT: Direction.DOWN,
    Direction.DOWN: Direction.RIGHT,
    Direction.RIGHT: Direction.UP,
}
RIGHT_TURN = {v: k for k, v in LEFT_TURN.items()}
ALL_ACTION_VALUES = [action.value for action in Action]


@dataclass
class EpisodeResult:
    seed: int
    mode: str
    total_return: float
    success: bool
    first_reward_step: int | None
    steps: int
    attempts_used: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run RWARE with /app-aligned greedy/retry/search style control."
    )
    parser.add_argument("--env-id", type=str, default="rware-tiny-2ag-easy-v2")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--early-reward-horizon", type=int, default=60)
    parser.add_argument("--seed", type=int, default=20260410)
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["greedy", "retry", "search"],
        choices=["random", "greedy", "retry", "search"],
    )
    parser.add_argument("--retry-attempts", type=int, default=3)
    parser.add_argument("--search-samples", type=int, default=5)
    parser.add_argument("--explore-epsilon", type=float, default=0.20)
    parser.add_argument("--out-dir", type=Path, default=PAPER_DIR / "rware_app_modes")

    parser.add_argument(
        "--openrouter-model",
        type=str,
        default="",
        help="Optional OpenRouter model (e.g., google/gemini-2.5-flash) for one-shot parameter hint.",
    )
    parser.add_argument(
        "--openrouter-base-url",
        type=str,
        default="https://openrouter.ai/api/v1",
    )
    parser.add_argument(
        "--openrouter-api-key",
        type=str,
        default="",
        help="If empty, fallback to env OPENROUTER_API_KEY.",
    )
    parser.add_argument(
        "--disable-openrouter-hint",
        action="store_true",
        help="Disable one-shot OpenRouter hint even if model/key are provided.",
    )
    return parser.parse_args()


def wilson_ci(successes: int, total: int) -> tuple[float, float]:
    if total <= 0:
        return 0.0, 0.0
    p = successes / total
    zz = Z95 * Z95
    denom = 1.0 + zz / total
    center = (p + zz / (2.0 * total)) / denom
    half = Z95 * math.sqrt((p * (1.0 - p) + zz / (4.0 * total)) / total) / denom
    return max(0.0, center - half), min(1.0, center + half)


def _binom_cdf(k: int, n: int) -> float:
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    denom = 2.0**n
    num = 0
    for i in range(k + 1):
        num += math.comb(n, i)
    return num / denom


def mcnemar_exact_pvalue(improved: int, worse: int) -> float:
    n = improved + worse
    if n == 0:
        return 1.0
    lo = min(improved, worse)
    hi = max(improved, worse)
    left_tail = _binom_cdf(lo, n)
    right_tail = 1.0 - _binom_cdf(hi - 1, n)
    p = 2.0 * min(left_tail, right_tail)
    return min(1.0, p)


def choose_turn(current: Direction, desired: Direction) -> Action:
    if current == desired:
        return Action.FORWARD
    if LEFT_TURN[current] == desired:
        return Action.LEFT
    if RIGHT_TURN[current] == desired:
        return Action.RIGHT
    return Action.LEFT


def desired_dir(ax: int, ay: int, tx: int, ty: int) -> Direction:
    dx = tx - ax
    dy = ty - ay
    if abs(dx) >= abs(dy):
        return Direction.RIGHT if dx > 0 else Direction.LEFT
    return Direction.DOWN if dy > 0 else Direction.UP


def nearest_target(
    pos: tuple[int, int],
    targets: list[tuple[int, int]],
    rng: random.Random,
) -> tuple[int, int]:
    x, y = pos
    by_distance = sorted(((abs(t[0] - x) + abs(t[1] - y), t) for t in targets), key=lambda item: item[0])
    best_distance = by_distance[0][0]
    tied = [target for distance, target in by_distance if distance == best_distance]
    return rng.choice(tied)


def random_action(env: gym.Env, rng: random.Random) -> list[int]:
    return [int(rng.randrange(int(env.action_space[i].n))) for i in range(len(env.action_space))]


def guided_action(env: gym.Env, rng: random.Random, *, epsilon: float) -> list[int]:
    uw = env.unwrapped
    requested_ids = {shelf.id for shelf in uw.request_queue}
    requested_positions = [(int(shelf.x), int(shelf.y)) for shelf in uw.request_queue]
    goals = [(int(goal[0]), int(goal[1])) for goal in uw.goals]
    actions: list[int] = []

    for agent in uw.agents:
        if rng.random() < epsilon:
            actions.append(rng.choice(ALL_ACTION_VALUES))
            continue

        ax, ay = int(agent.x), int(agent.y)
        carrying = agent.carrying_shelf

        if carrying is None:
            target = nearest_target((ax, ay), requested_positions, rng)
            if (ax, ay) == target:
                actions.append(Action.TOGGLE_LOAD.value)
            else:
                d = desired_dir(ax, ay, target[0], target[1])
                actions.append(choose_turn(agent.dir, d).value)
            continue

        carrying_id = carrying.id
        if carrying_id in requested_ids:
            target = nearest_target((ax, ay), goals, rng)
            if (ax, ay) == target:
                actions.append(Action.NOOP.value)
            else:
                d = desired_dir(ax, ay, target[0], target[1])
                actions.append(choose_turn(agent.dir, d).value)
            continue

        if not uw._is_highway(ax, ay):
            actions.append(Action.TOGGLE_LOAD.value)
        else:
            target = nearest_target((ax, ay), requested_positions, rng)
            d = desired_dir(ax, ay, target[0], target[1])
            actions.append(choose_turn(agent.dir, d).value)

    return actions


def run_policy_attempt(
    env_id: str,
    *,
    seed: int,
    max_steps: int,
    policy_name: str,
    epsilon: float,
    rng_seed: int,
) -> EpisodeResult:
    env = gym.make(env_id)
    env.reset(seed=seed)

    rng = random.Random(rng_seed)
    total_return = 0.0
    first_reward_step: int | None = None
    steps = 0

    try:
        for t in range(max_steps):
            if policy_name == "random":
                action = random_action(env, rng)
            elif policy_name == "guided":
                action = guided_action(env, rng, epsilon=epsilon)
            else:
                raise ValueError(f"Unknown policy: {policy_name}")

            _, reward, terminated, truncated, _ = env.step(action)
            r = float(sum(reward)) if isinstance(reward, (list, tuple)) else float(reward)
            total_return += r
            steps = t + 1
            if r > 0.0 and first_reward_step is None:
                first_reward_step = steps
            if bool(terminated) or bool(truncated):
                break
    finally:
        env.close()

    return EpisodeResult(
        seed=seed,
        mode=policy_name,
        total_return=total_return,
        success=total_return > 0.0,
        first_reward_step=first_reward_step,
        steps=steps,
        attempts_used=1,
    )


def pick_best_attempt(rows: list[EpisodeResult]) -> EpisodeResult:
    def key_fn(row: EpisodeResult) -> tuple[int, float, float]:
        return (
            1 if row.success else 0,
            row.total_return,
            -(row.first_reward_step if row.first_reward_step is not None else 10**9),
        )

    return max(rows, key=key_fn)


def run_episode_mode(
    env_id: str,
    *,
    seed: int,
    max_steps: int,
    mode: str,
    epsilon: float,
    retry_attempts: int,
    search_samples: int,
    base_rng_seed: int,
) -> EpisodeResult:
    if mode == "random":
        row = run_policy_attempt(
            env_id,
            seed=seed,
            max_steps=max_steps,
            policy_name="random",
            epsilon=0.0,
            rng_seed=base_rng_seed,
        )
        row.mode = mode
        return row

    if mode == "greedy":
        row = run_policy_attempt(
            env_id,
            seed=seed,
            max_steps=max_steps,
            policy_name="guided",
            epsilon=0.0,
            rng_seed=base_rng_seed,
        )
        row.mode = mode
        return row

    if mode == "retry":
        attempts = max(1, retry_attempts)
        candidates: list[EpisodeResult] = []
        for i in range(attempts):
            row = run_policy_attempt(
                env_id,
                seed=seed,
                max_steps=max_steps,
                policy_name="guided",
                epsilon=epsilon,
                rng_seed=base_rng_seed + i * 1009,
            )
            candidates.append(row)
            if row.success:
                break
        best = pick_best_attempt(candidates)
        best.mode = mode
        best.attempts_used = len(candidates)
        return best

    if mode == "search":
        samples = max(1, search_samples)
        candidates: list[EpisodeResult] = []
        for i in range(samples):
            row = run_policy_attempt(
                env_id,
                seed=seed,
                max_steps=max_steps,
                policy_name="guided",
                epsilon=epsilon,
                rng_seed=base_rng_seed + i * 2003,
            )
            candidates.append(row)
        best = pick_best_attempt(candidates)
        best.mode = mode
        best.attempts_used = len(candidates)
        return best

    raise ValueError(f"Unknown mode: {mode}")


def summarize_runs(rows: list[EpisodeResult], *, early_reward_horizon: int) -> dict[str, Any]:
    n = len(rows)
    successes = sum(1 for row in rows if row.success)
    ci_low, ci_high = wilson_ci(successes, n)
    mean_return = sum(row.total_return for row in rows) / max(n, 1)
    mean_steps = sum(row.steps for row in rows) / max(n, 1)
    mean_attempts = sum(row.attempts_used for row in rows) / max(n, 1)
    first_reward_steps = [row.first_reward_step for row in rows if row.first_reward_step is not None]
    early_reward_count = sum(
        1
        for row in rows
        if row.first_reward_step is not None and int(row.first_reward_step) <= early_reward_horizon
    )
    early_low, early_high = wilson_ci(early_reward_count, n)
    mean_first_reward_step = (
        sum(int(step) for step in first_reward_steps) / len(first_reward_steps)
        if first_reward_steps
        else None
    )
    return {
        "episodes": n,
        "success_count": successes,
        "success_rate": successes / max(n, 1),
        "success_ci_low": ci_low,
        "success_ci_high": ci_high,
        "mean_team_return": mean_return,
        "mean_episode_steps": mean_steps,
        "mean_attempts_used": mean_attempts,
        "mean_first_reward_step": mean_first_reward_step,
        "early_reward_horizon": early_reward_horizon,
        "early_reward_count": early_reward_count,
        "early_reward_rate": early_reward_count / max(n, 1),
        "early_reward_ci_low": early_low,
        "early_reward_ci_high": early_high,
    }


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _extract_json_block(text: str) -> dict[str, Any] | None:
    text = text.strip()
    if not text:
        return None
    if text.startswith("{") and text.endswith("}"):
        try:
            obj = json.loads(text)
            return obj if isinstance(obj, dict) else None
        except Exception:
            return None
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            obj = json.loads(text[start : end + 1])
            return obj if isinstance(obj, dict) else None
        except Exception:
            return None
    return None


def maybe_openrouter_hint(args: argparse.Namespace) -> dict[str, Any]:
    result = {
        "used": False,
        "source": "disabled",
        "epsilon": args.explore_epsilon,
        "retry_attempts": args.retry_attempts,
        "search_samples": args.search_samples,
        "raw_response": "",
    }
    if args.disable_openrouter_hint:
        return result
    model = args.openrouter_model.strip()
    if not model:
        result["source"] = "no_model"
        return result

    api_key = args.openrouter_api_key.strip()
    if not api_key:
        import os

        api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        result["source"] = "no_api_key"
        return result

    try:
        from openai import OpenAI
    except Exception:
        result["source"] = "openai_sdk_missing"
        return result

    client = OpenAI(base_url=args.openrouter_base_url, api_key=api_key)
    system_prompt = (
        "You are tuning robust exploration parameters for a symbolic multi-agent controller. "
        "Return strict JSON only: {\"epsilon\": float, \"retry_attempts\": int, \"search_samples\": int}."
    )
    user_prompt = (
        f"Environment={args.env_id}, max_steps={args.max_steps}, episodes={args.episodes}. "
        "Target: maximize success_rate while keeping attempts small. "
        "Use epsilon in [0.05, 0.30], retry_attempts in [2, 4], search_samples in [3, 6]."
    )
    try:
        response = client.chat.completions.create(
            model=model,
            temperature=0.0,
            max_tokens=120,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        content = (response.choices[0].message.content or "").strip()
        result["raw_response"] = content
        parsed = _extract_json_block(content)
        if parsed is None:
            result["source"] = "parse_failed"
            return result

        epsilon = _safe_float(parsed.get("epsilon"), args.explore_epsilon)
        retry_attempts = _safe_int(parsed.get("retry_attempts"), args.retry_attempts)
        search_samples = _safe_int(parsed.get("search_samples"), args.search_samples)

        result["epsilon"] = min(0.30, max(0.05, epsilon))
        result["retry_attempts"] = min(4, max(2, retry_attempts))
        result["search_samples"] = min(6, max(3, search_samples))
        result["source"] = "openrouter"
        result["used"] = True
        return result
    except Exception as exc:  # pragma: no cover - network/API dependent
        result["source"] = f"openrouter_error:{type(exc).__name__}"
        return result


def build_markdown(report: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# RWARE /app-Mode Generalization")
    lines.append("")
    lines.append(f"- Environment: `{report['env_id']}`")
    lines.append(f"- Episodes per mode: {report['episodes']}")
    lines.append(f"- Max steps: {report['max_steps']}")
    lines.append(f"- Early-reward horizon: {report['early_reward_horizon']} steps")
    lines.append(
        "- Effective params: "
        f"epsilon={report['effective_params']['epsilon']:.3f}, "
        f"retry_attempts={report['effective_params']['retry_attempts']}, "
        f"search_samples={report['effective_params']['search_samples']} "
        f"(source={report['effective_params']['source']})"
    )
    lines.append("")
    lines.append("| Mode | Success | 95% CI | Early-Reward@H | Mean Return | Mean Attempts | Mean First Reward Step |")
    lines.append("| --- | ---: | --- | ---: | ---: | ---: | ---: |")
    for mode, summary in report["modes"].items():
        lines.append(
            "| {mode} | {succ:.3f} ({cnt}/{n}) | [{low:.3f}, {high:.3f}] | {early:.3f} ({ecnt}/{n}) | {ret:.3f} | {att:.2f} | {fr} |".format(
                mode=mode,
                succ=summary["success_rate"],
                cnt=summary["success_count"],
                n=summary["episodes"],
                low=summary["success_ci_low"],
                high=summary["success_ci_high"],
                early=summary["early_reward_rate"],
                ecnt=summary["early_reward_count"],
                ret=summary["mean_team_return"],
                att=summary["mean_attempts_used"],
                fr="-" if summary["mean_first_reward_step"] is None else f"{summary['mean_first_reward_step']:.2f}",
            )
        )
    lines.append("")

    for key, stat in report["paired_tests_vs_greedy"].items():
        lines.append(
            f"- Paired exact McNemar ({key} vs greedy): improved={stat['improved']}, "
            f"worse={stat['worse']}, n={stat['n']}, p={stat['p_value']:.6f}."
        )

    target_pass = report["target_check"]["pass_search_success_ge_0_35"]
    lines.append("")
    lines.append(
        f"- Target check (`search success_rate >= 0.35`): "
        f"{'PASS' if target_pass else 'FAIL'} "
        f"({report['target_check']['search_success_rate']:.3f})."
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    hint = maybe_openrouter_hint(args)
    epsilon = float(hint["epsilon"])
    retry_attempts = int(hint["retry_attempts"])
    search_samples = int(hint["search_samples"])

    per_mode_rows: dict[str, list[EpisodeResult]] = {mode: [] for mode in args.modes}
    mode_seed_offset = {
        "random": 17,
        "greedy": 31,
        "retry": 47,
        "search": 61,
    }

    for ep in range(args.episodes):
        episode_seed = args.seed + ep
        for mode in args.modes:
            row = run_episode_mode(
                args.env_id,
                seed=episode_seed,
                max_steps=args.max_steps,
                mode=mode,
                epsilon=epsilon,
                retry_attempts=retry_attempts,
                search_samples=search_samples,
                base_rng_seed=args.seed * 100_000 + ep * 997 + mode_seed_offset[mode],
            )
            row.mode = mode
            per_mode_rows[mode].append(row)

    mode_summaries = {
        mode: summarize_runs(rows, early_reward_horizon=args.early_reward_horizon)
        for mode, rows in per_mode_rows.items()
    }

    paired_tests_vs_greedy: dict[str, dict[str, Any]] = {}
    if "greedy" in per_mode_rows:
        greedy_rows = per_mode_rows["greedy"]
        for mode, rows in per_mode_rows.items():
            if mode == "greedy":
                continue
            improved = 0
            worse = 0
            same = 0
            for gre, cur in zip(greedy_rows, rows):
                if cur.success and not gre.success:
                    improved += 1
                elif gre.success and not cur.success:
                    worse += 1
                else:
                    same += 1
            paired_tests_vs_greedy[mode] = {
                "n": len(rows),
                "improved": improved,
                "worse": worse,
                "same": same,
                "p_value": mcnemar_exact_pvalue(improved, worse),
            }

    search_success_rate = mode_summaries.get("search", {}).get("success_rate", 0.0)
    report = {
        "env_id": args.env_id,
        "episodes": args.episodes,
        "max_steps": args.max_steps,
        "early_reward_horizon": args.early_reward_horizon,
        "seed": args.seed,
        "effective_params": {
            "epsilon": epsilon,
            "retry_attempts": retry_attempts,
            "search_samples": search_samples,
            "source": hint["source"],
            "openrouter_used": hint["used"],
            "openrouter_raw_response": hint["raw_response"],
        },
        "modes": mode_summaries,
        "paired_tests_vs_greedy": paired_tests_vs_greedy,
        "target_check": {
            "search_success_rate": search_success_rate,
            "pass_search_success_ge_0_35": bool(search_success_rate >= 0.35),
        },
        "per_episode": {
            mode: [
                {
                    "seed": row.seed,
                    "mode": row.mode,
                    "success": row.success,
                    "total_return": row.total_return,
                    "first_reward_step": row.first_reward_step,
                    "steps": row.steps,
                    "attempts_used": row.attempts_used,
                }
                for row in rows
            ]
            for mode, rows in per_mode_rows.items()
        },
    }

    out_json = args.out_dir / "rware_app_modes_report.json"
    out_md = args.out_dir / "rware_app_modes_report.md"

    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    out_md.write_text(build_markdown(report), encoding="utf-8")

    print(f"[OK] Wrote {out_json}")
    print(f"[OK] Wrote {out_md}")
    print(
        "[CHECK] search_success_rate={:.3f}, target>=0.35: {}".format(
            report["target_check"]["search_success_rate"],
            "PASS" if report["target_check"]["pass_search_success_ge_0_35"] else "FAIL",
        )
    )


if __name__ == "__main__":
    main()
