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


@dataclass
class EpisodeResult:
    seed: int
    total_return: float
    success: bool
    first_reward_step: int | None
    steps: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RWARE standard-benchmark transfer pilot.")
    parser.add_argument("--env-id", type=str, default="rware-tiny-2ag-easy-v2", help="Gymnasium RWARE environment id.")
    parser.add_argument("--episodes", type=int, default=200, help="Number of episodes per policy.")
    parser.add_argument("--max-steps", type=int, default=300, help="Max steps per episode.")
    parser.add_argument(
        "--early-reward-horizon",
        type=int,
        default=60,
        help="Step horizon for early-reward rate metric.",
    )
    parser.add_argument("--seed", type=int, default=20260328, help="Base seed.")
    parser.add_argument("--out-dir", type=Path, default=PAPER_DIR, help="Output directory.")
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


def nearest_target(pos: tuple[int, int], targets: list[tuple[int, int]]) -> tuple[int, int]:
    x, y = pos
    return min(targets, key=lambda t: abs(t[0] - x) + abs(t[1] - y))


def verifier_heuristic_action(env: gym.Env) -> list[int]:
    uw = env.unwrapped
    requested_ids = {shelf.id for shelf in uw.request_queue}
    requested_positions = [(int(shelf.x), int(shelf.y)) for shelf in uw.request_queue]
    goals = [(int(goal[0]), int(goal[1])) for goal in uw.goals]
    actions: list[int] = []

    for agent in uw.agents:
        ax, ay = int(agent.x), int(agent.y)
        carrying = agent.carrying_shelf

        if carrying is None:
            target = nearest_target((ax, ay), requested_positions)
            if (ax, ay) == target:
                actions.append(Action.TOGGLE_LOAD.value)
            else:
                d = desired_dir(ax, ay, target[0], target[1])
                actions.append(choose_turn(agent.dir, d).value)
            continue

        carrying_id = carrying.id
        if carrying_id in requested_ids:
            target = nearest_target((ax, ay), goals)
            if (ax, ay) == target:
                # Delivery happens on goal occupancy; staying avoids accidental detours.
                actions.append(Action.NOOP.value)
            else:
                d = desired_dir(ax, ay, target[0], target[1])
                actions.append(choose_turn(agent.dir, d).value)
            continue

        # Drop non-request shelf once we are in a legal non-highway tile.
        if not uw._is_highway(ax, ay):
            actions.append(Action.TOGGLE_LOAD.value)
        else:
            target = nearest_target((ax, ay), requested_positions)
            d = desired_dir(ax, ay, target[0], target[1])
            actions.append(choose_turn(agent.dir, d).value)

    return actions


def random_action(env: gym.Env, rng: random.Random) -> list[int]:
    return [int(rng.randrange(int(env.action_space[i].n))) for i in range(len(env.action_space))]


def run_policy_episode(
    env_id: str,
    *,
    seed: int,
    max_steps: int,
    policy_name: str,
    rng: random.Random,
) -> EpisodeResult:
    env = gym.make(env_id)
    env.reset(seed=seed)
    total_return = 0.0
    first_reward_step: int | None = None
    steps = 0

    try:
        for t in range(max_steps):
            if policy_name == "random":
                action = random_action(env, rng)
            elif policy_name == "verifier_heuristic":
                action = verifier_heuristic_action(env)
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
        total_return=total_return,
        success=total_return > 0.0,
        first_reward_step=first_reward_step,
        steps=steps,
    )


def summarize_runs(rows: list[EpisodeResult], *, early_reward_horizon: int) -> dict[str, Any]:
    n = len(rows)
    successes = sum(1 for row in rows if row.success)
    ci_low, ci_high = wilson_ci(successes, n)
    mean_return = sum(row.total_return for row in rows) / max(n, 1)
    mean_steps = sum(row.steps for row in rows) / max(n, 1)
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
        "mean_first_reward_step": mean_first_reward_step,
        "early_reward_horizon": early_reward_horizon,
        "early_reward_count": early_reward_count,
        "early_reward_rate": early_reward_count / max(n, 1),
        "early_reward_ci_low": early_low,
        "early_reward_ci_high": early_high,
        "per_episode": [
            {
                "seed": row.seed,
                "total_return": row.total_return,
                "success": row.success,
                "first_reward_step": row.first_reward_step,
                "steps": row.steps,
            }
            for row in rows
        ],
    }


def build_markdown(report: dict[str, Any]) -> str:
    random_summary = report["policies"]["random"]
    heuristic_summary = report["policies"]["verifier_heuristic"]
    paired_success = report["paired_success_test"]
    paired_early = report["paired_early_reward_test"]
    lines: list[str] = []
    lines.append("# RWARE Generalization Pilot")
    lines.append("")
    lines.append(f"- Environment: `{report['env_id']}`")
    lines.append(f"- Episodes per policy: {report['episodes']}")
    lines.append(f"- Max steps: {report['max_steps']}")
    lines.append(f"- Early-reward horizon: {report['early_reward_horizon']} steps")
    lines.append("")
    lines.append("| Policy | Success | 95% CI | Early-Reward@H | Early-Reward 95% CI | Mean Team Return | Mean Steps | Mean First Reward Step |")
    lines.append("| --- | ---: | --- | ---: | --- | ---: | ---: | ---: |")
    lines.append(
        "| random | {succ:.3f} ({cnt}/{n}) | [{low:.3f}, {high:.3f}] | {ers:.3f} ({erc}/{n}) | [{erl:.3f}, {erh:.3f}] | {ret:.3f} | {steps:.2f} | {fr} |".format(
            succ=random_summary["success_rate"],
            cnt=random_summary["success_count"],
            n=random_summary["episodes"],
            low=random_summary["success_ci_low"],
            high=random_summary["success_ci_high"],
            ers=random_summary["early_reward_rate"],
            erc=random_summary["early_reward_count"],
            erl=random_summary["early_reward_ci_low"],
            erh=random_summary["early_reward_ci_high"],
            ret=random_summary["mean_team_return"],
            steps=random_summary["mean_episode_steps"],
            fr="-" if random_summary["mean_first_reward_step"] is None else f"{random_summary['mean_first_reward_step']:.2f}",
        )
    )
    lines.append(
        "| heuristic | {succ:.3f} ({cnt}/{n}) | [{low:.3f}, {high:.3f}] | {ers:.3f} ({erc}/{n}) | [{erl:.3f}, {erh:.3f}] | {ret:.3f} | {steps:.2f} | {fr} |".format(
            succ=heuristic_summary["success_rate"],
            cnt=heuristic_summary["success_count"],
            n=heuristic_summary["episodes"],
            low=heuristic_summary["success_ci_low"],
            high=heuristic_summary["success_ci_high"],
            ers=heuristic_summary["early_reward_rate"],
            erc=heuristic_summary["early_reward_count"],
            erl=heuristic_summary["early_reward_ci_low"],
            erh=heuristic_summary["early_reward_ci_high"],
            ret=heuristic_summary["mean_team_return"],
            steps=heuristic_summary["mean_episode_steps"],
            fr="-" if heuristic_summary["mean_first_reward_step"] is None else f"{heuristic_summary['mean_first_reward_step']:.2f}",
        )
    )
    lines.append("")
    lines.append(
        "- Paired exact McNemar (verifier_heuristic vs random): "
        f"success improved={paired_success['improved']}, worse={paired_success['worse']}, n={paired_success['n']}, p={paired_success['p_value']:.6f}; "
        f"early-reward improved={paired_early['improved']}, worse={paired_early['worse']}, n={paired_early['n']}, p={paired_early['p_value']:.6f}."
    )
    return "\n".join(lines) + "\n"


def build_latex_table(report: dict[str, Any]) -> str:
    random_summary = report["policies"]["random"]
    heuristic_summary = report["policies"]["verifier_heuristic"]
    paired_success = report["paired_success_test"]
    paired_early = report["paired_early_reward_test"]
    lines: list[str] = []
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append(
        "\\caption{Standard-benchmark transfer pilot on RWARE (%s).}" % report["env_id"].replace("_", "\\_")
    )
    lines.append("\\label{tab:rware-generalization}")
    lines.append("\\small")
    lines.append("\\begin{tabular}{lcccc}")
    lines.append("\\toprule")
    lines.append("Policy & Success & Early-Reward@H & Mean return & Mean first reward step \\\\")
    lines.append("\\midrule")
    lines.append(
        "random & {succ:.3f} ({cnt}/{n}) & {ers:.3f} ({erc}/{n}) & {ret:.3f} & {fr} \\\\".format(
            succ=random_summary["success_rate"],
            cnt=random_summary["success_count"],
            n=random_summary["episodes"],
            ers=random_summary["early_reward_rate"],
            erc=random_summary["early_reward_count"],
            ret=random_summary["mean_team_return"],
            fr="-" if random_summary["mean_first_reward_step"] is None else f"{random_summary['mean_first_reward_step']:.2f}",
        )
    )
    lines.append(
        "heuristic & {succ:.3f} ({cnt}/{n}) & {ers:.3f} ({erc}/{n}) & {ret:.3f} & {fr} \\\\".format(
            succ=heuristic_summary["success_rate"],
            cnt=heuristic_summary["success_count"],
            n=heuristic_summary["episodes"],
            ers=heuristic_summary["early_reward_rate"],
            erc=heuristic_summary["early_reward_count"],
            ret=heuristic_summary["mean_team_return"],
            fr="-" if heuristic_summary["mean_first_reward_step"] is None else f"{heuristic_summary['mean_first_reward_step']:.2f}",
        )
    )
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\vspace{1mm}")
    lines.append(
        "\\\\{\\footnotesize Paired exact McNemar (heuristic vs random): "
        f"success improved={paired_success['improved']}, worse={paired_success['worse']}, $p={paired_success['p_value']:.6f}$; "
        f"early-reward improved={paired_early['improved']}, worse={paired_early['worse']}, $p={paired_early['p_value']:.6f}$.}}"
    )
    lines.append("\\end{table}")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    policy_names = ("random", "verifier_heuristic")
    policy_runs: dict[str, list[EpisodeResult]] = {name: [] for name in policy_names}

    for ep in range(args.episodes):
        run_seed = args.seed + ep
        for policy_name in policy_names:
            policy_runs[policy_name].append(
                run_policy_episode(
                    args.env_id,
                    seed=run_seed,
                    max_steps=args.max_steps,
                    policy_name=policy_name,
                    rng=rng,
                )
            )

    summary_random = summarize_runs(policy_runs["random"], early_reward_horizon=args.early_reward_horizon)
    summary_heuristic = summarize_runs(policy_runs["verifier_heuristic"], early_reward_horizon=args.early_reward_horizon)

    improved = 0
    worse = 0
    same = 0
    for random_row, heuristic_row in zip(policy_runs["random"], policy_runs["verifier_heuristic"]):
        if heuristic_row.success and not random_row.success:
            improved += 1
        elif random_row.success and not heuristic_row.success:
            worse += 1
        else:
            same += 1

    improved_early = 0
    worse_early = 0
    same_early = 0
    for random_row, heuristic_row in zip(policy_runs["random"], policy_runs["verifier_heuristic"]):
        random_early = (
            random_row.first_reward_step is not None
            and int(random_row.first_reward_step) <= args.early_reward_horizon
        )
        heuristic_early = (
            heuristic_row.first_reward_step is not None
            and int(heuristic_row.first_reward_step) <= args.early_reward_horizon
        )
        if heuristic_early and not random_early:
            improved_early += 1
        elif random_early and not heuristic_early:
            worse_early += 1
        else:
            same_early += 1

    report = {
        "env_id": args.env_id,
        "episodes": args.episodes,
        "max_steps": args.max_steps,
        "early_reward_horizon": args.early_reward_horizon,
        "seed": args.seed,
        "policies": {
            "random": summary_random,
            "verifier_heuristic": summary_heuristic,
        },
        "paired_success_test": {
            "n": args.episodes,
            "improved": improved,
            "worse": worse,
            "same": same,
            "p_value": mcnemar_exact_pvalue(improved, worse),
        },
        "paired_early_reward_test": {
            "n": args.episodes,
            "improved": improved_early,
            "worse": worse_early,
            "same": same_early,
            "p_value": mcnemar_exact_pvalue(improved_early, worse_early),
        },
    }

    out_json = out_dir / "rware_generalization_report.json"
    out_md = out_dir / "rware_generalization_report.md"
    out_tex = out_dir / "rware_generalization_table.tex"

    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    out_md.write_text(build_markdown(report), encoding="utf-8")
    out_tex.write_text(build_latex_table(report), encoding="utf-8")

    print(f"[OK] Wrote {out_json}")
    print(f"[OK] Wrote {out_md}")
    print(f"[OK] Wrote {out_tex}")


if __name__ == "__main__":
    main()
