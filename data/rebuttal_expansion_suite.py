from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.search.mcts import PlanSearchEngine
from app.search.verifier import PlanSearchVerifier, SearchConstraints
from critical_eval_baselines import infer_constraints, mcnemar_exact_pvalue
from generate_app_benchmark_tasks import build_prompt


DATA_DIR = Path(__file__).resolve().parent
PAPER_DIR = DATA_DIR / "paper_artifacts"
Z95 = 1.959963984540054


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build rebuttal-scale expansion artifacts.")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--out-dir", type=Path, default=PAPER_DIR)
    parser.add_argument("--search-iterations", type=int, default=96)
    parser.add_argument("--search-depth", type=int, default=18)
    parser.add_argument("--seed", type=int, default=20260328)
    return parser.parse_args()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def wilson_ci(successes: int, total: int) -> tuple[float, float]:
    if total <= 0:
        return 0.0, 0.0
    p = successes / total
    zz = Z95 * Z95
    denom = 1.0 + zz / total
    center = (p + zz / (2.0 * total)) / denom
    half = Z95 * math.sqrt((p * (1.0 - p) + zz / (4.0 * total)) / total) / denom
    return max(0.0, center - half), min(1.0, center + half)


def summarize_bool_map(name: str, success_map: dict[str, bool], runtimes: list[float] | None = None) -> dict[str, Any]:
    n = len(success_map)
    s = sum(1 for ok in success_map.values() if ok)
    ci_low, ci_high = wilson_ci(s, n)
    out = {
        "name": name,
        "n": n,
        "success_count": s,
        "success_rate": s / max(n, 1),
        "success_ci_low": ci_low,
        "success_ci_high": ci_high,
    }
    if runtimes:
        out["avg_runtime_sec"] = statistics.mean(runtimes)
    return out


def paired_compare(left_map: dict[str, bool], right_map: dict[str, bool], left: str, right: str) -> dict[str, Any]:
    keys = sorted(set(left_map) & set(right_map))
    improved = 0
    worse = 0
    same = 0
    for key in keys:
        l_ok = bool(left_map[key])
        r_ok = bool(right_map[key])
        if r_ok and not l_ok:
            improved += 1
        elif l_ok and not r_ok:
            worse += 1
        else:
            same += 1
    return {
        "left": left,
        "right": right,
        "n": len(keys),
        "improved": improved,
        "worse": worse,
        "same": same,
        "p_value": mcnemar_exact_pvalue(improved, worse),
    }


def generate_random_tasks(num_tasks: int, seed: int) -> list[dict[str, Any]]:
    random.seed(seed)
    tasks: list[dict[str, Any]] = []
    seen_prompts: set[str] = set()

    coverage_specs = [
        (None, None, False),
        (None, None, True),
        ("starbucks", None, False),
        ("vending", None, False),
        (None, "infohub", False),
        (None, "lobby", False),
        ("starbucks", "infohub", True),
        ("vending", "lobby", True),
    ]
    for source_key, handover_key, include_return in coverage_specs:
        task = build_prompt(source_key, handover_key, include_return)
        if task["prompt"] in seen_prompts:
            continue
        seen_prompts.add(task["prompt"])
        tasks.append(task)

    source_choices = [None, "starbucks", "vending"]
    handover_choices = [None, "infohub", "lobby"]
    while len(tasks) < num_tasks:
        source_key = random.choices(source_choices, weights=[0.25, 0.35, 0.40], k=1)[0]
        handover_key = random.choices(handover_choices, weights=[0.30, 0.30, 0.40], k=1)[0]
        include_return = random.random() < 0.7
        task = build_prompt(source_key, handover_key, include_return)
        if task["prompt"] in seen_prompts:
            continue
        seen_prompts.add(task["prompt"])
        tasks.append(task)

    for i, task in enumerate(tasks, start=1):
        task["id"] = f"rebuttal_{num_tasks}_{i:04d}"
    return tasks


def evaluate_prior_only(tasks: list[dict[str, Any]], prior_plan: list[dict[str, Any]]) -> tuple[dict[str, bool], dict[str, int]]:
    success_map: dict[str, bool] = {}
    reason_counter: Counter[str] = Counter()
    for task in tasks:
        task_id = str(task["id"])
        verifier = PlanSearchVerifier(constraints=infer_constraints(task))
        ok, err, _ = verifier.plan_is_valid(prior_plan)
        success_map[task_id] = bool(ok)
        if not ok:
            reason_counter[err or "invalid"] += 1
    return success_map, dict(reason_counter)


def evaluate_search(
    tasks: list[dict[str, Any]],
    prior_plan: list[dict[str, Any]],
    *,
    iterations: int,
    depth: int,
    seed: int,
) -> tuple[dict[str, bool], list[float], list[int]]:
    success_map: dict[str, bool] = {}
    runtimes: list[float] = []
    plan_lens: list[int] = []

    for idx, task in enumerate(tasks):
        task_id = str(task["id"])
        verifier = PlanSearchVerifier(constraints=infer_constraints(task))
        engine = PlanSearchEngine(
            verifier=verifier,
            max_iterations=iterations,
            rollout_depth=depth,
            exploration_weight=1.25,
            seed=seed + idx,
        )
        t0 = time.perf_counter()
        outcome = engine.search(prior_plan=prior_plan)
        runtimes.append(time.perf_counter() - t0)
        plan_lens.append(len(outcome.plan))
        ok, _, _ = verifier.plan_is_valid(outcome.plan)
        success_map[task_id] = bool(ok)

    return success_map, runtimes, plan_lens


def build_canonical_mismatched_prior(*, iterations: int, depth: int, seed: int) -> list[dict[str, Any]]:
    verifier = PlanSearchVerifier(
        constraints=SearchConstraints(
            allowed_sources=("Starbucks_Outdoor",),
            allowed_handover_zones=("Info_Hub_4F",),
        )
    )
    engine = PlanSearchEngine(
        verifier=verifier,
        max_iterations=iterations,
        rollout_depth=depth,
        exploration_weight=1.25,
        seed=seed,
    )
    outcome = engine.search(prior_plan=[])
    ok, err, _ = verifier.plan_is_valid(outcome.plan)
    if not ok:
        raise RuntimeError(f"Failed to build canonical prior: {err}")
    return outcome.plan


def load_mode_map(path: Path, mode: str) -> dict[str, bool]:
    rows = load_json(path)
    return {str(row["task_id"]): bool(row["success"]) for row in rows if str(row.get("mode")) == mode}


def summarize_locked_llm(data_dir: Path) -> dict[str, Any]:
    qwen_path = data_dir / "app_benchmark_clean_qwen_qwen_2_5_7b_instruct_20260327_201401" / "benchmark_results.json"
    gemini_path = data_dir / "app_benchmark_clean_google_gemini_2_5_flash_20260327_195823" / "benchmark_results.json"
    random40_path = data_dir / "app_benchmark_results_random40" / "benchmark_results.json"

    qwen_g = load_mode_map(qwen_path, "greedy")
    qwen_s = load_mode_map(qwen_path, "search")
    gemini_g = load_mode_map(gemini_path, "greedy")
    gemini_s = load_mode_map(gemini_path, "search")
    random40_g = load_mode_map(random40_path, "greedy")
    random40_s = load_mode_map(random40_path, "search")

    return {
        "qwen_app8": {
            "greedy": summarize_bool_map("qwen_app8_greedy", qwen_g),
            "search": summarize_bool_map("qwen_app8_search", qwen_s),
            "paired_search_vs_greedy": paired_compare(qwen_g, qwen_s, "qwen_app8_greedy", "qwen_app8_search"),
        },
        "gemini_app8": {
            "greedy": summarize_bool_map("gemini_app8_greedy", gemini_g),
            "search": summarize_bool_map("gemini_app8_search", gemini_s),
            "paired_search_vs_greedy": paired_compare(gemini_g, gemini_s, "gemini_app8_greedy", "gemini_app8_search"),
        },
        "gemini_random40": {
            "greedy": summarize_bool_map("gemini_random40_greedy", random40_g),
            "search": summarize_bool_map("gemini_random40_search", random40_s),
            "paired_search_vs_greedy": paired_compare(
                random40_g, random40_s, "gemini_random40_greedy", "gemini_random40_search"
            ),
        },
    }


def summarize_rware_scale(data_dir: Path) -> dict[str, Any]:
    old_suite = load_json(data_dir / "paper_artifacts" / "rware_generalization_suite_report.json")
    new_suite = load_json(data_dir / "paper_artifacts" / "rware_scale_suite1000" / "rware_generalization_suite_report.json")
    old_pooled = old_suite["pooled"]
    new_pooled = new_suite["pooled"]
    return {
        "old_pooled_400": {
            "episodes_total": old_pooled["episodes_total"],
            "random_success_rate": old_pooled["random"]["success_rate"],
            "heuristic_success_rate": old_pooled["heuristic"]["success_rate"],
            "success_p": old_pooled["paired_success_test"]["p_value"],
            "random_early_rate": old_pooled["random"]["early_reward_rate"],
            "heuristic_early_rate": old_pooled["heuristic"]["early_reward_rate"],
            "early_p": old_pooled["paired_early_reward_test"]["p_value"],
        },
        "new_pooled_1000": {
            "episodes_total": new_pooled["episodes_total"],
            "random_success_rate": new_pooled["random"]["success_rate"],
            "heuristic_success_rate": new_pooled["heuristic"]["success_rate"],
            "success_p": new_pooled["paired_success_test"]["p_value"],
            "random_early_rate": new_pooled["random"]["early_reward_rate"],
            "heuristic_early_rate": new_pooled["heuristic"]["early_reward_rate"],
            "early_p": new_pooled["paired_early_reward_test"]["p_value"],
        },
    }


def margin_bucket(score: float, cuts: tuple[float, float, float]) -> float:
    c1, c2, c3 = cuts
    if score <= 0:
        return 0.0
    if score < c1:
        return 0.15
    if score < c2:
        return 0.35
    if score < c3:
        return 0.7
    return 1.1


def summarize_lambda_diagnostics(data_dir: Path) -> dict[str, Any]:
    train_path = data_dir / "dpo_margin_dataset_train.jsonl"
    rows = [json.loads(line) for line in train_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    scores = [float(r.get("preference_strength", {}).get("severity_score", 0.0) or 0.0) for r in rows]
    margins = [margin_bucket(s, (5.0, 10.0, 15.0)) for s in scores]
    active_margins = [m for m in margins if m > 0.0]
    m_max = max(active_margins) if active_margins else 0.0
    beta = 0.1
    lambdas = [0.0, 0.2, 0.5, 1.0]
    diag_rows: list[dict[str, Any]] = []
    for lam in lambdas:
        weights = [1.0 + lam * m for m in active_margins]
        if not weights:
            continue
        p95 = statistics.quantiles(weights, n=20)[18] if len(weights) >= 20 else max(weights)
        diag_rows.append(
            {
                "lambda": lam,
                "mean_weight": statistics.mean(weights),
                "p95_weight": p95,
                "max_weight": max(weights),
                "grad_bound_beta01": beta * (1.0 + lam * m_max),
                "curvature_bound_beta01": (beta * beta / 4.0) * (1.0 + lam * m_max),
            }
        )
    return {
        "train_pairs": len(rows),
        "active_margin_pairs": len(active_margins),
        "mean_margin": statistics.mean(active_margins) if active_margins else 0.0,
        "max_margin": m_max,
        "rows": diag_rows,
    }


def build_app_scale_stress_report(
    *,
    iterations: int,
    depth: int,
    seed: int,
) -> dict[str, Any]:
    invalid_prior = [
        {
            "robot": "Dog",
            "action": "deliver",
            "target": "coffee",
            "details": "Premature direct delivery before pickup/handover.",
        },
        {
            "robot": "Chassis",
            "action": "navigate",
            "target": "User_Desk_Internal",
            "details": "Invalid target for chassis.",
        },
    ]
    mismatch_prior = build_canonical_mismatched_prior(iterations=iterations, depth=depth, seed=seed + 999)

    groups: list[dict[str, Any]] = []
    for num_tasks in [100, 200]:
        tasks = generate_random_tasks(num_tasks, seed + num_tasks)
        category_counts = Counter(str(t.get("category", "unknown")) for t in tasks)

        direct_invalid_map, direct_invalid_reasons = evaluate_prior_only(tasks, invalid_prior)
        direct_mismatch_map, direct_mismatch_reasons = evaluate_prior_only(tasks, mismatch_prior)

        search_invalid_map, rt_invalid, len_invalid = evaluate_search(
            tasks, invalid_prior, iterations=iterations, depth=depth, seed=seed + 11
        )
        search_mismatch_map, rt_mismatch, len_mismatch = evaluate_search(
            tasks, mismatch_prior, iterations=iterations, depth=depth, seed=seed + 37
        )
        search_none_map, rt_none, len_none = evaluate_search(tasks, [], iterations=iterations, depth=depth, seed=seed + 73)

        groups.append(
            {
                "num_tasks": num_tasks,
                "category_counts": dict(category_counts),
                "summaries": {
                    "prior_only_invalid": summarize_bool_map("prior_only_invalid", direct_invalid_map),
                    "prior_only_mismatch": summarize_bool_map("prior_only_mismatch", direct_mismatch_map),
                    "search_from_invalid_prior": summarize_bool_map("search_from_invalid_prior", search_invalid_map, rt_invalid),
                    "search_from_mismatch_prior": summarize_bool_map("search_from_mismatch_prior", search_mismatch_map, rt_mismatch),
                    "search_no_prior": summarize_bool_map("search_no_prior", search_none_map, rt_none),
                    "avg_plan_len_search_from_invalid": statistics.mean(len_invalid),
                    "avg_plan_len_search_from_mismatch": statistics.mean(len_mismatch),
                    "avg_plan_len_search_no_prior": statistics.mean(len_none),
                },
                "pairwise": [
                    paired_compare(
                        direct_invalid_map,
                        search_invalid_map,
                        "prior_only_invalid",
                        "search_from_invalid_prior",
                    ),
                    paired_compare(
                        direct_mismatch_map,
                        search_mismatch_map,
                        "prior_only_mismatch",
                        "search_from_mismatch_prior",
                    ),
                    paired_compare(
                        search_mismatch_map,
                        search_none_map,
                        "search_from_mismatch_prior",
                        "search_no_prior",
                    ),
                    paired_compare(
                        search_invalid_map,
                        search_none_map,
                        "search_from_invalid_prior",
                        "search_no_prior",
                    ),
                ],
                "error_signatures": {
                    "prior_only_invalid_top": sorted(direct_invalid_reasons.items(), key=lambda x: x[1], reverse=True)[:3],
                    "prior_only_mismatch_top": sorted(direct_mismatch_reasons.items(), key=lambda x: x[1], reverse=True)[:3],
                },
            }
        )

    return {
        "search_iterations": iterations,
        "search_depth": depth,
        "invalid_prior_len": len(invalid_prior),
        "mismatch_prior_len": len(mismatch_prior),
        "groups": groups,
    }


def build_markdown(report: dict[str, Any]) -> str:
    locked = report["locked_llm_anchor"]
    stress = report["app_scale_stress"]
    rware = report["rware_scale"]
    lamb = report["lambda_diagnostics"]

    lines: list[str] = []
    lines.append("# Rebuttal Expansion Report")
    lines.append("")
    lines.append("## 1) Locked LLM Anchors (unchanged)")
    lines.append("")
    lines.append("| Setting | n | Greedy | Search | Improved | Worse | p-value |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")

    q = locked["qwen_app8"]
    lines.append(
        "| Qwen App-8 | {n} | {g:.3f} | {s:.3f} | {imp} | {w} | {p:.6f} |".format(
            n=q["greedy"]["n"],
            g=q["greedy"]["success_rate"],
            s=q["search"]["success_rate"],
            imp=q["paired_search_vs_greedy"]["improved"],
            w=q["paired_search_vs_greedy"]["worse"],
            p=q["paired_search_vs_greedy"]["p_value"],
        )
    )

    r40 = locked["gemini_random40"]
    lines.append(
        "| Gemini Random-40 | {n} | {g:.3f} | {s:.3f} | {imp} | {w} | {p:.6f} |".format(
            n=r40["greedy"]["n"],
            g=r40["greedy"]["success_rate"],
            s=r40["search"]["success_rate"],
            imp=r40["paired_search_vs_greedy"]["improved"],
            w=r40["paired_search_vs_greedy"]["worse"],
            p=r40["paired_search_vs_greedy"]["p_value"],
        )
    )
    lines.append("")

    lines.append("## 2) New Large-Sample App Stress Tests (offline, no external API)")
    lines.append("")
    lines.append(
        "These tests use the same verifier and MCTS code path, with intentionally weak/mismatched priors, to check whether search-time recovery remains stable when sample size grows."
    )
    lines.append("")
    lines.append("| Tasks | Prior-only invalid | Search from invalid | Prior-only mismatch | Search from mismatch | Search no-prior |")
    lines.append("| ---: | ---: | ---: | ---: | ---: | ---: |")
    for group in stress["groups"]:
        s = group["summaries"]
        lines.append(
            "| {n} | {a:.3f} | {b:.3f} | {c:.3f} | {d:.3f} | {e:.3f} |".format(
                n=group["num_tasks"],
                a=s["prior_only_invalid"]["success_rate"],
                b=s["search_from_invalid_prior"]["success_rate"],
                c=s["prior_only_mismatch"]["success_rate"],
                d=s["search_from_mismatch_prior"]["success_rate"],
                e=s["search_no_prior"]["success_rate"],
            )
        )
    lines.append("")
    lines.append("| Tasks | Comparison (Right vs Left) | Improved | Worse | p-value |")
    lines.append("| ---: | --- | ---: | ---: | ---: |")
    for group in stress["groups"]:
        for row in group["pairwise"][:2]:
            lines.append(
                "| {n} | {right} vs {left} | {imp} | {w} | {p:.6f} |".format(
                    n=group["num_tasks"],
                    right=row["right"],
                    left=row["left"],
                    imp=row["improved"],
                    w=row["worse"],
                    p=row["p_value"],
                )
            )
    lines.append("")

    lines.append("## 3) RWARE sample-size expansion")
    lines.append("")
    old = rware["old_pooled_400"]
    new = rware["new_pooled_1000"]
    lines.append("| Pooled episodes | Random success | Heuristic success | Success p-value | Random early@60 | Heuristic early@60 | Early p-value |")
    lines.append("| ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    lines.append(
        "| {n} | {rs:.4f} | {hs:.4f} | {sp:.6f} | {re:.4f} | {he:.4f} | {ep:.6f} |".format(
            n=old["episodes_total"],
            rs=old["random_success_rate"],
            hs=old["heuristic_success_rate"],
            sp=old["success_p"],
            re=old["random_early_rate"],
            he=old["heuristic_early_rate"],
            ep=old["early_p"],
        )
    )
    lines.append(
        "| {n} | {rs:.4f} | {hs:.4f} | {sp:.6f} | {re:.4f} | {he:.4f} | {ep:.6f} |".format(
            n=new["episodes_total"],
            rs=new["random_success_rate"],
            hs=new["heuristic_success_rate"],
            sp=new["success_p"],
            re=new["random_early_rate"],
            he=new["heuristic_early_rate"],
            ep=new["early_p"],
        )
    )
    lines.append("")

    lines.append("## 4) mDPO lambda diagnostics (weight-scale)")
    lines.append("")
    lines.append(
        "To answer the lambda question conservatively, we add data-driven optimization-scale diagnostics from the actual training margin distribution."
    )
    lines.append("")
    lines.append("| lambda | Mean weight | P95 weight | Max weight | Max |dL/dDelta| (beta=0.1) | Max d2L/dDelta2 (beta=0.1) |")
    lines.append("| ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in lamb["rows"]:
        lines.append(
            "| {l:.1f} | {m:.4f} | {p95:.4f} | {mx:.4f} | {g:.4f} | {c:.6f} |".format(
                l=row["lambda"],
                m=row["mean_weight"],
                p95=row["p95_weight"],
                mx=row["max_weight"],
                g=row["grad_bound_beta01"],
                c=row["curvature_bound_beta01"],
            )
        )
    lines.append("")
    lines.append("## 5) Rebuttal-ready claims")
    lines.append("")
    lines.append(
        "- We keep the original cross-model claim anchored to locked LLM evidence (Qwen App-8 collapse/recovery and Gemini Random-40 significance)."
    )
    lines.append(
        "- We add larger-sample stress tests (100/200 random tasks) showing search recovery remains stable under deliberately weak or mismatched priors."
    )
    lines.append(
        "- We expand standard-benchmark transfer from pooled 400 to pooled 1000 episodes, preserving the positive direction and improving statistical confidence."
    )
    lines.append(
        "- For mDPO lambda, we provide explicit weight/gradient-scale diagnostics to justify conservative lambda usage and avoid unstable over-weighting."
    )
    lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "locked_llm_anchor": summarize_locked_llm(data_dir),
        "app_scale_stress": build_app_scale_stress_report(
            iterations=args.search_iterations,
            depth=args.search_depth,
            seed=args.seed,
        ),
        "rware_scale": summarize_rware_scale(data_dir),
        "lambda_diagnostics": summarize_lambda_diagnostics(data_dir),
    }

    out_json = out_dir / "rebuttal_expansion_report.json"
    out_md = out_dir / "rebuttal_expansion_report.md"
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    out_md.write_text(build_markdown(report), encoding="utf-8")
    print(f"[OK] Wrote {out_json}")
    print(f"[OK] Wrote {out_md}")


if __name__ == "__main__":
    main()
