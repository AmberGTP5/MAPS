from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from eval_utils import dump_jsonl, extract_task_context, load_jsonl
from graph_engine import build_default_graph_engine
from oracle_planner import OraclePlanner
from symbolic_validator import SymbolicValidator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a unified prior evaluation protocol on one fixed split: "
            "greedy (single-shot) -> retry (invalid-only retries) -> search (oracle repair on failed subset)."
        )
    )
    parser.add_argument("--input-file", required=True, help="Evaluation jsonl file.")
    parser.add_argument("--output-dir", required=True, help="Directory for outputs.")
    parser.add_argument("--label", required=True, help="Run label, e.g. sft, dpo, mdpo.")
    parser.add_argument("--model-name-or-path", required=True, help="Base HF model path.")
    parser.add_argument("--adapter-path", default="", help="Optional LoRA adapter path.")
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--dtype", default="bfloat16", choices=("auto", "float16", "bfloat16", "float32"))
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--retry-on-invalid", type=int, default=2)
    parser.add_argument("--python-bin", default="python", help="Python executable for subprocess calls.")
    return parser.parse_args()


def run_cmd(cmd: List[str]) -> None:
    print("[cmd]", " ".join(cmd))
    subprocess.run(cmd, check=True)


def load_exact_fail_ids(scored_path: Path) -> Set[str]:
    rows = load_jsonl(scored_path)
    failed: Set[str] = set()
    for row in rows:
        if bool(row.get("exact_delivery")):
            continue
        key = row.get("pair_id") or row.get("task_id")
        if isinstance(key, str) and key:
            failed.add(key)
    return failed


def build_answer_text_from_plan(plan: List[Dict[str, Any]]) -> str:
    return "<answer>\n" + json.dumps(plan, ensure_ascii=False, indent=2) + "\n</answer>"


def repair_failed_subset_with_oracle(
    generated_rows: List[Dict[str, Any]],
    failed_ids: Set[str],
) -> tuple[List[Dict[str, Any]], int]:
    engine = build_default_graph_engine(Path(__file__).resolve().parent)
    validator = SymbolicValidator(engine)
    planner = OraclePlanner(engine=engine, validator=validator)

    repaired_count = 0
    output_rows: List[Dict[str, Any]] = []
    for row in generated_rows:
        key = row.get("pair_id") or row.get("task_id")
        if not isinstance(key, str) or not key or key not in failed_ids:
            output_rows.append(row)
            continue

        context = extract_task_context(row)
        start_node = context.get("start_node")
        end_node = context.get("end_node")
        target_item = context.get("target_item")
        if not (isinstance(start_node, str) and isinstance(end_node, str) and isinstance(target_item, str)):
            output_rows.append(row)
            continue

        oracle = planner.plan_task(
            start_node=start_node,
            end_node=end_node,
            target_item=target_item,
            preferred_robot=None,
            force_preferred_if_feasible=False,
        )
        if oracle is None:
            output_rows.append(row)
            continue

        feedback = oracle.feedback or {}
        if not bool(feedback.get("is_valid")):
            output_rows.append(row)
            continue

        patched = dict(row)
        patched["generated_text"] = build_answer_text_from_plan(oracle.plan)
        patched["search_repaired"] = True
        patched["search_repair_source"] = "oracle_planner"
        patched["search_repair_meta"] = {
            "trajectory_cost": feedback.get("trajectory_cost"),
            "selected_robot": oracle.robot,
            "step_count": len(oracle.plan),
        }
        output_rows.append(patched)
        repaired_count += 1

    return output_rows, repaired_count


def load_report(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def exact_count(report: Dict[str, Any]) -> int:
    overall = report.get("overall") or {}
    return int(overall.get("valid_plan_count", 0))


def total_count(report: Dict[str, Any]) -> int:
    overall = report.get("overall") or {}
    return int(overall.get("count", 0))


def main() -> None:
    args = parse_args()
    input_file = Path(args.input_file).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    greedy_generated = output_dir / f"{args.label}_greedy_generated.jsonl"
    greedy_prefix = output_dir / f"{args.label}_greedy"
    retry_generated = output_dir / f"{args.label}_retry_generated.jsonl"
    retry_prefix = output_dir / f"{args.label}_retry"
    search_generated = output_dir / f"{args.label}_search_generated.jsonl"
    search_prefix = output_dir / f"{args.label}_search"
    summary_path = output_dir / f"{args.label}_greedy_retry_search_summary.json"

    base_gen_cmd = [
        args.python_bin,
        "eval_generate_plans.py",
        "--input-file",
        str(input_file),
        "--model-name-or-path",
        args.model_name_or_path,
        "--dtype",
        args.dtype,
        "--device-map",
        args.device_map,
        "--temperature",
        str(args.temperature),
        "--max-new-tokens",
        str(args.max_new_tokens),
    ]
    if args.trust_remote_code:
        base_gen_cmd.append("--trust-remote-code")
    if args.adapter_path.strip():
        base_gen_cmd.extend(["--adapter-path", args.adapter_path.strip()])

    # 1) Greedy
    run_cmd(base_gen_cmd + ["--output-file", str(greedy_generated)])
    run_cmd(
        [
            args.python_bin,
            "eval_score_generations.py",
            "--input-file",
            str(greedy_generated),
            "--text-field",
            "generated_text",
            "--output-prefix",
            str(greedy_prefix),
        ]
    )

    # 2) Retry
    retry_cmd = base_gen_cmd + [
        "--output-file",
        str(retry_generated),
        "--retry-on-invalid",
        str(max(0, args.retry_on_invalid)),
        "--retry-if-nosolution",
        "--retry-if-parse-fail",
        "--retry-if-empty-plan",
        "--retry-if-schema-fail",
    ]
    run_cmd(retry_cmd)
    run_cmd(
        [
            args.python_bin,
            "eval_score_generations.py",
            "--input-file",
            str(retry_generated),
            "--text-field",
            "generated_text",
            "--output-prefix",
            str(retry_prefix),
        ]
    )

    # 3) Search (repair only failed subset from retry)
    retry_scored = Path(str(retry_prefix) + "_scored.jsonl")
    failed_ids = load_exact_fail_ids(retry_scored)
    retry_rows = load_jsonl(retry_generated)
    repaired_rows, repaired_count = repair_failed_subset_with_oracle(retry_rows, failed_ids)
    dump_jsonl(search_generated, repaired_rows)
    run_cmd(
        [
            args.python_bin,
            "eval_score_generations.py",
            "--input-file",
            str(search_generated),
            "--text-field",
            "generated_text",
            "--output-prefix",
            str(search_prefix),
        ]
    )

    greedy_report = load_report(Path(str(greedy_prefix) + "_report.json"))
    retry_report = load_report(Path(str(retry_prefix) + "_report.json"))
    search_report = load_report(Path(str(search_prefix) + "_report.json"))

    total = total_count(greedy_report)
    greedy_ok = exact_count(greedy_report)
    retry_ok = exact_count(retry_report)
    search_ok = exact_count(search_report)

    summary = {
        "label": args.label,
        "input_file": str(input_file),
        "total": total,
        "greedy_exact_delivery": {"count": greedy_ok, "rate": round(greedy_ok / max(total, 1), 6)},
        "retry_exact_delivery": {"count": retry_ok, "rate": round(retry_ok / max(total, 1), 6)},
        "search_exact_delivery": {"count": search_ok, "rate": round(search_ok / max(total, 1), 6)},
        "retry_failed_count": len(failed_ids),
        "search_repaired_count": repaired_count,
        "search_repair_rate_on_retry_fail": round(repaired_count / max(len(failed_ids), 1), 6),
        "paths": {
            "greedy_report": str(Path(str(greedy_prefix) + "_report.json")),
            "retry_report": str(Path(str(retry_prefix) + "_report.json")),
            "search_report": str(Path(str(search_prefix) + "_report.json")),
            "search_generated": str(search_generated),
        },
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[saved] {summary_path}")


if __name__ == "__main__":
    main()
