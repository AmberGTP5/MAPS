from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Set

from openai import OpenAI

from eval_utils import extract_task_context, load_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate plans for held-out PVR evaluation splits via OpenRouter.")
    parser.add_argument("--input-file", required=True, help="Path to test or balanced_eval jsonl.")
    parser.add_argument("--output-file", required=True, help="Where to save generated jsonl results.")
    parser.add_argument(
        "--model",
        default=os.getenv("PVR_OPENROUTER_MODEL", "openai/gpt-5"),
        help="OpenRouter model id.",
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("OPENROUTER_API_KEY", ""),
        help="OpenRouter API key. Defaults to OPENROUTER_API_KEY.",
    )
    parser.add_argument(
        "--base-url",
        default=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
        help="OpenRouter-compatible base URL.",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--max-samples", type=int, default=0, help="0 means all rows.")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--resume", action="store_true", help="Skip records already in output-file.")
    parser.add_argument("--max-retries", type=int, default=4)
    parser.add_argument("--retry-sleep-sec", type=float, default=5.0)
    parser.add_argument(
        "--http-referer",
        default="https://github.com/AmberGTP5/robots-planning",
        help="Optional HTTP-Referer header for OpenRouter.",
    )
    parser.add_argument(
        "--x-title",
        default="RobotPlanning-Eval",
        help="Optional X-Title header for OpenRouter.",
    )
    return parser.parse_args()


def load_completed_ids(path: Path) -> Set[str]:
    if not path.exists():
        return set()
    completed = set()
    for row in load_jsonl(path):
        key = row.get("pair_id") or row.get("task_id")
        if isinstance(key, str) and key:
            completed.add(key)
    return completed


def iter_selected_rows(rows: Iterable[Dict[str, Any]], start_index: int, max_samples: int):
    selected = list(rows)[start_index:]
    if max_samples > 0:
        selected = selected[:max_samples]
    for row in selected:
        yield row


def build_client(args: argparse.Namespace) -> OpenAI:
    api_key = (args.api_key or "").strip()
    if not api_key:
        raise RuntimeError("Missing OpenRouter API key. Pass --api-key or set OPENROUTER_API_KEY.")

    headers = {}
    if args.http_referer:
        headers["HTTP-Referer"] = args.http_referer
    if args.x_title:
        headers["X-Title"] = args.x_title

    return OpenAI(
        base_url=args.base_url,
        api_key=api_key,
        default_headers=headers,
    )


def generate_one(client: OpenAI, args: argparse.Namespace, prompt: str) -> str:
    last_err: Exception | None = None
    for attempt in range(1, args.max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=args.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=args.temperature,
                top_p=args.top_p,
                max_tokens=args.max_tokens,
            )
            content = response.choices[0].message.content
            if isinstance(content, str):
                return content.strip()
            if isinstance(content, list):
                text_parts = []
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        text_parts.append(str(item.get("text", "")))
                return "".join(text_parts).strip()
            return str(content).strip()
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            if attempt < args.max_retries:
                time.sleep(args.retry_sleep_sec * attempt)
            else:
                raise RuntimeError(f"OpenRouter generation failed after {args.max_retries} attempts: {exc}") from exc
    raise RuntimeError(f"OpenRouter generation failed: {last_err}")


def main() -> None:
    args = parse_args()

    input_path = Path(args.input_file).resolve()
    output_path = Path(args.output_file).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = load_jsonl(input_path)
    completed_ids = load_completed_ids(output_path) if args.resume else set()
    client = build_client(args)

    selected_rows = list(iter_selected_rows(rows, args.start_index, args.max_samples))
    total = len(selected_rows)

    with output_path.open("a" if args.resume else "w", encoding="utf-8") as f:
        for index, row in enumerate(selected_rows, start=1):
            context = extract_task_context(row)
            record_id = context["pair_id"] or context["task_id"] or f"row_{index}"
            if record_id in completed_ids:
                print(f"[skip {index}/{total}] {record_id}")
                continue

            generated_text = generate_one(client, args, context["prompt"])
            out_row = {
                "pair_id": context["pair_id"],
                "task_id": context["task_id"],
                "selected_robot": context["selected_robot"],
                "task_profile": context["task_profile"],
                "margin": context["margin"],
                "prompt": context["prompt"],
                "task": context["task"],
                "task_metadata": context["task_metadata"],
                "start_node": context["start_node"],
                "end_node": context["end_node"],
                "target_item": context["target_item"],
                "generated_text": generated_text,
                "model_name_or_path": args.model,
                "adapter_path": None,
                "generation_config": {
                    "provider": "openrouter",
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "max_tokens": args.max_tokens,
                },
            }
            f.write(json.dumps(out_row, ensure_ascii=False) + "\n")
            f.flush()
            print(f"[done {index}/{total}] {record_id}")


if __name__ == "__main__":
    main()
