from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Set

from eval_utils import extract_task_context, load_jsonl, parse_response_text, validate_plan_schema


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate plans for held-out PVR evaluation splits.")
    parser.add_argument("--input-file", required=True, help="Path to test or balanced_eval jsonl.")
    parser.add_argument("--output-file", required=True, help="Where to save generated jsonl results.")
    parser.add_argument("--model-name-or-path", required=True, help="Base HF model path.")
    parser.add_argument("--adapter-path", default="", help="Optional LoRA adapter path.")
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--dtype", default="auto", choices=("auto", "float16", "bfloat16", "float32"))
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--attn-implementation", default="")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--max-samples", type=int, default=0, help="0 means all rows.")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--resume", action="store_true", help="Skip records already in output-file.")
    parser.add_argument("--prompt-mode", default="chat", choices=("chat", "raw"))
    parser.add_argument(
        "--retry-on-invalid",
        type=int,
        default=0,
        help="Retry generation up to N times when retry conditions are met.",
    )
    parser.add_argument(
        "--retry-if-nosolution",
        action="store_true",
        help="Retry when output contains no-solution or infeasible claims.",
    )
    parser.add_argument(
        "--retry-if-parse-fail",
        action="store_true",
        help="Retry when JSON plan cannot be parsed.",
    )
    parser.add_argument(
        "--retry-if-empty-plan",
        action="store_true",
        help="Retry when parsed plan is an empty list.",
    )
    parser.add_argument(
        "--retry-if-schema-fail",
        action="store_true",
        help="Retry when parsed plan fails schema validation.",
    )
    parser.add_argument(
        "--retry-instruction",
        default=(
            "Task is feasible. Do NOT output no_solution/infeasible. "
            "Output exactly one JSON array plan in <answer>...</answer>. "
            "Each step must be an object with keys: id, robot, action, args, depends_on. "
            "Plan must start with pick_up at task start node and end with put_down at task end node."
        ),
        help="Instruction suffix appended to prompt on retries.",
    )
    return parser.parse_args()


def resolve_torch_dtype(dtype_name: str):
    if dtype_name == "auto":
        return "auto"

    import torch

    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    return mapping[dtype_name]


def build_generator(args: argparse.Namespace):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=args.trust_remote_code,
    )

    model_kwargs: Dict[str, Any] = {
        "device_map": args.device_map,
        "trust_remote_code": args.trust_remote_code,
    }
    torch_dtype = resolve_torch_dtype(args.dtype)
    if torch_dtype != "auto":
        model_kwargs["torch_dtype"] = torch_dtype
    if args.attn_implementation:
        model_kwargs["attn_implementation"] = args.attn_implementation

    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, **model_kwargs)

    if args.adapter_path:
        try:
            from peft import PeftModel
        except ImportError as exc:
            raise RuntimeError("PEFT is required when --adapter-path is set.") from exc
        model = PeftModel.from_pretrained(model, args.adapter_path)

    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model.eval()

    def format_prompt(prompt: str) -> str:
        if args.prompt_mode == "raw":
            return prompt
        if hasattr(tokenizer, "apply_chat_template"):
            try:
                return tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                return prompt
        return prompt

    @torch.inference_mode()
    def generate_one(prompt: str) -> str:
        formatted_prompt = format_prompt(prompt)
        inputs = tokenizer(formatted_prompt, return_tensors="pt")
        input_device = next(model.parameters()).device
        inputs = {name: tensor.to(input_device) for name, tensor in inputs.items()}

        generation_kwargs = {
            "max_new_tokens": args.max_new_tokens,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        if args.temperature > 0.0:
            generation_kwargs["do_sample"] = True
            generation_kwargs["temperature"] = args.temperature
            generation_kwargs["top_p"] = args.top_p
        else:
            generation_kwargs["do_sample"] = False

        generated = model.generate(**inputs, **generation_kwargs)
        new_tokens = generated[0][inputs["input_ids"].shape[-1] :]
        return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    return generate_one


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


NO_SOLUTION_PATTERN = re.compile(
    r"(no[_\s-]?solution|infeasible|unsat|cannot\s+deliver|no\s+feasible\s+plan)",
    re.IGNORECASE,
)


def check_retry_needed(text: str, args: argparse.Namespace) -> Optional[str]:
    if args.retry_if_nosolution and NO_SOLUTION_PATTERN.search(text or ""):
        return "no_solution_claim"

    parsed = parse_response_text(text)
    if args.retry_if_parse_fail and not parsed.get("parse_ok", False):
        return "parse_fail"

    if parsed.get("parse_ok", False):
        plan = parsed.get("plan")
        if args.retry_if_empty_plan and isinstance(plan, list) and len(plan) == 0:
            return "empty_plan"
        if args.retry_if_schema_fail:
            schema = validate_plan_schema(plan)
            if not schema.get("is_valid", False):
                return "schema_fail"

    return None


def main() -> None:
    args = parse_args()

    input_path = Path(args.input_file).resolve()
    output_path = Path(args.output_file).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = load_jsonl(input_path)
    completed_ids = load_completed_ids(output_path) if args.resume else set()
    generator = build_generator(args)

    selected_rows = list(iter_selected_rows(rows, args.start_index, args.max_samples))
    total = len(selected_rows)

    with output_path.open("a" if args.resume else "w", encoding="utf-8") as f:
        for index, row in enumerate(selected_rows, start=1):
            context = extract_task_context(row)
            record_id = context["pair_id"] or context["task_id"] or f"row_{index}"
            if record_id in completed_ids:
                print(f"[skip {index}/{total}] {record_id}")
                continue

            generated_text = generator(context["prompt"])
            retry_count = 0
            retry_reason = ""
            while retry_count < max(0, args.retry_on_invalid):
                reason = check_retry_needed(generated_text, args)
                if not reason:
                    break

                retry_count += 1
                retry_reason = reason
                retry_prompt = (
                    f"{context['prompt']}\n\n"
                    f"[RETRY {retry_count}/{args.retry_on_invalid}] {args.retry_instruction}"
                )
                generated_text = generator(retry_prompt)
                print(f"[retry {index}/{total}] {record_id} reason={reason} attempt={retry_count}")

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
                "model_name_or_path": args.model_name_or_path,
                "adapter_path": args.adapter_path or None,
                "generation_config": {
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "max_new_tokens": args.max_new_tokens,
                    "prompt_mode": args.prompt_mode,
                    "retry_on_invalid": args.retry_on_invalid,
                    "retry_if_nosolution": bool(args.retry_if_nosolution),
                    "retry_if_parse_fail": bool(args.retry_if_parse_fail),
                    "retry_if_empty_plan": bool(args.retry_if_empty_plan),
                    "retry_if_schema_fail": bool(args.retry_if_schema_fail),
                    "retry_instruction": args.retry_instruction,
                    "retry_count": retry_count,
                    "retry_last_reason": retry_reason,
                },
            }
            f.write(json.dumps(out_row, ensure_ascii=False) + "\n")
            f.flush()
            print(f"[done {index}/{total}] {record_id}")


if __name__ == "__main__":
    main()
