from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any


SEED_TASKS = {
    "procthor": [
        "Bring an item from the kitchen counter to the living room table.",
        "Pick up an object in the bedroom and place it in the bathroom.",
        "Navigate to a target object in the dining area and relocate it near the sofa.",
        "Find a mug in the kitchen and move it to the study desk.",
        "Collect an object from the hallway and deliver it to the office room.",
        "Retrieve a small object from the shelf and place it by the TV stand.",
        "Move an item from the entrance area to the bedroom nightstand.",
        "Take a kitchen object and deliver it to the dining table.",
        "Fetch an item from the bathroom sink and place it on a bedroom table.",
        "Carry an object from one room to another while avoiding unnecessary detours.",
    ],
    "alfworld": [
        "go to the kitchen and pick up a mug",
        "find a bottle and put it on the table",
        "pick up a slice of bread and place it in the microwave",
        "go to the bedroom and bring a book to the desk",
        "find a clean cup and deliver it to the counter",
        "retrieve an item from the fridge and place it on the dining table",
        "pick up an object in the living room and take it to the hallway",
        "find a kettle and move it to the kitchen counter",
        "get a plate and place it in the sink",
        "pick up an object and deliver it to the requested location",
    ],
}


INSTRUCTION_KEYS = {
    "instruction",
    "task",
    "goal",
    "goal_description",
    "task_desc",
    "task_description",
    "episode_instruction",
    "pddl_goal",
    "description",
}


SOURCE_CHOICES = [
    ("Starbucks_Outdoor", "Starbucks"),
    ("Vending_Machine_2F", "the 2F vending machine"),
]
HANDOVER_CHOICES = [
    ("Info_Hub_4F", "Info Hub 4F"),
    ("Elevator_Lobby_4F", "the 4F elevator lobby"),
]
RETURN_PHRASES = [
    "After delivery, return both robots to base.",
    "Then send both robots back to their stations.",
    "Finally, recover both robots to base.",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build /app-transfer task files from ProcTHOR/ALFWorld textual goals."
    )
    parser.add_argument(
        "--benchmark",
        required=True,
        choices=["procthor", "alfworld"],
        help="Source benchmark name for metadata and fallback seeds.",
    )
    parser.add_argument(
        "--input-files",
        nargs="*",
        type=Path,
        default=[],
        help="Optional JSON/JSONL/TXT files containing raw third-party task text.",
    )
    parser.add_argument("--num-tasks", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output JSON file compatible with run_app_benchmark.py.",
    )
    return parser.parse_args()


def _read_jsonl(path: Path) -> list[Any]:
    rows: list[Any] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except Exception:
            rows.append(line)
    return rows


def _read_generic(path: Path) -> list[Any]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        return _read_jsonl(path)
    if suffix in {".json"}:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            return payload
        if isinstance(payload, dict):
            return [payload]
        return []
    if suffix in {".txt", ".md"}:
        rows = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        return rows
    return []


def _extract_texts(obj: Any, out: list[str]) -> None:
    if isinstance(obj, str):
        text = " ".join(obj.split())
        if len(text) >= 8:
            out.append(text)
        return
    if isinstance(obj, list):
        for item in obj:
            _extract_texts(item, out)
        return
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key.lower() in INSTRUCTION_KEYS:
                _extract_texts(value, out)
        for value in obj.values():
            if isinstance(value, (dict, list)):
                _extract_texts(value, out)


def load_source_texts(benchmark: str, input_files: list[Path]) -> list[str]:
    texts: list[str] = []
    for path in input_files:
        if not path.exists():
            continue
        for obj in _read_generic(path):
            _extract_texts(obj, texts)

    deduped: list[str] = []
    seen: set[str] = set()
    for text in texts:
        if text in seen:
            continue
        seen.add(text)
        deduped.append(text)

    if deduped:
        return deduped

    return list(SEED_TASKS[benchmark])


def choose_source(text: str, rng: random.Random) -> tuple[str, str]:
    lower = text.lower()
    if any(token in lower for token in ["coffee", "drink", "cup", "mug"]):
        return SOURCE_CHOICES[0]
    if any(token in lower for token in ["can", "bottle", "vending", "soda"]):
        return SOURCE_CHOICES[1]
    return rng.choice(SOURCE_CHOICES)


def choose_handover(text: str, rng: random.Random) -> tuple[str, str]:
    lower = text.lower()
    if any(token in lower for token in ["hallway", "corridor", "lobby"]):
        return HANDOVER_CHOICES[1]
    if any(token in lower for token in ["desk", "office", "study"]):
        return HANDOVER_CHOICES[0]
    return rng.choice(HANDOVER_CHOICES)


def build_prompt(benchmark: str, source_text: str, rng: random.Random) -> dict[str, Any]:
    source_key, source_nl = choose_source(source_text, rng)
    handover_key, handover_nl = choose_handover(source_text, rng)
    add_return = rng.random() < 0.70

    prompt = (
        f"[{benchmark.upper()} transfer] Original task: \"{source_text}\". "
        f"In this campus setting, buy coffee from {source_nl}, perform handoff at {handover_nl}, "
        "and deliver to the office desk."
    )
    if add_return:
        prompt = f"{prompt} {rng.choice(RETURN_PHRASES)}"

    return {
        "category": f"{benchmark}_transfer_proxy",
        "prompt": " ".join(prompt.split()),
        "expected_source": source_key,
        "expected_handover": handover_key,
        "origin_benchmark": benchmark,
        "origin_instruction": source_text,
        "transfer_proxy": True,
    }


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    source_texts = load_source_texts(args.benchmark, args.input_files)
    if not source_texts:
        raise RuntimeError("No source texts were found and no fallback seeds available.")

    tasks: list[dict[str, Any]] = []
    for idx in range(args.num_tasks):
        text = source_texts[idx % len(source_texts)]
        payload = build_prompt(args.benchmark, text, rng)
        payload["id"] = f"{args.benchmark}_{idx + 1:03d}"
        tasks.append(payload)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(tasks, ensure_ascii=False, indent=2), encoding="utf-8")
    print(args.output.resolve())
    print(
        json.dumps(
            {
                "benchmark": args.benchmark,
                "tasks": len(tasks),
                "source_texts": len(source_texts),
                "output": str(args.output),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()

