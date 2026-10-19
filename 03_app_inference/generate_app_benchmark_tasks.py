from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


GENERIC_SOURCE_PHRASES = [
    "Buy me a coffee",
    "Please get me a coffee",
    "Bring me a coffee",
]

STARBUCKS_SOURCE_PHRASES = [
    "Buy me a Starbucks coffee",
    "Please pick up a coffee from Starbucks",
    "Get me coffee from Starbucks",
]

VENDING_SOURCE_PHRASES = [
    "Get me a coffee from the 2F vending machine",
    "Please pick up coffee from the vending machine on 2F",
    "Use the vending machine on 2F to get me a coffee",
]

GENERIC_DELIVERY_PHRASES = [
    "and bring it to my desk",
    "and deliver it to my desk",
    "then take it to my desk",
]

INFOHUB_HANDOVER_PHRASES = [
    "and do the handoff at Info Hub 4F before delivery",
    "and hand it over at the Info Hub on 4F before bringing it to my desk",
    "and use Info Hub 4F as the handoff point before the desk delivery",
]

LOBBY_HANDOVER_PHRASES = [
    "and do the handoff at the 4F elevator lobby before delivery",
    "and meet at the 4F elevator lobby for handoff before bringing it to my desk",
    "and use the 4F elevator lobby as the handoff point before the desk delivery",
]

RETURN_PHRASES = [
    "Afterward, return both robots to base.",
    "Then make sure both robots return to their stations.",
    "Finally, have both robots go back to base.",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate random prompts for the current /app coffee benchmark.")
    parser.add_argument("--num-tasks", type=int, default=24, help="Number of tasks to generate.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("app_benchmark_tasks_random.json"),
        help="Where to write the generated benchmark tasks.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--ensure-coverage",
        action="store_true",
        help="Seed the output with one example from each major constraint category.",
    )
    return parser.parse_args()


def choose_source_bundle(source_key: str | None) -> tuple[str, str | None]:
    if source_key == "starbucks":
        return random.choice(STARBUCKS_SOURCE_PHRASES), "Starbucks_Outdoor"
    if source_key == "vending":
        return random.choice(VENDING_SOURCE_PHRASES), "Vending_Machine_2F"
    return random.choice(GENERIC_SOURCE_PHRASES), None


def choose_handover_bundle(handover_key: str | None) -> tuple[str, str | None]:
    if handover_key == "infohub":
        return random.choice(INFOHUB_HANDOVER_PHRASES), "Info_Hub_4F"
    if handover_key == "lobby":
        return random.choice(LOBBY_HANDOVER_PHRASES), "Elevator_Lobby_4F"
    return random.choice(GENERIC_DELIVERY_PHRASES), None


def build_prompt(source_key: str | None, handover_key: str | None, include_return: bool) -> dict:
    source_phrase, expected_source = choose_source_bundle(source_key)
    handover_phrase, expected_handover = choose_handover_bundle(handover_key)

    prompt = f"{source_phrase} {handover_phrase}."
    if include_return:
        prompt = f"{prompt} {random.choice(RETURN_PHRASES)}"

    category = "generic"
    if expected_source and expected_handover:
        category = "combined_constraint"
    elif expected_source:
        category = "source_constraint"
    elif expected_handover:
        category = "handover_constraint"

    task = {
        "category": category,
        "prompt": " ".join(prompt.split()),
    }
    if expected_source:
        task["expected_source"] = expected_source
    if expected_handover:
        task["expected_handover"] = expected_handover
    return task


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    tasks: list[dict] = []
    seen_prompts: set[str] = set()

    if args.ensure_coverage:
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

    while len(tasks) < args.num_tasks:
        source_key = random.choices(source_choices, weights=[0.25, 0.35, 0.40], k=1)[0]
        handover_key = random.choices(handover_choices, weights=[0.30, 0.30, 0.40], k=1)[0]
        include_return = random.random() < 0.7

        task = build_prompt(source_key, handover_key, include_return)
        if task["prompt"] in seen_prompts:
            continue
        seen_prompts.add(task["prompt"])
        tasks.append(task)

    for index, task in enumerate(tasks, start=1):
        task["id"] = f"rand_{index:03d}"

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(tasks, ensure_ascii=False, indent=2), encoding="utf-8")
    print(args.output.resolve())


if __name__ == "__main__":
    main()
