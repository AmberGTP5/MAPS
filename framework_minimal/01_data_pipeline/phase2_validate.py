"""
Phase 2: validate rejected plans with rule engine first, LLM fallback second.
"""

from __future__ import annotations

import json
import os
import re
import time
from collections import Counter
from datetime import datetime
from typing import Dict, List, Optional

from openai import OpenAI
from tqdm import tqdm

from symbolic_validator import SymbolicValidator


# Prefer generic OpenRouter-compatible env vars, while preserving the older
# phase-2-specific names so existing user scripts keep working unchanged.
VALIDATOR_API_KEY = os.getenv("CLAUDE_VALIDATOR_API_KEY") or os.getenv("OPENROUTER_API_KEY", "")
VALIDATOR_BASE_URL = os.getenv("CLAUDE_VALIDATOR_BASE_URL") or os.getenv(
    "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
)
VALIDATOR_MODEL = os.getenv("CLAUDE_VALIDATOR_MODEL") or os.getenv(
    "PVR_PHASE2_MODEL", "anthropic/claude-3.7-sonnet"
)

INPUT_FILE = "dpo_dataset_phase1.jsonl"
OUTPUT_FILE = "dpo_dataset_validated.jsonl"
LOG_FILE = "validation_log.txt"

client = OpenAI(base_url=VALIDATOR_BASE_URL, api_key=VALIDATOR_API_KEY or "MISSING")
validator = SymbolicValidator()

log_file = None
stats = {
    "total_samples": 0,
    "rule_validated": 0,
    "llm_validated": 0,
    "validation_failures": 0,
    "perfect_matches": 0,
    "partial_mismatches": 0,
    "pair_type_distribution": Counter(),
    "soft_rule_validated": 0,
    "soft_validation_failures": 0,
}
VALID_PAIR_TYPES = {"hard_invalid", "soft_suboptimal"}


def log(message: str, level: str = "INFO"):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_message = f"[{timestamp}] [{level}] {message}"
    print(log_message)
    if log_file:
        log_file.write(log_message + "\n")
        log_file.flush()


def call_llm(prompt: str, max_retries: int = 3) -> Optional[str]:
    if not VALIDATOR_API_KEY:
        return None

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=VALIDATOR_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=1500,
            )
            return response.choices[0].message.content
        except Exception as exc:
            log(f"LLM validation call failed ({attempt + 1}/{max_retries}): {exc}", "WARNING")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                return None


def extract_json_object(text: str) -> Optional[Dict]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        fenced = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
        if fenced:
            try:
                return json.loads(fenced.group(1))
            except json.JSONDecodeError:
                return None
    return None


def llm_validate_plan(instruction: str, rejected_plan: List[Dict], target_errors: List[str]) -> Optional[Dict]:
    prompt = f"""You are a backup validator for graph-grounded multi-robot plans.

Task instruction:
{instruction}

Rejected plan:
{json.dumps(rejected_plan, indent=2, ensure_ascii=False)}

Expected error families:
{json.dumps(target_errors)}

Return ONLY a JSON object:
```json
{{
  "is_valid": false,
  "error_type": ["..."],
  "error_details": "Detailed explanation",
  "failed_steps": [1, 3],
  "penalty_score": 10,
  "validation_source": "llm"
}}
```
"""
    response = call_llm(prompt)
    if not response:
        return None

    parsed = extract_json_object(response)
    if parsed:
        parsed["validation_source"] = "llm"
    return parsed


def normalize_pair_type(sample: Dict) -> str:
    pair_type = sample.get("pair_type", "hard_invalid")
    if not isinstance(pair_type, str) or pair_type not in VALID_PAIR_TYPES:
        return "hard_invalid"
    return pair_type


def normalize_soft_feedback(rule_feedback: Dict) -> Dict:
    normalized = dict(rule_feedback)
    normalized["is_valid"] = True
    normalized["error_type"] = []
    normalized["failed_steps"] = []
    if "error_details" not in normalized or not isinstance(normalized.get("error_details"), str):
        normalized["error_details"] = "valid_plan_confirmed_for_soft_suboptimal_pair"
    normalized["penalty_score"] = float(normalized.get("penalty_score", 0.0) or 0.0)
    normalized["validation_source"] = normalized.get("validation_source", "rule_engine")
    return normalized


def validate_sample(sample: Dict) -> Optional[Dict]:
    instruction = sample["instruction"]
    target_errors = sample.get("target_errors", [])
    rejected_plan = sample["rejected"]["plan"]
    pair_type = normalize_pair_type(sample)
    stats["pair_type_distribution"][pair_type] += 1

    start_node = sample["start_node"]
    end_node = sample["end_node"]
    target_item = sample["target_item"]

    log(f"  Validating pair_type={pair_type} (targets: {', '.join(target_errors)})...")
    rule_feedback = validator.validate_plan(
        rejected_plan,
        start_node=start_node,
        end_node=end_node,
        target_item=target_item,
    )

    if pair_type == "soft_suboptimal":
        if rule_feedback.get("is_valid", False):
            stats["rule_validated"] += 1
            stats["soft_rule_validated"] += 1
            return normalize_soft_feedback(rule_feedback)

        stats["validation_failures"] += 1
        stats["soft_validation_failures"] += 1
        log(
            "  [FAIL] soft_suboptimal pair rejected by rule validation "
            f"(errors={rule_feedback.get('error_type', [])})",
            "WARNING",
        )
        return None

    target_set = set(target_errors)
    caught_by_rule = set(rule_feedback.get("error_type", [])) if not rule_feedback.get("is_valid", True) else set()

    if not rule_feedback.get("is_valid", True):
        log(f"  [RULE] Detected: {sorted(caught_by_rule)} (score={rule_feedback.get('penalty_score')})")
        if target_set.issubset(caught_by_rule):
            stats["rule_validated"] += 1
            stats["perfect_matches"] += 1
            return rule_feedback

    llm_feedback = llm_validate_plan(instruction, rejected_plan, target_errors)
    if llm_feedback:
        if not llm_feedback.get("violation_breakdown") and not rule_feedback.get("is_valid", True):
            llm_feedback["violation_breakdown"] = rule_feedback.get("violation_breakdown", {})
        if "trajectory_cost" not in llm_feedback and "trajectory_cost" in rule_feedback:
            llm_feedback["trajectory_cost"] = rule_feedback.get("trajectory_cost", 0.0)
        if "penalty_score" not in llm_feedback and "penalty_score" in rule_feedback:
            llm_feedback["penalty_score"] = rule_feedback.get("penalty_score", 0.0)

        stats["llm_validated"] += 1
        caught_by_llm = set(llm_feedback.get("error_type", []))
        if target_set.issubset(caught_by_llm):
            stats["perfect_matches"] += 1
        else:
            stats["partial_mismatches"] += 1
        return llm_feedback

    if not rule_feedback.get("is_valid", True):
        stats["rule_validated"] += 1
        stats["partial_mismatches"] += 1
        return rule_feedback

    stats["validation_failures"] += 1
    return None


def main():
    global log_file

    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    log_file = open(LOG_FILE, "w", encoding="utf-8")

    log("=" * 60)
    log("Phase 2: Validate rejected plans")
    log("=" * 60)
    log(f"Input: {INPUT_FILE}")
    log(f"Output: {OUTPUT_FILE}")
    log("=" * 60)

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        samples = [json.loads(line) for line in f]

    output_file = open(OUTPUT_FILE, "w", encoding="utf-8")
    for index, sample in enumerate(tqdm(samples, desc="Validating")):
        stats["total_samples"] += 1
        log(f"\n--- Sample {index + 1}/{len(samples)} ---")
        feedback = validate_sample(sample)
        if feedback:
            pair_type = normalize_pair_type(sample)
            sample["rejected"]["feedback"] = feedback
            sample["pair_type"] = pair_type
            chosen_feedback = sample.get("chosen", {}).get("feedback", {})
            sample["chosen_valid"] = bool(chosen_feedback.get("is_valid", True))
            sample["rejected_valid"] = bool(feedback.get("is_valid", False))
            output_file.write(json.dumps(sample, ensure_ascii=False) + "\n")
            output_file.flush()
        else:
            log("  [SKIP] Validation failed.", "WARNING")
    output_file.close()

    log("\n" + "=" * 60)
    log("Validation complete")
    log("=" * 60)
    log(f"Total samples: {stats['total_samples']}")
    log(f"Rule-validated: {stats['rule_validated']}")
    log(f"LLM-validated: {stats['llm_validated']}")
    log(f"Validation failures: {stats['validation_failures']}")
    log(f"Perfect matches: {stats['perfect_matches']}")
    log(f"Partial mismatches: {stats['partial_mismatches']}")
    log(f"Soft rule-validated: {stats['soft_rule_validated']}")
    log(f"Soft validation failures: {stats['soft_validation_failures']}")
    log("Pair-type distribution:")
    for pair_type, count in stats["pair_type_distribution"].items():
        log(f"  {pair_type}: {count}")
    log(f"Output saved to: {OUTPUT_FILE}")
    log("=" * 60)

    log_file.close()


if __name__ == "__main__":
    main()
