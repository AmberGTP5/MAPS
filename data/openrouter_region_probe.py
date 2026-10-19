from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from typing import Any

DEFAULT_MODELS = [
    "google/gemini-2.5-flash",
    "qwen/qwen-2.5-7b-instruct",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe OpenRouter model availability under current network/VPN node.")
    parser.add_argument("--api-key", type=str, default="", help="OpenRouter API key. If omitted, reads OPENROUTER_API_KEY.")
    parser.add_argument("--base-url", type=str, default="https://openrouter.ai/api/v1", help="OpenRouter-compatible base URL.")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS, help="Model ids to probe.")
    parser.add_argument("--timeout-sec", type=float, default=25.0, help="Per-request timeout in seconds.")
    parser.add_argument("--rounds", type=int, default=1, help="How many probe rounds to run (>=1).")
    parser.add_argument("--interval-sec", type=float, default=3.0, help="Sleep seconds between rounds.")
    parser.add_argument(
        "--strict-all-models",
        action="store_true",
        help="Require every tested model to pass in every round (recommended before long benchmark runs).",
    )
    parser.add_argument("--show-ip", action="store_true", help="Also query public egress IP info for this node.")
    parser.add_argument(
        "--prompt",
        type=str,
        default="Reply exactly with OK.",
        help="Short probe prompt. Keep this concise to avoid unnecessary token cost.",
    )
    return parser.parse_args()


def get_egress_ip_info(timeout_sec: float) -> dict[str, Any]:
    # ipinfo is simple and returns JSON without API key for basic lookups.
    url = "https://ipinfo.io/json"
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
        payload = json.loads(resp.read().decode("utf-8", errors="ignore"))
    return payload


def _read_error_payload(err: urllib.error.HTTPError) -> str:
    try:
        raw = err.read().decode("utf-8", errors="ignore")
    except Exception:
        return ""

    if not raw:
        return ""

    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        return raw.strip()

    message = obj.get("error", {}).get("message") or obj.get("message") or raw
    return str(message).strip()


def probe_model(
    *,
    base_url: str,
    api_key: str,
    model: str,
    prompt: str,
    timeout_sec: float,
) -> dict[str, Any]:
    url = f"{base_url.rstrip('/')}/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": 32,
        "stream": False,
    }
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url=url,
        data=body,
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/FoundationAgents/OpenManus",
            "X-Title": "OpenManus-Robot",
        },
    )

    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
            latency = time.perf_counter() - t0
            raw = resp.read().decode("utf-8", errors="ignore")
            parsed = json.loads(raw)
            content = (
                parsed.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
            )
            return {
                "model": model,
                "ok": True,
                "http_status": resp.getcode(),
                "latency_sec": round(latency, 3),
                "detail": f"response={repr(str(content)[:80])}",
            }
    except urllib.error.HTTPError as err:
        latency = time.perf_counter() - t0
        detail = _read_error_payload(err)
        return {
            "model": model,
            "ok": False,
            "http_status": err.code,
            "latency_sec": round(latency, 3),
            "detail": detail or str(err),
        }
    except urllib.error.URLError as err:
        latency = time.perf_counter() - t0
        return {
            "model": model,
            "ok": False,
            "http_status": "URLERR",
            "latency_sec": round(latency, 3),
            "detail": str(err.reason),
        }
    except Exception as err:
        latency = time.perf_counter() - t0
        return {
            "model": model,
            "ok": False,
            "http_status": "EXC",
            "latency_sec": round(latency, 3),
            "detail": f"{type(err).__name__}: {err}",
        }


def print_table(rows: list[dict[str, Any]]) -> None:
    header = f"{'model':42} {'ok':4} {'http':8} {'latency':8} detail"
    print(header)
    print("-" * len(header))
    for row in rows:
        model = str(row["model"])[:42]
        ok = "yes" if row["ok"] else "no"
        http = str(row["http_status"])
        latency = str(row["latency_sec"])
        detail = str(row["detail"]).replace("\n", " ")
        print(f"{model:42} {ok:4} {http:8} {latency:8} {detail}")


def summarize_rounds(round_results: list[list[dict[str, Any]]]) -> dict[str, dict[str, Any]]:
    per_model: dict[str, dict[str, Any]] = {}
    for rows in round_results:
        for row in rows:
            model = str(row["model"])
            item = per_model.setdefault(
                model,
                {
                    "rounds": 0,
                    "successes": 0,
                    "http_statuses": {},
                },
            )
            item["rounds"] += 1
            if bool(row["ok"]):
                item["successes"] += 1
            status = str(row["http_status"])
            item["http_statuses"][status] = int(item["http_statuses"].get(status, 0)) + 1
    return per_model


def print_summary(per_model: dict[str, dict[str, Any]]) -> None:
    print("Round Summary:")
    header = f"{'model':42} {'ok/rounds':10} {'success_rate':12} statuses"
    print(header)
    print("-" * len(header))
    for model, item in per_model.items():
        rounds = int(item["rounds"])
        successes = int(item["successes"])
        rate = 0.0 if rounds == 0 else successes / rounds
        statuses = ", ".join(f"{k}:{v}" for k, v in sorted(item["http_statuses"].items()))
        print(f"{model[:42]:42} {f'{successes}/{rounds}':10} {rate:12.3f} {statuses}")


def validate_api_key(api_key: str) -> str | None:
    if not api_key:
        return "missing API key. Pass --api-key or set OPENROUTER_API_KEY."
    if any(ord(ch) > 127 for ch in api_key):
        return "OPENROUTER_API_KEY contains non-ASCII characters. Please paste the raw key only (starts with sk-or-v1-...)."
    if re.search(r"\s", api_key):
        return "OPENROUTER_API_KEY contains whitespace/newline. Remove all spaces and line breaks."
    if not api_key.startswith("sk-or-v1-"):
        return "OPENROUTER_API_KEY format looks wrong. Expected prefix: sk-or-v1-"
    return None


def main() -> int:
    args = parse_args()
    if args.rounds < 1:
        print("ERROR: --rounds must be >= 1.", file=sys.stderr)
        return 2
    if args.interval_sec < 0:
        print("ERROR: --interval-sec must be >= 0.", file=sys.stderr)
        return 2

    api_key = args.api_key.strip() or os.getenv("OPENROUTER_API_KEY", "").strip()
    key_err = validate_api_key(api_key)
    if key_err:
        print(f"ERROR: {key_err}", file=sys.stderr)
        return 2

    print(f"Base URL: {args.base_url}")
    print(f"Models: {args.models}")
    print(f"Timeout: {args.timeout_sec}s")
    print("")

    if args.show_ip:
        try:
            ip_info = get_egress_ip_info(args.timeout_sec)
            print("Egress IP info:")
            print(json.dumps(ip_info, ensure_ascii=False, indent=2))
            print("")
        except Exception as err:
            print(f"Warning: failed to fetch egress IP info: {err}")
            print("")

    all_round_results: list[list[dict[str, Any]]] = []
    for ridx in range(1, args.rounds + 1):
        print(f"Round {ridx}/{args.rounds}")
        round_rows = [
            probe_model(
                base_url=args.base_url,
                api_key=api_key,
                model=model,
                prompt=args.prompt,
                timeout_sec=args.timeout_sec,
            )
            for model in args.models
        ]
        print_table(round_rows)
        print("")
        all_round_results.append(round_rows)
        if ridx < args.rounds and args.interval_sec > 0:
            time.sleep(args.interval_sec)

    per_model = summarize_rounds(all_round_results)
    print_summary(per_model)
    print("")

    if args.strict_all_models:
        all_models_stable = all(
            int(item["successes"]) == int(item["rounds"]) and int(item["rounds"]) > 0
            for item in per_model.values()
        )
        if all_models_stable:
            stable_models = sorted(per_model.keys())
            print(f"PASS: strict stable availability -> {stable_models}")
            return 0
        unstable = [model for model, item in per_model.items() if int(item["successes"]) != int(item["rounds"])]
        print(f"FAIL: strict stability check failed, unstable models -> {unstable}")
        return 1

    any_ok_models = [model for model, item in per_model.items() if int(item["successes"]) > 0]
    if any_ok_models:
        print(f"PASS: available models on current node -> {sorted(any_ok_models)}")
        return 0

    print("FAIL: no tested model is available on current node.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
