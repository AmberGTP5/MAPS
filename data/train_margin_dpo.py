"""
Minimal margin-aware DPO trainer for the exported jsonl preference data.

This implementation intentionally avoids TRL/PEFT dependencies so the core
loss is explicit and auditable. It uses the exported fields:
    - prompt
    - chosen
    - rejected
    - margin

Recommended objective:
    L_i = - w_i * log sigma(beta * (pref_logit_i - alpha * margin_i))

where
    pref_logit_i =
        (log p_theta(y_w | x) - log p_theta(y_l | x))
      - (log p_ref(y_w | x)   - log p_ref(y_l | x))

and by default
    alpha = margin_scale
    w_i = (1 + margin_weight * margin_i) * pair_weight_i

The default configuration only applies the logit shift and leaves
margin_weight at 0.0 to keep optimization stable and interpretation simple.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup


DEFAULT_TRAIN_FILE = "dpo_margin_dataset_train.jsonl"
DEFAULT_EVAL_FILE = "dpo_margin_dataset_val.jsonl"


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def dump_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def infer_torch_dtype(dtype_name: str) -> Optional[torch.dtype]:
    mapping = {
        "auto": None,
        "fp32": torch.float32,
        "float32": torch.float32,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
    }
    if dtype_name not in mapping:
        raise ValueError(f"Unsupported dtype={dtype_name}")
    return mapping[dtype_name]


def maybe_load_peft_adapter(
    model: nn.Module,
    adapter_path: str,
    *,
    is_trainable: bool,
) -> nn.Module:
    if not adapter_path:
        return model

    from peft import PeftModel

    resolved = str(Path(adapter_path).expanduser().resolve())
    print(f"Loading PEFT adapter: {resolved} (trainable={is_trainable})")
    return PeftModel.from_pretrained(model, resolved, is_trainable=is_trainable)


def truncate_head_tail(token_ids: Sequence[int], max_len: int, head_len: int) -> List[int]:
    token_ids = list(token_ids)
    if max_len <= 0:
        return []
    if len(token_ids) <= max_len:
        return token_ids
    head_len = max(0, min(head_len, max_len))
    if head_len == 0:
        return token_ids[-max_len:]
    tail_len = max_len - head_len
    if tail_len <= 0:
        return token_ids[:max_len]
    return token_ids[:head_len] + token_ids[-tail_len:]


def build_combined_sequence(
    tokenizer,
    prompt_text: str,
    response_text: str,
    max_length: int,
    max_prompt_length: int,
    max_response_length: int,
    prompt_head_length: int,
    response_head_length: int,
) -> Dict[str, List[int]]:
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    response_ids = tokenizer.encode(response_text, add_special_tokens=False)

    prompt_ids = truncate_head_tail(prompt_ids, max_prompt_length, prompt_head_length)
    response_ids = truncate_head_tail(response_ids, max_response_length, response_head_length)

    while len(prompt_ids) + len(response_ids) > max_length:
        prompt_over = max(0, len(prompt_ids) - prompt_head_length)
        response_over = max(0, len(response_ids) - response_head_length)

        if prompt_over >= response_over and len(prompt_ids) > prompt_head_length:
            prompt_ids.pop(prompt_head_length)
        elif len(response_ids) > response_head_length:
            response_ids.pop(response_head_length)
        elif len(prompt_ids) > 1:
            prompt_ids = prompt_ids[1:]
        elif len(response_ids) > 1:
            response_ids = response_ids[:-1]
        else:
            break

    input_ids = prompt_ids + response_ids
    attention_mask = [1] * len(input_ids)
    loss_mask = [0] * len(prompt_ids) + [1] * len(response_ids)
    if not response_ids:
        raise ValueError("Response text became empty after truncation.")

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "loss_mask": loss_mask,
        "prompt_tokens": len(prompt_ids),
        "response_tokens": len(response_ids),
        "total_tokens": len(input_ids),
    }


@dataclass
class EncodedPreferenceRecord:
    task_id: str
    pair_id: str
    pair_type: str
    pair_weight: float
    selected_robot: str
    task_profile: str
    margin: float
    chosen_input_ids: List[int]
    chosen_attention_mask: List[int]
    chosen_loss_mask: List[int]
    chosen_response_tokens: int
    rejected_input_ids: List[int]
    rejected_attention_mask: List[int]
    rejected_loss_mask: List[int]
    rejected_response_tokens: int
    chosen_ref_logp: float = 0.0
    rejected_ref_logp: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "pair_id": self.pair_id,
            "pair_type": self.pair_type,
            "pair_weight": self.pair_weight,
            "selected_robot": self.selected_robot,
            "task_profile": self.task_profile,
            "margin": self.margin,
            "chosen_input_ids": self.chosen_input_ids,
            "chosen_attention_mask": self.chosen_attention_mask,
            "chosen_loss_mask": self.chosen_loss_mask,
            "chosen_response_tokens": self.chosen_response_tokens,
            "rejected_input_ids": self.rejected_input_ids,
            "rejected_attention_mask": self.rejected_attention_mask,
            "rejected_loss_mask": self.rejected_loss_mask,
            "rejected_response_tokens": self.rejected_response_tokens,
            "chosen_ref_logp": self.chosen_ref_logp,
            "rejected_ref_logp": self.rejected_ref_logp,
        }


class JsonlPreferenceDataset(Dataset):
    def __init__(
        self,
        path: Path,
        tokenizer,
        max_length: int,
        max_prompt_length: int,
        max_response_length: int,
        prompt_head_length: int,
        response_head_length: int,
        prompt_response_separator: str,
        hard_pair_weight: float,
        soft_pair_weight: float,
    ) -> None:
        self.path = path
        self.rows = load_jsonl(path)
        self.records: List[EncodedPreferenceRecord] = []

        for row in self.rows:
            prompt = row["prompt"]
            chosen = row["chosen"]
            rejected = row["rejected"]
            margin = float(row.get("margin", 0.0) or 0.0)
            task = row.get("task", {})
            task_metadata = task.get("task_metadata", {})
            selected_robot = row.get("selected_robot", "unknown")
            task_profile = task_metadata.get("task_profile", "unknown")
            pair_type = str(row.get("pair_type", "hard_invalid"))
            pair_weight = float(soft_pair_weight if pair_type == "soft_suboptimal" else hard_pair_weight)

            chosen_pack = build_combined_sequence(
                tokenizer=tokenizer,
                prompt_text=prompt + prompt_response_separator,
                response_text=chosen,
                max_length=max_length,
                max_prompt_length=max_prompt_length,
                max_response_length=max_response_length,
                prompt_head_length=prompt_head_length,
                response_head_length=response_head_length,
            )
            rejected_pack = build_combined_sequence(
                tokenizer=tokenizer,
                prompt_text=prompt + prompt_response_separator,
                response_text=rejected,
                max_length=max_length,
                max_prompt_length=max_prompt_length,
                max_response_length=max_response_length,
                prompt_head_length=prompt_head_length,
                response_head_length=response_head_length,
            )

            self.records.append(
                EncodedPreferenceRecord(
                    task_id=row["task_id"],
                    pair_id=row["pair_id"],
                    pair_type=pair_type,
                    pair_weight=pair_weight,
                    selected_robot=selected_robot,
                    task_profile=task_profile,
                    margin=margin,
                    chosen_input_ids=chosen_pack["input_ids"],
                    chosen_attention_mask=chosen_pack["attention_mask"],
                    chosen_loss_mask=chosen_pack["loss_mask"],
                    chosen_response_tokens=chosen_pack["response_tokens"],
                    rejected_input_ids=rejected_pack["input_ids"],
                    rejected_attention_mask=rejected_pack["attention_mask"],
                    rejected_loss_mask=rejected_pack["loss_mask"],
                    rejected_response_tokens=rejected_pack["response_tokens"],
                )
            )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        return self.records[index].to_dict()

    def set_reference_logps(self, chosen_logps: Sequence[float], rejected_logps: Sequence[float]) -> None:
        if len(chosen_logps) != len(self.records) or len(rejected_logps) != len(self.records):
            raise ValueError("Reference log-prob arrays do not match dataset length.")
        for record, chosen_logp, rejected_logp in zip(self.records, chosen_logps, rejected_logps):
            record.chosen_ref_logp = float(chosen_logp)
            record.rejected_ref_logp = float(rejected_logp)

    def summarize(self) -> Dict[str, Any]:
        margins = [record.margin for record in self.records]
        pair_weights = [record.pair_weight for record in self.records]
        chosen_response_tokens = [record.chosen_response_tokens for record in self.records]
        rejected_response_tokens = [record.rejected_response_tokens for record in self.records]
        robot_counts: Dict[str, int] = {}
        profile_counts: Dict[str, int] = {}
        pair_type_counts: Dict[str, int] = {}
        for record in self.records:
            robot_counts[record.selected_robot] = robot_counts.get(record.selected_robot, 0) + 1
            profile_counts[record.task_profile] = profile_counts.get(record.task_profile, 0) + 1
            pair_type_counts[record.pair_type] = pair_type_counts.get(record.pair_type, 0) + 1
        return {
            "path": str(self.path.name),
            "count": len(self.records),
            "pair_type_distribution": pair_type_counts,
            "robot_distribution": robot_counts,
            "task_profile_distribution": profile_counts,
            "pair_weight_mean": round(sum(pair_weights) / len(pair_weights), 4) if pair_weights else 0.0,
            "margin_min": min(margins) if margins else 0.0,
            "margin_max": max(margins) if margins else 0.0,
            "margin_mean": round(sum(margins) / len(margins), 4) if margins else 0.0,
            "chosen_response_tokens_mean": round(sum(chosen_response_tokens) / len(chosen_response_tokens), 2)
            if chosen_response_tokens
            else 0.0,
            "rejected_response_tokens_mean": round(sum(rejected_response_tokens) / len(rejected_response_tokens), 2)
            if rejected_response_tokens
            else 0.0,
        }


class PreferenceCollator:
    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = pad_token_id

    def _pad_batch(self, sequences: List[List[int]], pad_value: int) -> torch.Tensor:
        max_len = max(len(seq) for seq in sequences)
        padded = [seq + [pad_value] * (max_len - len(seq)) for seq in sequences]
        return torch.tensor(padded, dtype=torch.long)

    def _pad_batch_float(self, sequences: List[List[int]]) -> torch.Tensor:
        max_len = max(len(seq) for seq in sequences)
        padded = [seq + [0] * (max_len - len(seq)) for seq in sequences]
        return torch.tensor(padded, dtype=torch.float32)

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        return {
            "task_id": [row["task_id"] for row in batch],
            "pair_id": [row["pair_id"] for row in batch],
            "pair_type": [row["pair_type"] for row in batch],
            "selected_robot": [row["selected_robot"] for row in batch],
            "task_profile": [row["task_profile"] for row in batch],
            "margin": torch.tensor([row["margin"] for row in batch], dtype=torch.float32),
            "pair_weight": torch.tensor([row["pair_weight"] for row in batch], dtype=torch.float32),
            "chosen_input_ids": self._pad_batch([row["chosen_input_ids"] for row in batch], self.pad_token_id),
            "chosen_attention_mask": self._pad_batch([row["chosen_attention_mask"] for row in batch], 0),
            "chosen_loss_mask": self._pad_batch_float([row["chosen_loss_mask"] for row in batch]),
            "rejected_input_ids": self._pad_batch([row["rejected_input_ids"] for row in batch], self.pad_token_id),
            "rejected_attention_mask": self._pad_batch([row["rejected_attention_mask"] for row in batch], 0),
            "rejected_loss_mask": self._pad_batch_float([row["rejected_loss_mask"] for row in batch]),
            "chosen_ref_logp": torch.tensor([row["chosen_ref_logp"] for row in batch], dtype=torch.float32),
            "rejected_ref_logp": torch.tensor([row["rejected_ref_logp"] for row in batch], dtype=torch.float32),
        }


def move_batch_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    output: Dict[str, Any] = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            output[key] = value.to(device)
        else:
            output[key] = value
    return output


def compute_sequence_logps(
    model: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    loss_mask: torch.Tensor,
    average_log_prob: bool,
) -> torch.Tensor:
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    shifted_logits = logits[:, :-1, :]
    shifted_labels = input_ids[:, 1:]
    shifted_loss_mask = loss_mask[:, 1:]

    token_logps = F.log_softmax(shifted_logits, dim=-1).gather(-1, shifted_labels.unsqueeze(-1)).squeeze(-1)
    seq_logps = (token_logps * shifted_loss_mask).sum(dim=-1)
    if average_log_prob:
        denom = shifted_loss_mask.sum(dim=-1).clamp_min(1.0)
        seq_logps = seq_logps / denom
    return seq_logps


def compute_dpo_terms(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    ref_chosen_logps: torch.Tensor,
    ref_rejected_logps: torch.Tensor,
    margins: torch.Tensor,
    pair_weights: torch.Tensor,
    beta: float,
    margin_scale: float,
    margin_weight: float,
) -> Dict[str, torch.Tensor]:
    preference_logit = (policy_chosen_logps - policy_rejected_logps) - (ref_chosen_logps - ref_rejected_logps)
    shifted_logit = preference_logit - margin_scale * margins
    weights = (1.0 + margin_weight * margins) * pair_weights
    per_example_loss = -F.logsigmoid(beta * shifted_logit) * weights
    normalized_loss = per_example_loss.sum() / weights.sum().clamp_min(1.0)

    return {
        "loss": normalized_loss,
        "per_example_loss": per_example_loss.detach(),
        "preference_logit": preference_logit.detach(),
        "shifted_logit": shifted_logit.detach(),
        "reward_accuracy": (preference_logit > 0).float().detach(),
        "margin_accuracy": (preference_logit > margin_scale * margins).float().detach(),
        "weights": weights.detach(),
    }


@torch.inference_mode()
def precompute_reference_logps(
    model: nn.Module,
    dataset: JsonlPreferenceDataset,
    collator: PreferenceCollator,
    batch_size: int,
    device: torch.device,
    average_log_prob: bool,
) -> None:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collator)
    chosen_logps: List[float] = []
    rejected_logps: List[float] = []

    model.eval()
    for batch in loader:
        batch = move_batch_to_device(batch, device)
        chosen = compute_sequence_logps(
            model=model,
            input_ids=batch["chosen_input_ids"],
            attention_mask=batch["chosen_attention_mask"],
            loss_mask=batch["chosen_loss_mask"],
            average_log_prob=average_log_prob,
        )
        rejected = compute_sequence_logps(
            model=model,
            input_ids=batch["rejected_input_ids"],
            attention_mask=batch["rejected_attention_mask"],
            loss_mask=batch["rejected_loss_mask"],
            average_log_prob=average_log_prob,
        )
        chosen_logps.extend(chosen.detach().cpu().tolist())
        rejected_logps.extend(rejected.detach().cpu().tolist())

    dataset.set_reference_logps(chosen_logps, rejected_logps)


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    dataset: JsonlPreferenceDataset,
    collator: PreferenceCollator,
    batch_size: int,
    device: torch.device,
    average_log_prob: bool,
    beta: float,
    margin_scale: float,
    margin_weight: float,
) -> Dict[str, float]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collator)
    model.eval()

    total_loss = 0.0
    total_examples = 0
    reward_acc_sum = 0.0
    margin_acc_sum = 0.0
    mean_pref_logit_sum = 0.0
    mean_shifted_logit_sum = 0.0

    for batch in loader:
        batch = move_batch_to_device(batch, device)
        policy_chosen = compute_sequence_logps(
            model=model,
            input_ids=batch["chosen_input_ids"],
            attention_mask=batch["chosen_attention_mask"],
            loss_mask=batch["chosen_loss_mask"],
            average_log_prob=average_log_prob,
        )
        policy_rejected = compute_sequence_logps(
            model=model,
            input_ids=batch["rejected_input_ids"],
            attention_mask=batch["rejected_attention_mask"],
            loss_mask=batch["rejected_loss_mask"],
            average_log_prob=average_log_prob,
        )

        terms = compute_dpo_terms(
            policy_chosen_logps=policy_chosen,
            policy_rejected_logps=policy_rejected,
            ref_chosen_logps=batch["chosen_ref_logp"],
            ref_rejected_logps=batch["rejected_ref_logp"],
            margins=batch["margin"],
            pair_weights=batch["pair_weight"],
            beta=beta,
            margin_scale=margin_scale,
            margin_weight=margin_weight,
        )
        batch_size_now = batch["margin"].shape[0]
        total_examples += batch_size_now
        total_loss += float(terms["loss"].item()) * batch_size_now
        reward_acc_sum += float(terms["reward_accuracy"].sum().item())
        margin_acc_sum += float(terms["margin_accuracy"].sum().item())
        mean_pref_logit_sum += float(terms["preference_logit"].sum().item())
        mean_shifted_logit_sum += float(terms["shifted_logit"].sum().item())

    denom = max(1, total_examples)
    return {
        "loss": round(total_loss / denom, 6),
        "reward_accuracy": round(reward_acc_sum / denom, 6),
        "margin_accuracy": round(margin_acc_sum / denom, 6),
        "mean_preference_logit": round(mean_pref_logit_sum / denom, 6),
        "mean_shifted_logit": round(mean_shifted_logit_sum / denom, 6),
        "count": total_examples,
    }


def save_checkpoint(model: nn.Module, tokenizer, output_dir: Path, step_name: str) -> Path:
    checkpoint_dir = output_dir / step_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(checkpoint_dir)
    tokenizer.save_pretrained(checkpoint_dir)
    return checkpoint_dir


def train(args: argparse.Namespace) -> None:
    base_dir = Path(__file__).resolve().parent
    output_dir = (base_dir / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.jsonl"
    if metrics_path.exists():
        metrics_path.unlink()

    set_seed(args.seed)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    torch_dtype = infer_torch_dtype(args.torch_dtype)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        use_fast=not args.use_slow_tokenizer,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

    collator = PreferenceCollator(pad_token_id=tokenizer.pad_token_id)

    train_dataset = JsonlPreferenceDataset(
        path=(base_dir / args.train_file).resolve(),
        tokenizer=tokenizer,
        max_length=args.max_length,
        max_prompt_length=args.max_prompt_length,
        max_response_length=args.max_response_length,
        prompt_head_length=args.prompt_head_length,
        response_head_length=args.response_head_length,
        prompt_response_separator=args.prompt_response_separator,
        hard_pair_weight=args.hard_pair_weight,
        soft_pair_weight=args.soft_pair_weight,
    )
    eval_dataset = JsonlPreferenceDataset(
        path=(base_dir / args.eval_file).resolve(),
        tokenizer=tokenizer,
        max_length=args.max_length,
        max_prompt_length=args.max_prompt_length,
        max_response_length=args.max_response_length,
        prompt_head_length=args.prompt_head_length,
        response_head_length=args.response_head_length,
        prompt_response_separator=args.prompt_response_separator,
        hard_pair_weight=args.hard_pair_weight,
        soft_pair_weight=args.soft_pair_weight,
    )

    config_payload = {
        "args": vars(args),
        "device": str(device),
        "dataset_summary": {
            "train": train_dataset.summarize(),
            "eval": eval_dataset.summarize(),
        },
    }
    dump_json(output_dir / "run_config.json", config_payload)

    print("=" * 72)
    print("Loading reference model and precomputing reference log-probs")
    print("=" * 72)
    ref_model_path = args.reference_model_path or args.model_name_or_path
    ref_model = AutoModelForCausalLM.from_pretrained(
        ref_model_path,
        torch_dtype=torch_dtype,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
    )
    ref_model = maybe_load_peft_adapter(
        ref_model,
        args.reference_adapter_path,
        is_trainable=False,
    )
    if len(tokenizer) > ref_model.get_input_embeddings().num_embeddings:
        ref_model.resize_token_embeddings(len(tokenizer))
    ref_model.to(device)
    precompute_reference_logps(
        model=ref_model,
        dataset=train_dataset,
        collator=collator,
        batch_size=args.eval_batch_size,
        device=device,
        average_log_prob=args.average_log_prob,
    )
    precompute_reference_logps(
        model=ref_model,
        dataset=eval_dataset,
        collator=collator,
        batch_size=args.eval_batch_size,
        device=device,
        average_log_prob=args.average_log_prob,
    )
    del ref_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    print("=" * 72)
    print("Loading policy model")
    print("=" * 72)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch_dtype,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
    )
    peft_training_enabled = False
    if args.policy_adapter_path:
        model = maybe_load_peft_adapter(
            model,
            args.policy_adapter_path,
            is_trainable=True,
        )
        peft_training_enabled = True
    if len(tokenizer) > model.get_input_embeddings().num_embeddings:
        model.resize_token_embeddings(len(tokenizer))

    if args.use_lora and not args.policy_adapter_path:
        from peft import LoraConfig, TaskType, get_peft_model

        target_modules = [name.strip() for name in args.lora_target_modules.split(",") if name.strip()]
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=target_modules,
        )
        model = get_peft_model(model, lora_config)
        peft_training_enabled = True
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"LoRA enabled: trainable={trainable}, total={total}, pct={100.0 * trainable / max(1, total):.4f}%")
    elif args.policy_adapter_path and args.use_lora:
        print("policy_adapter_path is provided, skip creating a new LoRA adapter and finetune the loaded adapter.")

    if args.gradient_checkpointing:
        # For LoRA + gradient checkpointing, embeddings must require grad; otherwise
        # checkpointed graph can become detached and backward() fails.
        if peft_training_enabled:
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
            else:
                def _make_inputs_require_grad(_module, _inputs, output):
                    output.requires_grad_(True)

                model.get_input_embeddings().register_forward_hook(_make_inputs_require_grad)
        model.gradient_checkpointing_enable()
    model.to(device)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.per_device_train_batch_size,
        shuffle=True,
        collate_fn=collator,
    )

    updates_per_epoch = math.ceil(len(train_loader) / args.gradient_accumulation_steps)
    max_train_steps = updates_per_epoch * args.num_train_epochs
    warmup_steps = int(max_train_steps * args.warmup_ratio)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=max_train_steps,
    )

    best_eval_loss = float("inf")
    global_step = 0

    print("=" * 72)
    print("Starting margin-aware DPO training")
    print("=" * 72)
    print(f"Train records: {len(train_dataset)}")
    print(f"Eval records: {len(eval_dataset)}")
    print(f"Device: {device}")
    print(f"Max train steps: {max_train_steps}")
    print(
        f"beta={args.beta}, margin_scale={args.margin_scale}, margin_weight={args.margin_weight}, "
        f"hard_pair_weight={args.hard_pair_weight}, soft_pair_weight={args.soft_pair_weight}"
    )
    print("=" * 72)

    for epoch in range(args.num_train_epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        running_reward_acc = 0.0
        running_margin_acc = 0.0
        running_examples = 0

        for step, batch in enumerate(train_loader):
            batch = move_batch_to_device(batch, device)

            policy_chosen = compute_sequence_logps(
                model=model,
                input_ids=batch["chosen_input_ids"],
                attention_mask=batch["chosen_attention_mask"],
                loss_mask=batch["chosen_loss_mask"],
                average_log_prob=args.average_log_prob,
            )
            policy_rejected = compute_sequence_logps(
                model=model,
                input_ids=batch["rejected_input_ids"],
                attention_mask=batch["rejected_attention_mask"],
                loss_mask=batch["rejected_loss_mask"],
                average_log_prob=args.average_log_prob,
            )

            terms = compute_dpo_terms(
                policy_chosen_logps=policy_chosen,
                policy_rejected_logps=policy_rejected,
                ref_chosen_logps=batch["chosen_ref_logp"],
                ref_rejected_logps=batch["rejected_ref_logp"],
                margins=batch["margin"],
                pair_weights=batch["pair_weight"],
                beta=args.beta,
                margin_scale=args.margin_scale,
                margin_weight=args.margin_weight,
            )

            loss = terms["loss"] / args.gradient_accumulation_steps
            loss.backward()

            batch_size_now = batch["margin"].shape[0]
            running_examples += batch_size_now
            running_loss += float(terms["loss"].item()) * batch_size_now
            running_reward_acc += float(terms["reward_accuracy"].sum().item())
            running_margin_acc += float(terms["margin_accuracy"].sum().item())

            if (step + 1) % args.gradient_accumulation_steps == 0 or (step + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

        train_metrics = {
            "epoch": epoch + 1,
            "global_step": global_step,
            "split": "train",
            "loss": round(running_loss / max(1, running_examples), 6),
            "reward_accuracy": round(running_reward_acc / max(1, running_examples), 6),
            "margin_accuracy": round(running_margin_acc / max(1, running_examples), 6),
            "learning_rate": round(float(scheduler.get_last_lr()[0]), 12),
        }
        append_jsonl(metrics_path, train_metrics)
        print(f"[Epoch {epoch + 1}] train: {train_metrics}")

        eval_metrics = evaluate(
            model=model,
            dataset=eval_dataset,
            collator=collator,
            batch_size=args.eval_batch_size,
            device=device,
            average_log_prob=args.average_log_prob,
            beta=args.beta,
            margin_scale=args.margin_scale,
            margin_weight=args.margin_weight,
        )
        eval_record = {"epoch": epoch + 1, "global_step": global_step, "split": "eval", **eval_metrics}
        append_jsonl(metrics_path, eval_record)
        print(f"[Epoch {epoch + 1}] eval: {eval_record}")

        save_checkpoint(model, tokenizer, output_dir, f"checkpoint-epoch{epoch + 1}")
        if eval_metrics["loss"] < best_eval_loss:
            best_eval_loss = eval_metrics["loss"]
            best_dir = save_checkpoint(model, tokenizer, output_dir, "best_checkpoint")
            dump_json(
                output_dir / "best_metrics.json",
                {
                    "best_eval_loss": best_eval_loss,
                    "best_checkpoint": str(best_dir),
                    "eval_metrics": eval_record,
                    "train_metrics": train_metrics,
                },
            )

    final_eval = evaluate(
        model=model,
        dataset=eval_dataset,
        collator=collator,
        batch_size=args.eval_batch_size,
        device=device,
        average_log_prob=args.average_log_prob,
        beta=args.beta,
        margin_scale=args.margin_scale,
        margin_weight=args.margin_weight,
    )
    dump_json(
        output_dir / "final_metrics.json",
        {
            "final_eval": final_eval,
            "best_eval_loss": best_eval_loss,
        },
    )
    save_checkpoint(model, tokenizer, output_dir, "final_checkpoint")
    print("=" * 72)
    print("Training complete")
    print("=" * 72)
    print(json.dumps({"final_eval": final_eval, "best_eval_loss": best_eval_loss}, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a margin-aware DPO model on exported jsonl pairs.")
    parser.add_argument("--model-name-or-path", required=True, help="Base policy model path or Hugging Face id.")
    parser.add_argument(
        "--reference-model-path",
        default="",
        help="Optional separate reference model path. Defaults to model-name-or-path.",
    )
    parser.add_argument(
        "--policy-adapter-path",
        default="",
        help="Optional PEFT adapter path to initialize policy from (e.g., SFT LoRA).",
    )
    parser.add_argument(
        "--reference-adapter-path",
        default="",
        help="Optional PEFT adapter path to initialize reference from.",
    )
    parser.add_argument("--train-file", default=DEFAULT_TRAIN_FILE)
    parser.add_argument("--eval-file", default=DEFAULT_EVAL_FILE)
    parser.add_argument("--output-dir", default="training_runs/margin_dpo")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--use-slow-tokenizer", action="store_true")
    parser.add_argument("--device", default="auto", help="auto/cpu/cuda/cuda:0")
    parser.add_argument("--torch-dtype", default="auto", choices=["auto", "fp32", "bf16", "fp16"])

    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--max-prompt-length", type=int, default=3072)
    parser.add_argument("--max-response-length", type=int, default=1024)
    parser.add_argument("--prompt-head-length", type=int, default=256)
    parser.add_argument("--response-head-length", type=int, default=256)
    parser.add_argument("--prompt-response-separator", default="\n\n")
    parser.add_argument(
        "--average-log-prob",
        action="store_true",
        default=True,
        help="Use mean token log-prob over response tokens instead of raw sum.",
    )
    parser.add_argument(
        "--no-average-log-prob",
        dest="average_log_prob",
        action="store_false",
        help="Use raw response-sequence log-prob sums.",
    )

    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--margin-scale", type=float, default=1.0)
    parser.add_argument("--margin-weight", type=float, default=0.0)
    parser.add_argument(
        "--hard-pair-weight",
        type=float,
        default=1.0,
        help="Multiplicative loss weight for hard_invalid pairs.",
    )
    parser.add_argument(
        "--soft-pair-weight",
        type=float,
        default=0.0,
        help="Multiplicative loss weight for soft_suboptimal pairs (default 0.0 = hard-only).",
    )

    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--num-train-epochs", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument(
        "--use-lora",
        action="store_true",
        default=True,
        help="Enable LoRA for policy model training (recommended on single GPU).",
    )
    parser.add_argument(
        "--no-use-lora",
        dest="use_lora",
        action="store_false",
        help="Disable LoRA and train full model parameters.",
    )
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora-target-modules",
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        help="Comma-separated target module names for LoRA.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
