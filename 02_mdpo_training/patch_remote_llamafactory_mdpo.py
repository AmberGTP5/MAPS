from __future__ import annotations

from pathlib import Path
import shutil


BACKUP_SUFFIX = ".bak_mdpo_20260406"


def replace_once(text: str, old: str, new: str, path: Path) -> str:
    if old not in text:
        raise RuntimeError(f"Pattern not found in {path}: {old[:120]!r}")
    return text.replace(old, new, 1)


def patch_finetuning_args(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    if "pref_margin_mode" in text:
        return
    anchor = "\n\n@dataclass\nclass GaloreArguments:\n"
    insert = """

    pref_margin_mode: Literal["none", "sample"] = field(
        default="none",
        metadata={"help": "Whether to apply sample-wise preference margins in DPO training."},
    )
    pref_margin_field: str = field(
        default="margin",
        metadata={"help": "The dataset field name that stores sample-wise preference margins."},
    )
    pref_margin_scale: float = field(
        default=1.0,
        metadata={"help": "The scaling coefficient for sample-wise preference margins."},
    )
    pref_margin_weight: float = field(
        default=0.0,
        metadata={"help": "Optional extra per-sample loss weight coefficient for preference margins."},
    )
"""
    text = replace_once(text, anchor, insert + anchor, path)
    path.write_text(text, encoding="utf-8")


def patch_pairwise(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    if 'model_inputs["margin"]' in text:
        return
    old = '            model_inputs["audios"].append(examples["_audios"][i])\n'
    new = """            model_inputs["audios"].append(examples["_audios"][i])\n            margin_value = 0.0\n            if "margin" in examples and examples["margin"][i] is not None:\n                try:\n                    margin_value = float(examples["margin"][i])\n                except (TypeError, ValueError):\n                    margin_value = 0.0\n            model_inputs["margin"].append(margin_value)\n"""
    text = replace_once(text, old, new, path)
    path.write_text(text, encoding="utf-8")


def patch_converter(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    if '"margin": float(example["margin"])' in text:
        return

    old_alpaca = """        output = {\n            \"_prompt\": prompt,\n            \"_response\": response,\n            \"_system\": example[self.dataset_attr.system] if self.dataset_attr.system else \"\",\n            \"_tools\": example[self.dataset_attr.tools] if self.dataset_attr.tools else \"\",\n            \"_images\": self._find_medias(example[self.dataset_attr.images]) if self.dataset_attr.images else None,\n            \"_videos\": self._find_medias(example[self.dataset_attr.videos]) if self.dataset_attr.videos else None,\n            \"_audios\": self._find_medias(example[self.dataset_attr.audios]) if self.dataset_attr.audios else None,\n        }\n"""
    new_alpaca = """        output = {\n            \"_prompt\": prompt,\n            \"_response\": response,\n            \"_system\": example[self.dataset_attr.system] if self.dataset_attr.system else \"\",\n            \"_tools\": example[self.dataset_attr.tools] if self.dataset_attr.tools else \"\",\n            \"_images\": self._find_medias(example[self.dataset_attr.images]) if self.dataset_attr.images else None,\n            \"_videos\": self._find_medias(example[self.dataset_attr.videos]) if self.dataset_attr.videos else None,\n            \"_audios\": self._find_medias(example[self.dataset_attr.audios]) if self.dataset_attr.audios else None,\n            \"margin\": float(example[\"margin\"]) if \"margin\" in example and example[\"margin\"] is not None else 0.0,\n        }\n"""
    text = replace_once(text, old_alpaca, new_alpaca, path)

    old_sharegpt = """        output = {\n            \"_prompt\": prompt,\n            \"_response\": response,\n            \"_system\": system,\n            \"_tools\": example[self.dataset_attr.tools] if self.dataset_attr.tools else \"\",\n            \"_images\": self._find_medias(example[self.dataset_attr.images]) if self.dataset_attr.images else None,\n            \"_videos\": self._find_medias(example[self.dataset_attr.videos]) if self.dataset_attr.videos else None,\n            \"_audios\": self._find_medias(example[self.dataset_attr.audios]) if self.dataset_attr.audios else None,\n        }\n"""
    new_sharegpt = """        output = {\n            \"_prompt\": prompt,\n            \"_response\": response,\n            \"_system\": system,\n            \"_tools\": example[self.dataset_attr.tools] if self.dataset_attr.tools else \"\",\n            \"_images\": self._find_medias(example[self.dataset_attr.images]) if self.dataset_attr.images else None,\n            \"_videos\": self._find_medias(example[self.dataset_attr.videos]) if self.dataset_attr.videos else None,\n            \"_audios\": self._find_medias(example[self.dataset_attr.audios]) if self.dataset_attr.audios else None,\n            \"margin\": float(example[\"margin\"]) if \"margin\" in example and example[\"margin\"] is not None else 0.0,\n        }\n"""
    text = replace_once(text, old_sharegpt, new_sharegpt, path)
    path.write_text(text, encoding="utf-8")


def patch_collator(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    if 'batch["margin"] = torch.tensor' in text:
        return
    old = """        return super().__call__(concatenated_features)\n"""
    new = """        batch = super().__call__(concatenated_features)\n        batch["margin"] = torch.tensor([float(feature.get("margin", 0.0)) for feature in features], dtype=torch.float32)\n        return batch\n"""
    text = replace_once(text, old, new, path)
    path.write_text(text, encoding="utf-8")


def patch_trainer(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    if 'margins: Optional["torch.Tensor"] = None' in text and 'model_batch = {k: v for k, v in batch.items() if k != "margin"}' in text:
        return

    old_concatenated = """        all_logits: torch.Tensor = model(**batch, return_dict=True, use_cache=False).logits.to(torch.float32)\n        all_logps, valid_length = get_batch_logps(\n            logits=all_logits, labels=batch[\"labels\"], ld_alpha=(self.ld_alpha if not is_ref_model else None)\n        )\n        if self.loss_type in [\"ipo\", \"orpo\", \"simpo\"]:\n            all_logps = all_logps / valid_length\n\n        batch_size = batch[\"input_ids\"].size(0) // 2\n"""
    new_concatenated = """        model_batch = {k: v for k, v in batch.items() if k != \"margin\"}\n        all_logits: torch.Tensor = model(**model_batch, return_dict=True, use_cache=False).logits.to(torch.float32)\n        all_logps, valid_length = get_batch_logps(\n            logits=all_logits, labels=model_batch[\"labels\"], ld_alpha=(self.ld_alpha if not is_ref_model else None)\n        )\n        if self.loss_type in [\"ipo\", \"orpo\", \"simpo\"]:\n            all_logps = all_logps / valid_length\n\n        batch_size = model_batch[\"input_ids\"].size(0) // 2\n"""
    text = replace_once(text, old_concatenated, new_concatenated, path)

    old_compute_preference = """    def compute_preference_loss(\n        self,\n        policy_chosen_logps: \"torch.Tensor\",\n        policy_rejected_logps: \"torch.Tensor\",\n        reference_chosen_logps: Optional[\"torch.Tensor\"],\n        reference_rejected_logps: Optional[\"torch.Tensor\"],\n    ) -> tuple[\"torch.Tensor\", \"torch.Tensor\", \"torch.Tensor\"]:\n        r\"\"\"Compute loss for preference learning.\"\"\"\n        if not self.finetuning_args.use_ref_model:\n            if self.loss_type == \"orpo\":\n                losses = self.odds_ratio_loss(policy_chosen_logps, policy_rejected_logps)\n            elif self.loss_type == \"simpo\":\n                losses = self.simpo_loss(policy_chosen_logps, policy_rejected_logps)\n            else:\n                raise NotImplementedError(f\"Unknown loss type: {self.loss_type}.\")\n\n            chosen_rewards = self.beta * policy_chosen_logps.to(self.accelerator.device).detach()\n            rejected_rewards = self.beta * policy_rejected_logps.to(self.accelerator.device).detach()\n        else:\n            losses, chosen_rewards, rejected_rewards = self.dpo_loss(\n                policy_chosen_logps, policy_rejected_logps, reference_chosen_logps, reference_rejected_logps\n            )\n\n        return losses, chosen_rewards, rejected_rewards\n"""
    new_compute_preference = """    def compute_preference_loss(\n        self,\n        policy_chosen_logps: \"torch.Tensor\",\n        policy_rejected_logps: \"torch.Tensor\",\n        reference_chosen_logps: Optional[\"torch.Tensor\"],\n        reference_rejected_logps: Optional[\"torch.Tensor\"],\n        margins: Optional[\"torch.Tensor\"] = None,\n    ) -> tuple[\"torch.Tensor\", \"torch.Tensor\", \"torch.Tensor\"]:\n        r\"\"\"Compute loss for preference learning.\"\"\"\n        if not self.finetuning_args.use_ref_model:\n            if self.loss_type == \"orpo\":\n                losses = self.odds_ratio_loss(policy_chosen_logps, policy_rejected_logps)\n            elif self.loss_type == \"simpo\":\n                losses = self.simpo_loss(policy_chosen_logps, policy_rejected_logps)\n            else:\n                raise NotImplementedError(f\"Unknown loss type: {self.loss_type}.\")\n\n            chosen_rewards = self.beta * policy_chosen_logps.to(self.accelerator.device).detach()\n            rejected_rewards = self.beta * policy_rejected_logps.to(self.accelerator.device).detach()\n        else:\n            use_margin = (\n                margins is not None\n                and self.loss_type == \"sigmoid\"\n                and getattr(self.finetuning_args, \"pref_margin_mode\", \"none\") != \"none\"\n            )\n            if use_margin:\n                margins = margins.to(policy_chosen_logps.device, dtype=policy_chosen_logps.dtype)\n                logits = (policy_chosen_logps - policy_rejected_logps) - (\n                    reference_chosen_logps - reference_rejected_logps\n                )\n                shifted_logits = logits - self.finetuning_args.pref_margin_scale * margins\n                weights = 1.0 + self.finetuning_args.pref_margin_weight * margins\n                losses = -F.logsigmoid(self.beta * shifted_logits) * weights\n                chosen_rewards = self.beta * (\n                    policy_chosen_logps.to(self.accelerator.device) - reference_chosen_logps.to(self.accelerator.device)\n                ).detach()\n                rejected_rewards = self.beta * (\n                    policy_rejected_logps.to(self.accelerator.device) - reference_rejected_logps.to(self.accelerator.device)\n                ).detach()\n            else:\n                losses, chosen_rewards, rejected_rewards = self.dpo_loss(\n                    policy_chosen_logps, policy_rejected_logps, reference_chosen_logps, reference_rejected_logps\n                )\n\n        return losses, chosen_rewards, rejected_rewards\n"""
    text = replace_once(text, old_compute_preference, new_compute_preference, path)

    old_loss_metrics = """        reference_chosen_logps, reference_rejected_logps = self.compute_reference_log_probs(model, batch)\n        losses, chosen_rewards, rejected_rewards = self.compute_preference_loss(\n            policy_chosen_logps,\n            policy_rejected_logps,\n            reference_chosen_logps,\n            reference_rejected_logps,\n        )\n"""
    new_loss_metrics = """        reference_chosen_logps, reference_rejected_logps = self.compute_reference_log_probs(model, batch)\n        margins = batch.get(\"margin\", None)\n        losses, chosen_rewards, rejected_rewards = self.compute_preference_loss(\n            policy_chosen_logps,\n            policy_rejected_logps,\n            reference_chosen_logps,\n            reference_rejected_logps,\n            margins=margins,\n        )\n"""
    text = replace_once(text, old_loss_metrics, new_loss_metrics, path)

    old_metric_block = """        metrics[f\"{prefix}rewards/margins\"] = (chosen_rewards - rejected_rewards).mean().item()\n        metrics[f\"{prefix}logps/chosen\"] = policy_chosen_logps.mean().item()\n"""
    new_metric_block = """        metrics[f\"{prefix}rewards/margins\"] = (chosen_rewards - rejected_rewards).mean().item()\n        if margins is not None:\n            margins = margins.to(policy_chosen_logps.device, dtype=policy_chosen_logps.dtype)\n            if reference_chosen_logps is None or reference_rejected_logps is None:\n                preference_logits = policy_chosen_logps - policy_rejected_logps\n            else:\n                preference_logits = (policy_chosen_logps - policy_rejected_logps) - (\n                    reference_chosen_logps - reference_rejected_logps\n                )\n            metrics[f\"{prefix}rewards/target_margins\"] = margins.mean().item()\n            metrics[f\"{prefix}rewards/margin_accuracies\"] = (\n                preference_logits > self.finetuning_args.pref_margin_scale * margins\n            ).float().mean().item()\n        metrics[f\"{prefix}logps/chosen\"] = policy_chosen_logps.mean().item()\n"""
    text = replace_once(text, old_metric_block, new_metric_block, path)

    path.write_text(text, encoding="utf-8")


def backup(path: Path) -> None:
    backup_path = path.with_name(path.name + BACKUP_SUFFIX)
    if not backup_path.exists():
        shutil.copy2(path, backup_path)


def main() -> None:
    root = Path("/root/miniconda3/envs/pvr/lib/python3.10/site-packages/llamafactory")
    targets = {
        "finetuning_args": root / "hparams" / "finetuning_args.py",
        "converter": root / "data" / "converter.py",
        "pairwise": root / "data" / "processor" / "pairwise.py",
        "collator": root / "data" / "collator.py",
        "trainer": root / "train" / "dpo" / "trainer.py",
    }

    for path in targets.values():
        backup(path)

    patch_finetuning_args(targets["finetuning_args"])
    patch_converter(targets["converter"])
    patch_pairwise(targets["pairwise"])
    patch_collator(targets["collator"])
    patch_trainer(targets["trainer"])

    print("patched:")
    for name, path in targets.items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
