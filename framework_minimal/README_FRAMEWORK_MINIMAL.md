# Framework-Minimal Code Pack (for Paper Architecture Figure)

This minimal pack is curated to support architecture/storyline visualization in the paper.
It keeps only core pipeline scripts and key runtime modules.

## Workflow Mapping
1. Data Construction (4-stage): `01_data_pipeline/phase1-4_*.py`
2. Severity-aware Preference Learning (mDPO): `02_training_mdpo/train_margin_dpo.py` + mdpo configs/scripts
3. Closed-loop Inference (Planner + Verifier + Search): `03_inference_app/run_app_benchmark.py` and `app_core/search/*`

## Included Directories
- `01_data_pipeline/`
- `02_training_mdpo/`
- `03_inference_app/`
- `03_inference_app/app_core/`
- `common/`

## Notes
- This is a **slim subset** for framework communication; not a full reproducibility artifact.
- For full scripts, use sibling folders under `code/`.

## Missing During Packaging
- None
