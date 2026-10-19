# NeurIPS2026 Code Bundle

## 1) Data Generation (4-Phase)
- phase1_generate_openrouter.py
- phase2_validate.py
- phase3_filter.py
- phase4_prepare_training.py
- negative_plan_mutator.py
- preference_utils.py
- graph_engine.py
- symbolic_validator.py
- oracle_planner.py
- task_generator_english.py
- planning_schema.py

## 2) mDPO Training
- train_margin_dpo.py
- prepare_lf_pairwise_jsonl.py
- prepare_llamafactory_data.py
- prepare_llamafactory_data_fixed.py
- patch_remote_llamafactory_mdpo.py
- train_mdpo.sh
- train_mdpo_optimized.yaml
- train_mdpo_optimized_fixed.yaml
- train_mdpo_hardonly_1000.yaml
- train_mdpo_hardonly_1000_3gpu.yaml
- train_mdpo_hardonly_1000_ctx8192_2gpu.yaml
- train_mdpo_llama31_8b_hardonly_1000_2gpu.yaml
- train_mdpo_p0_1000.yaml
- train_mdpo_p1_scale_1000.yaml
- train_mdpo_p1_scale_375.yaml
- run_hardonly_1000_mdpo.sh
- run_hardonly_1000_mdpo_3gpu.sh
- run_hardonly_1000_mdpo_ctx8192_2gpu.sh
- run_llama31_8b_hardonly_1000_mdpo_2gpu.sh
- run_eval_qwen_mdpo_parallel_2gpu.sh
- run_eval_llama31_8b_mdpo_4096.sh
- run_eval_llama31_8b_mdpo_smoke10_2gpu.sh
- run_eval_llama31_8b_mdpo_smoke10_retry_2gpu.sh
- lf_dpo_qwen7b_v3.yaml

## 3) App Inference
- run_app_benchmark.py
- run_app_search_ablation.ps1
- run_app_qwen_local_closure_2gpu.sh
- app_llm_scale_suite.py
- build_app_local_prior_closure.py
- run_prior_retry_search_eval.py
- eval_generate_plans.py
- eval_score_generations.py
- eval_utils.py
- generate_app_benchmark_tasks.py
- run_rware_app_modes.py
- run_graph_mode_benchmark.py
- graph_engine.py
- oracle_planner.py
- symbolic_validator.py
- planning_schema.py
- preference_utils.py
- task_generator_english.py

## 4) Full App Package Source
- Copied from project app/ into 03_app_inference/app/ (all .py files, preserving package structure).

## Missing Files During Copy
- None
