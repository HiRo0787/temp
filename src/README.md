# `src` Deep-Dive README

This README is flow-first and starts from the requested entrypoint:
`training/single_stage_finetune.py`.

It explains:
- what executes first,
- which modules are called next,
- how data and artifacts move through the pipeline,
- and how the rest of the `src` tree relates to that flow.

---

## 1) Real Execution Flow (Single-Stage Finetuning)

### 1.1 CLI entrypoint
Primary script: `training/single_stage_finetune.py`

Main control paths:
- `--list-models`: reads model registry and exits.
- `--test-only`: loads base model (+ optional LoRA checkpoint) and runs prompts.
- default training path: full train -> save -> optional merge -> optional eval.

### 1.2 Model selection + loading
`QwenVLLMFineTuner.__init__` calls:
- `infra.model_registry.get_registry()`:
  - reads `config/model/models.yaml`,
  - resolves model by `--model-key` or (`--model-version`, `--model-size`),
  - exposes defaults (dtype, unsloth, LoRA config).
- `utilities.training_config.TrainingConfig.get_torch_dtype()`.
- `utilities.model_loader.ModelLoader`:
  - `load_with_unsloth()` if enabled,
  - otherwise `load_standard()`,
  - supports fallback models from registry chain.

### 1.3 LoRA preparation
`prepare_model_for_training()` calls `utilities.lora_preparer.LoRAPreparer`:
- Unsloth path: `FastLanguageModel.get_peft_model(...)`.
- Standard path: PEFT `LoraConfig` + `get_peft_model(...)`.
- LoRA target modules are sourced from `models.yaml` model config.

### 1.4 Data loading + ordering
`load_training_data()` calls `utilities.data_loader.DataLoader`:
- Reads JSONL records (`messages`, optional `metadata`).
- Splits train/test (90/10).
- Uses `utilities.data_formatter.DataFormatter.format_training_data()` to apply tokenizer chat template.
- Tokenizes with max length from CLI/default.
- Assigns difficulty buckets by context length.
- Applies `--learning-type` strategy:
  - `curriculum`, `random-shuffled`, `blocked-*`, `interleaved-*`.

### 1.5 Run directory + provenance
Before training:
- `utilities.single_stage_helpers.resolve_output_dir()` picks run dir.
- `utilities.artifact_paths.ensure_artifact_run_subdirs()` creates:
  - `checkpoints/`, `lora/`, `merged/`, `logs/`, `data_versions/`.
- `copy_data_manifest(...)` stores dataset provenance manifest with the run.
- `build_run_config_payload(...)` + `write_run_config_file(...)` writes run config JSON.

### 1.6 Trainer construction
Training path branches:
- Unsloth: TRL `SFTTrainer`.
- Standard: custom HF `Trainer` with custom loss/data-collator safeguards.

Shared setup:
- `TrainingConfig.get_eval_config(...)`
- `single_stage_helpers.compute_save_steps(...)`
- `TrainingConfig.get_optimizer_config(...)`
- `single_stage_helpers.get_eval_strategy_kwargs(...)`
- `TrainingArguments(...)` with eval/save/logging/checkpoint config.

### 1.7 Train + checkpoint resume behavior
If resuming:
- `single_stage_helpers.patch_checkpoint_trainer_state(...)` updates old checkpoint eval/save cadence.

Then:
- `trainer.train(...)`
- save LoRA to `lora/`
- save tokenizer
- generate LoRA README via `utilities.lora_readme.write_lora_readme(...)`
- optional merge via `_merge_lora_adapters(...)`.

### 1.8 Optional post-train benchmark eval
If `--eval-after-train`:
- resolve checkpoint (`best` or `final`) via `resolve_post_train_eval_checkpoint_path(...)`.
- release GPU from trainer process.
- run `python -m src.eval.run_eval` once per selected benchmark (subprocess).
- restore model back to CUDA for local prompt test.

### 1.9 Final smoke test
`test_model(...)` runs fixed prompts and logs outputs.

---

## 2) Import Dependency Graph (from the starting script)

Direct imports used by `training/single_stage_finetune.py`:
- `infra/model_registry.py`
- `infra/project_paths.py`
- `utilities/training_config.py`
- `utilities/training_logger.py`
- `utilities/model_loader.py`
- `utilities/data_formatter.py`
- `utilities/data_loader.py`
- `utilities/lora_preparer.py`
- `utilities/artifact_paths.py`
- `utilities/lora_readme.py`
- `utilities/model_config_serialization.py`
- `utilities/single_stage_helpers.py`

Second-level runtime dependencies reached through these:
- `eval/run_eval.py`
- `eval/checkpoint_runner.py`
- `eval/reporting.py`
- `eval/benchmarks/*` (selected by `--eval-bench`)
- `config/model/models.yaml`
- `config/data_prep/*` (indirectly via data prep and schema modules)

---

## 3) Data Flow and Artifact Flow

### Data flow
1. `data_prep/combine_training_data.py` creates JSONL in `combine_output/<version_id>/all_training_data.jsonl`.
2. Optional manifest (`manifest.json`) is produced alongside JSONL.
3. `single_stage_finetune.py --data ...` consumes that JSONL.
4. Manifest is copied into run artifacts (`data_versions/manifest.json`) for provenance.

### Artifact flow
Per training run (`artifacts/<model>/<run>/`):
- `checkpoints/`: HF checkpoint-* directories.
- `lora/`: final adapter + tokenizer.
- `merged/`: full merged model when requested.
- `logs/`: tensorboard events.
- `data_versions/manifest.json`: dataset version trace.
- `config.json`: run metadata/hyperparameters.

---

## 4) Folder-by-Folder Deep Explanation

## `training/`
- `single_stage_finetune.py`: primary one-run finetuner (flow described above).
- `multi_stage_finetune.py`: stage-wise continuation with LoRA replay support.
- `README.md`: command-level training docs.

## `utilities/`
- `training_config.py`: dtype mapping, optimizer config, single-process env helpers.
- `training_logger.py`: centralized colored logging + tee-to-file run logs.
- `model_loader.py`: unsloth/HF model loading + fallback chain logic.
- `data_formatter.py`: chat-template-based training/inference prompt formatting.
- `data_loader.py`: JSONL load, train/test split, tokenization, curriculum ordering.
- `lora_preparer.py`: PEFT/Unsloth LoRA attachment and trainable parameter setup.
- `artifact_paths.py`: run dir allocation, subdir creation, latest marker, manifest copy.
- `single_stage_helpers.py`: single-stage-specific defaults, run config writing, post-train eval runner.
- `multi_stage_helpers.py`: stage learning rates, replay mixing, load-existing-lora continuation.
- `lora_readme.py`: writes metadata README into LoRA output.
- `model_config_serialization.py`: patches non-JSON-safe model configs before trainer serialization.
- `colours.py`: terminal color/formatter support.

## `infra/`
- `model_registry.py`: config-driven model abstraction (`ModelConfig`, aliases, fallback chains).
- `project_paths.py`: canonical root/artifacts/config/log path resolver.
- `taxonomy.py`: large security taxonomy and consistency rules.
- `tool_registry.py`: standardized tool descriptions and category mappings.
- `mitre_owasp_mappings.py`: topic->MITRE, topic->OWASP, category->tactic mappings.
- `validator.py`: dataset quality and consistency validator against taxonomy/mappings.
- `refusal_coverage.py`: tracks coverage of high-refusal attack topics.
- `specialized_templates.py`: scenario-template functions for dataset generation pipelines.

## `data_prep/`
- `combine_training_data.py`: orchestrates scenario loading + conversion + output writing.
- `json_source_loader.py`: parses JSON/YAML and validates to schema types.
- `data_schema.py`: Pydantic schemas + training-example generators.
- `data_versioning.py`: version IDs, content hashes, archives, index, manifest management.
- `README.md`: full operational data pipeline guide.

## `eval/`
- `run_eval.py`: unified benchmark CLI for base model or LoRA checkpoints.
- `checkpoint_runner.py`: load-once base model and in-place adapter swap loop.
- `reporting.py`: manifest/report generation for eval runs.
- `prompt.py`: interactive prompt REPL for model/checkpoint.
- `eval_checkpoint_loss.py`: per-datapoint loss diagnostics for a checkpoint.
- `evaluate.py`: broader legacy evaluator (OpenAI/local scoring harness).
- `evaluate_qwen3guard.py`: vLLM endpoint test harness.
- `benchmarks/`: benchmark-specific runners grouped into:
  - `cybersecurity/`
  - `safeguards/`
  - `coding/`
  - `reasoning/`
- `benchmarks/benchmarks_catalog.yaml`: task catalog used by helper task limiting.

## `serve/`
- `serve_rabit0_vllm.py`: starts vLLM OpenAI-compatible server with base model + optional LoRA.
- `merge_lora_for_vllm.py`: merges adapter into base model for pure merged deployment.
- `python_json_tool_parser.py`: custom vLLM tool-call parser plugin for JSON-in-text extraction.

## `review/`
- `query_rabit0_model.py`: direct local query script for base+LoRA testing.

## `config/`
- `config/model/models.yaml`: source of truth for model keys, aliases, fallbacks, and LoRA defaults.
- `config/data_prep/datapoints.yaml`: active dataset source list for combiner.
- `config/data_prep/system_prompts.yaml`, `training_categories.yaml`, `quality_guidelines.yaml`: generation/runtime data-shaping configs.

---

## 5) How the Rest of `src` Connects Back to the Starting Script

From `single_stage_finetune.py`, the practical downstream loop is:
1. Train adapters (`training` + `utilities` + `infra`).
2. Optionally evaluate checkpoints (`eval` + `benchmarks`).
3. Serve resulting model (`serve`).
4. Regenerate improved datasets (`data_prep`) and retrain.

So the repository is not isolated scripts; it is one iterative system:
`data_prep -> train -> eval -> serve -> data improvements -> retrain`.

---

## 6) Notes on Current Codebase Shape

- The single-stage script is the most modern orchestrator and the best "source-of-truth" runtime path.
- Some modules (for example `evaluate.py` / `evaluate_qwen3guard.py`) are auxiliary evaluators outside the main `run_eval` checkpoint loop.
- Several scripts include backward-compatible path/import fallbacks; functionally this does not change the flow above.

---

## 7) Quick Start Path (Recommended)

1. Build data:
   - `python src/data_prep/combine_training_data.py`
2. Train:
   - `python src/training/single_stage_finetune.py --data combine_output/<version_id>/all_training_data.jsonl --model-key qwen2.5-7b`
3. Eval:
   - `python -m src.eval.run_eval --run-dir artifacts/<model>/<run_id> --bench secqa`
4. Serve:
   - `python src/serve/serve_rabit0_vllm.py --base-model <base> --adapter-path artifacts/<model>/<run_id>/lora`

