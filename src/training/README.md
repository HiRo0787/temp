# Training

Fine-tuning scripts for **Rabit0** (Red Team Security AI), built on Qwen2.5-Coder / Qwen3 and optimized for vLLM serving. Training uses LoRA adapters; you can merge them into a full model for vLLM or keep them separate for lighter storage.

## Contents

- [Scripts](#scripts)
- [Full workflow and detailed report](#full-workflow-and-detailed-report)
- [When to use single-stage vs multi-stage](#when-to-use-single-stage-vs-multi-stage)
- [Output structure](#output-structure)
- [Training data format](#training-data-format)
- [Single-stage training](#single-stage-training)
- [Post-training benchmark eval (single-stage)](#post-training-benchmark-eval-single-stage)
- [Multi-stage training](#multi-stage-training)
- [Resource requirements](#resource-requirements)
- [Environment and setup](#environment-and-setup)
- [Troubleshooting and notes](#troubleshooting-and-notes)
- [Related documentation](#related-documentation)

---

## Scripts

| Script | Description |
|--------|-------------|
| `single_stage_finetune.py` | One-shot LoRA fine-tuning: one dataset, one learning rate, one run. Best for a single curated dataset. |
| `multi_stage_finetune.py` | Multi-stage LoRA: stage 1 trains a fresh LoRA; stage 2+ load the previous stage LoRA and continue. Best for curriculum or phased data. |

Both support optional LoRA merge for vLLM-ready full models. Model choices (size, version, registry key) and training knobs (epochs, batch size, dtype, etc.) are shared where applicable.

---

## Full workflow and detailed report

### Files in this folder

| File | Role |
|------|------|
| `single_stage_finetune.py` | Entry point for one-shot LoRA fine-tuning. Defines `QwenVLLMFineTuner`, parses CLI, runs: model load -> LoRA prep -> data load -> train -> save (optional merge). Optional `--eval-after-train` runs `run_eval` per benchmark via `single_stage_helpers.run_post_training_eval`. |
| `multi_stage_finetune.py` | Entry point for multi-stage LoRA. Defines same `QwenVLLMFineTuner` (inlined, no import from single_stage). Implements all-in-one (`--stage-data`) and resumable (`--stage` + `--continue-from-lora`) modes; calls `multi_stage_finetune()` for all-in-one. |
| `README.md` | This documentation. |

### End-to-end workflow (high-level)

```
Versioned JSONL training data (+ optional manifest.json)
       |
       v
+------+------+
| single_stage |                    +------------------+
| or multi    | --(stage data)-->  | Model registry   |
| _finetune   |                    | (model key/size) |
+------+------+                    +------------------+
       |
       v
+------------------+     +------------------+
| ModelLoader      | --> | Base model +     |
| (Unsloth/standard)|     | tokenizer        |
+------------------+     +------------------+
       |
       v
+------------------+     +------------------+
| LoRAPreparer     | --> | PEFT/LoRA model  |
+------------------+     +------------------+
       |
       v
+------------------+     +------------------+
| DataLoader       | --> | Tokenized        |
| + DataFormatter  |     | train/test split |
+------------------+     +------------------+
       |                         |
       |                         +--> difficulty buckets
       |                               (easy / medium / hard)
       |                               from context_length
       v
+------------------+
| Train ordering   |  <-- --learning-type
| strategy         |      random-shuffled / blocked-learning /
+------------------+      interleaved-learning / curriculum /
       |                  blocked-curriculum / interleaved-curriculum
       v
+------------------+     +------------------+
| Trainer          | --> | checkpoints/,    |
| (SFTTrainer or   |     | lora/, logs/     |
|  CustomTrainer)  |     | under run dir    |
+------------------+     +------------------+
       |
       v
[Optional, single-stage] --merge-lora inside train(): merged/ (vLLM-ready)
       |
       v
[Optional, single-stage] --eval-after-train in main():
  resolve_post_train_eval_checkpoint_path + run_post_training_eval
  (subprocess: python -m src.eval.run_eval per bench; retry once, then skip)
       |
       v
Post-train smoke test: test_model(one prompt) in main()
```

See **`docs/POST_TRAIN_EVAL_WORKFLOW.md`** for checkpoint choice (`best` vs `final`), bench lists, and artifact output under `eval/`.

### Single-stage workflow (step-by-step)

1. **CLI** (`single_stage_finetune.py` `main()`): Parse `--data`, `--output`, `--epochs`, `--model-size`, `--model-version`, `--model-key`, `--merge-lora`, `--learning-rate`, `--weight-decay`, `--learning-type`, `--easy-context-max`, `--medium-context-max`, etc. If `--list-models`, print registry and exit. If `--test-only`, load model (and optional LoRA), run test prompts and exit.
2. **Tuner init** (`QwenVLLMFineTuner.__init__`): `get_registry()` -> select model by key or (version, size). `ModelLoader(model_name, torch_dtype, gpu_memory, use_4bit, fallback_models).load_with_unsloth()` or `load_standard()` -> `self.model`, `self.tokenizer`.
3. **Prepare for training**: `prepare_model_for_training()` -> `LoRAPreparer(...).prepare()` -> LoRA adapters applied (Unsloth or PEFT).
4. **Load data**: `load_training_data(data_file)` -> `DataLoader(tokenizer, use_4bit).load_and_tokenize(...)` -> load JSONL, 90/10 train/test split, tokenize with `DataFormatter.format_training_data`, max_length 1024 (4-bit) or 2048, then apply train ordering from `--learning-type` -> `DatasetDict`.
5. **Resolve output**: `resolve_output_dir(output_dir, model_config)` (from `single_stage_helpers`): if `output_dir` is None -> `get_artifact_run_dir(model_name, run_prefix="rabit0-v1", ...)` -> `artifacts/<model_name>/rabit0-v1-run1`, run2, ...
6. **Train** (returns `(trainer, dirs)`): `TrainingConfig.setup_single_process_environment()`, `TrainingConfig.create_accelerate_config(dtype)`. Eval settings from `TrainingConfig.get_eval_config()` (eval_steps, per_device_eval_batch_size, max_eval_samples). Save steps are set to a multiple of eval_steps so `load_best_model_at_end` is valid. Build `TrainingArguments`, then either `SFTTrainer` (Unsloth) or `CustomTrainer` (standard). `trainer.train(resume_from_checkpoint=...)` -> `trainer.save_model(dirs["lora"])`, `tokenizer.save_pretrained(dirs["lora"])`, `write_lora_readme(...)`. Optional `resume_from_checkpoint`: `True` (latest in output dir) or path to a checkpoint dir. If `--merge-lora`, `_merge_lora_adapters(lora_dir, merged_output_dir=dirs["merged"])`.
7. **Optional post-train benchmarks** (`main()`, single-stage only): If `--eval-after-train`, `resolve_post_train_eval_checkpoint_path(dirs, --eval-checkpoint-source, best_model_checkpoint=trainer.state.best_model_checkpoint)` then `run_post_training_eval(path, benches, gpu=..., dtype=..., use_4bit=...)`. Each bench runs `python -m src.eval.run_eval` in a subprocess; one retry on failure, then skip with a warning. See `docs/POST_TRAIN_EVAL_WORKFLOW.md`.
8. **Post-train smoke test**: `test_model` with one fixed prompt and response printed.

### Multi-stage workflow (step-by-step)

**All-in-one mode** (`--stage-data file1.jsonl file2.jsonl ...`):

1. **CLI**: Parse args. If no `--output`, `output_base = get_artifact_run_dir("rabit0-multistage-vllm", run_prefix="rabit0-v1", update_latest=True)`.
2. **multi_stage_finetune()**: For each stage index and data path:
   - **Stage 1**: `tuner.prepare_model_for_training()` (fresh LoRA), `tuner.load_training_data(data_path)` -> dataset. Train with `get_stage_learning_rate(1)` (2e-4), save to `<output_base>/stage_1`. Keep `previous_stage_dir` and `previous_train_datasets`.
   - **Stage 2+**: `_load_existing_lora_into_tuner(tuner, previous_stage_dir)` via `multi_stage_helpers.load_existing_lora()` (base model + PeftModel.from_pretrained(continue_from_lora)). Load current stage data; if `replay_ratio > 0`, `create_mixed_dataset(previous_train_datasets, current, replay_ratio)`. Train with stage LR (stage 2: 5e-5, stage 3+: 1e-5), save to `<output_base>/stage_<N>`. Append current train to `previous_train_datasets`.
3. If `--merge-lora`, merge from final stage dir.

**Resumable mode** (`--stage N --continue-from-lora DIR --data file.jsonl --output DIR`):

1. **CLI**: Requires `--output`. Load tuner, then `_load_existing_lora_into_tuner(tuner, args.continue_from_lora)`.
2. Load data for this stage only; no replay.
3. Train with `get_stage_learning_rate(args.stage)`, save to `args.output`, optionally merge if `--merge-lora`.

**Test-only** (`--test-only`): Same as single-stage: build tuner, optionally load LoRA from `--continue-from-lora`, run fixed test prompts.

### Dependencies (internal)

| Module | Used by | Purpose |
|--------|---------|--------|
| `src.infra.model_registry` | Both scripts | Model selection (key or version+size), fallback chain, default config. |
| `src.infra.project_paths` | Both scripts | Paths and naming (`generate_model_name`, `get_model_path` for merge). |
| `src.utilities.training_config` | Both scripts | Torch dtype, optimizer config, single-process env, accelerate config, eval defaults (eval_steps, max_eval_samples). |
| `src.utilities.training_logger` | Both scripts | Logging and section headers. |
| `src.utilities.model_loader` | Both scripts | Load base model + tokenizer (Unsloth or standard, with fallbacks). |
| `src.utilities.data_formatter` | Both scripts | Format messages for tokenizer (Qwen chat template). |
| `src.utilities.data_loader` | Both scripts | Load JSONL, train/test split, tokenize. |
| `src.utilities.lora_preparer` | Both scripts | Apply LoRA (Unsloth or PEFT). |
| `src.utilities.single_stage_helpers` | single_stage only | `get_default_learning_rate`, `get_default_weight_decay`, `resolve_output_dir`, `resolve_post_train_eval_checkpoint_path`, `supported_eval_benches`, `run_post_training_eval`, checkpoint patching, save/eval alignment. |
| `src.utilities.multi_stage_helpers` | multi_stage only | `get_stage_learning_rate`, `create_mixed_dataset`, `load_existing_lora`. |
| `src.utilities.artifact_paths` | single_stage (via helpers), multi_stage | `get_artifact_run_dir`, `ensure_artifact_run_subdirs`, `resolve_latest_artifact` (doc only). |

### Detailed report summary

| Aspect | Single-stage | Multi-stage |
|--------|--------------|-------------|
| **Input** | One JSONL file (`--data`). | Multiple JSONL files (`--stage-data`) or one per stage with `--stage` + `--data`. |
| **Output (default)** | `artifacts/<model_name>/rabit0-v1-run1`, run2, ... (e.g. `artifacts/qwen2.5-7b/rabit0-v1-run1`). | `artifacts/rabit0-multistage-vllm/rabit0-v1-run1` with `stage_1`, `stage_2`, ... inside. |
| **Output (explicit)** | `--output <dir>`. | `--output <base_dir>` (all-in-one) or `--output <stage_dir>` (resumable). |
| **Latest pointer** | Yes (`<model_name>/rabit0-v1-latest`). | Yes when output not specified. |
| **Learning rate** | Single: default 2e-5 (`get_default_learning_rate`). | Stage 1: 2e-4; Stage 2: 5e-5; Stage 3+: 1e-5. |
| **Weight decay** | Single: default 0.0 (`get_default_weight_decay`, AdamW). | Not exposed on multi-stage CLI (HF defaults apply in `TrainingArguments` unless extended). |
| **Replay** | N/A. | Optional: `--replay-ratio` mixes fraction of previous stage data into next. |
| **Merge** | `--merge-lora` after training. | `--merge-lora` after last stage (all-in-one or resumable). |
| **Resumable** | Yes: `--resume-from-checkpoint` (latest in output dir) or `--resume-from-checkpoint <path>`. | Yes: `--stage N --continue-from-lora DIR --data file.jsonl --output DIR`. |
| **Test-only** | `--test-only` [path or flag]. | `--test-only` with optional `--continue-from-lora` to test a trained LoRA. |
| **Post-train benchmark eval** | Optional `--eval-after-train` (+ checkpoint source, bench list, eval GPU/dtype). Runs after training, before `test_model`. | Not built into CLI (run `run_eval` manually on stage LoRA dirs). |

---

## When to use single-stage vs multi-stage

- **Single-stage**: One dataset, one learning rate (default 2e-5) and optional AdamW weight decay via `--weight-decay` (default 0.0). Use when you have a single training set and want a single run. Output (when `--output` is omitted) goes to `artifacts/<model_name>/rabit0-v1-run1`, run2, ... (e.g. `artifacts/qwen2.5-7b/rabit0-v1-run1`). You can resume an interrupted run with `--resume-from-checkpoint` (same `--output` and `--data`).
- **Multi-stage**: Multiple datasets in sequence; each stage continues from the previous LoRA. Use for curriculum (e.g. basics then advanced) or phased data. Stage 1 uses 2e-4, stage 2 uses 5e-5, stage 3+ uses 1e-5. Optional replay mixes a fraction of previous stage data into the next to reduce forgetting. Output (when `--output` is omitted) goes to `artifacts/rabit0-multistage-vllm/rabit0-v1-run1` with `stage_1`, `stage_2`, ... inside each run.

---

## Input and output examples

### Single-stage

| | |
|---|---|
| **Command** | `poetry run python src/training/single_stage_finetune.py --data combine_output/dpv_v1_20260325_091340_5102ffcc/all_training_data.jsonl --model-key qwen2.5-7b --epochs 3 --batch-size 1 --gradient-accumulation-steps 8 --merge-lora` |
| **Input** | `combine_output/dpv_v1_.../all_training_data.jsonl` (JSONL: one object per line with `messages` array). `manifest.json` in the same directory is auto-detected and copied to `trainable_data/`. Base model: registry entry for `qwen2.5-7b`. |
| **Output (default, no --output)** | `artifacts/qwen2.5-7b/rabit0-v1-run18/`: `checkpoints/`, `lora/`, `merged/`, `logs/`, `trainable_data/` (with `manifest.json`). Subsequent runs: `rabit0-v1-run19/`, etc. `artifacts/qwen2.5-7b/rabit0-v1-latest` points to the latest run dir. |
| **Output (with --output ./my-run)** | `./my-run/`: same layout; no artifact run numbering or `latest` marker. |
| **Data provenance** | `trainable_data/manifest.json` records `version_id`, `data_points_dir_sha256`, `output_file_sha256`, and the full `datapoints` list. Query with `jq .version_id artifacts/qwen2.5-7b/rabit0-v1-run18/trainable_data/manifest.json`. |

### Multi-stage

| | |
|---|---|
| **Command** | `poetry run python src/training/multi_stage_finetune.py --stage-data stage1_data.jsonl stage2_data.jsonl --model-size 8b --epochs 3 --batch-size 2 --merge-lora` |
| **Input** | `stage1_data.jsonl`, `stage2_data.jsonl` (JSONL with `messages`). Base model from registry for the given size. |
| **Output (default, no --output)** | `artifacts/rabit0-multistage-vllm/rabit0-v1-run1/` with `stage_1/`, `stage_2/` (each: LoRA adapters, tokenizer, logs). Next run: `rabit0-v1-run2/`, etc. `artifacts/rabit0-multistage-vllm/rabit0-v1-latest` points to latest run. |
| **Output (with --output ./multistage-out)** | `./multistage-out/`: same layout (`stage_1/`, `stage_2/`, ...); no artifact numbering. |

---

## Output structure

When you do **not** pass `--output`, runs are stored under `artifacts/<model_name>/rabit0-v1-run1`, `rabit0-v1-run2`, ... and a `<model_name>/rabit0-v1-latest` pointer is updated (symlink or fallback file). These paths are for final finetune outputs; logs use a separate folder.

### Single-stage (no `--output`)

```
artifacts/
  qwen2.5-7b/
    rabit0-v1-latest    -> rabit0-v1-run3 (symlink or fallback file)
    rabit0-v1-run1/
      checkpoints/      # Trainer checkpoints (e.g. checkpoint-100, checkpoint-200); used by --resume-from-checkpoint
      lora/             # Final LoRA adapters (after training completes)
      merged/           # Merged model (if --merge-lora)
      adapter_config.json, adapter_model.safetensors, tokenizer*.json  # when saved at run root
      logs/             # TensorBoard logs
    rabit0-v1-run2/
      ...
    rabit0-v1-run3/
      ...
```

Terminal logs for a run are written to `logs/<model_name>_<run_id>.log` (e.g. `logs/qwen2.5-7b_rabit0-v1-run12.log`). When you resume with `--resume-from-checkpoint`, new checkpoints and final LoRA/merged outputs continue under the same run dir, and the same log file is used.

### Multi-stage (no `--output`)

```
artifacts/
  rabit0-multistage-vllm/
    rabit0-v1-latest    -> rabit0-v1-run2
    rabit0-v1-run1/
      stage_1/
        adapter_config.json
        adapter_model.safetensors
        tokenizer*.json
        logs/
      stage_2/
        ...
    rabit0-v1-run2/
      stage_1/
      stage_2/
      ...
```

### Resolving latest in code

```python
from src.utilities.artifact_paths import resolve_latest_artifact

# Single-stage: path to latest run dir (e.g. .../qwen2.5-7b/rabit0-v1-run3)
latest_run = resolve_latest_artifact("qwen2.5-7b", run_prefix="rabit0-v1")

# Multi-stage: path to latest run dir; final stage is e.g. latest_run / "stage_2"
latest_run = resolve_latest_artifact("rabit0-multistage-vllm", run_prefix="rabit0-v1")
```

If you pass `--output`, that path is used as-is and no artifact run numbering or `latest` is applied.

---

## Training data format

Training data is **JSONL**: one JSON object per line. Each object must have a **`messages`** array. Each message has:

- **`role`**: `"system"`, `"user"`, or `"assistant"`
- **`content`**: string (the message text)

Example line:

```json
{"messages": [{"role": "user", "content": "How do I test for SQL injection?"}, {"role": "assistant", "content": "..."}]}
```

The pipeline uses Qwen chat template tokens (`<|im_start|>`, `<|im_end|>`). Data is split 90% train / 10% validation; max sequence length is 2048 (1024 when using 4-bit). For full schema, sources, and how to generate combined data, see **`docs/TRAINING_DATA_README.md`**.

---

## Single-stage training

One dataset, one learning rate. Default output when `--output` is omitted: `artifacts/<model_name>/rabit0-v1-run1`, run2, ... (e.g. `artifacts/qwen2.5-7b/rabit0-v1-run1`).

### Quick start

```bash
# Pass a versioned data file; manifest.json in the same folder is auto-linked into trainable_data/
poetry run python src/training/single_stage_finetune.py \
    --data combine_output/dpv_v1_20260325_091340_5102ffcc/all_training_data.jsonl \
    --model-key qwen2.5-7b \
    --epochs 3 \
    --batch-size 1 \
    --gradient-accumulation-steps 8 \
    --merge-lora
```

### Post-training benchmark eval (single-stage)

If you pass **`--eval-after-train`**, training will spawn **`python -m src.eval.run_eval`** for each selected benchmark after **`train()`** succeeds and the LoRA is saved. Checkpoint choice: **`--eval-checkpoint-source best`** (default; best eval loss on disk or **`lora/`**) vs **`final`** (latest **`checkpoint-*`**). Benches: omit **`--eval-bench`** for all supported names, or pass a subset. Failures: one automatic retry per bench, then skip with a warning. **`coconot`** is skipped when **`OPENAI_API_KEY`** is unset.

Full workflow (diagram, flags, output paths): **`docs/POST_TRAIN_EVAL_WORKFLOW.md`**.

```bash
poetry run python src/training/single_stage_finetune.py \
  --data combine_output/dpv_v1_20260325_091340_5102ffcc/all_training_data.jsonl \
  --model-key qwen2.5-7b \
  --epochs 3 \
  --batch-size 1 \
  --gradient-accumulation-steps 8 \
  --eval-after-train \
  --eval-gpu cuda:0
```

### All single-stage options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--data` | str | `combine_output/all_training_data.jsonl` | Path to training JSONL. |
| `--output` | str | None | Output directory. If omitted, uses `artifacts/<model_name>/rabit0-v1-run1`, run2, ... |
| `--epochs` | int | 3 | Number of training epochs. |
| `--batch-size` | int | 2 | Per-device batch size. |
| `--gradient-accumulation-steps` | int | 8 | Gradient accumulation steps (effective batch = batch_size * this). |
| `--learning-rate` | float | 2e-5 | Learning rate. |
| `--weight-decay` | float | 0.0 | AdamW weight decay (`>= 0`). |
| `--model-size` | choice | 8b | Model size: 7b, 8b, 14b, 30b. |
| `--model-version` | choice | qwen3 | Base model: qwen3, qwen2.5. |
| `--model-key` | str | None | Registry key (e.g. qwen2.5-7b); overrides version/size. |
| `--dtype` | choice | bfloat16 | bfloat16, float16, float32 (training and post-train eval base load). |
| `--gpu-memory` | int | 40 | GPU memory in GiB (used for defaults). |
| `--no-unsloth` | flag | False | Disable Unsloth. |
| `--use-4bit` | flag | False | Use 4-bit quantization (QLoRA) for large models. |
| `--use-8bit` | flag | False | Use 8-bit quantization to reduce VRAM usage. |
| `--lr-scheduler-type` | choice | cosine | LR scheduler type: `constant`, `linear`, `cosine`, `step` (mapped internally to HF scheduler names). |
| `--merge-lora` | flag | False | Merge LoRA into base model after training (vLLM-ready). |
| `--eval-steps` | int | 500 (TrainingConfig) | Run evaluation every N steps. |
| `--eval-batch-size` | int | same as `--batch-size` | Per-device eval batch size. |
| `--max-eval-samples` | int | None (full set) | Cap validation set size for each eval (faster evals, less OOM risk). |
| `--resume-from-checkpoint` | optional path | None | Resume training: no value = latest checkpoint in output dir; or path to checkpoint dir (e.g. `.../checkpoints/checkpoint-100`). Use same `--output` and `--data` as the run that created the checkpoint. |
| `--easy-context-max` | int | 512 | Context-length threshold for easy datapoints. |
| `--medium-context-max` | int | 1024 | Context-length threshold for medium datapoints (hard is above this). |
| `--learning-type` | choice | curriculum | Train data ordering mode: `random-shuffled`, `blocked-learning`, `interleaved-learning`, `curriculum`, `blocked-curriculum`, `interleaved-curriculum`. |
| `--list-models` | flag | False | List registry models and exit. |
| `--test-only` | str | None | Run test prompts only (no training). If set to a checkpoint path (e.g. `.../checkpoints/checkpoint-1500`), loads that LoRA and runs test prompts; use `--model-key` to match the run (e.g. `--model-key qwen2.5-7b`). |
| `--eval-after-train` | flag | False | After successful training, run `run_eval` once per benchmark (subprocess). Incompatible with `--test-only`. |
| `--eval-checkpoint-source` | choice | `best` | `best` (HF best checkpoint or `lora/`) or `final` (latest `checkpoint-*`). |
| `--eval-bench` | str... | all | Benchmark names (space-separated in shell). Default: all supported by `run_eval`. |
| `--eval-gpu` | str | None | Passed as `run_eval --gpu` (e.g. `cuda:0`, `cpu`). |
| `--eval-use-4bit` | flag | False | Pass `--use-4bit` to `run_eval`. |
| `--eval-task-count` | int | 3 | First N tasks per benchmark (from `benchmarks_catalog.yaml`). If a bench has fewer than N tasks, all available tasks are used. |

Save steps are set automatically so they are a multiple of `eval_steps` (required when `load_best_model_at_end=True`). With default `--eval-steps 500`, checkpoints are saved every 500 steps. The value is logged at startup as "Save steps".

### Single-stage examples

```bash
# 7B for vLLM (e.g. 20GB GPU), auto output dir
poetry run python src/training/single_stage_finetune.py \
  --data redteam_training_data.jsonl \
  --model-size 7b \
  --model-version qwen3 \
  --epochs 3 \
  --batch-size 2 \
  --gpu-memory 20 \
  --merge-lora

# 14B, explicit output
poetry run python src/training/single_stage_finetune.py \
  --data redteam_training_data.jsonl \
  --output ./rabit0-v1.0-qwen3-14b-vllm \
  --model-size 14b \
  --epochs 3 \
  --batch-size 1 \
  --gpu-memory 40 \
  --merge-lora

# No merge; merge later with merge_lora_for_vllm.py
poetry run python src/training/single_stage_finetune.py \
  --data redteam_training_data.jsonl \
  --output ./rabit0-v1.0-qwen3-7b-vllm \
  --epochs 3
# Then:
# python src/main/serve/merge_lora_for_vllm.py --adapter-path ./rabit0-v1.0-qwen3-7b-vllm --output-path ./rabit0-v1.0-qwen3-7b-vllm-merged

# Eval tuning: fewer evals (every 500 steps), larger eval batch, cap eval samples (faster, less OOM)
poetry run python src/training/single_stage_finetune.py \
  --data redteam_training_data.jsonl \
  --model-key qwen2.5-7b \
  --output ./artifacts/qwen2.5-7b/rabit0-v1-run13 \
  --epochs 3 --batch-size 1 \
  --eval-steps 500 --eval-batch-size 2 --max-eval-samples 500

# Resume from latest checkpoint in output dir (same --output and --data as original run)
poetry run python src/training/single_stage_finetune.py \
  --data redteam_training_data.jsonl \
  --model-key qwen2.5-7b \
  --output ./artifacts/qwen2.5-7b/rabit0-v1-run12 \
  --epochs 3 --batch-size 1 \
  --resume-from-checkpoint

# Resume from a specific checkpoint directory
poetry run python src/training/single_stage_finetune.py \
  --data redteam_training_data.jsonl \
  --model-key qwen2.5-7b \
  --output ./artifacts/qwen2.5-7b/rabit0-v1-run12 \
  --epochs 3 --batch-size 1 \
  --resume-from-checkpoint ./artifacts/qwen2.5-7b/rabit0-v1-run12/checkpoints/checkpoint-100

# Test a specific checkpoint (no training; runs fixed security test prompts)
poetry run python src/training/single_stage_finetune.py --test-only \
  ./artifacts/qwen2.5-7b/rabit0-v1-run12/checkpoints/checkpoint-1500 \
  --model-key qwen2.5-7b
# Add --use-4bit or --use-8bit if you need to reduce VRAM.
# Do not pass both flags at the same time.

# (MLflow staging examples removed.)
```

### Example run — 10 data points (versioned data)

Command:

```bash
poetry run python src/training/single_stage_finetune.py \
    --data combine_output/dpv_v1_20260325_091340_5102ffcc/all_training_data.jsonl \
    --model-key qwen2.5-7b \
    --epochs 3 \
    --batch-size 1 \
    --gradient-accumulation-steps 8 \
    --merge-lora
```

Terminal output (10 examples total: 9 train, 1 eval after 90/10 split; 27 forward steps, logging every 10):

```
================================================================================
Training Rabit0: Red Team Security AI (vLLM-Compatible)
================================================================================
Base Model              : Qwen/Qwen2.5-Coder-7B-Instruct
Model Key               : qwen2.5-7b
Model Version           : QWEN2.5
Model Size              : 7b
Output                  : Rabit0-v1.0-qwen2.5-7b-vllm
Architecture            : Dense
Precision               : bfloat16 (vLLM-compatible)
Quantization            : None (full precision for vLLM)
Unsloth Optimization    : True
License                 : Apache 2.0 (Fully unrestricted for SaaS)
Expected VRAM           : 14GB
================================================================================
Data manifest found     : /home/rabit0_rsr/combine_output/dpv_v1_20260325_091340_5102ffcc/manifest.json
Unsloth: Loading Qwen/Qwen2.5-Coder-7B-Instruct ...
Unsloth: bfloat16 support detected.
Starting training...
Output directory        : /home/rabit0_rsr/artifacts/qwen2.5-7b/rabit0-v1-run18
Epochs                  : 3
Batch size              : 1
Gradient accumulation   : 8
Effective batch size    : 8
Learning rate           : 2e-05
Merge LoRA after training: True
Eval steps              : 500
Per-device eval batch size: 1
Save steps              : 500
Trainable data trace    : /home/rabit0_rsr/artifacts/qwen2.5-7b/rabit0-v1-run18/trainable_data
Data version manifest   : /home/rabit0_rsr/artifacts/qwen2.5-7b/rabit0-v1-run18/trainable_data/manifest.json
Eval samples            : full validation set (1)
Training in progress...
Monitor with            : tensorboard --logdir /home/rabit0_rsr/artifacts/qwen2.5-7b/rabit0-v1-run18/logs

{'loss': 2.4128, 'grad_norm': 0.8234, 'learning_rate': 1.11e-05, 'epoch': 1.11, 'step': 10}
{'loss': 2.1053, 'grad_norm': 0.6891, 'learning_rate': 2.22e-06, 'epoch': 2.22, 'step': 20}
{'eval_loss': 2.0341, 'eval_runtime': 0.45, 'eval_samples_per_second': 2.22,
 'eval_steps_per_second': 2.22, 'epoch': 3.0, 'step': 27}
{'train_runtime': 14.31, 'train_samples_per_second': 1.89, 'train_steps_per_second': 1.89,
 'train_loss': 2.1744, 'epoch': 3.0, 'step': 27}

Saving LoRA adapters to /home/rabit0_rsr/artifacts/qwen2.5-7b/rabit0-v1-run18/lora
SUCCESS: Training complete!
Model saved to          : /home/rabit0_rsr/artifacts/qwen2.5-7b/rabit0-v1-run18/lora
Merging LoRA adapters into base model (for vLLM compatibility)...
Saving merged model to /home/rabit0_rsr/artifacts/qwen2.5-7b/rabit0-v1-run18/merged
SUCCESS: Merged model saved! Use this with vLLM: /home/rabit0_rsr/artifacts/qwen2.5-7b/rabit0-v1-run18/merged
Merged model (vLLM-ready): /home/rabit0_rsr/artifacts/qwen2.5-7b/rabit0-v1-run18/merged
```

Resulting artifact folder:

```
artifacts/qwen2.5-7b/
  rabit0-v1-latest  ->  rabit0-v1-run18
  rabit0-v1-run18/
    trainable_data/
      manifest.json              <- dpv_v1_20260325_091340_5102ffcc (SHA + datapoints)
      trace_meta.json
      epoch_summary.json
      datapoint_usage.jsonl
      datapoint_exposure_score.csv
      datapoint_learning_score.csv
    checkpoints/                 <- empty (27 steps < save_steps=500)
    lora/
      adapter_config.json
      adapter_model.safetensors
      tokenizer.json / tokenizer_config.json
      README.md
    merged/
      config.json
      model-00001-of-00004.safetensors
      ...
      tokenizer.json
    logs/
      events.out.tfevents.*

logs/
  qwen2.5-7b_rabit0-v1-run18.log  <- full terminal capture
```

Verify which data version was used:

```bash
jq '{version_id, run_message, data_points_dir_sha256}' \
    artifacts/qwen2.5-7b/rabit0-v1-run18/trainable_data/manifest.json
# {
#   "version_id": "dpv_v1_20260325_091340_5102ffcc",
#   "run_message": "testing full pipeline with tools data sets",
#   "data_points_dir_sha256": "5102ffcc07dd9f87e76f0f1aad1777eba5c1852304576bc2562e459df17b1c99"
# }
```

---

## Multi-stage training

Progressive fine-tuning: stage 1 trains a fresh LoRA; stage 2+ load the previous stage LoRA and continue. Default output when `--output` is omitted: `artifacts/rabit0-multistage-vllm/rabit0-v1-run1` (with `stage_1`, `stage_2`, ... inside each run).

### Modes

1. **All-in-one** – `--stage-data file1.jsonl file2.jsonl` (optional `--output`). Runs all stages in one process; each stage writes to `<base>/stage_1`, `<base>/stage_2`, ...
2. **Resumable** – Run stage 1 with `--stage 1 --data file1.jsonl --output <dir>/stage_1`, then stage 2 with `--stage 2 --continue-from-lora <dir>/stage_1 --data file2.jsonl --output <dir>/stage_2`. Requires `--output` when using `--stage` and `--continue-from-lora`.

### Learning rates per stage

- Stage 1: 2e-4  
- Stage 2: 5e-5  
- Stage 3+: 1e-5  

### All multi-stage options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--stage-data` | str+ | None | Data files per stage (all-in-one mode). |
| `--stage` | int | None | Current stage number (1, 2, ...) for resumable mode. |
| `--continue-from-lora` | str | None | Path to previous stage LoRA directory (stage 2+). |
| `--data` | str | None | Single data file for current stage (with `--stage`). |
| `--output` | str | None | Base dir (all-in-one) or stage output dir (resumable). Omit for artifact. |
| `--replay-ratio` | float | 0.0 | Fraction of previous stage data to mix into next (0.0–1.0). |
| `--merge-lora` | flag | False | Merge LoRA after the last stage. |
| `--epochs` | int | 3 | Epochs per stage. |
| `--batch-size` | int | 2 | Batch size. |
| `--gradient-accumulation-steps` | int | 8 | Gradient accumulation steps. |
| `--model-size` | choice | 8b | 7b, 8b, 14b, 30b. |
| `--model-version` | choice | qwen3 | qwen3, qwen2.5. |
| `--model-key` | str | None | Registry key. |
| `--dtype` | choice | bfloat16 | bfloat16, float16, float32 (training and post-train eval base load). |
| `--gpu-memory` | int | 40 | GPU memory in GiB. |
| `--no-unsloth` | flag | False | Disable Unsloth. |
| `--use-4bit` | flag | False | 4-bit quantization (QLoRA). |
| `--list-models` | flag | False | List registry models and exit. |
| `--test-only` | flag | False | Run test prompts only; use `--continue-from-lora` to test a trained LoRA. |

### Multi-stage examples

```bash
# All-in-one, auto output (artifacts/rabit0-multistage-vllm/rabit0-v1-run1)
poetry run python src/training/multi_stage_finetune.py \
  --stage-data stage1.jsonl stage2.jsonl \
  --model-size 7b \
  --model-version qwen3 \
  --epochs 3 \
  --gpu-memory 20

# All-in-one with replay and merge
poetry run python src/training/multi_stage_finetune.py \
  --stage-data stage1.jsonl stage2.jsonl \
  --output ./rabit0-multistage-vllm \
  --replay-ratio 0.2 \
  --merge-lora \
  --epochs 3

# Resumable: stage 1 then stage 2
poetry run python src/training/multi_stage_finetune.py \
  --stage 1 \
  --data stage1.jsonl \
  --output ./rabit0-multistage-vllm/stage_1

poetry run python src/training/multi_stage_finetune.py \
  --stage 2 \
  --continue-from-lora ./rabit0-multistage-vllm/stage_1 \
  --data stage2.jsonl \
  --output ./rabit0-multistage-vllm/stage_2 \
  --merge-lora

# Test with a trained LoRA
poetry run python src/training/multi_stage_finetune.py --test-only \
  --continue-from-lora ./rabit0-multistage-vllm/stage_1 --model-size 7b
```

---

## Resource requirements

### GPU memory (training)

| Model | Approx. VRAM | Typical use |
|-------|--------------|-------------|
| 7B | ~14 GB | Fits on 20 GB GPU (e.g. RTX 4000, 3090). |
| 8B | ~16 GB | 20 GB+ GPU. |
| 14B | ~28 GB | 40 GB+ GPU (e.g. A100 40GB). |
| 30B | ~60 GB+ | 80 GB+ or use `--use-4bit` (QLoRA) / `--use-8bit`. |

### GPU memory (vLLM serving, merged model)

- 7B: ~14 GB (fits in 20 GB GPU).  
- 14B: ~28 GB (40 GB+ or quantization).

### Training time (rough)

- 7B on 1x RTX 4000 20 GB: about 3–4 hours.  
- 7B on 1x A100 40 GB: about 2–3 hours.  
- 14B on 1x A100 40 GB: about 4–5 hours.  

Times depend on dataset size, batch size, and gradient accumulation.

---

## Environment and setup

- **Python**: Use the project environment (e.g. `poetry install` from repo root). See `requirements.txt` and `pyproject.toml`.
- **CUDA**: Required for GPU training; ensure drivers and PyTorch CUDA build match.
- **Unsloth**: Used when available for faster training; disable with `--no-unsloth` if you hit issues.
- **Single process**: Scripts set up single-process training by default. For multi-GPU, use `accelerate launch` and the same script + args.
- **Artifacts and latest**: On Windows, if creating a symlink for `<run_prefix>-latest` fails (e.g. permissions), the code falls back to a file containing the run directory name; `resolve_latest_artifact()` still works.

---

## Troubleshooting and notes

- **Why is training slow (e.g. ~22 s/step)?** Each step runs 8 micro-batches (gradient accumulation) with sequences up to 2048 tokens on a 7B model, plus gradient checkpointing and Unsloth gradient offload to fit 19GB VRAM. To speed up when you have headroom: (1) Try `--batch-size 2` (same effective batch with `--gradient-accumulation-steps 4`) so fewer steps and better GPU utilization; (2) Use `--eval-steps 1000` and `--max-eval-samples 500` so evaluation runs less often and on a subset; (3) If you hit OOM after increasing batch size, revert to `--batch-size 1` and `--gradient-accumulation-steps 8`. Expect roughly 60 hours for 9,903 steps at ~22 s/step on a single RTX 4000-class GPU.
- **Out of memory**: Reduce `--batch-size` (e.g. 1), increase `--gradient-accumulation-steps` to keep effective batch size, or use `--use-4bit` / `--use-8bit` for large models. Do not enable both quantization flags together.
- **Unsloth errors**: Run with `--no-unsloth` to use the standard Hugging Face Trainer.
- **Explicit output**: Use `--output` to write to a fixed path (e.g. CI or custom layout); artifact run numbering and `latest` are then skipped.
- **Multi-stage resumable**: When using `--stage` and `--continue-from-lora`, `--output` is required so the script knows where to save the current stage.
- **Merge after training**: If you did not use `--merge-lora`, merge later with `src/main/serve/merge_lora_for_vllm.py` (see single-stage examples above).
- **TensorBoard**: Metrics and events are under `<output_dir>/logs` (e.g. `artifacts/qwen2.5-7b/rabit0-v1-run1/logs`). Run `tensorboard --logdir <output_dir>/logs` to monitor training.
- **Text log file**: Training log is written to `logs/<model_name>_<run_id>.log` (e.g. `logs/qwen2.5-7b_rabit0-v1-run12.log`). When resuming, the same run dir and log file are used.
- **Long runs / OOM during eval**: Use `--eval-steps 500` (or higher), `--eval-batch-size 2` (if VRAM allows), and `--max-eval-samples 500` to run evaluation less often and on a subset so training can finish.
- **Resume after interrupt**: Use `--resume-from-checkpoint` with the same `--output` and `--data` as the interrupted run. No value = resume from latest checkpoint in that output dir; or pass the path to a specific checkpoint dir (e.g. `.../checkpoints/checkpoint-100`).
- **Save steps vs eval steps**: Checkpoints are saved at a step count that is a multiple of `eval_steps` so that `load_best_model_at_end` can load the best model by eval loss. With default `eval_steps=500`, save happens every 500 steps. You do not set save frequency manually.
- **Resume: "arguments do not match" warning**: When resuming from a checkpoint created with older settings (e.g. eval_steps=50, save_steps=100), the Trainer may warn that args in the checkpoint differ from current args. This is expected: the *current* run uses your current eval_steps/save_steps (e.g. 500/500). Training continues correctly; you can ignore the warning.

---

## Related documentation

| Topic | Location |
|-------|----------|
| Training data format, sources, combine pipeline | `docs/TRAINING_DATA_README.md` |
| Project layout, config, paths | `docs/PROJECT_STRUCTURE.md`, `src/infra/project_paths.py` |
| Model registry and available models | `src/config/model/`, `src/infra/model_registry.py` |
| Artifact paths and `latest` | `src/utilities/artifact_paths.py` |
| Serving (merge LoRA, vLLM server) | `src/main/serve/` |
| SecQA evaluation (standalone eval script) | `docs/SECQA_EVAL.md`, `src/eval/eval_secqa_checkpoints.py` |
| Post-train benchmark eval (train then `run_eval`) | `docs/POST_TRAIN_EVAL_WORKFLOW.md`, `src/eval/README.md` |
