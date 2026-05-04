# Source Directory (src)

This directory contains the Rabit0 project source code, organized into subdirectories.

## Directory Structure

```
src/
├── config/data_prep/        # Data prep config (datapoints.yaml, system_prompts, etc.)
├── data_prep/               # Data preparation and schema
├── data_creation/           # Dataset generation
├── eval/                    # Evaluation
├── training/                # Model training (see training/README.md)
│   ├── README.md            # Training scripts, options, artifact layout
│   ├── single_stage_finetune.py  # Primary single-stage training
│   └── multi_stage_finetune.py   # Multi-stage LoRA chaining
├── serve/                   # Model serving
├── infra/                   # Infrastructure and registry
├── utilities/               # Reusable utilities
├── docker/                  # Docker management scripts (shell)
└── review/                  # Review and analysis
```

## Main Scripts (Production)

### Training
- **`training/single_stage_finetune.py`** - **PRIMARY SINGLE-STAGE TRAINING SCRIPT**
  - Refactored to follow SOLID and DRY principles
  - Uses ModelRegistry (SOLID-compliant)
  - Supports all Qwen models via `src/config/model/models.yaml`
  - vLLM-compatible output
  - When `--output` is omitted, saves to `artifacts/<model_name>/rabit0-v1-run1`, `rabit0-v1-run2`, ... with a `rabit0-v1-latest` marker
  - **Resume**: `--resume-from-checkpoint` (latest in output dir) or `--resume-from-checkpoint <path>`; use same `--output` and `--data` as the original run
  - **Learning/data-loading strategies**: `--learning-type random-shuffled|blocked-learning|interleaved-learning|curriculum|blocked-curriculum|interleaved-curriculum`
  - **Quantization**: `--use-4bit` (QLoRA) or `--use-8bit` (mutually exclusive)
  - **LR scheduler**: `--lr-scheduler-type constant|linear|cosine|step`
  - **Optimizer**: `--learning-rate` (default 2e-5), `--weight-decay` (default 0.0, AdamW)
  - Usage: `python src/training/single_stage_finetune.py --data combine_output/all_training_data.jsonl --model-key qwen2.5-7b --epochs 3`
- **`training/multi_stage_finetune.py`** - Multi-stage LoRA chaining across multiple data files; when `--output` is omitted uses `artifacts/rabit0-multistage-vllm/rabit0-v1-runN`; supports `--test-only` and `--continue-from-lora`
- **`training/README.md`** - Full documentation: options, artifact output structure, data format, resource requirements, troubleshooting

### Data Preparation
Located in `data_prep/` (see **data_prep/README.md** for full details):
- **`combine_training_data.py`** - Combines training data from `data_points/` (JSON/YAML) into a single JSONL; supports `--list`, `--validate`, per-source selection, and snapshot switching by `--use-version-id` or `--use-data-points-sha`.
- **`data_schema.py`** - Pydantic models (SecurityScenario, QAScenario, PayloadScenario, TrainingExample) and generators that build training examples from each scenario type.
- **`json_source_loader.py`** - Loads and validates JSON/YAML files from `data_points/`; returns SecurityScenario, QAScenario, or PayloadScenario instances.
- **`data_versioning.py`** - Version artifact manager (manifest/checksums/archive), hash-based version reuse, `index.json` lookup (`sha <-> version_id`), and auto-extraction of archived `data_points/` when switching versions.

Legacy: RESK-FR Hugging Face transformer moved to **archived/dataset_transformer.py** (used only by archived/external_dataset.py).

### Serving
Located in `serve/`:
- **`serve_rabit0_vllm.py`** - vLLM OpenAI-compatible API server
- **`merge_lora_for_vllm.py`** - Merges LoRA adapters for vLLM deployment
- **`python_json_tool_parser.py`** - Custom vLLM tool parser plugin (extracts JSON tool calls from text)

### Evaluation
Located in `eval/`:
- **`secqa_eval.py`** - Shared SecQA evaluation utilities: prompt formatting, dataset loading, checkpoint discovery, result persistence, and the `run_secqa_on_model()` evaluation loop. Used by `training/single_stage_finetune.py` via `--secqa-eval` / `--secqa-run-dir` flags. See `docs/SECQA_EVAL.md` for full usage.
- **`evaluate_qwen3guard.py`** - Evaluates models via vLLM API (simple, focused)
- **`evaluate.py`** - Comprehensive evaluation framework (supports OpenAI API and local models)

## Infrastructure Scripts

Located in `infra/`:
- **`model_registry.py`** - Model configuration registry (SOLID)
- **`project_paths.py`** - Standardized path management (SOLID)
- **`validator.py`** - Data validation utilities
- **`taxonomy.py`** - Security taxonomy definitions
- **`tool_registry.py`** - Security tool registry
- **`mitre_owasp_mappings.py`** - MITRE ATT&CK and OWASP mappings
- **`refusal_coverage.py`** - Refusal coverage tracking
- **`specialized_templates.py`** - Template definitions

## Utility Scripts

Located in `utilities/`:

### Training Utilities (Active - Used by training scripts)
- **`training_config.py`** - Training configuration utilities (dtype conversion, optimizer config, environment setup)
- **`training_logger.py`** - Centralized logging utility (headers, info, success, warning, error messages)
- **`model_loader.py`** - Model loading utility (Unsloth and standard loading with fallback support)
- **`data_formatter.py`** - Data formatting utility (training data and prompt formatting for Qwen models)
- **`data_loader.py`** - Data loading utility (dataset loading and tokenization)
- **`lora_preparer.py`** - LoRA preparation utility (LoRA adapter setup for fine-tuning)
- **`lora_readme.py`** - Auto-generate LoRA README.md with training metadata (base model, hyperparameters, steps)
- **`artifact_paths.py`** - Artifact paths: `artifacts/<model_name>/rabit0-v1-run1`, `run2`, ... (finetune outputs; logs elsewhere), `ensure_artifact_run_subdirs(run_dir)` (checkpoints, lora, merged, logs), and `resolve_latest_artifact(model_name, run_prefix="rabit0-v1")`
- **`single_stage_helpers.py`** - Helpers for single-stage fine-tuning (e.g. `resolve_output_dir` using artifact paths when `--output` is omitted)
- **`multi_stage_helpers.py`** - Helpers for multi-stage LoRA chaining (stage LR, mixed dataset, load existing LoRA)
- **`colours.py`** - Terminal colour helpers for logging

## Data Generation Scripts

Located in `data_creation/`:
- **`generate_comprehensive_dataset.py`** - Comprehensive dataset generation (5,000+ examples)
- **`generate_premium_detailed.py`** - Premium detailed examples
- **`generate_premium_unrestriction.py`** - Premium unrestriction examples
- **`generate_unrestriction_batches.py`** - Batch unrestriction generation
- **`generate_refusal_only_dataset.py`** - Refusal-only dataset
- **`generate_5k_dataset.sh`** - Shell wrapper for dataset generation

## Docker Scripts

Located in `docker/`:
- **`docker-build.sh`** - Build Docker images
- **`docker-start.sh`** - Start containers
- **`docker-stop.sh`** - Stop containers
- **`docker-exec.sh`** - Execute commands in containers
- **`docker-logs.sh`** - View container logs
- **`setup-server.sh`** - Server setup

## Review Scripts

Located in `review/`:
- **`query_rabit0_model.py`** - Direct model querying (alternative to vLLM API)

## Import Paths

All imports have been updated to reflect the new directory structure:

### Main Scripts
- **Data Schema**: `from src.data_prep.data_schema import ...`

### Infrastructure
- **Taxonomy**: `from src.infra.taxonomy import ...`
- **Tool Registry**: `from src.infra.tool_registry import ...`
- **Validator**: `from src.infra.validator import ...`
- **MITRE/OWASP Mappings**: `from src.infra.mitre_owasp_mappings import ...`
- **Refusal Coverage**: `from src.infra.refusal_coverage import ...`
- **Model Registry**: `from src.infra.model_registry import ...`
- **Project Paths**: `from src.infra.project_paths import ...`

### Training Utilities
- **Training Config**: `from src.utilities.training_config import TrainingConfig`
- **Training Logger**: `from src.utilities.training_logger import Logger`
- **Model Loader**: `from src.utilities.model_loader import ModelLoader`
- **Data Formatter**: `from src.utilities.data_formatter import DataFormatter`
- **Data Loader**: `from src.utilities.data_loader import DataLoader`
- **LoRA Preparer**: `from src.utilities.lora_preparer import LoRAPreparer`
- **Artifact Paths**: `from src.utilities.artifact_paths import get_artifact_run_dir, resolve_latest_artifact`

## Notes

### Training
- **Single-stage**: Use **`training/single_stage_finetune.py`** for one data file; **multi-stage**: use **`training/multi_stage_finetune.py`** for chained LoRA stages.
- **Data ordering**: Single-stage uses `--learning-type` for train ordering strategy (`random-shuffled`, `blocked-learning`, `interleaved-learning`, `curriculum`, `blocked-curriculum`, `interleaved-curriculum`).
- **Resume**: Single-stage supports **`--resume-from-checkpoint`** (latest in output dir) or **`--resume-from-checkpoint <path>`**; use the same `--output` and `--data` as the run that created the checkpoint.
- **Output layout**: Omitting `--output` writes to **`artifacts/<model_name>/rabit0-v1-run1`**, `run2`, ... (e.g. `artifacts/qwen2.5-7b/rabit0-v1-run1`). Use **`resolve_latest_artifact("qwen2.5-7b", run_prefix="rabit0-v1")`** (single-stage) or **`resolve_latest_artifact("rabit0-multistage-vllm", run_prefix="rabit0-v1")`** (multi-stage) to get the latest run path in code.
- Full training docs (options, data format, resources, troubleshooting): **`training/README.md`**.
- Training scripts follow SOLID and DRY principles; utility classes are in `utilities/`.
- Model configuration is in **`src/config/model/models.yaml`**.

### Code Organization
- Paths are standardized via **`infra/project_paths.py`**
- Training utilities are modular and shared by single- and multi-stage scripts

### Serving
- The `python_json_tool_parser.py` is loaded as a plugin by vLLM via `--tool-parser-plugin` flag

### Import Structure
- Use package imports: `from src.data_prep.*`, `from src.infra.*`, `from src.utilities.<module> import <Class>`
- See `docs/SCRIPT_ANALYSIS.md` for detailed script analysis

