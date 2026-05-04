# Data Preparation (data_prep)

This folder contains the pipeline that loads training data from `data_points/` and produces a single JSONL file in OpenAI chat format for fine-tuning. Sources and related settings are configuration-driven via `src/config/data_prep/`.

## Contents

| File | Purpose |
|------|---------|
| **combine_training_data.py** | Entry point. Reads datapoints from `src/config/data_prep/datapoints.yaml`, combines them into one JSONL. Supports `--list`, `--validate`, `--no-data-version`, and switching source snapshots with `--use-version-id` or `--use-data-points-sha`. Data versioning is enabled by default. |
| **data_versioning.py** | Data versioning helpers. Creates version manifest/checksums/compressed snapshot, maintains `data_versions/index.json` (`sha -> version_id` and `version_id -> metadata`), reuses existing version ids when `data_points` hash is unchanged, and auto-extracts archives when switching if `data_points/` is missing. |
| **json_source_loader.py** | Loads JSON/YAML from `data_points/`, validates with Pydantic (SecurityScenario, QAScenario, PayloadScenario, ToolScenario), returns model instances. |
| **data_schema.py** | Pydantic models (Message, TrainingExample, SecurityScenario, QAScenario, PayloadScenario, ToolScenario). Loads system prompts, training categories, and quality guidelines from `src/config/data_prep/`. Generators turn each scenario type into a TrainingExample. |

## Configuration (src/config/data_prep/)

| File | Purpose |
|------|---------|
| **datapoints.yaml** | List of datapoints to combine: each has `id`, `path` (under `data_points/`), `loader`, and optional `scenario_type` (security, qa, payload, tool). |
| **system_prompts.yaml** | System prompts per scenario type (security, qa, payload, tool). |
| **training_categories.yaml** | Security/training categories and topics used in scenario generation. |
| **quality_guidelines.yaml** | Quality guidelines for generated assistant content. |

## Data flow

1. **Datapoints** are defined in `src/config/data_prep/datapoints.yaml` (path under `data_points/`, optional `scenario_type`).
2. **combine_training_data** loads the config and, for each datapoint, calls **json_source_loader** to read files (or directories), parse JSON/YAML, and validate each item as SecurityScenario, QAScenario, PayloadScenario, or ToolScenario.
3. **data_schema** generators convert each scenario type into a `TrainingExample` (system/user/assistant messages + metadata), using prompts and categories from config.
4. **combine_training_data** writes one JSON object per line (OpenAI-style `messages` + `metadata`) to the output JSONL.
5. **data_versioning** (default on) saves artifacts under `data_versions/<version_id>/`, updates `data_versions/index.json`, and reuses the same `version_id` when the `data_points` SHA256 has not changed.

## Source types

- **security** – Exploit/step-based scenarios (PortSwigger, Invicti, Juice Shop, n8n). Schema: `SecurityScenario`; generator: `generate_training_example`.
- **qa** – Q&A pairs (e.g. `data_points/QA/cybersec_QA_dataset-1.yaml`, `cybersec_QA_dataset-2.yaml`). Schema: `QAScenario`; generator: `generate_qa_training_example`.
- **payload** – Payload datapoints (e.g. `data_points/payload/payload_dataset.json`). Schema: `PayloadScenario`; generator: `generate_payload_training_example`.
- **tool** – Kali Linux / CLI tool commands and use cases (e.g. `data_points/tool/kali_linux_dataset.json`). Schema: `ToolScenario`; generator: `generate_tool_training_example`.

## Usage

```bash
# Default run: combine + version snapshot
# JSONL -> combine_output/<version_id>/all_training_data.jsonl
# manifest.json written alongside the JSONL (auto-linked by training script)
poetry run python src/data_prep/combine_training_data.py

# Attach a human-readable note stored in manifest.json as "run_message"
poetry run python src/data_prep/combine_training_data.py \
  --run-message "added 500 juice_shop exploits, removed stale portswigger set"

# Plain combine only (no versioning, JSONL to --output path)
poetry run python src/data_prep/combine_training_data.py \
  --no-data-version --output combine_output/all_training_data.jsonl

# Use tar.gz (faster, ~74M) instead of default tar.xz (~53M, slower)
poetry run python src/data_prep/combine_training_data.py \
  --archive-format tar.gz

# Pin an explicit version id
poetry run python src/data_prep/combine_training_data.py \
  --version-id dpv_v3_manual

# Re-process a versioned or demo snapshot without touching the live data_points/ folder
# The leading "data_points" segment in every datapoints.yaml entry is replaced with the
# supplied path, so no config editing is required.
poetry run python src/data_prep/combine_training_data.py \
  --data-points-dir demo_datapoints/dpv_v1_20260325_091340_5102ffcc/data_points

# Switch by saved version id (resolved from data_versions/<id>/manifest.json)
# If data_versions/<id>/data_points is missing, archive is extracted automatically.
poetry run python src/data_prep/combine_training_data.py \
  --use-version-id dpv_v1_20260325_091340_5102ffcc

# Switch by exact data_points SHA256 (resolved through data_versions/index.json)
poetry run python src/data_prep/combine_training_data.py \
  --use-data-points-sha 5102ffcc................................................

# Same, but skip creating a new version snapshot of the demo copy
poetry run python src/data_prep/combine_training_data.py \
  --data-points-dir demo_datapoints/dpv_v1_20260325_091340_5102ffcc/data_points \
  --no-data-version --output combine_output/restored_v1.jsonl

# List datapoints and scenario counts
poetry run python src/data_prep/combine_training_data.py --list

# Validate all datapoints (step count, tools, etc.)
poetry run python src/data_prep/combine_training_data.py --validate

# Pass versioned output directly to the training script
# manifest.json in the same folder is auto-detected and copied to trainable_data/
poetry run python src/training/single_stage_finetune.py \
    --data combine_output/dpv_v1_20260325_091340_5102ffcc/all_training_data.jsonl \
    --model-key qwen2.5-7b \
    --epochs 3 \
    --batch-size 1 \
    --gradient-accumulation-steps 8 \
    --merge-lora
```

## Data versioning

Every default combine run produces three outputs:

```
data_versions/
  dpv_v1_20260325_181530_5102ffcc/
    manifest.json                                    <- version metadata + all SHA256s
    SHA256SUMS.txt                                   <- checksums for archive + manifest + JSONL
    data_points_dpv_v1_20260325_181530_5102ffcc.tar.xz  <- full data_points/ snapshot
  index.json                                         <- sha <-> version lookup index

combine_output/
  dpv_v1_20260325_181530_5102ffcc/
    all_training_data.jsonl                          <- combined training file
    manifest.json                                    <- copy of manifest (JSONL and metadata together)
```

The manifest is written to both locations so the training file and its metadata always travel together.

### Version id format

```
dpv_v{N}_{YYYYMMDD}_{HHMMSS}_{hash8}
     ^                         ^
     auto-incremented run      first 8 chars of SHA256 of data_points/ contents
```

- `N` increments by scanning existing `data_versions/dpv_vN_*` folders.
- Same `hash8` on two runs means data content is unchanged; a different hash means at least one source file changed.

### manifest.json fields

| Field | Description |
|-------|-------------|
| `version_id` | Full version id string |
| `run` | Integer run number (same as N in version id) |
| `version_method` | Current version id method (`data_points_sha256`) |
| `reused_existing_version` | `true` when unchanged `data_points` reused an existing version id |
| `run_message` | Optional free-text note passed via `--run-message`; `null` when not supplied |
| `created_at_utc` | ISO 8601 timestamp |
| `output_file` | Absolute path to the combined JSONL |
| `output_file_sha256` | SHA256 of the combined JSONL |
| `archive_format` | `tar.gz` or `tar.xz` |
| `archive_file` | Archive filename |
| `archive_size_mb` | Compressed size in MB |
| `archive_sha256` | SHA256 of the archive |
| `data_points_dir_sha256` | SHA256 of entire `data_points/` tree |
| `datapoints` | List of `{id, path, loader, scenario_type}` active for this run |

### Training linkage

When `single_stage_finetune.py` is run with `--data` pointing to a versioned combine output, it automatically copies `manifest.json` into the training artifact folder so the data version is permanently recorded alongside the model weights:

```
combine_output/dpv_v4_20260325_084021_5102ffcc/
  all_training_data.jsonl        <- passed as --data
  manifest.json                  <- auto-detected and copied to trainable_data/

artifacts/qwen2.5-7b/rabit0-v1-run18/
  trainable_data/
    manifest.json                <- provenance: which data version trained this run
    ...
```

No extra flags are needed. If the data file was produced without versioning, the copy step is silently skipped. See `docs/DATA_VERSIONING.md` for full details.

### Restoring a version

```bash
# 1. Restore a snapshot to demo_datapoints/ (recommended; leaves live data_points/ untouched)
poetry run python scripts/restore_datapoints.py \
  --version-id dpv_v1_20260325_091340_5102ffcc

# 2. Rebuild JSONL directly from the restored folder using --data-points-dir
#    No symlinks, no config changes needed.
poetry run python src/data_prep/combine_training_data.py \
  --data-points-dir demo_datapoints/dpv_v1_20260325_091340_5102ffcc/data_points \
  --no-data-version --output combine_output/restored_v1.jsonl

# Verify archive integrity before restoring
sha256sum -c data_versions/dpv_v1_.../SHA256SUMS.txt
```

You can also skip manual restore and switch directly with:

```bash
poetry run python src/data_prep/combine_training_data.py \
  --use-version-id dpv_v1_...
```

If `data_versions/dpv_v1_.../data_points` is not present, the archive is extracted automatically into that folder.

### Comparing two versions

```bash
# 1. Diff manifest datapoints lists (which ids/paths are active)
diff <(jq .datapoints data_versions/dpv_v1_.../manifest.json) \
     <(jq .datapoints data_versions/dpv_v2_.../manifest.json)

# 2. Compare data_points_dir_sha256 fields: same = no data change, different = content changed
jq .data_points_dir_sha256 data_versions/dpv_v1_.../manifest.json
jq .data_points_dir_sha256 data_versions/dpv_v2_.../manifest.json
```

## CLI flags reference

| Flag | Default | Description |
|------|---------|-------------|
| `--output` | `combine_output/all_training_data.jsonl` | JSONL path (ignored when versioning is on) |
| `--format` | `openai` | Output format: `openai`, `jsonl`, `json` |
| `--save-data-version` | on | Enable versioning (default) |
| `--no-data-version` | — | Disable versioning; write JSONL to `--output` |
| `--version-id` | auto | Explicit version id (skips auto-increment) |
| `--data-version-dir` | `data_versions` | Root dir for version metadata + archive |
| `--archive-format` | `tar.xz` | Snapshot compression: `tar.gz` (~74M, fast) or `tar.xz` (~53M, slow) |
| `--run-message` | `null` | Free-text note stored in `manifest.json` as `run_message` |
| `--data-points-dir` | `data_points` | Data directory to load from. Replaces the leading `data_points` segment in every `datapoints.yaml` entry. Use to re-process a versioned snapshot (e.g. `demo_datapoints/dpv_v1_.../data_points`) without editing config. |
| `--use-version-id` | `null` | Load datapoints from an existing saved version id in `data_versions/` (auto-extracts archive if needed). Cannot be used with `--data-points-dir`. |
| `--use-data-points-sha` | `null` | Load datapoints by exact `data_points_dir_sha256` via `data_versions/index.json` (latest matching version). Cannot be used with `--data-points-dir`. |
| `--list` | — | List datapoints and counts; no combine |
| `--validate` | — | Validate datapoints quality; no combine |

## Adding a datapoint

1. Add a JSON or YAML file under `data_points/` (or use an existing path).
2. In `src/config/data_prep/datapoints.yaml`, add a new entry under `datapoints:` with:
   - `id: my_datapoint` (used in breakdown output and version manifest)
   - `path: data_points/...` (relative to project root)
   - `loader: json`
   - `scenario_type: security` | `qa` | `payload` | `tool` (default is `security` if omitted)
3. For a new scenario type (not security/qa/payload/tool), add a Pydantic model and generator in `data_schema.py`, register the type in the combiner loop and in `json_source_loader.load_scenarios_from_json`, and optionally support `scenario_type="auto"` for inference.

## Current datapoints (from datapoints.yaml)

The combiner reads `src/config/data_prep/datapoints.yaml`. As of the latest config, included sources are:

| Id | Path | Type |
|----|------|------|
| exploit_portswigger | data_points/exploit/portswigger_dataset.json | security |
| exploit_invicti | data_points/exploit/invicti_dataset.json | security |
| exploit_juice_shop | data_points/exploit/juiceshop_dataset.json | security |
| exploit_n8n | data_points/exploit/n8n_dataset.json | security |
| exploit_web_app | data_points/exploit/web_app_dataset.json | security |
| exploit_org | data_points/exploit/org_dataset.json | security |
| payload | data_points/payload/payload_dataset.json | payload |
| tool | data_points/tool | tool |

QA sources (`qa`, `qa_cybersecurity`) are defined in `datapoints.yaml` but currently commented out.

See `data_points/README.md` for folder layout and file descriptions.

## Archived / legacy

- **Hugging Face (resk-fr)**: The former `dataset_transformer.py` that fetched and transformed `resk-fr/pentesting-for-agents` has been moved to `archived/dataset_transformer.py`. It is only used by `archived/external_dataset.py`. The main pipeline uses only local `data_points/` (JSON/YAML).
- **sources.yaml**: Replaced by `datapoints.yaml`; datapoints are now listed there with `id`, `path`, `loader`, and `scenario_type`.
