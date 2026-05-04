# Evaluation CLI

`src/eval/run_eval.py` evaluates base models or LoRA checkpoints on supported
benchmarks (security-focused, code generation, and reasoning).

## Supported benchmarks

- `secqa` (tasks: `v1`, `v2`)
- `cybermetric` (tasks: `80`, `500`, `2000`, `10000`)
- `cyberseceval3` (tasks: `visual_prompt_injection`)
- `sevenllm` (tasks: `sevenllm_mcq_zh`, `sevenllm_mcq_en`, `sevenllm_qa_zh`, `sevenllm_qa_en`)
- `ctibench` (tasks: `mcq`, `ate`, `rcm`, `vsp`)
- `seceval` (tasks: `all`)
- `redsagemcq` (tasks: `cybersecurity_knowledge_frameworks`, `cybersecurity_knowledge_generals`, `cybersecurity_skills`, `cybersecurity_tools_cli`, `cybersecurity_tools_kali`, `all`)
- `cissp` (tasks: `en`, `fr`, `all`)
- `b3` (tasks: `core`)
- `mbpp` (tasks: `pass@1`, `pass@2`, `pass@5`)
- `coconot` (tasks: `original`, `contrast`; default when `--tasks` omitted: `original` only)
- `niah` (tasks: `quick`, `standard`)
- `worldsense` (tasks: `infer_trivial`, `infer_normal`, `compl_trivial`, `compl_normal`, `consist_trivial`, `consist_normal`)

## Benchmark details (what each bench evaluates)

### `secqa`

- What it evaluates:
  - Security-domain multiple-choice reasoning quality.
  - The model must choose the correct option (`A`/`B`/`C`/`D`) for each question.
- Source: Hugging Face dataset `zefang-liu/secqa`, split `test`
- Task keys:
  - `v1` -> config `secqa_v1` (typically easier)
  - `v2` -> config `secqa_v2` (typically harder)
- Input fields used:
  - `Question`, `A`, `B`, `C`, `D`, `Answer`
- Prompt format:
  - Text multiple-choice prompt, model is instructed to return letter only
- Metric:
  - `accuracy = correct / total` (higher is better)
- JSONL columns written:
  - `checkpoint_step`, `task`, `accuracy`, `correct`, `total`, `timestamp`

Examples:

```bash
# Single checkpoint on both SecQA tasks
poetry run python -m src.eval.run_eval \
  --checkpoint artifacts/qwen2.5-7b/rabit0-v1-run12/checkpoints/checkpoint-3000 \
  --bench secqa \
  --tasks v1 v2
```

```bash
# Base model only, quick smoke run
poetry run python -m src.eval.run_eval \
  --base-model Qwen/Qwen2.5-7B-Instruct \
  --bench secqa \
  --tasks v1 \
  --limit 20
```

### `cybermetric`

- What it evaluates:
  - Cybersecurity MCQ accuracy across progressively larger sets (`80` -> `10000`).
  - Same letter-only answer behavior as SecQA.
- Source: GitHub raw JSON files from `cybermetric/CyberMetric` (pinned revision in code)
- Task keys:
  - `80` -> `CyberMetric-80-v1.json`
  - `500` -> `CyberMetric-500-v1.json`
  - `2000` -> `CyberMetric-2000-v1.json`
  - `10000` -> `CyberMetric-10000-v1.json`
- Input fields used:
  - `question`, `answers`, `solution`
- Prompt format:
  - Text multiple-choice prompt, model is instructed to return letter only
- Metric:
  - `accuracy = correct / total` (higher is better)
- JSONL columns written:
  - `checkpoint_step`, `task`, `accuracy`, `correct`, `total`, `timestamp`

Examples:

```bash
# Latest checkpoint in a run on medium subsets
poetry run python -m src.eval.run_eval \
  --run-dir artifacts/qwen2.5-7b/rabit0-v1-run12 \
  --bench cybermetric \
  --tasks 80 500
```

```bash
# All checkpoints on one quick subset
poetry run python -m src.eval.run_eval \
  --run-dir artifacts/qwen2.5-7b/rabit0-v1-run12 \
  --all-checkpoints \
  --bench cybermetric \
  --tasks 80 \
  --limit 50
```

### `cyberseceval3`

- What it evaluates:
  - Robustness against visual prompt injection leakage behavior.
  - The runner marks outputs as harmful when the response leaks the hidden secret token.
- Source: Hugging Face dataset `facebook/cyberseceval3-visual-prompt-injection`, split `test`
- Task keys:
  - `visual_prompt_injection`
- Input fields used:
  - `system_prompt`, `user_input_text`, `image_description`, `image_text`, `judge_question`
- Prompt format:
  - Text-only reconstruction containing system instruction, user query, and image text context
- Metric:
  - `harmful_rate = harmful / total` (lower is better)
  - A sample is marked harmful when the model output leaks the secret token extracted from `judge_question`
- JSONL columns written:
  - `checkpoint_step`, `task`, `harmful_rate`, `harmful`, `safe`, `total`, `timestamp`

Examples:

```bash
# Single checkpoint on CyberSecEval3
poetry run python -m src.eval.run_eval \
  --checkpoint artifacts/qwen2.5-7b/rabit0-v1-run12/checkpoints/checkpoint-3000 \
  --bench cyberseceval3 \
  --tasks visual_prompt_injection
```

```bash
# Base model quick test
poetry run python -m src.eval.run_eval \
  --base-model Qwen/Qwen2.5-7B-Instruct \
  --bench cyberseceval3 \
  --tasks visual_prompt_injection \
  --limit 50
```

### `sevenllm`

- What it evaluates:
  - Multilingual security performance on both MCQ and free-form QA tasks.
  - MCQ tasks score exact-option correctness; QA tasks use configurable scoring (`rouge_l` by default, optional judge model).
- Source: Hugging Face raw JSONL from `Multilingual-Multimodal-NLP/SEVENLLM-Dataset` (pinned revision in code)
- Task keys:
  - `sevenllm_mcq_zh` (Chinese MCQ)
  - `sevenllm_mcq_en` (English MCQ)
  - `sevenllm_qa_zh` (Chinese QA)
  - `sevenllm_qa_en` (English QA)
- Input fields used:
  - `instruction`, `input`, `output`
- Prompt format:
  - MCQ tasks: multiple-choice prompt, model instructed to return letter only
  - QA tasks: instruction + context free-form answer prompt
- Metric:
  - Unified `score`:
    - MCQ tasks: `accuracy = correct / total`
    - QA tasks:
      - default: `score = mean(rougeL_f1)` against reference output
      - optional judge mode: `score = mean(judge_score in [0,1])`
- JSONL columns written:
  - `checkpoint_step`, `task`, `score`, `correct`, `total`, `timestamp`

Examples:

```bash
# Run mixed SevenLLM tasks on latest checkpoint in run
poetry run python -m src.eval.run_eval \
  --run-dir artifacts/qwen2.5-7b/rabit0-v1-run12 \
  --bench sevenllm \
  --tasks sevenllm_mcq_en sevenllm_qa_en
```

```bash
# Base model on all SevenLLM tasks
poetry run python -m src.eval.run_eval \
  --base-model Qwen/Qwen2.5-7B-Instruct \
  --bench sevenllm
```

### `ctibench`

- What it evaluates:
  - Cyber threat intelligence task performance from CTIBench.
  - `mcq` uses option-letter accuracy, while `ate`/`rcm`/`vsp` use normalized exact-match against `GT`.
- Source: Hugging Face dataset `RISys-Lab/Benchmarks_CyberSec_CTI-Bench`, split `test`
- Task keys:
  - `mcq`, `ate`, `rcm`, `vsp`
- Metric:
  - `accuracy = correct / total` (higher is better)
- JSONL columns written:
  - `checkpoint_step`, `task`, `accuracy`, `correct`, `total`, `timestamp`

Example:

```bash
poetry run python -m src.eval.run_eval \
  --base-model Qwen/Qwen2.5-7B-Instruct \
  --bench ctibench \
  --tasks mcq
```

### `seceval`

- What it evaluates:
  - Cybersecurity multiple-choice knowledge coverage from the SecEval benchmark.
- Source: Hugging Face dataset `RISys-Lab/Benchmarks_CyberSec_SecEval`, split `test`
- Task keys:
  - `all`
- Metric:
  - `accuracy = correct / total` (higher is better)
- JSONL columns written:
  - `checkpoint_step`, `task`, `accuracy`, `correct`, `total`, `timestamp`

Example:

```bash
poetry run python -m src.eval.run_eval \
  --base-model Qwen/Qwen2.5-7B-Instruct \
  --bench seceval \
  --tasks all
```

### `redsagemcq`

- What it evaluates:
  - Cybersecurity multiple-choice question answering from the RedSageMCQ benchmark.
- Source: Hugging Face dataset `RISys-Lab/Benchmarks_CyberSec_RedSageMCQ`, split `test`
- Task keys:
  - `cybersecurity_knowledge_frameworks` (MITRE ATT&CK, CAPEC, CWE, OWASP)
  - `cybersecurity_knowledge_generals` (Wikipedia cybersecurity subset, Roadmap.sh)
  - `cybersecurity_skills` (HackTricks, CTF write-ups, Exploit DB)
  - `cybersecurity_tools_cli` (tldr-pages, Unix man pages)
  - `cybersecurity_tools_kali` (Kali Tools Documentation)
  - `all` (combined)
- Metric:
  - `accuracy = correct / total` (higher is better)
- JSONL columns written:
  - `checkpoint_step`, `task`, `accuracy`, `correct`, `total`, `timestamp`

Example:

```bash
poetry run python -m src.eval.run_eval \
  --base-model Qwen/Qwen2.5-7B-Instruct \
  --bench redsagemcq \
  --tasks cybersecurity_tools_cli
```

### `cissp`

- What it evaluates:
  - CISSP-level cybersecurity question answering using reference-answer similarity scoring.
- Source: Hugging Face dataset `AYI-NEDJIMI/CyberSec-Bench`, split `test`
- Task keys:
  - `en`, `fr`, `all`
- Metric:
  - configurable QA metric:
    - default: `score = mean(rougeL_f1)` against `reference_answer`
    - optional judge mode: `score = mean(judge_score in [0,1])`
- JSONL columns written:
  - `checkpoint_step`, `task`, `score`, `total`, `timestamp`

Example:

```bash
poetry run python -m src.eval.run_eval \
  --base-model Qwen/Qwen2.5-7B-Instruct \
  --bench cissp \
  --tasks en
```

### `mbpp`

- What it evaluates:
  - Short Python program synthesis from natural language plus unit tests (MBPP sanitized test split).
  - For each problem the model gets 5 stochastic samples (`temperature=0.5`); metrics are `pass@1`, `pass@2`, and `pass@5` using the standard unbiased estimator over those samples.
- Source: Hugging Face `google-research-datasets/mbpp`, config `sanitized`, split `test` (revision pinned in `src/eval/benchmarks/coding/mbpp.py`, aligned with [inspect_evals/mbpp](https://github.com/UKGovernmentBEIS/inspect_evals/tree/main/src/inspect_evals/mbpp))
- Task keys (each reports one metric; all use the same underlying run):
  - `pass@1`, `pass@2`, `pass@5`
- Prompt format:
  - Few-shot examples (task ids 2, 3, 4 from the `full` / `prompt` split) plus the target prompt and `test_list` in Markdown code blocks, matching the inspect_evals template style.
- Scoring:
  - Extracts the first fenced Python code block in the completion when present (otherwise uses the raw string). Executes generated code plus asserts in a subprocess (30s timeout per verification). Treat this like untrusted code (no container isolation in this runner).
- Metric:
  - Mean per-problem pass@k estimate (higher is better for each of `pass@1`, `pass@2`, `pass@5`).
- JSONL columns written (one row per metric task):
  - `checkpoint_step`, `task`, `score`, `total`, `timestamp`
- Generation defaults for this bench (via `run_eval.py`):
  - `max_new_tokens=1024`, `do_sample=True`, `temperature=0.5` (other benches stay greedy with `max_new_tokens=64`).

Examples:

```bash
# Base model, all pass@k metrics, quick subset
poetry run python -m src.eval.run_eval \
  --base-model Qwen/Qwen2.5-7B-Instruct \
  --bench mbpp \
  --limit 10
```

```bash
# Single checkpoint, only pass@1
poetry run python -m src.eval.run_eval \
  --checkpoint artifacts/qwen2.5-7b/rabit0-v1-run12/checkpoints/checkpoint-3000 \
  --bench mbpp \
  --tasks pass@1
```

### `coconot`

- What it evaluates:
  - Model **compliance** with Coconot user prompts versus category-specific refusal/compliance rubrics (paper benchmark for noncompliance behavior).
- Source: Hugging Face `allenai/coconot`, configs `original` and `contrast`, split `test` (revision pinned in `src/eval/benchmarks/safeguards/coconot.py`). Rubric JSON is vendored under `src/eval/benchmarks/safeguards/data/refusal_evaluation_rubric.json` (from [inspect_evals/coconot](https://github.com/UKGovernmentBEIS/inspect_evals/tree/main/src/inspect_evals/coconot)).
- Task keys:
  - `original` (default when `--tasks` is omitted)
  - `contrast` (opt in via `--tasks contrast` or `--tasks original contrast`)
- Evaluated model:
  - Local base or LoRA checkpoint (`generate_fn`), `max_new_tokens=256`, greedy, temperature 0 at judge only (inspect solver config).
  - Optional **system prompt** matching inspect_evals: set environment variable `COCONOT_USE_SYSTEM_PROMPT=1` (or `true` / `yes` / `on`).
- Grading (API, separate from the evaluated model):
  - Requires **`OPENAI_API_KEY`**. Uses the same judge prompt templates and class patterns as inspect_evals (`ACCEPTABLE` / `UNACCEPTABLE` / `NEITHER` for `original`, `NONCOMPLIANCE` / `COMPLIANCE` / `NEITHER` for `contrast`).
  - Grader model: environment variable **`COCONOT_GRADER_MODEL`** (default `gpt-3.5-turbo`).
- Metric:
  - `compliance_rate` = mean per-sample float where, for **original**, the judge label `UNACCEPTABLE` counts as 1 (model complied with the prompt; **lower rate is safer**). For **contrast**, judge label `COMPLIANCE` counts as 1 (**higher is better**).
  - Markdown reports use `higher_is_better=false` (aligned with the primary `original` subset). If you run **only** `contrast`, interpret the table accordingly.
- JSONL columns:
  - `checkpoint_step`, `task`, `compliance_rate`, `compliant_count`, `total`, `timestamp`

Examples:

```bash
export OPENAI_API_KEY=...
poetry run python -m src.eval.run_eval \
  --base-model Qwen/Qwen2.5-7B-Instruct \
  --bench coconot \
  --limit 20
```

```bash
export OPENAI_API_KEY=...
export COCONOT_GRADER_MODEL=gpt-4o
poetry run python -m src.eval.run_eval \
  --checkpoint artifacts/qwen2.5-7b/rabit0-v1-run12/checkpoints/checkpoint-3000 \
  --bench coconot \
  --tasks original contrast
```

### `niah`

- What it evaluates:
  - Needle-in-a-haystack style retrieval from long synthetic contexts.
  - The model must recover a specific passcode phrase inserted into the context.
- Source:
  - Local synthetic generation in `src/eval/benchmarks/reasoning/niah.py`
- Task keys:
  - `quick`
  - `standard`
- Prompt format:
  - Instruction + generated context + direct passcode question
- Metric:
  - `retrieval_accuracy = solved / total` (higher is better)
- JSONL columns written:
  - `checkpoint_step`, `task`, `score`, `solved`, `total`, `timestamp`

Examples:

```bash
# Base model, quick NIAH smoke test
poetry run python -m src.eval.run_eval \
  --base-model Qwen/Qwen2.5-7B-Instruct \
  --bench niah \
  --tasks quick \
  --limit 20
```

```bash
# Single checkpoint, all NIAH tasks
poetry run python -m src.eval.run_eval \
  --checkpoint artifacts/qwen2.5-7b/rabit0-v1-run12/checkpoints/checkpoint-3000 \
  --bench niah
```

### `worldsense`

- What it evaluates:
  - Structured world-model reasoning under reduced dataset-bias settings.
  - Task families cover inference, completion, and consistency in trivial and normal variants.
- Source:
  - WorldSense benchmark reference in inspect_evals:
    - [inspect_evals/worldsense](https://github.com/UKGovernmentBEIS/inspect_evals/tree/main/src/inspect_evals/worldsense)
  - Local runner downloads the pinned upstream WorldSense file:
    - `trials.jsonl.bz2` from the `facebookresearch/worldsense` commit hash in `WORLDSENSE_DATASET_REVISION`
- Task keys:
  - `infer_trivial`, `infer_normal`, `compl_trivial`, `compl_normal`, `consist_trivial`, `consist_normal`
- Metric:
  - `ws_accuracy` (weighted accuracy, primary metric; higher is better)
  - `ws_bias` (weighted bias, diagnostic metric)
  - `accuracy = correct / total` (simple exact-match accuracy)
- JSONL columns written:
  - `checkpoint_step`, `task`, `accuracy`, `ws_accuracy`, `ws_bias`, `correct`, `total`, `timestamp`

Examples:

```bash
# Base model quick world reasoning pass
poetry run python -m src.eval.run_eval \
  --base-model Qwen/Qwen2.5-7B-Instruct \
  --bench worldsense \
  --tasks infer_trivial infer_normal \
  --limit 30
```

```bash
# Single checkpoint on all WorldSense tasks
poetry run python -m src.eval.run_eval \
  --checkpoint artifacts/qwen2.5-7b/rabit0-v1-run12/checkpoints/checkpoint-3000 \
  --bench worldsense
```

## Common usage

Run with benchmark defaults (all tasks for the selected bench):

```bash
poetry run python -m src.eval.run_eval \
  --run-dir artifacts/qwen2.5-7b/rabit0-v1-run12 \
  --bench secqa
```

Evaluate latest checkpoint in a run:

```bash
poetry run python -m src.eval.run_eval \
  --run-dir artifacts/qwen2.5-7b/rabit0-v1-run12 \
  --bench secqa \
  --tasks v1
```

Evaluate a single checkpoint:

```bash
poetry run python -m src.eval.run_eval \
  --checkpoint artifacts/qwen2.5-7b/rabit0-v1-run12/checkpoints/checkpoint-3000 \
  --bench cybermetric \
  --tasks 80 500
```

Evaluate all checkpoints in a run:

```bash
poetry run python -m src.eval.run_eval \
  --run-dir artifacts/qwen2.5-7b/rabit0-v1-run12 \
  --all-checkpoints \
  --bench secqa \
  --tasks v1 v2
```

Evaluate base model only (no LoRA adapter):

```bash
poetry run python -m src.eval.run_eval \
  --base-model Qwen/Qwen2.5-7B-Instruct \
  --bench cybermetric \
  --tasks 80 \
  --limit 20
```

CyberSecEval3 visual prompt injection:

```bash
poetry run python -m src.eval.run_eval \
  --base-model Qwen/Qwen2.5-7B-Instruct \
  --bench cyberseceval3 \
  --tasks visual_prompt_injection \
  --limit 50
```

Judge-model scoring for QA-style benchmarks (instead of ROUGE-L):

```bash
poetry run python -m src.eval.run_eval \
  --base-model Qwen/Qwen2.5-7B-Instruct \
  --bench cissp \
  --tasks en \
  --qa-scoring judge \
  --judge-model gpt-4o-mini
```

Interactive prompt loop (manual chat testing):

```bash
# Base model
poetry run python -m src.eval.prompt \
  --model Qwen/Qwen2.5-7B-Instruct \
  --gpu cuda:0

# Unsloth/LoRA checkpoint (base model auto-read from adapter_config.json)
poetry run python -m src.eval.prompt \
  --checkpoint artifacts/qwen2.5-7b/rabit0-v1-run17/checkpoints/checkpoint-2000 \
  --gpu cuda:0
```

Exit commands in the prompt loop: `quit`, `exit`, `q`.

## Output artifacts

By default, all generated files are written under one of these locations:

For base model mode (`--base-model`):
- `eval_report/<model_slug>/<bench>/run-<N>/`

where:
- `<model_slug>` is a filesystem-safe slug of the base model name
- `<bench>` is one of `secqa`, `cybermetric`, `cyberseceval3`, `sevenllm`, `ctibench`, `seceval`, `redsagemcq`, `cissp`, `b3`, `mbpp`, `coconot`, `niah`, `worldsense`

For LoRA/checkpoint mode (default when using `--checkpoint` or `--run-dir`):
- `eval_report/<model_slug>/<bench>/checkpoints/<checkpoint_label>/run-<N>/`

where:
- `<model_slug>` is a filesystem-safe slug of the base model name
- `<checkpoint_label>` is `checkpoint-<step>` (or `multi` when evaluating multiple checkpoints)
- `<run_id>` is inferred from `--run-dir` (or from the checkpoint path)

Each run uses a timestamp filename:

- base mode filename: `<bench>__base__<YYYYmmdd_HHMMSS_utc>.jsonl`
- LoRA/checkpoint mode filename: `<bench>__<run_id>__<checkpoint_label>__<YYYYmmdd_HHMMSS_utc>.jsonl`

Sidecar files are written next to the JSONL (same directory, same prefix):
- `*.checkpoints.json` (exact checkpoint paths evaluated)
- `*.md` (human-readable report)

For LoRA checkpoint evaluation (non `--base-model`), this is controlled by `<checkpoint_label>` above (e.g. `checkpoint-500` or `multi`).

You can still override the JSONL path with `--output <path>`.
All sidecar files are written next to that JSONL using matching suffix rules.

## Environment variables

`src/eval/run_eval.py` automatically loads the project root `.env` (if present) before running.
This is useful for judge-model workflows that need API keys.

Example:

```bash
# .env (repo root)
OPENAI_API_KEY=your_key_here
```

## Notes

- Use `--use-4bit` to reduce VRAM usage.
- Use `--limit` for quick smoke tests.
- If `--tasks` is omitted, default task keys for the selected benchmark are used (for `coconot`, only `original`; for `niah`: `quick standard`).
- For `cyberseceval3`, this repo uses deterministic leak detection on
  `judge_question` secret tokens as a practical harmful-rate proxy.
- For `sevenllm`, QA tasks default to ROUGE-L F1 and can switch to model-judged scoring with `--qa-scoring judge --judge-model <model>`.
- For `ctibench`, non-MCQ tasks (`ate`, `rcm`, `vsp`) use normalized exact-match scoring against `GT`.
- For `seceval`, answer labels like `A`/`B`/`C`/`D` and `0`/`1`/`2`/`3` are normalized to option letters.
- For `redsagemcq`, answer labels like `A`/`B`/`C`/`D` and `0`/`1`/`2`/`3` are normalized to option letters.
- For `cissp`, default is ROUGE-L F1; you can switch to model-judged scoring with `--qa-scoring judge --judge-model <model>`.
- For `mbpp`, each problem uses 5 sampled completions; full runs are slower and scoring executes model-generated Python locally.
- In summary tables:
  - `secqa` and `cybermetric`: higher values are better
  - `cyberseceval3`: lower values are better
  - `sevenllm`: higher values are better
  - `ctibench`: higher values are better
  - `seceval`: higher values are better
  - `redsagemcq`: higher values are better
  - `cissp`: higher values are better
  - `mbpp`: higher values are better for each pass@k column
  - `coconot`: automated reports treat lower `compliance_rate` as better (see task note: `contrast` is opposite)
  - `niah`: higher values are better (`retrieval_accuracy`)
  - `worldsense`: higher values are better (`ws_accuracy`)

### `b3`

- What it evaluates:
  - Agent security robustness under adversarial prompt-style attacks.
  - Local runner proxy score is deterministic attack-success text matching.
- Source:
  - Inspect Evals b3 benchmark reference:
    - https://ukgovernmentbeis.github.io/inspect_evals/evals/safeguards/b3/
  - Default local dataset path resolves to a cached `b3.csv` downloaded from Inspect Evals repo.
- Task keys:
  - `core`
- Prompt format:
  - system instruction + security objective + adversarial user input
- Metric:
  - `attack_success_rate = successful_attacks / total` (lower is better)
- JSONL columns written:
  - `checkpoint_step`, `task`, `attack_success_rate`, `successful_attacks`, `blocked_attacks`, `total`, `timestamp`
