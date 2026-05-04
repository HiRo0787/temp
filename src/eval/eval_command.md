# Evaluation commands (all benches, full tasks)

Run from the repo root.

Defaults:
- If you **omit** `--tasks`, the evaluator runs the **default tasks for that benchmark** (most benches: all tasks; `coconot`: `original` only unless you add `contrast`).
- Outputs go under:
  - base-model mode: `eval_report/<base-model>/<bench>/...` (auto timestamped)
  - LoRA/checkpoint mode: `artifacts/<model>/<run>/eval/<bench>/...` (auto timestamped)


Supported benches in this repo:
- `secqa`
- `cybermetric`
- `cyberseceval3`
- `sevenllm`
- `ctibench`
- `seceval`
- `redsagemcq`
- `cissp`
- `b3`
- `mbpp` (code generation; 5 samples per problem, slower than MCQ benches)
- `coconot` (requires `OPENAI_API_KEY` for the judge model; evaluated model is still local)
- `worldsense` (reasoning benchmark; tasks: `infer_trivial`, `infer_normal`, `compl_trivial`, `compl_normal`, `consist_trivial`, `consist_normal`)

---

## Target 1: LoRA checkpoint (run12 @ checkpoint-3000)

Checkpoint:

`artifacts/qwen2.5-7b/rabit0-v1-run12/checkpoints/checkpoint-3000`

Run all benches (all tasks):

```bash
poetry run python -m src.eval.run_eval \
  --checkpoint artifacts/qwen2.5-7b/rabit0-v1-run12/checkpoints/checkpoint-3000 \
  --gpu cuda:0 \
  --bench secqa
```

```bash
poetry run python -m src.eval.run_eval \
  --checkpoint artifacts/qwen2.5-7b/rabit0-v1-run12/checkpoints/checkpoint-3000 \
  --gpu cuda:0 \
  --bench cybermetric
```

```bash
poetry run python -m src.eval.run_eval \
  --checkpoint artifacts/qwen2.5-7b/rabit0-v1-run12/checkpoints/checkpoint-3000 \
  --gpu cuda:0 \
  --bench cyberseceval3
```

```bash
poetry run python -m src.eval.run_eval \
  --checkpoint artifacts/qwen2.5-7b/rabit0-v1-run12/checkpoints/checkpoint-3000 \
  --gpu cuda:0 \
  --bench sevenllm
```

```bash
poetry run python -m src.eval.run_eval \
  --checkpoint artifacts/qwen2.5-7b/rabit0-v1-run12/checkpoints/checkpoint-3000 \
  --gpu cuda:0 \
  --bench ctibench
```

```bash
poetry run python -m src.eval.run_eval \
  --checkpoint artifacts/qwen2.5-7b/rabit0-v1-run12/checkpoints/checkpoint-3000 \
  --gpu cuda:0 \
  --bench seceval
```

```bash
poetry run python -m src.eval.run_eval \
  --checkpoint artifacts/qwen2.5-7b/rabit0-v1-run12/checkpoints/checkpoint-3000 \
  --gpu cuda:0 \
  --bench redsagemcq
```

```bash
poetry run python -m src.eval.run_eval \
  --checkpoint artifacts/qwen2.5-7b/rabit0-v1-run12/checkpoints/checkpoint-3000 \
  --gpu cuda:0 \
  --bench cissp
```

```bash
poetry run python -m src.eval.run_eval \
  --checkpoint artifacts/qwen2.5-7b/rabit0-v1-run12/checkpoints/checkpoint-3000 \
  --gpu cuda:0 \
  --bench b3
```

```bash
poetry run python -m src.eval.run_eval \
  --checkpoint artifacts/qwen2.5-7b/rabit0-v1-run12/checkpoints/checkpoint-3000 \
  --gpu cuda:0 \
  --bench mbpp
```

```bash
export OPENAI_API_KEY=...
poetry run python -m src.eval.run_eval \
  --checkpoint artifacts/qwen2.5-7b/rabit0-v1-run12/checkpoints/checkpoint-3000 \
  --gpu cuda:0 \
  --bench coconot
```

```bash
poetry run python -m src.eval.run_eval \
  --checkpoint artifacts/qwen2.5-7b/rabit0-v1-run12/checkpoints/checkpoint-3000 \
  --gpu cuda:0 \
  --bench niah
```

```bash
poetry run python -m src.eval.run_eval \
  --checkpoint artifacts/qwen2.5-7b/rabit0-v1-run12/checkpoints/checkpoint-3000 \
  --gpu cuda:0 \
  --bench worldsense
```

---

## Target 2: LoRA checkpoint (run17 @ checkpoint-2000)

Checkpoint:

`artifacts/qwen2.5-7b/rabit0-v1-run17/checkpoints/checkpoint-2000`

Run all benches (all tasks):

```bash
poetry run python -m src.eval.run_eval \
  --checkpoint artifacts/qwen2.5-7b/rabit0-v1-run17/checkpoints/checkpoint-2000 \
  --gpu cuda:0 \
  --bench secqa
```

```bash
poetry run python -m src.eval.run_eval \
  --checkpoint artifacts/qwen2.5-7b/rabit0-v1-run17/checkpoints/checkpoint-2000 \
  --gpu cuda:0 \
  --bench cybermetric
```

```bash
poetry run python -m src.eval.run_eval \
  --checkpoint artifacts/qwen2.5-7b/rabit0-v1-run17/checkpoints/checkpoint-2000 \
  --gpu cuda:0 \
  --bench cyberseceval3
```

```bash
poetry run python -m src.eval.run_eval \
  --checkpoint artifacts/qwen2.5-7b/rabit0-v1-run17/checkpoints/checkpoint-2000 \
  --gpu cuda:0 \
  --bench sevenllm
```

```bash
poetry run python -m src.eval.run_eval \
  --checkpoint artifacts/qwen2.5-7b/rabit0-v1-run17/checkpoints/checkpoint-2000 \
  --gpu cuda:0 \
  --bench ctibench
```

```bash
poetry run python -m src.eval.run_eval \
  --checkpoint artifacts/qwen2.5-7b/rabit0-v1-run17/checkpoints/checkpoint-2000 \
  --gpu cuda:0 \
  --bench seceval
```

```bash
poetry run python -m src.eval.run_eval \
  --checkpoint artifacts/qwen2.5-7b/rabit0-v1-run17/checkpoints/checkpoint-2000 \
  --gpu cuda:0 \
  --bench redsagemcq
```

```bash
poetry run python -m src.eval.run_eval \
  --checkpoint artifacts/qwen2.5-7b/rabit0-v1-run17/checkpoints/checkpoint-2000 \
  --gpu cuda:0 \
  --bench cissp
```

```bash
poetry run python -m src.eval.run_eval \
  --checkpoint artifacts/qwen2.5-7b/rabit0-v1-run17/checkpoints/checkpoint-2000 \
  --gpu cuda:0 \
  --bench b3
```

```bash
poetry run python -m src.eval.run_eval \
  --checkpoint artifacts/qwen2.5-7b/rabit0-v1-run17/checkpoints/checkpoint-2000 \
  --gpu cuda:0 \
  --bench mbpp
```

```bash
export OPENAI_API_KEY=...
poetry run python -m src.eval.run_eval \
  --checkpoint artifacts/qwen2.5-7b/rabit0-v1-run17/checkpoints/checkpoint-2000 \
  --gpu cuda:0 \
  --bench coconot
```

```bash
poetry run python -m src.eval.run_eval \
  --checkpoint artifacts/qwen2.5-7b/rabit0-v1-run17/checkpoints/checkpoint-2000 \
  --gpu cuda:0 \
  --bench niah
```

```bash
poetry run python -m src.eval.run_eval \
  --checkpoint artifacts/qwen2.5-7b/rabit0-v1-run17/checkpoints/checkpoint-2000 \
  --gpu cuda:0 \
  --bench worldsense
```

---

## Target 3: Base model (no LoRA)

Base model:

`Qwen/Qwen2.5-7B-Instruct`

Run all benches (all tasks):

```bash
poetry run python -m src.eval.run_eval \
  --base-model Qwen/Qwen2.5-7B-Instruct \
  --gpu cuda:0 \
  --bench secqa
```

```bash
poetry run python -m src.eval.run_eval \
  --base-model Qwen/Qwen2.5-7B-Instruct \
  --gpu cuda:0 \
  --bench cybermetric
```

```bash
poetry run python -m src.eval.run_eval \
  --base-model Qwen/Qwen2.5-7B-Instruct \
  --gpu cuda:0 \
  --bench cyberseceval3
```

```bash
poetry run python -m src.eval.run_eval \
  --base-model Qwen/Qwen2.5-7B-Instruct \
  --gpu cuda:0 \
  --bench sevenllm
```

```bash
poetry run python -m src.eval.run_eval \
  --base-model Qwen/Qwen2.5-7B-Instruct \
  --gpu cuda:0 \
  --bench ctibench
```

```bash
poetry run python -m src.eval.run_eval \
  --base-model Qwen/Qwen2.5-7B-Instruct \
  --gpu cuda:0 \
  --bench seceval
```

```bash
poetry run python -m src.eval.run_eval \
  --base-model Qwen/Qwen2.5-7B-Instruct \
  --gpu cuda:0 \
  --bench redsagemcq
```

```bash
poetry run python -m src.eval.run_eval \
  --base-model Qwen/Qwen2.5-7B-Instruct \
  --gpu cuda:0 \
  --bench cissp
```

```bash
poetry run python -m src.eval.run_eval \
  --base-model Qwen/Qwen2.5-7B-Instruct \
  --gpu cuda:0 \
  --bench b3
```

```bash
poetry run python -m src.eval.run_eval \
  --base-model Qwen/Qwen2.5-7B-Instruct \
  --gpu cuda:0 \
  --bench mbpp
```

```bash
export OPENAI_API_KEY=...
poetry run python -m src.eval.run_eval \
  --base-model Qwen/Qwen2.5-7B-Instruct \
  --gpu cuda:0 \
  --bench coconot
```

```bash
poetry run python -m src.eval.run_eval \
  --base-model Qwen/Qwen2.5-7B-Instruct \
  --gpu cuda:0 \
  --bench niah
```

```bash
poetry run python -m src.eval.run_eval \
  --base-model Qwen/Qwen2.5-7B-Instruct \
  --gpu cuda:0 \
  --bench worldsense
```

---

## Optional flags

- 4-bit load (less VRAM, slower):

```bash
poetry run python -m src.eval.run_eval --help
```

Common additions (examples):
- Add `--use-4bit` to any command to reduce VRAM.
- Add `--dtype float16` (or `float32`) if needed.
- Add `--gpu cuda:0` to pin eval to a specific GPU (`--gpu cuda` also works).
- Add `--limit N` for a quick smoke test.
- Add `--qa-scoring judge --judge-model gpt-4o-mini` for judge-model scoring on free-form QA benchmarks (default is `--qa-scoring rouge_l`).
- For `coconot`: set `OPENAI_API_KEY`; optional `COCONOT_GRADER_MODEL` (default `gpt-3.5-turbo`); optional `COCONOT_USE_SYSTEM_PROMPT=1` for the inspect_evals-style system message on the **evaluated** model; add `--tasks contrast` to run the contrast subset.
- For `mbpp`, add `--tasks pass@1` to log only pass@1 (still runs 5 samples per problem; omits pass@2 and pass@5 JSONL rows).
- For `niah`, add `--tasks quick` for a faster run or `--tasks standard` for the larger default variant.
- For `worldsense`, use task filters such as `--tasks infer_trivial infer_normal` for faster subsets.

---

## Interactive prompt (`src.eval.prompt`)

Use this when you want manual chat-style testing in terminal instead of benchmark scoring.

Base model:

```bash
poetry run python -m src.eval.prompt \
  --model Qwen/Qwen2.5-7B-Instruct \
  --gpu cuda:0
```

Unsloth/LoRA checkpoint (base model auto-resolved from `adapter_config.json`):

```bash
poetry run python -m src.eval.prompt \
  --checkpoint artifacts/qwen2.5-7b/rabit0-v1-run17/checkpoints/checkpoint-2000 \
  --gpu cuda:0
```

Helpful options:
- Add `--use-4bit` to reduce VRAM usage.
- Add `--max-new-tokens 512` for longer answers.
- Add `--do-sample --temperature 0.7 --top-p 0.9` for creative sampling.
- Exit with `quit`, `exit`, or `q`.