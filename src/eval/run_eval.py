"""
Single CLI entry point for LoRA checkpoint evaluation.

Evaluates one or more LoRA checkpoints against a configurable security
benchmark. The base model is loaded once; adapter weights are swapped
in-place for each checkpoint.

Adding a new benchmark
----------------------
1. Add a module under ``src/eval/benchmarks/`` (e.g. ``cybersecurity/``,
   ``coding/``, or ``safeguards/``) implementing ``make_bench_fn()``,
   ``preload_*_tasks()``, and ``print_summary()`` (see ``secqa.py``).
2. Add the bench name to ``_SUPPORTED_BENCHES`` below.
3. Add an ``elif`` branch in ``main()`` and extend ``_default_tasks_for_bench()``
   when task keys differ from SecQA.

Usage
-----
# Latest checkpoint, SecQA v1 (default):
poetry run python -m src.eval.run_eval \\
    --run-dir artifacts/qwen2.5-7b/rabit0-v1-run12

# Single specific checkpoint:
poetry run python -m src.eval.run_eval \\
    --checkpoint artifacts/qwen2.5-7b/rabit0-v1-run12/checkpoints/checkpoint-3000

# All checkpoints in a run:
poetry run python -m src.eval.run_eval \\
    --run-dir artifacts/qwen2.5-7b/rabit0-v1-run12 \\
    --all-checkpoints

# Both SecQA tasks, limited to 10 questions:
poetry run python -m src.eval.run_eval \\
    --run-dir artifacts/qwen2.5-7b/rabit0-v1-run12 \\
    --bench secqa --tasks v1 v2 --limit 10

See docs/SECQA_EVAL.md for the full reference.
"""

import argparse
import os
import random
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.eval.checkpoint_runner import (
    read_base_model_name,
    resolve_checkpoints,
    run_base_model,
    run_checkpoints,
)
from src.eval.benchmarks.cybersecurity.cybermetric import (
    CYBERMETRIC_TASK_VERSIONS,
    make_bench_fn as make_cybermetric_bench_fn,
    preload_cybermetric_tasks,
    print_summary as print_cybermetric_summary,
)
from src.eval.benchmarks.cybersecurity.cyberseceval3 import (
    CYBERSECEVAL3_TASK_VERSIONS,
    make_bench_fn as make_cyberseceval3_bench_fn,
    preload_cyberseceval3_tasks,
    print_summary as print_cyberseceval3_summary,
)
from src.eval.benchmarks.cybersecurity.secqa import (
    SECQA_TASK_VERSIONS,
    make_bench_fn,
    preload_secqa_tasks,
    print_summary,
)
from src.eval.benchmarks.cybersecurity.sevenllm import (
    SEVENLLM_TASK_VERSIONS,
    make_bench_fn as make_sevenllm_bench_fn,
    preload_sevenllm_tasks,
    print_summary as print_sevenllm_summary,
)
from src.eval.benchmarks.cybersecurity.ctibench import (
    CTIBENCH_TASK_VERSIONS,
    make_bench_fn as make_ctibench_bench_fn,
    preload_ctibench_tasks,
    print_summary as print_ctibench_summary,
)
from src.eval.benchmarks.cybersecurity.seceval import (
    SECEVAL_TASK_VERSIONS,
    make_bench_fn as make_seceval_bench_fn,
    preload_seceval_tasks,
    print_summary as print_seceval_summary,
)
from src.eval.benchmarks.cybersecurity.cissp import (
    CISSP_TASK_VERSIONS,
    make_bench_fn as make_cissp_bench_fn,
    preload_cissp_tasks,
    print_summary as print_cissp_summary,
)
from src.eval.benchmarks.cybersecurity.redsagemcq import (
    REDSAGEMCQ_TASK_VERSIONS,
    make_bench_fn as make_redsagemcq_bench_fn,
    preload_redsagemcq_tasks,
    print_summary as print_redsagemcq_summary,
)
from src.eval.benchmarks.safeguards.b3 import (
    B3_TASK_VERSIONS,
    make_bench_fn as make_b3_bench_fn,
    preload_b3_tasks,
    print_summary as print_b3_summary,
)
from src.eval.benchmarks.safeguards.coconot import (
    COCONOT_TASK_VERSIONS,
    OPTIONAL_SYSTEM_PROMPT,
    default_grader_model,
    make_bench_fn as make_coconot_bench_fn,
    preload_coconot_tasks,
    print_summary as print_coconot_summary,
    use_optional_system_prompt_from_env,
)
from src.eval.benchmarks.coding.mbpp import (
    MBPP_TASK_VERSIONS,
    make_bench_fn as make_mbpp_bench_fn,
    preload_mbpp_tasks,
    print_summary as print_mbpp_summary,
)
from src.eval.benchmarks.reasoning.niah import (
    NIAH_TASK_VERSIONS,
    make_bench_fn as make_niah_bench_fn,
    preload_niah_tasks,
    print_summary as print_niah_summary,
)
from src.eval.benchmarks.reasoning.worldsense import (
    WORLDSENSE_TASK_VERSIONS,
    make_bench_fn as make_worldsense_bench_fn,
    preload_worldsense_tasks,
    print_summary as print_worldsense_summary,
)
from src.utilities.training_logger import Logger
from src.eval.reporting import write_eval_manifest, write_markdown_report

_SUPPORTED_BENCHES = [
    "secqa",
    "cybermetric",
    "cyberseceval3",
    "sevenllm",
    "ctibench",
    "seceval",
    "cissp",
    "redsagemcq",
    "b3",
    "mbpp",
    "coconot",
    "niah",
    "worldsense",
]


def _default_tasks_for_bench(bench: str):
    """Default --tasks when omitted: run all tasks for the benchmark."""
    if bench == "cybermetric":
        return list(CYBERMETRIC_TASK_VERSIONS.keys())
    if bench == "cyberseceval3":
        return list(CYBERSECEVAL3_TASK_VERSIONS.keys())
    if bench == "sevenllm":
        return list(SEVENLLM_TASK_VERSIONS.keys())
    if bench == "ctibench":
        return list(CTIBENCH_TASK_VERSIONS.keys())
    if bench == "seceval":
        return list(SECEVAL_TASK_VERSIONS.keys())
    if bench == "cissp":
        return list(CISSP_TASK_VERSIONS.keys())
    if bench == "redsagemcq":
        return list(REDSAGEMCQ_TASK_VERSIONS.keys())
    if bench == "b3":
        return list(B3_TASK_VERSIONS.keys())
    if bench == "mbpp":
        return list(MBPP_TASK_VERSIONS.keys())
    if bench == "coconot":
        return ["original"]
    if bench == "niah":
        return list(NIAH_TASK_VERSIONS.keys())
    if bench == "worldsense":
        return list(WORLDSENSE_TASK_VERSIONS.keys())
    return list(SECQA_TASK_VERSIONS.keys())


def _model_slug(model_name: str) -> str:
    """Create a filesystem-safe model slug for report paths."""
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", model_name.strip().lower())
    slug = slug.strip("._-")
    return slug or "unknown_model"


def _load_eval_env_file() -> None:
    """
    Load project-root .env key=value pairs into os.environ.

    Existing environment variables are not overridden.
    """
    env_path = _PROJECT_ROOT / ".env"
    if not env_path.is_file():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key:
            continue
        value = value.strip().strip('"').strip("'")
        if key not in os.environ:
            os.environ[key] = value


def _safe_label(value: str | None, *, default: str) -> str:
    """Filesystem-safe label for filenames."""
    if value is None:
        return default
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip().lower())
    slug = slug.strip("._-")
    return slug or default


def _next_run_subdir(parent: Path) -> Path:
    """
    Return next run directory under parent as run-<N>.

    Examples:
      if parent has run-1, run-2 -> returns parent/run-3
      if none exist -> returns parent/run-1
    """
    parent = Path(parent)
    max_idx = 0
    for p in parent.iterdir() if parent.exists() else []:
        if not p.is_dir():
            continue
        m = re.fullmatch(r"run-(\d+)", p.name)
        if not m:
            continue
        max_idx = max(max_idx, int(m.group(1)))
    return parent / f"run-{max_idx + 1}"


def _default_output_path(bench: str, base_model_name: str) -> Path:
    """
    Default output path for a run.

    Uses root eval_report/<model>/<bench>/run-<N>/<bench>__base__<timestamp>.jsonl.
    """
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = _next_run_subdir(Path("eval_report") / _model_slug(base_model_name) / bench)
    filename = f"{_safe_label(bench, default='bench')}__base__{stamp}.jsonl"
    return run_dir / filename


def _default_output_path_for_checkpoints(
    bench: str,
    base_model_name: str,
    checkpoint_pairs: list[tuple[object, Path]],
    run_id: str | None = None,
    eval_run_root: Path | None = None,
) -> Path:
    """
    Default output path when evaluating LoRA checkpoints.

    Layout:
      <eval_run_root>/eval/<bench>/checkpoints/<checkpoint-name>/run-<N>/
        <bench>__<run-id>__<checkpoint-name>__<timestamp>.jsonl

    When ``eval_run_root`` is not provided, falls back to:
      eval_report/<model>/<bench>/checkpoints/<checkpoint-name>/run-<N>/<bench>__<run-id>__<checkpoint-name>__<timestamp>.jsonl

    This makes it obvious which checkpoint was evaluated, even for single-checkpoint runs.
    """
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    if len(checkpoint_pairs) == 1:
        step, ckpt_path = checkpoint_pairs[0]
        if step is not None:
            ckpt_label = f"checkpoint-{step}"
        else:
            ckpt_label = ckpt_path.name or "checkpoint"
    else:
        ckpt_label = "multi"
    filename = (
        f"{_safe_label(bench, default='bench')}__"
        f"{_safe_label(run_id, default='unknown_run')}__"
        f"{_safe_label(ckpt_label, default='checkpoint')}__"
        f"{stamp}.jsonl"
    )
    if eval_run_root is not None:
        base_dir = (
            Path(eval_run_root)
            / "eval"
            / bench
            / "checkpoints"
            / ckpt_label
        )
    else:
        base_dir = (
            Path("eval_report")
            / _model_slug(base_model_name)
            / bench
            / "checkpoints"
            / ckpt_label
        )
    run_dir = _next_run_subdir(base_dir)
    return run_dir / filename


def _infer_run_id(run_dir: str | None, checkpoint: str | None) -> str | None:
    """Infer training run id from --run-dir/--checkpoint path."""
    if run_dir:
        return Path(run_dir).name or None
    if checkpoint:
        ckpt_path = Path(checkpoint)
        # .../<run_id>/checkpoints/checkpoint-<step>
        parent = ckpt_path.parent
        if parent.name == "checkpoints":
            return parent.parent.name or None
    return None


def _checkpoint_label(checkpoint_pairs: list[tuple[object, Path]]) -> str | None:
    """Human-readable checkpoint label for metadata tags."""
    if not checkpoint_pairs:
        return None
    if len(checkpoint_pairs) == 1:
        step, ckpt_path = checkpoint_pairs[0]
        if step is not None:
            return f"checkpoint-{step}"
        return ckpt_path.name or "checkpoint"
    return "multi"


def _eval_target_tag(
    *,
    base_model_mode: bool,
    run_id: str | None,
    checkpoint_label: str | None,
    bench: str,
) -> str:
    """Compact eval tag shown in logs and persisted in manifest/report."""
    if base_model_mode:
        return f"base_model|bench={bench}"
    rid = run_id or "unknown_run"
    ck = checkpoint_label or "unknown_checkpoint"
    return f"lora|run_id={rid}|checkpoint={ck}|bench={bench}"


def _apply_random_limit(
    preloaded: dict[str, list[dict]],
    limit: int | None,
) -> dict[str, list[dict]]:
    """Apply deterministic random sampling per task when --limit is set."""
    if limit is None:
        return preloaded
    sampled: dict[str, list[dict]] = {}
    for task, rows in preloaded.items():
        if len(rows) <= limit:
            sampled[task] = rows
            continue
        # Keep sampling deterministic across repeated runs with same task+limit.
        rng = random.Random(f"{task}:{limit}:eval-limit")
        indices = sorted(rng.sample(range(len(rows)), k=limit))
        sampled[task] = [rows[i] for i in indices]
    return sampled


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate LoRA checkpoints on security benchmarks.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Latest checkpoint in a run (default — v1 runs once):
  poetry run python -m src.eval.run_eval \\
      --run-dir artifacts/qwen2.5-7b/rabit0-v1-run12

  # Single specific checkpoint:
  poetry run python -m src.eval.run_eval \\
      --checkpoint artifacts/qwen2.5-7b/rabit0-v1-run12/checkpoints/checkpoint-3000

  # All checkpoints in a run (v1 runs once per checkpoint):
  poetry run python -m src.eval.run_eval \\
      --run-dir artifacts/qwen2.5-7b/rabit0-v1-run12 \\
      --all-checkpoints

  # Quick test (first 10 questions per task):
  poetry run python -m src.eval.run_eval \\
      --checkpoint artifacts/qwen2.5-7b/rabit0-v1-run12/checkpoints/checkpoint-3000 \\
      --tasks v1 --limit 10
""",
    )

    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument(
        "--checkpoint",
        metavar="CHECKPOINT_DIR",
        help=(
            "Path to a single LoRA checkpoint directory "
            "(e.g. artifacts/qwen2.5-7b/rabit0-v1-run12/checkpoints/checkpoint-3000)."
        ),
    )
    target.add_argument(
        "--run-dir",
        metavar="RUN_DIR",
        help=(
            "Path to a run directory. Scans <run_dir>/checkpoints/checkpoint-* "
            "in ascending step order. Example: artifacts/qwen2.5-7b/rabit0-v1-run12"
        ),
    )
    target.add_argument(
        "--base-model",
        metavar="MODEL_ID_OR_PATH",
        help=(
            "Evaluate a base model directly (no LoRA adapters). "
            "Pass a HuggingFace model id or local path."
        ),
    )

    parser.add_argument(
        "--bench",
        choices=_SUPPORTED_BENCHES,
        default="secqa",
        help=f"Benchmark to run (default: secqa). Supported: {_SUPPORTED_BENCHES}.",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=None,
        help=(
            "Task keys for the selected benchmark "
            "(default: all tasks for the selected benchmark)."
        ),
    )
    parser.add_argument(
        "--all-checkpoints",
        action="store_true",
        default=False,
        help=(
            "Evaluate every checkpoint found in the run directory. "
            "By default only the latest checkpoint is evaluated. "
            "Only applies when --run-dir is used."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="Limit to N random questions per task (deterministic sample).",
    )
    parser.add_argument(
        "--output",
        default=None,
        metavar="PATH",
        help=(
            "JSONL file path to append results to. "
            "Default: base-model mode -> eval_report/<model>/<bench>/...; "
            "LoRA/checkpoint mode -> artifacts/<model>/<run>/eval/<bench>/..."
        ),
    )
    parser.add_argument(
        "--dtype",
        choices=["bfloat16", "float16", "float32"],
        default="bfloat16",
        help="Model precision (default: bfloat16).",
    )
    parser.add_argument(
        "--use-4bit",
        action="store_true",
        default=False,
        help="Load base model in 4-bit quantization (BitsAndBytes) to save VRAM.",
    )
    parser.add_argument(
        "--gpu",
        type=str,
        default=None,
        metavar="DEVICE",
        help=(
            "Device selector: cpu, cuda, or cuda:<index> (e.g. cuda:0). "
            "By default device mapping is automatic."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for deterministic sampling (overrides env/default).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.5,
        help="Generation temperature for evaluation (default: 0.5).",
    )
    parser.add_argument(
        "--qa-scoring",
        choices=["rouge_l", "judge"],
        default="rouge_l",
        help=(
            "Scoring mode for free-form QA tasks (used by sevenllm QA and cissp). "
            "Default: rouge_l."
        ),
    )
    parser.add_argument(
        "--judge-model",
        default=None,
        metavar="MODEL",
        help=(
            "Judge model name for --qa-scoring judge (e.g. gpt-4o-mini). "
            "Requires OPENAI_API_KEY."
        ),
    )

    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    _load_eval_env_file()
    parser = _build_parser()
    args = parser.parse_args()
    tasks = args.tasks if args.tasks is not None else _default_tasks_for_bench(args.bench)

    system_message = None
    if args.bench == "mbpp":
        gen_kw = {
            "max_new_tokens": 1024,
            "do_sample": True,
            "temperature": args.temperature,
            "top_p": 0.95,
        }
    elif args.bench == "coconot":
        gen_kw = {
            "max_new_tokens": 256,
            "do_sample": False,
            "temperature": args.temperature,
            "top_p": 0.95,
        }
        if use_optional_system_prompt_from_env():
            system_message = OPTIONAL_SYSTEM_PROMPT
    else:
        gen_kw = {
            "max_new_tokens": 64,
            "do_sample": False,
            "temperature": args.temperature,
            "top_p": 0.95,
        }

    # Resolve base model / checkpoints.
    if args.base_model:
        base_model_name = args.base_model
        checkpoint_pairs = []
    else:
        checkpoint_pairs = resolve_checkpoints(
            args.checkpoint, args.run_dir, args.all_checkpoints
        )
        # Read base model name from the first checkpoint (all checkpoints share the same base).
        first_ckpt = checkpoint_pairs[0][1]
        base_model_name = read_base_model_name(first_ckpt)

    # Resolve output path (depends on resolved base_model_name for default layout).
    eval_run_root = None
    if not args.base_model:
        # Training artifact root for eval: --run-dir or parent of checkpoints/.
        if args.run_dir:
            p = Path(args.run_dir).resolve()
            eval_run_root = p if p.is_dir() else None
        elif args.checkpoint:
            ckpt = Path(args.checkpoint).resolve()
            if ckpt.parent.name == "checkpoints":
                eval_run_root = ckpt.parent.parent
            else:
                eval_run_root = ckpt if ckpt.is_dir() else ckpt.parent
    if args.output:
        output_path = Path(args.output)
    else:
        if args.base_model:
            output_path = _default_output_path(args.bench, base_model_name)
        else:
            output_path = _default_output_path_for_checkpoints(
                args.bench,
                base_model_name,
                checkpoint_pairs,
                run_id=_infer_run_id(args.run_dir, args.checkpoint),
                eval_run_root=eval_run_root,
            )
    run_id = _infer_run_id(args.run_dir, args.checkpoint)
    ckpt_label = _checkpoint_label(checkpoint_pairs)
    eval_tag = _eval_target_tag(
        base_model_mode=bool(args.base_model),
        run_id=run_id,
        checkpoint_label=ckpt_label,
        bench=args.bench,
    )

    Logger.print_header("Checkpoint Evaluation")
    Logger.print_info("bench", args.bench)
    Logger.print_info("base model", base_model_name)
    Logger.print_info("tasks", " ".join(tasks))
    Logger.print_info("checkpoints", str(len(checkpoint_pairs) if checkpoint_pairs else 1))
    Logger.print_info("eval tag", eval_tag)
    Logger.print_info("output", str(output_path))
    if checkpoint_pairs:
        for step, path in checkpoint_pairs:
            step_label = str(step) if step is not None else "unknown_step"
            Logger.print_info(f"checkpoint[{step_label}]", str(path))
    if args.limit:
        Logger.print_info("limit", f"{args.limit} random questions per task")
    if args.gpu:
        Logger.print_info("gpu", str(args.gpu))
    if args.qa_scoring != "rouge_l":
        Logger.print_info("qa scoring", args.qa_scoring)
    if args.judge_model:
        Logger.print_info("judge model", args.judge_model)

    trace_callback = None

    # Dispatch to the selected benchmark.
    if args.bench == "secqa":
        invalid = [t for t in tasks if t not in SECQA_TASK_VERSIONS]
        if invalid:
            Logger.print_error(
                f"Unknown SecQA tasks: {invalid}. Valid: {list(SECQA_TASK_VERSIONS)}"
            )
            sys.exit(1)

        preloaded = _apply_random_limit(preload_secqa_tasks(tasks, None), args.limit)
        bench_fn = make_bench_fn(
            tasks=tasks,
            preloaded=preloaded,
            output_path=output_path,
        )
        summary = (
            run_base_model(
                bench_fn=bench_fn,
                base_model_name=base_model_name,
                dtype=args.dtype,
                use_4bit=args.use_4bit,
                gpu=args.gpu,
                system_message=system_message,
                trace_callback=trace_callback,
                **gen_kw,
            )
            if args.base_model
            else run_checkpoints(
                checkpoint_pairs=checkpoint_pairs,
                bench_fn=bench_fn,
                base_model_name=base_model_name,
                dtype=args.dtype,
                use_4bit=args.use_4bit,
                gpu=args.gpu,
                system_message=system_message,
                trace_callback=trace_callback,
                **gen_kw,
            )
        )
        print_summary(summary, tasks)

    elif args.bench == "cybermetric":
        invalid = [t for t in tasks if t not in CYBERMETRIC_TASK_VERSIONS]
        if invalid:
            Logger.print_error(
                f"Unknown CyberMetric tasks: {invalid}. "
                f"Valid: {list(CYBERMETRIC_TASK_VERSIONS)}"
            )
            sys.exit(1)

        preloaded = _apply_random_limit(preload_cybermetric_tasks(tasks, None), args.limit)
        bench_fn = make_cybermetric_bench_fn(
            tasks=tasks,
            preloaded=preloaded,
            output_path=output_path,
        )
        summary = (
            run_base_model(
                bench_fn=bench_fn,
                base_model_name=base_model_name,
                dtype=args.dtype,
                use_4bit=args.use_4bit,
                gpu=args.gpu,
                system_message=system_message,
                trace_callback=trace_callback,
                **gen_kw,
            )
            if args.base_model
            else run_checkpoints(
                checkpoint_pairs=checkpoint_pairs,
                bench_fn=bench_fn,
                base_model_name=base_model_name,
                dtype=args.dtype,
                use_4bit=args.use_4bit,
                gpu=args.gpu,
                system_message=system_message,
                trace_callback=trace_callback,
                **gen_kw,
            )
        )
        print_cybermetric_summary(summary, tasks)

    elif args.bench == "cyberseceval3":
        invalid = [t for t in tasks if t not in CYBERSECEVAL3_TASK_VERSIONS]
        if invalid:
            Logger.print_error(
                f"Unknown CyberSecEval3 tasks: {invalid}. "
                f"Valid: {list(CYBERSECEVAL3_TASK_VERSIONS)}"
            )
            sys.exit(1)

        preloaded = _apply_random_limit(preload_cyberseceval3_tasks(tasks, None), args.limit)
        bench_fn = make_cyberseceval3_bench_fn(
            tasks=tasks,
            preloaded=preloaded,
            output_path=output_path,
        )
        summary = (
            run_base_model(
                bench_fn=bench_fn,
                base_model_name=base_model_name,
                dtype=args.dtype,
                use_4bit=args.use_4bit,
                gpu=args.gpu,
                system_message=system_message,
                trace_callback=trace_callback,
                **gen_kw,
            )
            if args.base_model
            else run_checkpoints(
                checkpoint_pairs=checkpoint_pairs,
                bench_fn=bench_fn,
                base_model_name=base_model_name,
                dtype=args.dtype,
                use_4bit=args.use_4bit,
                gpu=args.gpu,
                system_message=system_message,
                trace_callback=trace_callback,
                **gen_kw,
            )
        )
        print_cyberseceval3_summary(summary, tasks)

    elif args.bench == "sevenllm":
        invalid = [t for t in tasks if t not in SEVENLLM_TASK_VERSIONS]
        if invalid:
            Logger.print_error(
                f"Unknown SEvenLLM tasks: {invalid}. "
                f"Valid: {list(SEVENLLM_TASK_VERSIONS)}"
            )
            sys.exit(1)

        preloaded = _apply_random_limit(preload_sevenllm_tasks(tasks, None), args.limit)
        if args.qa_scoring == "rouge_l" and args.judge_model is None:
            bench_fn = make_sevenllm_bench_fn(
                tasks=tasks,
                preloaded=preloaded,
                output_path=output_path,
            )
        else:
            bench_fn = make_sevenllm_bench_fn(
                tasks=tasks,
                preloaded=preloaded,
                output_path=output_path,
                qa_scoring=args.qa_scoring,
                judge_model=args.judge_model,
            )
        summary = (
            run_base_model(
                bench_fn=bench_fn,
                base_model_name=base_model_name,
                dtype=args.dtype,
                use_4bit=args.use_4bit,
                gpu=args.gpu,
                system_message=system_message,
                trace_callback=trace_callback,
                **gen_kw,
            )
            if args.base_model
            else run_checkpoints(
                checkpoint_pairs=checkpoint_pairs,
                bench_fn=bench_fn,
                base_model_name=base_model_name,
                dtype=args.dtype,
                use_4bit=args.use_4bit,
                gpu=args.gpu,
                system_message=system_message,
                trace_callback=trace_callback,
                **gen_kw,
            )
        )
        print_sevenllm_summary(summary, tasks)

    elif args.bench == "ctibench":
        invalid = [t for t in tasks if t not in CTIBENCH_TASK_VERSIONS]
        if invalid:
            Logger.print_error(
                f"Unknown CTIBench tasks: {invalid}. "
                f"Valid: {list(CTIBENCH_TASK_VERSIONS)}"
            )
            sys.exit(1)

        preloaded = _apply_random_limit(preload_ctibench_tasks(tasks, None), args.limit)
        bench_fn = make_ctibench_bench_fn(
            tasks=tasks,
            preloaded=preloaded,
            output_path=output_path,
        )
        summary = (
            run_base_model(
                bench_fn=bench_fn,
                base_model_name=base_model_name,
                dtype=args.dtype,
                use_4bit=args.use_4bit,
                gpu=args.gpu,
                system_message=system_message,
                trace_callback=trace_callback,
                **gen_kw,
            )
            if args.base_model
            else run_checkpoints(
                checkpoint_pairs=checkpoint_pairs,
                bench_fn=bench_fn,
                base_model_name=base_model_name,
                dtype=args.dtype,
                use_4bit=args.use_4bit,
                gpu=args.gpu,
                system_message=system_message,
                trace_callback=trace_callback,
                **gen_kw,
            )
        )
        print_ctibench_summary(summary, tasks)

    elif args.bench == "seceval":
        invalid = [t for t in tasks if t not in SECEVAL_TASK_VERSIONS]
        if invalid:
            Logger.print_error(
                f"Unknown SecEval tasks: {invalid}. "
                f"Valid: {list(SECEVAL_TASK_VERSIONS)}"
            )
            sys.exit(1)

        preloaded = _apply_random_limit(preload_seceval_tasks(tasks, None), args.limit)
        bench_fn = make_seceval_bench_fn(
            tasks=tasks,
            preloaded=preloaded,
            output_path=output_path,
        )
        summary = (
            run_base_model(
                bench_fn=bench_fn,
                base_model_name=base_model_name,
                dtype=args.dtype,
                use_4bit=args.use_4bit,
                gpu=args.gpu,
                system_message=system_message,
                trace_callback=trace_callback,
                **gen_kw,
            )
            if args.base_model
            else run_checkpoints(
                checkpoint_pairs=checkpoint_pairs,
                bench_fn=bench_fn,
                base_model_name=base_model_name,
                dtype=args.dtype,
                use_4bit=args.use_4bit,
                gpu=args.gpu,
                system_message=system_message,
                trace_callback=trace_callback,
                **gen_kw,
            )
        )
        print_seceval_summary(summary, tasks)

    elif args.bench == "cissp":
        invalid = [t for t in tasks if t not in CISSP_TASK_VERSIONS]
        if invalid:
            Logger.print_error(
                f"Unknown CISSP tasks: {invalid}. "
                f"Valid: {list(CISSP_TASK_VERSIONS)}"
            )
            sys.exit(1)

        preloaded = _apply_random_limit(preload_cissp_tasks(tasks, None), args.limit)
        if args.qa_scoring == "rouge_l" and args.judge_model is None:
            bench_fn = make_cissp_bench_fn(
                tasks=tasks,
                preloaded=preloaded,
                output_path=output_path,
            )
        else:
            bench_fn = make_cissp_bench_fn(
                tasks=tasks,
                preloaded=preloaded,
                output_path=output_path,
                qa_scoring=args.qa_scoring,
                judge_model=args.judge_model,
            )
        summary = (
            run_base_model(
                bench_fn=bench_fn,
                base_model_name=base_model_name,
                dtype=args.dtype,
                use_4bit=args.use_4bit,
                gpu=args.gpu,
                system_message=system_message,
                trace_callback=trace_callback,
                **gen_kw,
            )
            if args.base_model
            else run_checkpoints(
                checkpoint_pairs=checkpoint_pairs,
                bench_fn=bench_fn,
                base_model_name=base_model_name,
                dtype=args.dtype,
                use_4bit=args.use_4bit,
                gpu=args.gpu,
                system_message=system_message,
                trace_callback=trace_callback,
                **gen_kw,
            )
        )
        print_cissp_summary(summary, tasks)

    elif args.bench == "redsagemcq":
        invalid = [t for t in tasks if t not in REDSAGEMCQ_TASK_VERSIONS]
        if invalid:
            Logger.print_error(
                f"Unknown RedSageMCQ tasks: {invalid}. "
                f"Valid: {list(REDSAGEMCQ_TASK_VERSIONS)}"
            )
            sys.exit(1)

        preloaded = _apply_random_limit(preload_redsagemcq_tasks(tasks, None), args.limit)
        bench_fn = make_redsagemcq_bench_fn(
            tasks=tasks,
            preloaded=preloaded,
            output_path=output_path,
        )
        summary = (
            run_base_model(
                bench_fn=bench_fn,
                base_model_name=base_model_name,
                dtype=args.dtype,
                use_4bit=args.use_4bit,
                gpu=args.gpu,
                system_message=system_message,
                trace_callback=trace_callback,
                **gen_kw,
            )
            if args.base_model
            else run_checkpoints(
                checkpoint_pairs=checkpoint_pairs,
                bench_fn=bench_fn,
                base_model_name=base_model_name,
                dtype=args.dtype,
                use_4bit=args.use_4bit,
                gpu=args.gpu,
                system_message=system_message,
                trace_callback=trace_callback,
                **gen_kw,
            )
        )
        print_redsagemcq_summary(summary, tasks)

    elif args.bench == "b3":
        invalid = [t for t in tasks if t not in B3_TASK_VERSIONS]
        if invalid:
            Logger.print_error(
                f"Unknown b3 tasks: {invalid}. "
                f"Valid: {list(B3_TASK_VERSIONS)}"
            )
            sys.exit(1)

        preloaded = _apply_random_limit(preload_b3_tasks(tasks, None), args.limit)
        bench_fn = make_b3_bench_fn(
            tasks=tasks,
            preloaded=preloaded,
            output_path=output_path,
        )
        summary = (
            run_base_model(
                bench_fn=bench_fn,
                base_model_name=base_model_name,
                dtype=args.dtype,
                use_4bit=args.use_4bit,
                gpu=args.gpu,
                system_message=system_message,
                trace_callback=trace_callback,
                **gen_kw,
            )
            if args.base_model
            else run_checkpoints(
                checkpoint_pairs=checkpoint_pairs,
                bench_fn=bench_fn,
                base_model_name=base_model_name,
                dtype=args.dtype,
                use_4bit=args.use_4bit,
                gpu=args.gpu,
                system_message=system_message,
                trace_callback=trace_callback,
                **gen_kw,
            )
        )
        print_b3_summary(summary, tasks)

    elif args.bench == "mbpp":
        invalid = [t for t in tasks if t not in MBPP_TASK_VERSIONS]
        if invalid:
            Logger.print_error(
                f"Unknown MBPP tasks: {invalid}. "
                f"Valid: {list(MBPP_TASK_VERSIONS)}"
            )
            sys.exit(1)

        preloaded = _apply_random_limit(preload_mbpp_tasks(tasks, None), args.limit)
        bench_fn = make_mbpp_bench_fn(
            tasks=tasks,
            preloaded=preloaded,
            output_path=output_path,
        )
        summary = (
            run_base_model(
                bench_fn=bench_fn,
                base_model_name=base_model_name,
                dtype=args.dtype,
                use_4bit=args.use_4bit,
                gpu=args.gpu,
                system_message=system_message,
                trace_callback=trace_callback,
                **gen_kw,
            )
            if args.base_model
            else run_checkpoints(
                checkpoint_pairs=checkpoint_pairs,
                bench_fn=bench_fn,
                base_model_name=base_model_name,
                dtype=args.dtype,
                use_4bit=args.use_4bit,
                gpu=args.gpu,
                system_message=system_message,
                trace_callback=trace_callback,
                **gen_kw,
            )
        )
        print_mbpp_summary(summary, tasks)

    elif args.bench == "coconot":
        if not os.environ.get("OPENAI_API_KEY"):
            Logger.print_error(
                "Coconot requires OPENAI_API_KEY for the judge model (see inspect_evals coconot)."
            )
            sys.exit(1)
        invalid = [t for t in tasks if t not in COCONOT_TASK_VERSIONS]
        if invalid:
            Logger.print_error(
                f"Unknown Coconot tasks: {invalid}. "
                f"Valid: {list(COCONOT_TASK_VERSIONS)}"
            )
            sys.exit(1)

        preloaded = _apply_random_limit(preload_coconot_tasks(tasks, None), args.limit)
        grader = default_grader_model()
        Logger.print_info("coconot grader", grader)
        bench_fn = make_coconot_bench_fn(
            tasks=tasks,
            preloaded=preloaded,
            output_path=output_path,
            grader_model=grader,
        )
        summary = (
            run_base_model(
                bench_fn=bench_fn,
                base_model_name=base_model_name,
                dtype=args.dtype,
                use_4bit=args.use_4bit,
                gpu=args.gpu,
                system_message=system_message,
                trace_callback=trace_callback,
                **gen_kw,
            )
            if args.base_model
            else run_checkpoints(
                checkpoint_pairs=checkpoint_pairs,
                bench_fn=bench_fn,
                base_model_name=base_model_name,
                dtype=args.dtype,
                use_4bit=args.use_4bit,
                gpu=args.gpu,
                system_message=system_message,
                trace_callback=trace_callback,
                **gen_kw,
            )
        )
        print_coconot_summary(summary, tasks)

    elif args.bench == "niah":
        invalid = [t for t in tasks if t not in NIAH_TASK_VERSIONS]
        if invalid:
            Logger.print_error(
                f"Unknown NIAH tasks: {invalid}. "
                f"Valid: {list(NIAH_TASK_VERSIONS)}"
            )
            sys.exit(1)

        preloaded = _apply_random_limit(preload_niah_tasks(tasks, None), args.limit)
        bench_fn = make_niah_bench_fn(
            tasks=tasks,
            preloaded=preloaded,
            output_path=output_path,
        )
        summary = (
            run_base_model(
                bench_fn=bench_fn,
                base_model_name=base_model_name,
                dtype=args.dtype,
                use_4bit=args.use_4bit,
                gpu=args.gpu,
                system_message=system_message,
                trace_callback=trace_callback,
                **gen_kw,
            )
            if args.base_model
            else run_checkpoints(
                checkpoint_pairs=checkpoint_pairs,
                bench_fn=bench_fn,
                base_model_name=base_model_name,
                dtype=args.dtype,
                use_4bit=args.use_4bit,
                gpu=args.gpu,
                system_message=system_message,
                trace_callback=trace_callback,
                **gen_kw,
            )
        )
        print_niah_summary(summary, tasks)

    elif args.bench == "worldsense":
        invalid = [t for t in tasks if t not in WORLDSENSE_TASK_VERSIONS]
        if invalid:
            Logger.print_error(
                f"Unknown WorldSense tasks: {invalid}. "
                f"Valid: {list(WORLDSENSE_TASK_VERSIONS)}"
            )
            sys.exit(1)

        preloaded = _apply_random_limit(preload_worldsense_tasks(tasks, None), args.limit)
        bench_fn = make_worldsense_bench_fn(
            tasks=tasks,
            preloaded=preloaded,
            output_path=output_path,
        )
        summary = (
            run_base_model(
                bench_fn=bench_fn,
                base_model_name=base_model_name,
                dtype=args.dtype,
                use_4bit=args.use_4bit,
                gpu=args.gpu,
                system_message=system_message,
                trace_callback=trace_callback,
                **gen_kw,
            )
            if args.base_model
            else run_checkpoints(
                checkpoint_pairs=checkpoint_pairs,
                bench_fn=bench_fn,
                base_model_name=base_model_name,
                dtype=args.dtype,
                use_4bit=args.use_4bit,
                gpu=args.gpu,
                system_message=system_message,
                trace_callback=trace_callback,
                **gen_kw,
            )
        )
        print_worldsense_summary(summary, tasks)

    all_scores = [
        score
        for _, task_scores in summary
        for score in task_scores.values()
        if score is not None
    ]
    if all_scores:
        avg_accuracy = sum(all_scores) / len(all_scores)
        Logger.print_info("avg accuracy (all outputs)", f"{avg_accuracy:.4f}")
    else:
        Logger.print_warning(
            "No numeric task scores were produced; overall average accuracy is unavailable."
        )

    Logger.print_info("results written to", str(output_path))
    manifest_path = write_eval_manifest(
        output_path=output_path,
        bench=args.bench,
        base_model_name=base_model_name,
        tasks=tasks,
        checkpoint_pairs=checkpoint_pairs,
        base_model_mode=bool(args.base_model),
        run_dir=args.run_dir,
        checkpoint=args.checkpoint,
        all_checkpoints=bool(args.all_checkpoints),
        eval_target_tag=eval_tag,
        run_id=run_id,
        checkpoint_label=ckpt_label,
    )
    Logger.print_info("checkpoint manifest", str(manifest_path))

    # Write human-friendly reports next to the JSONL.
    if args.output:
        stem = output_path.stem
        prefix = stem[:-8] if stem.endswith("_results") else stem
        md_path = output_path.with_name(f"{prefix}_report.md")
    else:
        md_path = output_path.with_suffix(".md")
    write_markdown_report(
        report_path=md_path,
        bench=args.bench,
        base_model_name=base_model_name,
        tasks=tasks,
        output_jsonl_path=output_path,
        summary=summary,
        manifest_path=manifest_path,
    )


if __name__ == "__main__":
    main()
