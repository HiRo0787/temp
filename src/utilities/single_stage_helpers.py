"""
Single-Stage Training Helpers (single-stage logic only).

Used by single_stage_finetune.py. Multi-stage training
(multi_stage_finetune.py) does not use this module.

Helpers: default learning rate, output dir resolution, save_steps/eval_strategy,
checkpoint state patching, latest checkpoint resolution, post-train benchmark subprocess eval.
Output is stored under artifacts/<model_name>/rabit0-v1-run1, rabit0-v1-run2, ...
(e.g. artifacts/qwen2.5-7b/rabit0-v1-run1). Logs use a separate folder.
"""

import gc
import json
import logging
import os
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from packaging import version

from src.infra.project_paths import get_paths
from src.utilities.artifact_paths import get_artifact_run_dir
from src.utilities.training_config import TrainingConfig
from src.utilities.training_logger import Logger

logger = logging.getLogger(__name__)

# Repo root: src/utilities/single_stage_helpers.py -> parent of src
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class CheckpointStateError(Exception):
    """Raised when checkpoint trainer_state.json cannot be read or written."""
    pass


def get_default_learning_rate() -> float:
    """Default learning rate for single-stage LoRA fine-tuning (one run, one LR).

    Uses a conservative LR (5e-5) to reduce instability on instruction tuning
    while still allowing LoRA adapters to converge reliably.
    """
    return 2e-5


def build_launch_command(argv: List[str]) -> str:
    """Build a shell-safe launch command string from argv values."""
    return "python " + " ".join(shlex.quote(str(arg)) for arg in argv)


def write_run_config_file(output_dir: Union[str, Path], config_payload: Dict[str, Any]) -> Path:
    """
    Persist run hyperparameters and launch metadata as output_dir/config.json.
    """
    config_path = Path(output_dir).resolve() / "config.json"
    config_path.write_text(
        json.dumps(config_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return config_path


def build_run_config_payload(
    *,
    run_id: str,
    launch_command: Optional[str],
    model_config: Any,
    model_name: str,
    output_dir: Union[str, Path],
    num_epochs: int,
    batch_size: int,
    gradient_accumulation_steps: int,
    learning_rate: float,
    weight_decay: float,
    lr_scheduler_type: str,
    warmup_ratio: float,
    logging_steps: int,
    eval_steps: int,
    per_device_eval_batch_size: int,
    max_eval_samples: Optional[int],
    save_steps: int,
    resume_from_checkpoint: Optional[Union[bool, str]],
    merge_lora: bool,
    dtype: str,
    use_unsloth: bool,
    use_4bit: bool,
    use_8bit: bool,
    data_manifest: Optional[Union[str, Path]],
    data_manifest_dest: Union[str, Path],
    cli_args: Optional[Dict[str, Any]],
    post_train_eval: Optional[Dict[str, Any]] = None,
    lora_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Construct normalized config payload for run-level provenance."""
    return {
        "run_id": run_id,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "launch_command": launch_command,
        "model": {
            "key": model_config.key,
            "name": model_name,
            "version": model_config.version,
            "size": model_config.size,
        },
        "lora": lora_config or {},
        "training": {
            "output_dir": str(output_dir),
            "num_epochs": num_epochs,
            "per_device_train_batch_size": batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "effective_batch_size": batch_size * gradient_accumulation_steps,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "lr_scheduler_type": lr_scheduler_type,
            "warmup_ratio": warmup_ratio,
            "logging_steps": logging_steps,
            "eval_steps": eval_steps,
            "per_device_eval_batch_size": per_device_eval_batch_size,
            "max_eval_samples": max_eval_samples,
            "save_steps": save_steps,
            "resume_from_checkpoint": resume_from_checkpoint,
            "merge_lora": merge_lora,
            "dtype": dtype,
            "use_unsloth": use_unsloth,
            "use_4bit": use_4bit,
            "use_8bit": use_8bit,
        },
        "data": {
            "manifest": str(data_manifest_dest) if data_manifest else None,
        },
        "post_train_eval": post_train_eval,
        "cli_args": cli_args or {},
    }


def get_default_weight_decay() -> float:
    """Default AdamW weight decay for single-stage training (matches HF TrainingArguments default)."""
    return 0.0


def get_default_eval_steps() -> int:
    """Default evaluation interval (run eval every N steps) for single-stage training."""
    return TrainingConfig.EVAL_STEPS_DEFAULT


def resolve_output_dir(
    output_dir: Optional[str],
    model_config: Any,
    version: str = "v1.0",
    suffix: str = "vllm",
) -> str:
    """
    Resolve output directory for single-stage training.
    If output_dir is None, use artifacts/<model_name>/rabit0-v1-runN (final finetune outputs).

    Args:
        output_dir: User-provided output dir or None.
        model_config: Model config with .version and .size (e.g. from registry).
        version: Model version prefix (e.g. "v1.0").
        suffix: Name suffix (e.g. "vllm"); unused when using artifact path.

    Returns:
        Resolved output directory path (string).
    """
    if output_dir is not None:
        return output_dir
    if model_config is None:
        raise ValueError("resolve_output_dir: model_config is required when output_dir is None")
    if not hasattr(model_config, "version") or not hasattr(model_config, "size"):
        raise ValueError(
            "resolve_output_dir: model_config must have .version and .size "
            "(e.g. from registry)"
        )
    try:
        paths = get_paths()
        model_name = paths.generate_artifact_model_name(
            model_version=model_config.version,
            model_size=model_config.size,
        )
        run_prefix = paths.generate_artifact_prefix(version=version)
        return get_artifact_run_dir(model_name, run_prefix=run_prefix, update_latest=True)
    except OSError as e:
        raise ValueError(
            f"Failed to create or resolve output directory (artifacts run dir): {e}"
        ) from e
    except (ValueError, TypeError) as e:
        raise ValueError(
            f"Failed to resolve output directory: {e}"
        ) from e


def compute_save_steps(eval_steps: int) -> int:
    """Compute save_steps so load_best_model_at_end works (save_steps multiple of eval_steps).

    When eval_steps >= 100, save_steps = eval_steps. Otherwise round up so we save at least
    every 100 steps while keeping save_steps a multiple of eval_steps.
    """
    if not isinstance(eval_steps, int) or eval_steps <= 0:
        raise ValueError(
            f"compute_save_steps: eval_steps must be a positive integer, got {eval_steps!r}"
        )
    if eval_steps >= 100:
        return eval_steps
    return ((100 + eval_steps - 1) // eval_steps) * eval_steps


def get_eval_strategy_kwargs() -> Dict[str, str]:
    """Return the correct eval strategy kwarg for the installed transformers version.

    Transformers >= 4.46 uses 'eval_strategy'; older versions use 'evaluation_strategy'.

    Raises:
        ImportError: If transformers is not installed.
        ValueError: If transformers version cannot be parsed.
    """
    try:
        import transformers
    except ImportError as e:
        raise ImportError(
            "get_eval_strategy_kwargs requires transformers. Install with: pip install transformers"
        ) from e
    try:
        v = version.parse(transformers.__version__)
    except (TypeError, ValueError) as e:
        raise ValueError(
            f"Cannot parse transformers version {getattr(transformers, '__version__', '?')}: {e}"
        ) from e
    if v >= version.parse("4.46.0"):
        return {"eval_strategy": "steps"}
    return {"evaluation_strategy": "steps"}


def get_latest_checkpoint_path(checkpoints_dir: Union[str, Path]) -> Optional[Path]:
    """Return the path to the latest checkpoint directory, or None if none exist.

    Latest is the checkpoint-* dir whose step number (e.g. 100 in checkpoint-100) is largest.
    Returns None if checkpoints_dir does not exist or is not a directory.
    """
    if checkpoints_dir is None:
        return None
    try:
        checkpoints_dir = Path(checkpoints_dir)
    except TypeError:
        raise TypeError(
            f"get_latest_checkpoint_path: checkpoints_dir must be str or Path, got {type(checkpoints_dir).__name__}"
        )
    if not checkpoints_dir.is_dir():
        return None
    try:
        entries = list(checkpoints_dir.iterdir())
    except OSError as e:
        logger.warning("Cannot list checkpoints dir %s: %s", checkpoints_dir, e)
        return None
    pattern = "checkpoint-"
    candidates = [
        d for d in entries
        if d.is_dir() and d.name.startswith(pattern)
    ]
    if not candidates:
        return None

    def step_from_name(d: Path) -> int:
        try:
            return int(d.name[len(pattern):])
        except ValueError:
            return -1

    return max(candidates, key=step_from_name)


def resolve_post_train_eval_checkpoint_path(
    dirs: Dict[str, Path],
    source: str,
    *,
    best_model_checkpoint: Optional[str] = None,
) -> Path:
    """
    Resolve a LoRA directory for post-training benchmark evaluation.

    Args:
        dirs: Artifact subdirs dict (keys ``checkpoints``, ``lora``, ...) from
            ``ensure_artifact_run_subdirs``.
        source: ``"best"`` or ``"final"``.
        best_model_checkpoint: Optional ``TrainerState.best_model_checkpoint`` path
            from Hugging Face (directory with adapter files).

    Returns:
        Resolved path to a directory containing ``adapter_config.json``.

    Raises:
        ValueError: If ``source`` is not ``best`` or ``final``.
        FileNotFoundError: If no usable adapter directory exists.
    """
    if source not in ("best", "final"):
        raise ValueError(
            "resolve_post_train_eval_checkpoint_path: source must be "
            f"'best' or 'final', got {source!r}"
        )

    def _has_adapter(p: Path) -> bool:
        return p.is_dir() and (p / "adapter_config.json").is_file()

    lora = Path(dirs["lora"]).resolve()

    if source == "final":
        latest = get_latest_checkpoint_path(dirs["checkpoints"])
        if latest is not None and _has_adapter(latest):
            return latest.resolve()
        Logger.print_warning(
            "No valid latest checkpoint-* dir; using lora/ for post-train eval."
        )
        if _has_adapter(lora):
            return lora
        raise FileNotFoundError(
            f"No LoRA adapter (adapter_config.json) in lora dir: {lora}"
        )

    if best_model_checkpoint:
        p = Path(best_model_checkpoint).resolve()
        if _has_adapter(p):
            return p
        Logger.print_warning(
            "best_model_checkpoint path missing or has no adapter; "
            "using lora/ for post-train eval."
        )
    if _has_adapter(lora):
        return lora
    raise FileNotFoundError(
        f"No LoRA adapter (adapter_config.json) for post-train eval under {lora}"
    )


def get_default_warmup_ratio() -> float:
    """Default warmup ratio for single-stage training.

    A ratio scales with total training steps across different dataset sizes and
    effective batch configurations, providing a stable optimization ramp.
    """
    return 0.01


def get_default_save_total_limit() -> Optional[int]:
    """Default max number of checkpoints to keep."""
    return 3


def patch_checkpoint_trainer_state(
    resume_from_checkpoint: Union[bool, str, None],
    checkpoints_dir: Union[str, Path],
    eval_steps: int,
    save_steps: int,
) -> None:
    """Update checkpoint trainer_state.json so resumed training uses current eval_steps/save_steps.

    When resuming, the Trainer loads scheduler from trainer_state.json; if that file has
    eval_steps=50/save_steps=100, it keeps evaluating/saving every 50/100 steps instead of
    the current args. Patching the file before train() ensures the current schedule is used.

    Raises:
        CheckpointStateError: If trainer_state.json exists but cannot be read or written.
        ValueError: If resume_from_checkpoint is a path that does not exist or is not a directory.
    """
    if resume_from_checkpoint is None:
        return
    if resume_from_checkpoint is True:
        checkpoint_dir = get_latest_checkpoint_path(checkpoints_dir)
        if checkpoint_dir is None:
            logger.debug(
                "patch_checkpoint_trainer_state: no latest checkpoint found in %s",
                checkpoints_dir,
            )
            return
    else:
        try:
            checkpoint_dir = Path(resume_from_checkpoint)
        except TypeError:
            raise TypeError(
                f"resume_from_checkpoint must be bool, str, or Path, got {type(resume_from_checkpoint).__name__}"
            )
        if not checkpoint_dir.exists():
            raise ValueError(
                f"resume_from_checkpoint path does not exist: {checkpoint_dir}"
            )
        if not checkpoint_dir.is_dir():
            raise ValueError(
                f"resume_from_checkpoint path is not a directory: {checkpoint_dir}"
            )
    state_file = checkpoint_dir / "trainer_state.json"
    if not state_file.is_file():
        logger.debug(
            "patch_checkpoint_trainer_state: no trainer_state.json at %s",
            state_file,
        )
        return
    try:
        with open(state_file, "r", encoding="utf-8") as f:
            state = json.load(f)
    except json.JSONDecodeError as e:
        raise CheckpointStateError(
            f"Invalid JSON in {state_file}: {e}"
        ) from e
    except OSError as e:
        raise CheckpointStateError(
            f"Cannot read {state_file}: {e}"
        ) from e
    state["eval_steps"] = eval_steps
    state["save_steps"] = save_steps
    try:
        with open(state_file, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)
    except OSError as e:
        raise CheckpointStateError(
            f"Cannot write {state_file}: {e}"
        ) from e


def supported_eval_benches() -> List[str]:
    """Names accepted by ``src.eval.run_eval`` (lazy import to limit import cost)."""
    from src.eval.run_eval import _SUPPORTED_BENCHES

    return list(_SUPPORTED_BENCHES)


def _benchmark_tasks_from_catalog(bench: str) -> Optional[List[str]]:
    """
    Read task keys for a benchmark from ``src/eval/benchmarks/benchmarks_catalog.yaml``.

    Returns None when the file or parser is unavailable, so callers can fallback
    to ``run_eval`` defaults.
    """
    try:
        import yaml  # type: ignore
    except ImportError:
        return None

    catalog_path = _PROJECT_ROOT / "src" / "eval" / "benchmarks" / "benchmarks_catalog.yaml"
    if not catalog_path.is_file():
        return None

    try:
        with open(catalog_path, "r", encoding="utf-8") as f:
            doc = yaml.safe_load(f) or {}
    except OSError:
        return None

    grouped = doc.get("tasks_by_benchmark")
    if not isinstance(grouped, dict):
        return None
    bench_meta = grouped.get(bench)
    if not isinstance(bench_meta, dict):
        return None
    task_rows = bench_meta.get("tasks")
    if not isinstance(task_rows, list):
        return None

    out: List[str] = []
    for row in task_rows:
        if isinstance(row, dict):
            key = row.get("task_key")
            if isinstance(key, str) and key.strip():
                out.append(key.strip())
    return out or None


def _default_tasks_for_bench(bench: str) -> List[str]:
    """Fallback task list from run_eval defaults."""
    from src.eval.run_eval import _default_tasks_for_bench

    return list(_default_tasks_for_bench(bench))


def tasks_for_bench_limit(bench: str, task_count: Optional[int]) -> Optional[List[str]]:
    """
    Resolve task keys for one bench with optional count limit.

    If task_count is None, returns None so run_eval keeps its own defaults.
    If task_count is set, returns the first N task keys from catalog; falls back
    to run_eval defaults when catalog lookup is unavailable.
    """
    if task_count is None:
        return None
    if task_count < 1:
        raise ValueError(f"task_count must be >= 1, got {task_count}")

    tasks = _benchmark_tasks_from_catalog(bench) or _default_tasks_for_bench(bench)
    return list(tasks[:task_count])


def release_training_gpu_for_subprocess_eval(trainer: Any, tuner: Any) -> None:
    """Move the training model to CPU and clear CUDA cache before ``run_eval`` subprocess.

    The Hugging Face ``Trainer`` keeps optimizer state on GPU. Callers must drop their
    reference after this returns, for example ``del trainer``, then run post-train eval.

    Without this, a child ``python -m src.eval.run_eval`` process often hits
    ``CUDA ... busy or unavailable`` or OOM while loading LoRA weights because the
    parent process still holds most of the device memory.
    """
    try:
        import torch
    except ImportError:
        return

    model = getattr(tuner, "model", None)
    if model is None:
        return
    if not torch.cuda.is_available():
        return

    Logger.print_info(
        "Post-train eval GPU handoff",
        "moving training model to CPU; caller should delete the Trainer next",
    )
    try:
        model.to("cpu")
    except Exception as exc:
        Logger.print_warning(
            f"Training model could not be moved to CPU before post-train eval: {exc}"
        )
        return
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    time.sleep(1)


def restore_training_model_cuda(tuner: Any) -> None:
    """Move ``tuner.model`` back to CUDA after post-train subprocess eval (for ``test_model``)."""
    try:
        import torch
    except ImportError:
        return
    model = getattr(tuner, "model", None)
    if model is None or not torch.cuda.is_available():
        return
    try:
        model.to(torch.device("cuda:0"))
    except Exception as exc:
        Logger.print_warning(
            f"Could not restore training model to CUDA after eval: {exc}"
        )
        return
    gc.collect()
    Logger.print_success("Training model restored on CUDA for inference tests.")


def run_post_training_eval(
    checkpoint_path: Union[str, Path],
    benches: List[str],
    *,
    gpu: Optional[str] = None,
    dtype: str = "bfloat16",
    use_4bit: bool = False,
    task_count: Optional[int] = None,
) -> None:
    """Run ``python -m src.eval.run_eval`` once per benchmark in a subprocess.

    On non-zero exit, runs the same command once more; if it still fails, logs a
    warning and continues with the next bench.
    """
    checkpoint_path = Path(checkpoint_path).resolve()
    env = os.environ.copy()
    for bench in benches:
        if bench == "coconot" and not env.get("OPENAI_API_KEY"):
            Logger.print_warning(
                "Skipping post-train eval bench coconot: OPENAI_API_KEY is not set."
            )
            continue
        cmd = [
            sys.executable,
            "-m",
            "src.eval.run_eval",
            "--checkpoint",
            str(checkpoint_path),
            "--bench",
            bench,
            "--dtype",
            dtype,
        ]
        if gpu:
            cmd.extend(["--gpu", gpu])
        if use_4bit:
            cmd.append("--use-4bit")
        selected_tasks = tasks_for_bench_limit(bench, task_count)
        if selected_tasks:
            cmd.extend(["--tasks", *selected_tasks])
        Logger.print_section(f"Post-train eval: bench={bench}")
        Logger.print_info("checkpoint", str(checkpoint_path))
        Logger.print_info("command", " ".join(cmd))
        proc: Optional[subprocess.CompletedProcess] = None
        for attempt in range(2):
            if attempt == 1:
                prev_code = proc.returncode if proc is not None else "?"
                Logger.print_warning(
                    f"Post-train eval bench={bench} failed (exit {prev_code}); retrying once."
                )
                Logger.print_info("command (retry)", " ".join(cmd))
            proc = subprocess.run(cmd, cwd=str(_PROJECT_ROOT), env=env)
            if proc.returncode == 0:
                break
        if proc is not None and proc.returncode != 0:
            Logger.print_warning(
                f"Post-train eval bench={bench} failed after retry (exit {proc.returncode}); "
                "skipping this bench."
            )
