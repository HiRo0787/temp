"""
SecQA benchmark implementation.

Runs the SecQA (Security Question Answering) multiple-choice benchmark
from zefang-liu/secqa on HuggingFace against any checkpoint supplied by
checkpoint_runner.run_checkpoints().

Adding a new benchmark
----------------------
Create a new file alongside this one (e.g. cybermetric.py) that exports:
    TASK_VERSIONS    — dict mapping short keys to HuggingFace config names.
    make_bench_fn()  — factory returning a BenchFn-compatible callable.
    preload_tasks()  — load all dataset splits once.
    print_summary()  — print the per-checkpoint accuracy table.
Then register the new benchmark name in src/eval/run_eval.py.

Public API
----------
SECQA_TASK_VERSIONS   Dict mapping short task keys ("v1", "v2") to HF config names.
extract_answer_letter Parse A/B/C/D from raw model output.
format_secqa_prompt   Format one dataset record as a MCQ prompt.
load_secqa_split      Load one SecQA split from HuggingFace.
preload_secqa_tasks   Load all requested task splits once.
append_result         Append one result row to a JSONL file.
make_bench_fn         Create a BenchFn(generate_fn, step) -> results callable.
print_summary         Print the accuracy table for all evaluated checkpoints.
"""

import json
import re
import sys
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.utilities.training_logger import Logger


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SECQA_TASK_VERSIONS: Dict[str, str] = {
    "v1": "secqa_v1",
    "v2": "secqa_v2",
}

_ANSWER_PATTERN = re.compile(
    r"(?:answer\s*(?:is|:)\s*|^|\s)([ABCD])(?:\)|\.|\s|$)",
    re.IGNORECASE | re.MULTILINE,
)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def extract_answer_letter(text: str) -> Optional[str]:
    """
    Extract a single A/B/C/D answer letter from model output.

    Tries explicit patterns first (e.g. "Answer: B", "The answer is C"),
    then falls back to the first standalone letter in the text.

    Args:
        text: Raw model output string.

    Returns:
        Uppercase letter "A", "B", "C", or "D", or None if not found.
    """
    if not text:
        return None
    m = _ANSWER_PATTERN.search(text.strip())
    if m:
        return m.group(1).upper()
    m2 = re.search(r"\b([ABCD])\b", text.strip(), re.IGNORECASE)
    if m2:
        return m2.group(1).upper()
    return None


# ---------------------------------------------------------------------------
# Prompt formatting
# ---------------------------------------------------------------------------

def format_secqa_prompt(record: dict) -> str:
    """
    Format a SecQA dataset record as a multiple-choice prompt.

    Instructs the model to respond with only the answer letter so that
    extract_answer_letter can reliably parse the output.

    Args:
        record: A row from zefang-liu/secqa with keys Question, A, B, C, D, Answer.

    Returns:
        Plain-text prompt string. Callers are responsible for applying
        any chat template before passing to the model.
    """
    return (
        "Answer the following multiple choice security question. "
        "Respond with only the letter of the correct answer (A, B, C, or D).\n\n"
        f"Question: {record['Question']}\n\n"
        f"A) {record['A']}\n"
        f"B) {record['B']}\n"
        f"C) {record['C']}\n"
        f"D) {record['D']}\n\n"
        "Answer:"
    )


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_secqa_split(version: str, limit: Optional[int] = None) -> List[dict]:
    """
    Load the SecQA HuggingFace dataset for a given version config.

    Args:
        version: HuggingFace config name — "secqa_v1" (110 questions, easier)
                 or "secqa_v2" (100 questions, harder). Use SECQA_TASK_VERSIONS
                 to map short task keys ("v1", "v2") to these config names.
        limit: Optional cap on number of records returned (for quick testing).

    Returns:
        List of record dicts with keys Question, A, B, C, D, Answer, Explanation.

    Raises:
        ImportError: If the datasets library is not installed.
        ValueError: If version is not a recognised SecQA config name.
    """
    valid = tuple(SECQA_TASK_VERSIONS.values())
    if version not in valid:
        raise ValueError(
            f"load_secqa_split: version must be one of {valid}, got {version!r}"
        )
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "datasets library is required: pip install datasets"
        ) from exc

    Logger.print_section(f"Loading SecQA {version} from zefang-liu/secqa ...")
    ds = load_dataset("zefang-liu/secqa", version, split="test")
    records = list(ds)
    if limit is not None:
        records = records[:limit]
    Logger.print_info(f"SecQA {version}", f"{len(records)} questions loaded")
    return records


def preload_secqa_tasks(
    tasks: List[str],
    limit: Optional[int] = None,
) -> Dict[str, List[dict]]:
    """
    Load all requested task splits exactly once.

    Pass the returned dict to make_bench_fn() so that datasets are not
    re-downloaded on every checkpoint iteration.

    Args:
        tasks: List of short task keys ("v1", "v2").
        limit: Optional cap on questions per task.

    Returns:
        Dict mapping each task key to its list of records.
    """
    Logger.print_section("Pre-loading SecQA task datasets ...")
    preloaded: Dict[str, List[dict]] = {}
    for task in tasks:
        version = SECQA_TASK_VERSIONS.get(task)
        if version:
            preloaded[task] = load_secqa_split(version, limit)
    return preloaded


# ---------------------------------------------------------------------------
# Result persistence
# ---------------------------------------------------------------------------

def append_result(
    output_path: Path,
    checkpoint_step: Optional[int],
    task: str,
    accuracy: float,
    correct: int,
    total: int,
) -> None:
    """
    Append a single SecQA result row as a JSON line to the output file.

    Creates parent directories if they do not exist.

    Args:
        output_path: Path to the JSONL results file.
        checkpoint_step: Training step number, or None for the final model.
        task: Task key (e.g. "v1", "v2").
        accuracy: Fraction of questions answered correctly (0-1).
        correct: Number of correct answers.
        total: Total number of questions evaluated.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "checkpoint_step": checkpoint_step,
        "task": task,
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    with open(output_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

def _run_secqa(
    generate_fn: Callable[[str], str],
    checkpoint_step: Optional[int],
    tasks: List[str],
    output_path: Optional[Path],
    preloaded: Dict[str, List[dict]],
) -> Dict[str, Optional[float]]:
    """
    Core SecQA evaluation loop. Use make_bench_fn() to create the callable.

    Args:
        generate_fn: Callable that accepts a plain-text prompt and returns
                     the model's response string.
        checkpoint_step: Training step number recorded in result rows.
        tasks: List of short task keys to evaluate.
        output_path: JSONL file to append per-task results (optional).
        preloaded: Dict from preload_secqa_tasks() — avoids re-downloading.

    Returns:
        Dict mapping each task key to accuracy in [0, 1], or None on error.
    """
    results: Dict[str, Optional[float]] = {}

    for task in tasks:
        version = SECQA_TASK_VERSIONS.get(task)
        if version is None:
            Logger.print_warning(f"Unknown SecQA task: {task!r}, skipping.")
            results[task] = None
            continue

        records = preloaded.get(task)
        if records is None:
            Logger.print_warning(f"SecQA {task} not in preloaded datasets, skipping.")
            results[task] = None
            continue

        total = len(records)
        correct = 0

        Logger.print_section(f"--- Evaluating SecQA {task} ({total} questions) ---")
        for idx, rec in enumerate(records):
            question_text = rec.get("Question", rec.get("question", "")).strip()
            prompt = format_secqa_prompt(rec)
            try:
                response = generate_fn(prompt)
            except Exception as exc:
                Logger.print_error(
                    f"[{idx + 1}/{total}] ERROR  | Q: {question_text} | {exc}"
                )
                continue
            predicted = extract_answer_letter(response)
            expected = str(rec.get("Answer", rec.get("answer", ""))).strip().upper()
            if predicted == expected:
                correct += 1
                Logger.print_success(
                    f"[{idx + 1}/{total}] SOLVED     | predicted={predicted}"
                    f" expected={expected} | {question_text}"
                )
            else:
                Logger.print_warning(
                    f"[{idx + 1}/{total}] NOT SOLVED | predicted={predicted}"
                    f" expected={expected} | {question_text}"
                )

        accuracy = correct / total if total > 0 else 0.0
        results[task] = accuracy
        Logger.print_info(
            f"SecQA {task}", f"accuracy={accuracy:.4f} ({correct}/{total})"
        )

        if output_path is not None:
            append_result(
                output_path=output_path,
                checkpoint_step=checkpoint_step,
                task=task,
                accuracy=accuracy,
                correct=correct,
                total=total,
            )

    return results


# ---------------------------------------------------------------------------
# bench_fn factory
# ---------------------------------------------------------------------------

def make_bench_fn(
    tasks: List[str],
    preloaded: Dict[str, List[dict]],
    output_path: Optional[Path] = None,
) -> Callable:
    """
    Create a BenchFn compatible with checkpoint_runner.run_checkpoints().

    The returned callable accepts (generate_fn, checkpoint_step) and runs
    SecQA evaluation using the preloaded records.

    Args:
        tasks: List of short task keys to evaluate ("v1", "v2").
        preloaded: Dict from preload_secqa_tasks() — datasets loaded once.
        output_path: JSONL file to append results to (optional).

    Returns:
        bench_fn(generate_fn, checkpoint_step) -> Dict[str, Optional[float]]
    """
    return partial(
        _run_secqa,
        tasks=tasks,
        output_path=output_path,
        preloaded=preloaded,
    )


# ---------------------------------------------------------------------------
# Summary printing
# ---------------------------------------------------------------------------

def print_summary(
    summary: List[Tuple[Optional[int], Dict[str, Optional[float]]]],
    tasks: List[str],
) -> None:
    """Print the accuracy table for all evaluated checkpoints."""
    Logger.print_header("SecQA Evaluation Summary")
    col_w = max(10, max(len(t) for t in tasks))
    Logger.print_section("  step     " + "  ".join(f"{t:>{col_w}}" for t in tasks))
    for step, res in summary:
        step_label = str(step) if step is not None else "final"
        cols = [
            f"{res.get(t):.4f}" if res.get(t) is not None else "failed"
            for t in tasks
        ]
        Logger.print_section(
            f"{step_label:>8}  " + "  ".join(f"{c:>{col_w}}" for c in cols)
        )
