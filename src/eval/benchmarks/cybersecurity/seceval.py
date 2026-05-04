"""
SecEval benchmark implementation.

Implements the SecEval cybersecurity multiple-choice benchmark using a
Hugging Face dataset mirror for stable loading.
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

from src.eval.benchmarks.cybersecurity.secqa import extract_answer_letter
from src.utilities.training_logger import Logger

SECEVAL_DATASET_ID = "RISys-Lab/Benchmarks_CyberSec_SecEval"
SECEVAL_TASK_VERSIONS: Dict[str, str] = {
    "all": "default",
}


def _choice_letter(idx: int) -> str:
    return chr(ord("A") + idx)


def _normalize_answer_letter(value: object) -> Optional[str]:
    text = str(value or "").strip()
    if not text:
        return None
    m = re.search(r"\b([ABCD])\b", text, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    m = re.search(r"\b([0-3])\b", text)
    if m:
        return _choice_letter(int(m.group(1)))
    return extract_answer_letter(text)


def format_seceval_prompt(record: dict) -> str:
    """Format one SecEval row as a MCQ prompt."""
    question = str(record.get("question", "")).strip()
    choices = record.get("choices", [])
    lines: List[str] = []
    if isinstance(choices, list):
        for idx, choice in enumerate(choices[:4]):
            lines.append(f"{_choice_letter(idx)}) {choice}")
    options_block = "\n".join(lines)
    return (
        "Answer the following cybersecurity multiple choice question. "
        "Respond with only the letter of the correct answer (A, B, C, or D).\n\n"
        f"Question: {question}\n\n"
        f"{options_block}\n\n"
        "Answer:"
    )


def load_seceval_split(task_key: str, limit: Optional[int] = None) -> List[dict]:
    """Load the SecEval test split."""
    subset = SECEVAL_TASK_VERSIONS.get(task_key)
    if subset is None:
        valid = tuple(SECEVAL_TASK_VERSIONS.keys())
        raise ValueError(
            f"load_seceval_split: task_key must be one of {valid}, got {task_key!r}"
        )
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError("datasets library is required: pip install datasets") from exc

    Logger.print_section(f"Loading SecEval {task_key} ({subset}) ...")
    ds = load_dataset(SECEVAL_DATASET_ID, subset, split="test")
    records = list(ds)
    if limit is not None:
        records = records[:limit]
    Logger.print_info(f"SecEval {task_key}", f"{len(records)} records loaded")
    return records


def preload_seceval_tasks(
    tasks: List[str],
    limit: Optional[int] = None,
) -> Dict[str, List[dict]]:
    """Load all requested SecEval tasks once."""
    Logger.print_section("Pre-loading SecEval task datasets ...")
    preloaded: Dict[str, List[dict]] = {}
    for task in tasks:
        if task in SECEVAL_TASK_VERSIONS:
            preloaded[task] = load_seceval_split(task, limit=limit)
    return preloaded


def append_result(
    output_path: Path,
    checkpoint_step: Optional[int],
    task: str,
    accuracy: float,
    correct: int,
    total: int,
) -> None:
    """Append one SecEval result row as JSONL."""
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


def _run_seceval(
    generate_fn: Callable[[str], str],
    checkpoint_step: Optional[int],
    tasks: List[str],
    output_path: Optional[Path],
    preloaded: Dict[str, List[dict]],
) -> Dict[str, Optional[float]]:
    results: Dict[str, Optional[float]] = {}
    for task in tasks:
        if task not in SECEVAL_TASK_VERSIONS:
            Logger.print_warning(f"Unknown SecEval task: {task!r}, skipping.")
            results[task] = None
            continue
        records = preloaded.get(task)
        if records is None:
            Logger.print_warning(f"SecEval {task} not in preloaded datasets, skipping.")
            results[task] = None
            continue

        total = len(records)
        correct = 0
        Logger.print_section(f"--- Evaluating SecEval {task} ({total} records) ---")
        for idx, rec in enumerate(records):
            prompt = format_seceval_prompt(rec)
            try:
                response = generate_fn(prompt)
            except Exception as exc:
                Logger.print_error(f"[{idx + 1}/{total}] ERROR  | {exc}")
                continue
            expected = _normalize_answer_letter(rec.get("answer", ""))
            predicted = extract_answer_letter(response)
            if predicted is not None and predicted == expected:
                correct += 1
                Logger.print_success(f"[{idx + 1}/{total}] SOLVED")
            else:
                Logger.print_warning(f"[{idx + 1}/{total}] NOT SOLVED")

        accuracy = correct / total if total > 0 else 0.0
        results[task] = accuracy
        Logger.print_info(f"SecEval {task}", f"accuracy={accuracy:.4f} ({correct}/{total})")
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


def make_bench_fn(
    tasks: List[str],
    preloaded: Dict[str, List[dict]],
    output_path: Optional[Path] = None,
) -> Callable:
    """BenchFn compatible with checkpoint_runner.run_checkpoints()."""
    return partial(
        _run_seceval,
        tasks=tasks,
        output_path=output_path,
        preloaded=preloaded,
    )


def print_summary(
    summary: List[Tuple[Optional[int], Dict[str, Optional[float]]]],
    tasks: List[str],
) -> None:
    """Print the accuracy table for all evaluated checkpoints."""
    Logger.print_header("SecEval Evaluation Summary")
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
