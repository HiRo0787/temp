"""
CTIBench benchmark implementation.

Provides a practical local-eval integration for the CTIBench dataset family:
- cti-mcq: multiple-choice CTI knowledge questions
- cti-ate: attack technique extraction
- cti-rcm: root-cause mapping
- cti-vsp: vulnerability severity prediction

This implementation uses deterministic local scoring:
- cti-mcq: exact option-letter accuracy
- other tasks: normalized exact-match against GT field
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

CTIBENCH_DATASET_ID = "RISys-Lab/Benchmarks_CyberSec_CTI-Bench"
CTIBENCH_TASK_VERSIONS: Dict[str, str] = {
    "mcq": "cti-mcq",
    "ate": "cti-ate",
    "rcm": "cti-rcm",
    "vsp": "cti-vsp",
}


def _normalize_text(text: object) -> str:
    s = str(text or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def _extract_gt_letter(gt_text: object) -> Optional[str]:
    text = str(gt_text or "")
    m = re.search(r"\boption\s*([abcd])\b", text, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    return extract_answer_letter(text)


def format_ctibench_prompt(task_key: str, record: dict) -> str:
    """Format one CTIBench sample to a model prompt."""
    if task_key == "mcq":
        return (
            "Answer the following CTI multiple choice question. "
            "Respond with only the letter of the correct answer (A, B, C, or D).\n\n"
            f"Question: {record.get('Question', '')}\n\n"
            f"A) {record.get('Option A', '')}\n"
            f"B) {record.get('Option B', '')}\n"
            f"C) {record.get('Option C', '')}\n"
            f"D) {record.get('Option D', '')}\n\n"
            "Answer:"
        )
    prompt = str(record.get("Prompt", "")).strip()
    if prompt:
        return prompt
    return f"Question: {record.get('Question', '')}\n\nAnswer:"


def _is_correct(task_key: str, response: str, record: dict) -> bool:
    gt = record.get("GT", "")
    if task_key == "mcq":
        expected = _extract_gt_letter(gt)
        predicted = extract_answer_letter(response)
        return predicted is not None and predicted == expected
    return _normalize_text(response) == _normalize_text(gt)


def load_ctibench_split(task_key: str, limit: Optional[int] = None) -> List[dict]:
    """Load one CTIBench subset from Hugging Face."""
    subset = CTIBENCH_TASK_VERSIONS.get(task_key)
    if subset is None:
        valid = tuple(CTIBENCH_TASK_VERSIONS.keys())
        raise ValueError(
            f"load_ctibench_split: task_key must be one of {valid}, got {task_key!r}"
        )
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError("datasets library is required: pip install datasets") from exc

    Logger.print_section(f"Loading CTIBench {task_key} ({subset}) ...")
    ds = load_dataset(CTIBENCH_DATASET_ID, subset, split="test")
    records = list(ds)
    if limit is not None:
        records = records[:limit]
    Logger.print_info(f"CTIBench {task_key}", f"{len(records)} records loaded")
    return records


def preload_ctibench_tasks(
    tasks: List[str],
    limit: Optional[int] = None,
) -> Dict[str, List[dict]]:
    """Load all requested CTIBench subsets once."""
    Logger.print_section("Pre-loading CTIBench task datasets ...")
    preloaded: Dict[str, List[dict]] = {}
    for task in tasks:
        if task in CTIBENCH_TASK_VERSIONS:
            preloaded[task] = load_ctibench_split(task, limit=limit)
    return preloaded


def append_result(
    output_path: Path,
    checkpoint_step: Optional[int],
    task: str,
    accuracy: float,
    correct: int,
    total: int,
) -> None:
    """Append one CTIBench result row as JSONL."""
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


def _run_ctibench(
    generate_fn: Callable[[str], str],
    checkpoint_step: Optional[int],
    tasks: List[str],
    output_path: Optional[Path],
    preloaded: Dict[str, List[dict]],
) -> Dict[str, Optional[float]]:
    results: Dict[str, Optional[float]] = {}

    for task in tasks:
        if task not in CTIBENCH_TASK_VERSIONS:
            Logger.print_warning(f"Unknown CTIBench task: {task!r}, skipping.")
            results[task] = None
            continue
        records = preloaded.get(task)
        if records is None:
            Logger.print_warning(f"CTIBench {task} not in preloaded datasets, skipping.")
            results[task] = None
            continue

        total = len(records)
        correct = 0
        Logger.print_section(f"--- Evaluating CTIBench {task} ({total} records) ---")
        for idx, rec in enumerate(records):
            prompt = format_ctibench_prompt(task, rec)
            try:
                response = generate_fn(prompt)
            except Exception as exc:
                Logger.print_error(f"[{idx + 1}/{total}] ERROR  | {exc}")
                continue
            if _is_correct(task, response, rec):
                correct += 1
                Logger.print_success(f"[{idx + 1}/{total}] SOLVED")
            else:
                Logger.print_warning(f"[{idx + 1}/{total}] NOT SOLVED")

        accuracy = correct / total if total > 0 else 0.0
        results[task] = accuracy
        Logger.print_info(f"CTIBench {task}", f"accuracy={accuracy:.4f} ({correct}/{total})")
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
        _run_ctibench,
        tasks=tasks,
        output_path=output_path,
        preloaded=preloaded,
    )


def print_summary(
    summary: List[Tuple[Optional[int], Dict[str, Optional[float]]]],
    tasks: List[str],
) -> None:
    """Print the accuracy table for all evaluated checkpoints."""
    Logger.print_header("CTIBench Evaluation Summary")
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
