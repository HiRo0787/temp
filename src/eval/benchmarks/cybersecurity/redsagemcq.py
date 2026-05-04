"""
RedSageMCQ benchmark implementation.

Implements the RedSageMCQ cybersecurity multiple-choice benchmark using a
Hugging Face dataset.
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

REDSAGEMCQ_DATASET_ID = "RISys-Lab/Benchmarks_CyberSec_RedSageMCQ"
REDSAGEMCQ_TASK_VERSIONS: Dict[str, str] = {
    "cybersecurity_knowledge_frameworks": "cybersecurity_knowledge_frameworks",
    "cybersecurity_knowledge_generals": "cybersecurity_knowledge_generals",
    "cybersecurity_skills": "cybersecurity_skills",
    "cybersecurity_tools_cli": "cybersecurity_tools_cli",
    "cybersecurity_tools_kali": "cybersecurity_tools_kali",
    "all": "all",
}
REDSAGEMCQ_TASK_NAMES: Dict[str, str] = {
    "cybersecurity_knowledge_frameworks": "Cybersecurity Knowledge Frameworks",
    "cybersecurity_knowledge_generals": "Cybersecurity General Knowledge",
    "cybersecurity_skills": "Cybersecurity Practical Skills",
    "cybersecurity_tools_cli": "Cybersecurity CLI Tools",
    "cybersecurity_tools_kali": "Cybersecurity Kali Tools",
    "all": "All RedSageMCQ Configurations",
}
REDSAGEMCQ_TASK_SOURCES: Dict[str, str] = {
    "cybersecurity_knowledge_frameworks": "MITRE ATT&CK, CAPEC, CWE, OWASP",
    "cybersecurity_knowledge_generals": "Wikipedia (Cybersecurity subset), Roadmap.sh",
    "cybersecurity_skills": "HackTricks, CTF write-ups, Exploit DB",
    "cybersecurity_tools_cli": "tldr-pages, Unix man pages",
    "cybersecurity_tools_kali": "Kali Tools Documentation",
    "all": "All sources combined",
}
_REDSAGEMCQ_BASE_TASKS: Tuple[str, ...] = (
    "cybersecurity_knowledge_frameworks",
    "cybersecurity_knowledge_generals",
    "cybersecurity_skills",
    "cybersecurity_tools_cli",
    "cybersecurity_tools_kali",
)


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


def _extract_choices(record: dict) -> List[str]:
    answers = record.get("answers")
    if isinstance(answers, dict):
        vals = [str(answers.get(k, "")) for k in ("A", "B", "C", "D")]
        if any(v.strip() for v in vals):
            return vals

    choices = record.get("choices")
    if isinstance(choices, list):
        return [str(c) for c in choices[:4]]

    options = record.get("options")
    if isinstance(options, list):
        return [str(c) for c in options[:4]]

    values: List[str] = []
    for key in ("A", "B", "C", "D"):
        if key in record:
            values.append(str(record.get(key, "")))
    return values[:4]


def _extract_expected_answer(record: dict) -> Optional[str]:
    return _normalize_answer_letter(
        record.get(
            "solution",
            record.get("answer", record.get("Answer", record.get("label", ""))),
        )
    )


def format_redsagemcq_prompt(record: dict) -> str:
    """Format one RedSageMCQ row as a MCQ prompt."""
    question = str(record.get("question", record.get("Question", ""))).strip()
    choices = _extract_choices(record)
    lines = [f"{_choice_letter(idx)}) {choice}" for idx, choice in enumerate(choices)]
    options_block = "\n".join(lines)
    return (
        "Answer the following cybersecurity multiple choice question. "
        "Respond with only the letter of the correct answer (A, B, C, or D).\n\n"
        f"Question: {question}\n\n"
        f"{options_block}\n\n"
        "Answer:"
    )


def load_redsagemcq_split(task_key: str, limit: Optional[int] = None) -> List[dict]:
    """Load the RedSageMCQ test split."""
    subset = REDSAGEMCQ_TASK_VERSIONS.get(task_key)
    if subset is None:
        valid = tuple(REDSAGEMCQ_TASK_VERSIONS.keys())
        raise ValueError(
            f"load_redsagemcq_split: task_key must be one of {valid}, got {task_key!r}"
        )
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError("datasets library is required: pip install datasets") from exc

    task_name = REDSAGEMCQ_TASK_NAMES.get(task_key, task_key)
    Logger.print_section(f"Loading RedSageMCQ {task_name} ({subset}) ...")
    if subset == "all":
        # The upstream "all" config currently has a schema mismatch across files.
        # Build an equivalent aggregate split from stable per-domain configs.
        records: List[dict] = []
        for base_task in _REDSAGEMCQ_BASE_TASKS:
            base_subset = REDSAGEMCQ_TASK_VERSIONS[base_task]
            ds = load_dataset(REDSAGEMCQ_DATASET_ID, base_subset, split="test")
            records.extend(list(ds))
    else:
        ds = load_dataset(REDSAGEMCQ_DATASET_ID, subset, split="test")
        records = list(ds)
    if limit is not None:
        records = records[:limit]
    Logger.print_info(f"RedSageMCQ {task_key}", f"{len(records)} records loaded")
    return records


def preload_redsagemcq_tasks(
    tasks: List[str],
    limit: Optional[int] = None,
) -> Dict[str, List[dict]]:
    """Load all requested RedSageMCQ tasks once."""
    Logger.print_section("Pre-loading RedSageMCQ task datasets ...")
    preloaded: Dict[str, List[dict]] = {}
    for task in tasks:
        if task in REDSAGEMCQ_TASK_VERSIONS:
            preloaded[task] = load_redsagemcq_split(task, limit=limit)
    return preloaded


def append_result(
    output_path: Path,
    checkpoint_step: Optional[int],
    task: str,
    accuracy: float,
    correct: int,
    total: int,
) -> None:
    """Append one RedSageMCQ result row as JSONL."""
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


def _run_redsagemcq(
    generate_fn: Callable[[str], str],
    checkpoint_step: Optional[int],
    tasks: List[str],
    output_path: Optional[Path],
    preloaded: Dict[str, List[dict]],
) -> Dict[str, Optional[float]]:
    results: Dict[str, Optional[float]] = {}
    for task in tasks:
        if task not in REDSAGEMCQ_TASK_VERSIONS:
            Logger.print_warning(f"Unknown RedSageMCQ task: {task!r}, skipping.")
            results[task] = None
            continue
        records = preloaded.get(task)
        if records is None:
            Logger.print_warning(f"RedSageMCQ {task} not in preloaded datasets, skipping.")
            results[task] = None
            continue

        total = len(records)
        correct = 0
        Logger.print_section(f"--- Evaluating RedSageMCQ {task} ({total} records) ---")
        for idx, rec in enumerate(records):
            prompt = format_redsagemcq_prompt(rec)
            try:
                response = generate_fn(prompt)
            except Exception as exc:
                Logger.print_error(f"[{idx + 1}/{total}] ERROR  | {exc}")
                continue
            expected = _extract_expected_answer(rec)
            predicted = extract_answer_letter(response)
            if predicted is not None and predicted == expected:
                correct += 1
                Logger.print_success(f"[{idx + 1}/{total}] SOLVED")
            else:
                Logger.print_warning(f"[{idx + 1}/{total}] NOT SOLVED")

        accuracy = correct / total if total > 0 else 0.0
        results[task] = accuracy
        Logger.print_info(
            f"RedSageMCQ {task}", f"accuracy={accuracy:.4f} ({correct}/{total})"
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


def make_bench_fn(
    tasks: List[str],
    preloaded: Dict[str, List[dict]],
    output_path: Optional[Path] = None,
) -> Callable:
    """BenchFn compatible with checkpoint_runner.run_checkpoints()."""
    return partial(
        _run_redsagemcq,
        tasks=tasks,
        output_path=output_path,
        preloaded=preloaded,
    )


def print_summary(
    summary: List[Tuple[Optional[int], Dict[str, Optional[float]]]],
    tasks: List[str],
) -> None:
    """Print the accuracy table for all evaluated checkpoints."""
    Logger.print_header("RedSageMCQ Evaluation Summary")
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
