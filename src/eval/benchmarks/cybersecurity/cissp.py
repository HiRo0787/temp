"""
CISSP-level benchmark implementation.

Uses CyberSec-Bench, which targets professional certification-level cybersecurity
knowledge (including CISSP-equivalent difficulty), and evaluates free-form
answers with ROUGE-L F1 by default, or judge-model scoring when configured.
"""

import json
import os
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

CISSP_DATASET_ID = "AYI-NEDJIMI/CyberSec-Bench"
CISSP_TASK_VERSIONS: Dict[str, str] = {
    "en": "en",
    "fr": "fr",
    "all": "all",
}


def _rouge_l_f1(prediction: str, target: str) -> float:
    try:
        from rouge_score import rouge_scorer
    except ImportError as exc:
        raise ImportError("rouge-score is required: pip install rouge-score") from exc
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    scores = scorer.score(target or "", prediction or "")
    return float(scores["rougeL"].fmeasure)


def _judge_score_with_model(
    prediction: str,
    target: str,
    *,
    judge_model: str,
) -> float:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ImportError("openai is required for judge scoring: pip install openai") from exc
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required for judge scoring")
    client = OpenAI(api_key=api_key)
    prompt = (
        "You are grading a cybersecurity answer against a reference answer.\n"
        "Return only JSON: {\"score\": <float between 0 and 1>}.\n"
        "Use semantic correctness and completeness.\n\n"
        f"Reference answer:\n{target}\n\n"
        f"Model answer:\n{prediction}\n"
    )
    resp = client.chat.completions.create(
        model=judge_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=64,
    )
    content = (resp.choices[0].message.content or "").strip()
    try:
        payload = json.loads(content)
        score = float(payload.get("score"))
    except Exception as exc:
        raise ValueError(f"Invalid judge response: {content!r}") from exc
    return max(0.0, min(1.0, score))


def _is_language_match(task_key: str, row: dict) -> bool:
    if task_key == "all":
        return True
    return str(row.get("language", "")).strip().lower() == task_key


def format_cissp_prompt(record: dict) -> str:
    """Format one CyberSec-Bench record as an evaluation prompt."""
    question = str(record.get("question", "")).strip()
    category = str(record.get("category", "")).strip()
    subcategory = str(record.get("subcategory", "")).strip()
    difficulty = str(record.get("difficulty", "")).strip()
    return (
        "Answer the following cybersecurity question clearly and concisely.\n\n"
        f"Category: {category}\n"
        f"Subcategory: {subcategory}\n"
        f"Difficulty: {difficulty}\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )


def load_cissp_split(task_key: str, limit: Optional[int] = None) -> List[dict]:
    """Load one CISSP-level task split."""
    if task_key not in CISSP_TASK_VERSIONS:
        valid = tuple(CISSP_TASK_VERSIONS.keys())
        raise ValueError(
            f"load_cissp_split: task_key must be one of {valid}, got {task_key!r}"
        )
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError("datasets library is required: pip install datasets") from exc

    Logger.print_section(f"Loading CISSP benchmark {task_key} ...")
    ds = load_dataset(CISSP_DATASET_ID, split="test")
    rows = [row for row in ds if _is_language_match(task_key, row)]
    if limit is not None:
        rows = rows[:limit]
    Logger.print_info(f"CISSP {task_key}", f"{len(rows)} records loaded")
    return rows


def preload_cissp_tasks(
    tasks: List[str],
    limit: Optional[int] = None,
) -> Dict[str, List[dict]]:
    """Load all requested CISSP benchmark tasks once."""
    Logger.print_section("Pre-loading CISSP benchmark datasets ...")
    preloaded: Dict[str, List[dict]] = {}
    for task in tasks:
        if task in CISSP_TASK_VERSIONS:
            preloaded[task] = load_cissp_split(task, limit=limit)
    return preloaded


def append_result(
    output_path: Path,
    checkpoint_step: Optional[int],
    task: str,
    score: float,
    total: int,
) -> None:
    """Append one CISSP result row as JSONL."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "checkpoint_step": checkpoint_step,
        "task": task,
        "score": score,
        "total": total,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    with open(output_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _run_cissp(
    generate_fn: Callable[[str], str],
    checkpoint_step: Optional[int],
    tasks: List[str],
    output_path: Optional[Path],
    preloaded: Dict[str, List[dict]],
    qa_scoring: str,
    judge_model: Optional[str],
) -> Dict[str, Optional[float]]:
    results: Dict[str, Optional[float]] = {}
    for task in tasks:
        if task not in CISSP_TASK_VERSIONS:
            Logger.print_warning(f"Unknown CISSP task: {task!r}, skipping.")
            results[task] = None
            continue
        records = preloaded.get(task)
        if records is None:
            Logger.print_warning(f"CISSP {task} not in preloaded datasets, skipping.")
            results[task] = None
            continue

        total = len(records)
        score_sum = 0.0
        Logger.print_section(f"--- Evaluating CISSP {task} ({total} records) ---")
        for idx, rec in enumerate(records):
            prompt = format_cissp_prompt(rec)
            try:
                response = generate_fn(prompt)
            except Exception as exc:
                Logger.print_error(f"[{idx + 1}/{total}] ERROR  | {exc}")
                continue
            target = str(rec.get("reference_answer", "")).strip()
            if qa_scoring == "judge":
                if not judge_model:
                    raise ValueError("judge_model must be set when qa_scoring='judge'")
                sample_score = _judge_score_with_model(
                    response, target, judge_model=judge_model
                )
                score_name = "judge_score"
            else:
                sample_score = _rouge_l_f1(response, target)
                score_name = "rougeL_f1"
            score_sum += sample_score
            Logger.print_info(
                f"[{idx + 1}/{total}] CISSP score",
                f"{score_name}={sample_score:.4f}",
            )

        score = score_sum / total if total > 0 else 0.0
        results[task] = score
        label = "judge_score" if qa_scoring == "judge" else "rougeL_f1"
        Logger.print_info(f"CISSP {task}", f"{label}={score:.4f}")
        if output_path is not None:
            append_result(
                output_path=output_path,
                checkpoint_step=checkpoint_step,
                task=task,
                score=score,
                total=total,
            )
    return results


def make_bench_fn(
    tasks: List[str],
    preloaded: Dict[str, List[dict]],
    output_path: Optional[Path] = None,
    *,
    qa_scoring: str = "rouge_l",
    judge_model: Optional[str] = None,
) -> Callable:
    """BenchFn compatible with checkpoint_runner.run_checkpoints()."""
    if qa_scoring not in ("rouge_l", "judge"):
        raise ValueError(f"Unsupported qa_scoring: {qa_scoring!r}")
    return partial(
        _run_cissp,
        tasks=tasks,
        output_path=output_path,
        preloaded=preloaded,
        qa_scoring=qa_scoring,
        judge_model=judge_model,
    )


def print_summary(
    summary: List[Tuple[Optional[int], Dict[str, Optional[float]]]],
    tasks: List[str],
) -> None:
    """Print the score table for all evaluated checkpoints."""
    Logger.print_header("CISSP Evaluation Summary")
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
