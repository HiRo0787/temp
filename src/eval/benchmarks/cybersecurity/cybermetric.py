"""
CyberMetric benchmark implementation.

Loads the multiple-choice CyberMetric datasets (80, 500, 2000, 10000 questions)
from the public GitHub repo (same pinned revision as UKGovernmentBEIS/inspect_evals
cybermetric task). Paper: https://arxiv.org/abs/2402.07688

Public API
----------
CYBERMETRIC_TASK_VERSIONS   Short task keys to CyberMetric-*.json dataset names.
format_cybermetric_prompt Format one record as an MCQ prompt.
load_cybermetric_split    Download (if needed) and load one subset.
preload_cybermetric_tasks Load all requested subsets once.
make_bench_fn             BenchFn factory for checkpoint_runner.run_checkpoints.
print_summary             Print per-checkpoint accuracy table.
"""

import json
import os
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.eval.benchmarks.cybersecurity.secqa import append_result, extract_answer_letter
from src.utilities.training_logger import Logger

# Same revision as inspect_evals/src/inspect_evals/cybermetric/cybermetric.py
CYBERMETRIC_DATASET_REVISION = "205262cdf5022ba890e792efd176fb19d42913fa"
_GITHUB_RAW = "https://raw.githubusercontent.com/cybermetric/CyberMetric"

CYBERMETRIC_TASK_VERSIONS: Dict[str, str] = {
    "80": "CyberMetric-80",
    "500": "CyberMetric-500",
    "2000": "CyberMetric-2000",
    "10000": "CyberMetric-10000",
}


def default_cybermetric_cache_dir() -> Path:
    """Cache directory for downloaded CyberMetric-*.json files."""
    xdg = os.environ.get("XDG_CACHE_HOME")
    base = Path(xdg) if xdg else Path.home() / ".cache"
    return base / "rabit0_rsr" / "cybermetric"


def format_cybermetric_prompt(record: dict) -> str:
    """
    Format a CyberMetric JSON record as a multiple-choice prompt.

    Args:
        record: Dict with keys question, answers (map letter -> text), solution.

    Returns:
        Plain-text prompt for the model.
    """
    answers = record["answers"]
    ordered_keys = sorted(answers.keys(), key=lambda k: (len(str(k)), str(k)))
    lines = [f"{key}) {answers[key]}" for key in ordered_keys]
    options_block = "\n".join(lines)
    return (
        "Answer the following multiple choice question. "
        "Respond with only the letter of the correct answer (A, B, C, or D).\n\n"
        f"Question: {record['question']}\n\n"
        f"{options_block}\n\n"
        "Answer:"
    )


def _dataset_url(display_name: str) -> str:
    return (
        f"{_GITHUB_RAW}/{CYBERMETRIC_DATASET_REVISION}/"
        f"{display_name}-v1.json"
    )


def _ensure_json_file(json_path: Path, url: str) -> None:
    if json_path.exists():
        return
    json_path.parent.mkdir(parents=True, exist_ok=True)
    Logger.print_info("Downloading CyberMetric data", f"{url}")
    try:
        with urllib.request.urlopen(url, timeout=120) as resp:
            json_path.write_bytes(resp.read())
    except (urllib.error.URLError, OSError) as exc:
        raise RuntimeError(f"Failed to download CyberMetric dataset from {url}: {exc}") from exc


def load_cybermetric_split(
    task_key: str,
    limit: Optional[int] = None,
    cache_dir: Optional[Path] = None,
) -> List[dict]:
    """
    Load one CyberMetric subset by short task key.

    Args:
        task_key: One of "80", "500", "2000", "10000".
        limit: Optional max number of questions.
        cache_dir: Override cache directory (for tests).

    Returns:
        List of records with question, answers, solution.

    Raises:
        ValueError: Unknown task_key.
        RuntimeError: Download failure.
    """
    display = CYBERMETRIC_TASK_VERSIONS.get(task_key)
    if not display:
        valid = tuple(CYBERMETRIC_TASK_VERSIONS.keys())
        raise ValueError(
            f"load_cybermetric_split: task_key must be one of {valid}, got {task_key!r}"
        )
    cache = cache_dir if cache_dir is not None else default_cybermetric_cache_dir()
    json_path = cache / f"{display}-v1.json"
    _ensure_json_file(json_path, _dataset_url(display))

    Logger.print_section(f"Loading CyberMetric {task_key} ({display}) ...")
    with open(json_path, encoding="utf-8") as f:
        payload = json.load(f)
    questions = payload.get("questions")
    if not isinstance(questions, list):
        raise ValueError(f"Invalid CyberMetric JSON: missing 'questions' list in {json_path}")

    records: List[dict] = []
    for row in questions:
        if isinstance(row, dict) and "question" in row and "answers" in row and "solution" in row:
            records.append(row)

    if limit is not None:
        records = records[:limit]
    Logger.print_info(f"CyberMetric {task_key}", f"{len(records)} questions loaded")
    return records


def preload_cybermetric_tasks(
    tasks: List[str],
    limit: Optional[int] = None,
    cache_dir: Optional[Path] = None,
) -> Dict[str, List[dict]]:
    """Load all requested CyberMetric subsets once."""
    Logger.print_section("Pre-loading CyberMetric task datasets ...")
    preloaded: Dict[str, List[dict]] = {}
    for task in tasks:
        if task in CYBERMETRIC_TASK_VERSIONS:
            preloaded[task] = load_cybermetric_split(task, limit, cache_dir=cache_dir)
    return preloaded


def _run_cybermetric(
    generate_fn: Callable[[str], str],
    checkpoint_step: Optional[int],
    tasks: List[str],
    output_path: Optional[Path],
    preloaded: Dict[str, List[dict]],
) -> Dict[str, Optional[float]]:
    results: Dict[str, Optional[float]] = {}

    for task in tasks:
        if task not in CYBERMETRIC_TASK_VERSIONS:
            Logger.print_warning(f"Unknown CyberMetric task: {task!r}, skipping.")
            results[task] = None
            continue

        records = preloaded.get(task)
        if records is None:
            Logger.print_warning(
                f"CyberMetric {task} not in preloaded datasets, skipping."
            )
            results[task] = None
            continue

        total = len(records)
        correct = 0
        Logger.print_section(f"--- Evaluating CyberMetric {task} ({total} questions) ---")

        for idx, rec in enumerate(records):
            qtext = str(rec.get("question", "")).strip()
            prompt = format_cybermetric_prompt(rec)
            try:
                response = generate_fn(prompt)
            except Exception as exc:
                Logger.print_error(
                    f"[{idx + 1}/{total}] ERROR  | Q: {qtext} | {exc}"
                )
                continue
            predicted = extract_answer_letter(response)
            expected = str(rec.get("solution", "")).strip().upper()
            if predicted == expected:
                correct += 1
                Logger.print_success(
                    f"[{idx + 1}/{total}] SOLVED     | predicted={predicted}"
                    f" expected={expected} | {qtext}"
                )
            else:
                Logger.print_warning(
                    f"[{idx + 1}/{total}] NOT SOLVED | predicted={predicted}"
                    f" expected={expected} | {qtext}"
                )

        accuracy = correct / total if total > 0 else 0.0
        results[task] = accuracy
        Logger.print_info(
            f"CyberMetric {task}", f"accuracy={accuracy:.4f} ({correct}/{total})"
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
        _run_cybermetric,
        tasks=tasks,
        output_path=output_path,
        preloaded=preloaded,
    )


def print_summary(
    summary: List[Tuple[Optional[int], Dict[str, Optional[float]]]],
    tasks: List[str],
) -> None:
    """Print the accuracy table for all evaluated checkpoints."""
    Logger.print_header("CyberMetric Evaluation Summary")
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
