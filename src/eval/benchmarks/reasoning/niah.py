"""
Needle-in-a-Haystack (NIAH) benchmark.

Inspired by the Inspect Evals NIAH task:
https://github.com/UKGovernmentBEIS/inspect_evals/tree/main/src/inspect_evals/niah

This local variant generates synthetic haystacks with one embedded fact ("needle")
and scores whether the model retrieves the exact answer string.
"""

from __future__ import annotations

import json
import random
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

NIAH_TASK_VERSIONS: Dict[str, str] = {
    "quick": "quick",
    "standard": "standard",
}

_QUESTION = "According to the context, what is the passcode?"
_INSTRUCTIONS = (
    "Read the following context and answer only with the passcode phrase. "
    "Do not explain."
)
_WORD_BANK = [
    "policy",
    "meeting",
    "project",
    "analysis",
    "timeline",
    "document",
    "section",
    "process",
    "review",
    "result",
    "system",
    "context",
    "record",
    "update",
    "status",
]
_NEEDLES = [
    ("atlas", "The passcode is amber-atlas-739."),
    ("lumen", "The passcode is cobalt-lumen-184."),
    ("orion", "The passcode is silver-orion-552."),
    ("delta", "The passcode is crimson-delta-907."),
]
_TASK_CONFIG = {
    "quick": {"contexts": [180, 320], "positions": [0.2, 0.8], "runs": 1},
    "standard": {"contexts": [300, 600, 900], "positions": [0.1, 0.5, 0.9], "runs": 1},
}


def _token_count(text: str) -> int:
    return len((text or "").split())


def _extract_answer_phrase(needle_sentence: str) -> str:
    match = re.search(r"passcode is (.+?)\.", needle_sentence, flags=re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return needle_sentence.strip().rstrip(".")


def _normalize(text: str) -> str:
    value = (text or "").strip().lower()
    value = re.sub(r"[^a-z0-9\- ]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def _preview_text(text: str, limit: int = 220) -> str:
    one_line = re.sub(r"\s+", " ", (text or "").strip())
    if len(one_line) <= limit:
        return one_line
    return one_line[: limit - 3] + "..."


def _build_haystack_words(target_words: int, *, rng: random.Random) -> str:
    if target_words <= 0:
        return ""
    words = [rng.choice(_WORD_BANK) for _ in range(target_words)]
    return " ".join(words)


def _insert_needle(haystack: str, needle_sentence: str, position: float) -> str:
    words = haystack.split()
    if not words:
        return needle_sentence
    p = min(1.0, max(0.0, position))
    idx = int(round(p * len(words)))
    idx = max(0, min(len(words), idx))
    prefix = " ".join(words[:idx]).strip()
    suffix = " ".join(words[idx:]).strip()
    if prefix and suffix:
        return f"{prefix}. {needle_sentence} {suffix}."
    if prefix:
        return f"{prefix}. {needle_sentence}"
    if suffix:
        return f"{needle_sentence} {suffix}."
    return needle_sentence


def _build_prompt(context: str) -> str:
    return f"{_INSTRUCTIONS}\n\nContext:\n{context}\n\nQuestion: {_QUESTION}"


def _score_retrieval(response: str, expected_answer: str) -> float:
    return 1.0 if _normalize(expected_answer) in _normalize(response) else 0.0


def _task_rows(task: str, limit: Optional[int] = None, *, seed: int = 7) -> List[dict]:
    cfg = _TASK_CONFIG[task]
    rng = random.Random(seed)
    rows: List[dict] = []
    needle_idx = 0
    for ctx_len in cfg["contexts"]:
        for pos in cfg["positions"]:
            for _ in range(cfg["runs"]):
                _, needle_sentence = _NEEDLES[needle_idx % len(_NEEDLES)]
                needle_idx += 1
                answer = _extract_answer_phrase(needle_sentence)
                haystack = _build_haystack_words(ctx_len, rng=rng)
                context = _insert_needle(haystack, needle_sentence, pos)
                rows.append(
                    {
                        "context_length_words": _token_count(context),
                        "position": pos,
                        "prompt": _build_prompt(context),
                        "needle": needle_sentence,
                        "answer": answer,
                    }
                )
                if limit is not None and len(rows) >= limit:
                    return rows
    return rows


def preload_niah_tasks(tasks: List[str], limit: Optional[int] = None) -> Dict[str, List[dict]]:
    Logger.print_section("Pre-loading NIAH synthetic datasets ...")
    preloaded: Dict[str, List[dict]] = {}
    for task in tasks:
        if task not in NIAH_TASK_VERSIONS:
            continue
        rows = _task_rows(task, limit=limit, seed=7)
        Logger.print_info(f"NIAH {task}", f"{len(rows)} samples loaded")
        preloaded[task] = rows
    return preloaded


def append_result(
    output_path: Path,
    checkpoint_step: Optional[int],
    task: str,
    score: float,
    solved: int,
    total: int,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "checkpoint_step": checkpoint_step,
        "task": task,
        "score": score,
        "solved": solved,
        "total": total,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    with open(output_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _run_niah(
    generate_fn: Callable[[str], str],
    checkpoint_step: Optional[int],
    tasks: List[str],
    output_path: Optional[Path],
    preloaded: Dict[str, List[dict]],
) -> Dict[str, Optional[float]]:
    results: Dict[str, Optional[float]] = {}
    for task in tasks:
        if task not in NIAH_TASK_VERSIONS:
            Logger.print_warning(f"Unknown NIAH task: {task!r}, skipping.")
            results[task] = None
            continue
        rows = preloaded.get(task, [])
        if not rows:
            Logger.print_warning(f"NIAH {task}: no rows found, skipping.")
            results[task] = None
            continue

        total = len(rows)
        solved = 0
        Logger.print_section(f"--- Evaluating NIAH {task} ({total} samples) ---")
        for idx, row in enumerate(rows):
            try:
                answer = generate_fn(str(row["prompt"]))
            except Exception as exc:
                Logger.print_error(f"[{idx + 1}/{total}] generate ERROR | {exc}")
                continue
            Logger.print_info(f"[{idx + 1}/{total}] model_output", _preview_text(str(answer)))
            one = _score_retrieval(str(answer), str(row["answer"]))
            solved += int(one)
            Logger.print_info(
                f"[{idx + 1}/{total}]",
                f"score={one:.1f} expected={row['answer']!r}",
            )

        score = solved / total if total > 0 else 0.0
        results[task] = score
        Logger.print_info(f"NIAH {task}", f"retrieval_accuracy={score:.4f} ({solved}/{total})")
        if output_path is not None:
            append_result(
                output_path=output_path,
                checkpoint_step=checkpoint_step,
                task=task,
                score=score,
                solved=solved,
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
        _run_niah,
        tasks=tasks,
        output_path=output_path,
        preloaded=preloaded,
    )


def print_summary(
    summary: List[Tuple[Optional[int], Dict[str, Optional[float]]]],
    tasks: List[str],
) -> None:
    Logger.print_header("NIAH Evaluation Summary (retrieval accuracy)")
    col_w = max(8, max(len(t) for t in tasks))
    Logger.print_section("  step     " + "  ".join(f"{t:>{col_w}}" for t in tasks))
    for step, res in summary:
        step_label = str(step) if step is not None else "final"
        cols = [f"{res.get(t):.4f}" if res.get(t) is not None else "failed" for t in tasks]
        Logger.print_section(f"{step_label:>8}  " + "  ".join(f"{c:>{col_w}}" for c in cols))
