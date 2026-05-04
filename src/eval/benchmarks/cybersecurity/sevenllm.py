"""
SEvenLLM benchmark implementation.

Implements four tasks aligned with inspect_evals:
- sevenllm_mcq_zh
- sevenllm_mcq_en
- sevenllm_qa_zh
- sevenllm_qa_en

Scoring:
- MCQ tasks use accuracy (parsed answer letter).
- QA tasks use ROUGE-L F1 by default, or judge-model scoring when configured.

To keep a single report metric across mixed task types, this module reports a
single `score` in [0, 1] where higher is better.
"""

import json
import os
import re
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

from src.eval.benchmarks.cybersecurity.secqa import extract_answer_letter
from src.utilities.training_logger import Logger

SEVENLLM_DATASET_REVISION = "1de23ce55cadc984d3f3a7b52c4035a68c6cd5b0"
SEVENLLM_DATASET_URL = (
    "https://huggingface.co/datasets/Multilingual-Multimodal-NLP/"
    f"SEVENLLM-Dataset/raw/{SEVENLLM_DATASET_REVISION}/test.jsonl"
)

SEVENLLM_TASK_VERSIONS: Dict[str, Tuple[str, str]] = {
   
    "sevenllm_mcq_en": ("en", "mcq"),
    
    "sevenllm_qa_en": ("en", "qa"),
}

_ZH_PATTERN = re.compile(r"[\u4e00-\u9fff]")


def default_sevenllm_cache_dir() -> Path:
    """Cache directory for the SEvenLLM dataset JSONL file."""
    xdg = os.environ.get("XDG_CACHE_HOME")
    base = Path(xdg) if xdg else Path.home() / ".cache"
    return base / "rabit0_rsr" / "sevenllm"


def _contains_zh(text: str) -> bool:
    return bool(_ZH_PATTERN.search(text or ""))


def _record_language(record: dict) -> str:
    instruction = record.get("instruction")
    if isinstance(instruction, str):
        text = instruction
    elif isinstance(instruction, dict):
        text = str(instruction.get("question", ""))
    else:
        text = ""
    return "zh" if _contains_zh(text) else "en"


def _record_format(record: dict) -> str:
    return "qa" if isinstance(record.get("instruction"), str) else "mcq"


def _ensure_dataset_file(jsonl_path: Path) -> None:
    if jsonl_path.exists():
        return
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    Logger.print_info("Downloading SEvenLLM data", SEVENLLM_DATASET_URL)
    try:
        with urllib.request.urlopen(SEVENLLM_DATASET_URL, timeout=120) as resp:
            jsonl_path.write_bytes(resp.read())
    except (urllib.error.URLError, OSError) as exc:
        raise RuntimeError(
            f"Failed to download SEvenLLM dataset from {SEVENLLM_DATASET_URL}: {exc}"
        ) from exc


def _load_all_records(cache_dir: Optional[Path] = None) -> List[dict]:
    cache = cache_dir if cache_dir is not None else default_sevenllm_cache_dir()
    jsonl_path = cache / "test.jsonl"
    _ensure_dataset_file(jsonl_path)
    records: List[dict] = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if isinstance(row, dict):
                records.append(row)
    return records


def _format_mcq_prompt(record: dict) -> str:
    instruction = record["instruction"]
    question = str(instruction["question"])
    choices = instruction["choice"]
    context = str(record.get("input", "")).strip()
    return (
        "Answer the following multiple choice cybersecurity question. "
        "Respond with only the letter of the correct answer (A, B, C, or D).\n\n"
        f"Question: {question}\n\n"
        f"Context: {context}\n\n"
        f"A) {choices['A']}\n"
        f"B) {choices['B']}\n"
        f"C) {choices['C']}\n"
        f"D) {choices['D']}\n\n"
        "Answer:"
    )


def _format_qa_prompt(record: dict) -> str:
    instruction = str(record.get("instruction", ""))
    context = str(record.get("input", "")).strip()
    return (
        "Below is a cybersecurity instruction and context. "
        "Write a concise and correct answer.\n\n"
        f"Instruction: {instruction}\n\n"
        f"Context: {context}\n\n"
        "Answer:"
    )


def _qa_target_text(output_obj: object) -> str:
    if isinstance(output_obj, str):
        return output_obj
    return json.dumps(output_obj, ensure_ascii=False, sort_keys=True)


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
    """
    Score one QA sample via an external judge model.

    Expects OPENAI_API_KEY and the openai package.
    """
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ImportError("openai is required for judge scoring: pip install openai") from exc
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required for judge scoring")
    client = OpenAI(api_key=api_key)
    prompt = (
        "You are grading a model answer against a reference answer.\n"
        "Return only a JSON object: {\"score\": <float between 0 and 1>}.\n"
        "Use 1.0 for semantically equivalent answers, 0.0 for fully incorrect answers.\n\n"
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


def load_sevenllm_split(
    task_key: str,
    limit: Optional[int] = None,
    cache_dir: Optional[Path] = None,
) -> List[dict]:
    """Load one SEvenLLM task by key."""
    task_meta = SEVENLLM_TASK_VERSIONS.get(task_key)
    if task_meta is None:
        valid = tuple(SEVENLLM_TASK_VERSIONS.keys())
        raise ValueError(
            f"load_sevenllm_split: task_key must be one of {valid}, got {task_key!r}"
        )
    language, data_format = task_meta
    Logger.print_section(f"Loading SEvenLLM {task_key} ...")
    all_rows = _load_all_records(cache_dir=cache_dir)
    filtered = [
        row
        for row in all_rows
        if _record_language(row) == language and _record_format(row) == data_format
    ]
    if limit is not None:
        filtered = filtered[:limit]
    Logger.print_info(f"SEvenLLM {task_key}", f"{len(filtered)} records loaded")
    return filtered


def preload_sevenllm_tasks(
    tasks: List[str],
    limit: Optional[int] = None,
    cache_dir: Optional[Path] = None,
) -> Dict[str, List[dict]]:
    """Load all requested SEvenLLM tasks once."""
    Logger.print_section("Pre-loading SEvenLLM task datasets ...")
    preloaded: Dict[str, List[dict]] = {}
    for task in tasks:
        if task in SEVENLLM_TASK_VERSIONS:
            preloaded[task] = load_sevenllm_split(task, limit=limit, cache_dir=cache_dir)
    return preloaded


def append_result(
    output_path: Path,
    checkpoint_step: Optional[int],
    task: str,
    score: float,
    correct: Optional[int],
    total: int,
) -> None:
    """Append one SEvenLLM result row as JSONL."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "checkpoint_step": checkpoint_step,
        "task": task,
        "score": score,
        "correct": correct,
        "total": total,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    with open(output_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _run_sevenllm(
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
        task_meta = SEVENLLM_TASK_VERSIONS.get(task)
        if task_meta is None:
            Logger.print_warning(f"Unknown SEvenLLM task: {task!r}, skipping.")
            results[task] = None
            continue

        records = preloaded.get(task)
        if records is None:
            Logger.print_warning(f"SEvenLLM {task} not in preloaded datasets, skipping.")
            results[task] = None
            continue

        _, data_format = task_meta
        total = len(records)
        correct = 0
        score_sum = 0.0

        Logger.print_section(f"--- Evaluating SEvenLLM {task} ({total} records) ---")
        for idx, rec in enumerate(records):
            prompt = _format_mcq_prompt(rec) if data_format == "mcq" else _format_qa_prompt(rec)
            try:
                response = generate_fn(prompt)
            except Exception as exc:
                Logger.print_error(f"[{idx + 1}/{total}] ERROR | {exc}")
                continue

            if data_format == "mcq":
                predicted = extract_answer_letter(response)
                expected = str(rec.get("output", "")).strip().upper()
                if predicted == expected:
                    correct += 1
                    Logger.print_success(
                        f"[{idx + 1}/{total}] SOLVED     | predicted={predicted} expected={expected}"
                    )
                else:
                    Logger.print_warning(
                        f"[{idx + 1}/{total}] NOT SOLVED | predicted={predicted} expected={expected}"
                    )
            else:
                target = _qa_target_text(rec.get("output", ""))
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
                    f"[{idx + 1}/{total}] QA score",
                    f"{score_name}={sample_score:.4f}",
                )

        if data_format == "mcq":
            score = correct / total if total > 0 else 0.0
            row_correct: Optional[int] = correct
            Logger.print_info(f"SEvenLLM {task}", f"accuracy={score:.4f} ({correct}/{total})")
        else:
            score = score_sum / total if total > 0 else 0.0
            row_correct = None
            label = "judge_score" if qa_scoring == "judge" else "rougeL_f1"
            Logger.print_info(f"SEvenLLM {task}", f"{label}={score:.4f}")

        results[task] = score
        if output_path is not None:
            append_result(
                output_path=output_path,
                checkpoint_step=checkpoint_step,
                task=task,
                score=score,
                correct=row_correct,
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
        _run_sevenllm,
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
    Logger.print_header("SEvenLLM Evaluation Summary")
    col_w = max(14, max(len(t) for t in tasks))
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
