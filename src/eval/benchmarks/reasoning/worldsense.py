"""
WorldSense reasoning benchmark.

Reference:
https://github.com/UKGovernmentBEIS/inspect_evals/tree/main/src/inspect_evals/worldsense
"""

from __future__ import annotations

import bz2
import json
import re
import sys
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import pandas as pd
import requests

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.utilities.training_logger import Logger

WORLDSENSE_DATASET_REVISION = "bd81d945077f169cf95ff39207f788f86e4645e9"
WORLDSENSE_DATA_URL = (
    "https://github.com/facebookresearch/worldsense/raw/"
    f"{WORLDSENSE_DATASET_REVISION}/data/worldsense/test_set/trials.jsonl.bz2"
)

WORLDSENSE_TASK_VERSIONS: Dict[str, str] = {
    "infer_trivial": "Infer.trivial",
    "infer_normal": "Infer.normal",
    "compl_trivial": "Compl.trivial",
    "compl_normal": "Compl.normal",
    "consist_trivial": "Consist.trivial",
    "consist_normal": "Consist.normal",
}


def _normalize(text: str) -> str:
    s = (text or "").strip().lower()
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _preview_text(text: str, limit: int = 220) -> str:
    one_line = re.sub(r"\s+", " ", (text or "").strip())
    if len(one_line) <= limit:
        return one_line
    return one_line[: limit - 3] + "..."


def _extract_choice_label(text: str) -> str:
    t = (text or "").strip()
    if not t:
        return ""
    m = re.search(r"\(([123])\)", t)
    if m:
        return m.group(1)
    m = re.search(r"\b([123])\b", t)
    if m:
        return m.group(1)
    if "true" in t.lower():
        return "true"
    if "false" in t.lower():
        return "false"
    if "impossible" in t.lower():
        return "impossible"
    if "possible" in t.lower():
        return "possible"
    return _normalize(t)


def load_worldsense_rows(limit: Optional[int] = None) -> List[dict]:
    Logger.print_section("Loading WorldSense trials.jsonl.bz2 from pinned source ...")
    response = requests.get(WORLDSENSE_DATA_URL, timeout=120)
    response.raise_for_status()
    decompressed = bz2.decompress(response.content).decode("utf-8")
    rows = [json.loads(line) for line in decompressed.strip().splitlines() if line.strip()]
    if limit is not None:
        rows = rows[:limit]
    Logger.print_info("WorldSense", f"{len(rows)} records loaded")
    return rows


def _goldresp_mapping(goldresp_obfusc: str) -> str:
    mapping = {
        "Emmanuel": "TRUE",
        "Megi": "FALSE",
        "Dieuwke": "POSSIBLE",
        "Pascal": "IMPOSSIBLE",
        "Mark": "1",
        "Youssef": "2",
        "Yoda": "3",
    }
    return mapping.get(str(goldresp_obfusc), "")


def _preprocess_scores(records: List[dict]) -> pd.DataFrame:
    if not records:
        return pd.DataFrame(
            columns=["tuple_ID", "problemname", "problemsize", "value", "bias", "weight"]
        )

    score_df = pd.DataFrame(records)
    bias_mapping = {
        "1": 1,
        "2": 1,
        "TRUE": 1,
        "POSSIBLE": 1,
        "3": -1,
        "FALSE": -1,
        "IMPOSSIBLE": -1,
    }
    weight_mapping = {
        "1": 0.25,
        "2": 0.25,
        "3": 0.5,
        "TRUE": 0.5,
        "POSSIBLE": 0.5,
        "FALSE": 0.5,
        "IMPOSSIBLE": 0.5,
    }

    score_df["weight"] = score_df["answer"].map(weight_mapping).astype(float)
    score_df["bias"] = score_df["answer"].map(bias_mapping).astype(float) * score_df["weight"]
    score_df["value"] = score_df["value"].astype(float) * score_df["weight"]

    grouped = (
        score_df.groupby(["tuple_ID", "problemname", "problemsize"])
        .agg({"value": "sum", "bias": "sum", "weight": "sum"})
        .reset_index()
    )
    grouped["value"] = grouped["value"] / grouped["weight"].where(grouped["weight"] != 0, 1)
    grouped["bias"] = grouped["bias"] / grouped["weight"].where(grouped["weight"] != 0, 1)
    return grouped


def _compute_ws_accuracy(grouped_scores: pd.DataFrame) -> float:
    if grouped_scores.empty:
        return 0.0
    problem_summary = (
        grouped_scores.groupby(["problemname", "problemsize"]).agg({"value": "mean"}).reset_index()
    )
    final_summary = problem_summary.groupby("problemname").agg({"value": "mean"}).reset_index()
    return float(final_summary["value"].mean())


def _compute_ws_bias(grouped_scores: pd.DataFrame) -> float:
    if grouped_scores.empty:
        return 0.0
    problem_summary = (
        grouped_scores.groupby(["problemname", "problemsize"]).agg({"bias": "mean"}).reset_index()
    )
    final_summary = problem_summary.groupby("problemname").agg({"bias": "mean"}).reset_index()
    return float(final_summary["bias"].mean())


def preload_worldsense_tasks(
    tasks: List[str],
    limit: Optional[int] = None,
) -> Dict[str, List[dict]]:
    Logger.print_section("Pre-loading WorldSense datasets ...")
    preloaded: Dict[str, List[dict]] = {}
    valid_tasks = [t for t in tasks if t in WORLDSENSE_TASK_VERSIONS]
    if not valid_tasks:
        return preloaded

    all_rows = load_worldsense_rows(limit=None)
    for task in valid_tasks:
        problemname = WORLDSENSE_TASK_VERSIONS.get(task)
        if problemname is None:
            continue
        filt = [r for r in all_rows if str(r.get("problemname", "")) == problemname]
        if limit is not None:
            filt = filt[:limit]
        preloaded[task] = filt
        Logger.print_info(f"WorldSense {task}", f"{len(filt)} samples loaded")
    return preloaded


def append_result(
    output_path: Path,
    checkpoint_step: Optional[int],
    task: str,
    accuracy: float,
    ws_accuracy: float,
    ws_bias: float,
    correct: int,
    total: int,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "checkpoint_step": checkpoint_step,
        "task": task,
        "accuracy": accuracy,
        "ws_accuracy": ws_accuracy,
        "ws_bias": ws_bias,
        "correct": correct,
        "total": total,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    with open(output_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _run_worldsense(
    generate_fn: Callable[[str], str],
    checkpoint_step: Optional[int],
    tasks: List[str],
    output_path: Optional[Path],
    preloaded: Dict[str, List[dict]],
) -> Dict[str, Optional[float]]:
    results: Dict[str, Optional[float]] = {}
    for task in tasks:
        if task not in WORLDSENSE_TASK_VERSIONS:
            Logger.print_warning(f"Unknown WorldSense task: {task!r}, skipping.")
            results[task] = None
            continue
        rows = preloaded.get(task, [])
        if not rows:
            Logger.print_warning(f"WorldSense {task}: no rows found, skipping.")
            results[task] = None
            continue

        total = len(rows)
        correct = 0
        ws_records: List[dict] = []
        Logger.print_section(f"--- Evaluating WorldSense {task} ({total} samples) ---")
        for idx, row in enumerate(rows):
            prompt = str(row.get("text", "")).strip()
            canonical_answer = _goldresp_mapping(str(row.get("goldresp_obfusc", "")))
            expected = _extract_choice_label(canonical_answer)
            try:
                answer = generate_fn(prompt)
            except Exception as exc:
                Logger.print_error(f"[{idx + 1}/{total}] generate ERROR | {exc}")
                continue
            pred = _extract_choice_label(str(answer))
            Logger.print_info(f"[{idx + 1}/{total}] model_output", _preview_text(str(answer)))
            ok = pred == expected and expected != ""
            correct += int(ok)
            ws_records.append(
                {
                    "value": float(int(ok)),
                    "answer": canonical_answer,
                    "tuple_ID": row.get("tuple_ID"),
                    "problemname": row.get("problemname"),
                    "problemsize": row.get("problemsize"),
                }
            )
            Logger.print_info(
                f"[{idx + 1}/{total}]",
                f"correct={int(ok)} pred={pred!r} expected={expected!r}",
            )

        score = correct / total if total > 0 else 0.0
        grouped = _preprocess_scores(ws_records)
        ws_accuracy = _compute_ws_accuracy(grouped)
        ws_bias = _compute_ws_bias(grouped)
        results[task] = ws_accuracy
        Logger.print_info(
            f"WorldSense {task}",
            f"accuracy={score:.4f} ws_accuracy={ws_accuracy:.4f} ws_bias={ws_bias:.4f} ({correct}/{total})",
        )
        if output_path is not None:
            append_result(
                output_path=output_path,
                checkpoint_step=checkpoint_step,
                task=task,
                accuracy=score,
                ws_accuracy=ws_accuracy,
                ws_bias=ws_bias,
                correct=correct,
                total=total,
            )
    return results


def make_bench_fn(
    tasks: List[str],
    preloaded: Dict[str, List[dict]],
    output_path: Optional[Path] = None,
) -> Callable:
    return partial(
        _run_worldsense,
        tasks=tasks,
        output_path=output_path,
        preloaded=preloaded,
    )


def print_summary(
    summary: List[Tuple[Optional[int], Dict[str, Optional[float]]]],
    tasks: List[str],
) -> None:
    Logger.print_header("WorldSense Evaluation Summary (ws_accuracy)")
    col_w = max(10, max(len(t) for t in tasks))
    Logger.print_section("  step     " + "  ".join(f"{t:>{col_w}}" for t in tasks))
    for step, res in summary:
        step_label = str(step) if step is not None else "final"
        cols = [f"{res.get(t):.4f}" if res.get(t) is not None else "failed" for t in tasks]
        Logger.print_section(f"{step_label:>8}  " + "  ".join(f"{c:>{col_w}}" for c in cols))
