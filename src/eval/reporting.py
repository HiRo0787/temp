"""
Evaluation reporting helpers.

Converts the in-memory eval summary (produced by run_checkpoints/run_base_model)
into Markdown + CSV reports for quick comparison across checkpoints.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from src.utilities.training_logger import Logger


@dataclass(frozen=True)
class BenchSpec:
    metric_key: str
    higher_is_better: bool


_BENCH_SPECS: Dict[str, BenchSpec] = {
    "secqa": BenchSpec(metric_key="accuracy", higher_is_better=True),
    "cybermetric": BenchSpec(metric_key="accuracy", higher_is_better=True),
    "cyberseceval3": BenchSpec(metric_key="harmful_rate", higher_is_better=False),
    "sevenllm": BenchSpec(metric_key="score", higher_is_better=True),
    "ctibench": BenchSpec(metric_key="accuracy", higher_is_better=True),
    "seceval": BenchSpec(metric_key="accuracy", higher_is_better=True),
    "redsagemcq": BenchSpec(metric_key="accuracy", higher_is_better=True),
    "cissp": BenchSpec(metric_key="score", higher_is_better=True),
    "b3": BenchSpec(metric_key="attack_success_rate", higher_is_better=False),
    "mbpp": BenchSpec(metric_key="pass@k", higher_is_better=True),
    "coconot": BenchSpec(metric_key="compliance_rate", higher_is_better=False),
    "niah": BenchSpec(metric_key="retrieval_accuracy", higher_is_better=True),
    "worldsense": BenchSpec(metric_key="ws_accuracy", higher_is_better=True),
}


def bench_spec(bench: str) -> BenchSpec:
    spec = _BENCH_SPECS.get(bench)
    if spec is None:
        raise ValueError(f"Unknown bench for reporting: {bench!r}")
    return spec


def _safe_float(x: object) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def _mean(values: Iterable[Optional[float]]) -> Optional[float]:
    nums = [v for v in values if v is not None]
    if not nums:
        return None
    return sum(nums) / len(nums)


def _step_label(step: Optional[int]) -> str:
    return str(step) if step is not None else "final"


def _best_row_for_task(
    summary: List[Tuple[Optional[int], Dict[str, Optional[float]]]],
    task: str,
    higher_is_better: bool,
) -> Tuple[Optional[int], Optional[float]]:
    best_step: Optional[int] = None
    best_val: Optional[float] = None
    for step, res in summary:
        val = _safe_float(res.get(task))
        if val is None:
            continue
        if best_val is None:
            best_step = step
            best_val = val
            continue
        if (higher_is_better and val > best_val) or (
            (not higher_is_better) and val < best_val
        ):
            best_step = step
            best_val = val
    return best_step, best_val


def select_best_step(
    summary: List[Tuple[Optional[int], Dict[str, Optional[float]]]],
    tasks: List[str],
    higher_is_better: bool,
) -> Optional[int]:
    """Pick best checkpoint step by mean across tasks. Returns step or None."""
    scored: List[Tuple[Optional[int], float]] = []
    for step, res in summary:
        mean_val = _mean([_safe_float(res.get(t)) for t in tasks])
        if mean_val is None:
            continue
        scored.append((step, mean_val))
    if not scored:
        return None
    scored.sort(key=lambda x: x[1], reverse=higher_is_better)
    return scored[0][0]


def write_markdown_report(
    *,
    report_path: Path,
    bench: str,
    base_model_name: str,
    tasks: List[str],
    output_jsonl_path: Path,
    summary: List[Tuple[Optional[int], Dict[str, Optional[float]]]],
    manifest_path: Optional[Path] = None,
) -> Path:
    spec = bench_spec(bench)
    created = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    best_step = select_best_step(summary, tasks, higher_is_better=spec.higher_is_better)
    direction = "higher is better" if spec.higher_is_better else "lower is better"

    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    headers = ["step", *tasks, "mean"]
    rows: List[List[str]] = []
    for step, res in summary:
        vals = [_safe_float(res.get(t)) for t in tasks]
        mean_val = _mean(vals)
        row = [_step_label(step)]
        row += [f"{v:.4f}" if v is not None else "failed" for v in vals]
        row += [f"{mean_val:.4f}" if mean_val is not None else "failed"]
        rows.append(row)
    all_scores = [
        score
        for _, res in summary
        for score in (_safe_float(res.get(task)) for task in tasks)
        if score is not None
    ]
    avg_metric_all_outputs = _mean(all_scores)

    lines: List[str] = []
    lines.append(f"# Eval report: {bench}")
    lines.append("")
    lines.append("## Overview")
    lines.append("")
    lines.append(f"- **created**: {created}")
    lines.append(f"- **base model**: `{base_model_name}`")
    lines.append(f"- **metric**: `{spec.metric_key}` ({direction})")
    lines.append(f"- **tasks**: {' '.join(f'`{t}`' for t in tasks)}")
    lines.append(f"- **results jsonl**: `{output_jsonl_path}`")
    if avg_metric_all_outputs is not None:
        lines.append(
            f"- **average {spec.metric_key} (all outputs)**: `{avg_metric_all_outputs:.4f}`"
        )
    else:
        lines.append(f"- **average {spec.metric_key} (all outputs)**: `unavailable`")
    lines.append(f"- **best step**: `{_step_label(best_step)}`")

    if manifest_path is not None and Path(manifest_path).exists():
        try:
            manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
            source = manifest.get("source") if isinstance(manifest, dict) else None
            if isinstance(source, dict):
                run_dir = source.get("run_dir")
                checkpoint = source.get("checkpoint")
                all_checkpoints = source.get("all_checkpoints")
                eval_target_tag = manifest.get("eval_target_tag")
                run_id = manifest.get("run_id")
                checkpoint_label = manifest.get("checkpoint_label")
                if run_dir:
                    lines.append(f"- **run dir**: `{run_dir}`")
                if checkpoint:
                    lines.append(f"- **checkpoint**: `{checkpoint}`")
                if all_checkpoints is True:
                    lines.append(f"- **all checkpoints**: `true`")
                if run_id:
                    lines.append(f"- **run id**: `{run_id}`")
                if checkpoint_label:
                    lines.append(f"- **checkpoint label**: `{checkpoint_label}`")
                if eval_target_tag:
                    lines.append(f"- **eval tag**: `{eval_target_tag}`")
        except Exception:
            pass
    lines.append("")

    if manifest_path is not None and Path(manifest_path).exists():
        try:
            manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
            evaluated = manifest.get("evaluated_checkpoints") if isinstance(manifest, dict) else None
            if isinstance(evaluated, list) and evaluated:
                lines.append("## Evaluated checkpoints")
                lines.append("")
                for item in evaluated:
                    if not isinstance(item, dict):
                        continue
                    step = item.get("checkpoint_step")
                    path = item.get("checkpoint_path")
                    step_label = _step_label(step) if isinstance(step, (int, type(None))) else str(step)
                    if path:
                        lines.append(f"- step `{step_label}`: `{path}`")
                    else:
                        lines.append(f"- step `{step_label}`")
                lines.append("")
        except Exception:
            pass

    lines.append("## Task highlights")
    lines.append("")
    for task in tasks:
        task_best_step, task_best_val = _best_row_for_task(
            summary, task, higher_is_better=spec.higher_is_better
        )
        if task_best_val is None:
            lines.append(f"- `{task}`: no valid score")
        else:
            lines.append(
                f"- `{task}`: best `{spec.metric_key}` = `{task_best_val:.4f}` "
                f"at step `{_step_label(task_best_step)}`"
            )
    lines.append("")
    lines.append("## Checkpoint comparison")
    lines.append("")

    # Markdown table
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for r in rows:
        lines.append("| " + " | ".join(r) + " |")
    lines.append("")
    lines.append("## Reading guide")
    lines.append("")
    lines.append(f"- Compare checkpoints by the **mean** column ({direction}).")
    lines.append("- `failed` means that task did not produce a valid metric value.")
    lines.append("")

    report_path.write_text("\n".join(lines), encoding="utf-8")
    Logger.print_success(f"report written: {report_path}")
    return report_path


def write_eval_manifest(
    *,
    output_path: Path,
    bench: str,
    base_model_name: str,
    tasks: List[str],
    checkpoint_pairs: List[Tuple[Optional[int], Path]],
    base_model_mode: bool,
    run_dir: str | None,
    checkpoint: str | None,
    all_checkpoints: bool,
    eval_target_tag: str,
    run_id: str | None,
    checkpoint_label: str | None,
    sample_indices_path: str | None = None,
    sample_indices_by_task: Dict[str, List[int]] | None = None,
) -> Path:
    """
    Write sidecar manifest containing exact evaluated checkpoints and sampling info.
    """
    manifest_path = output_path.with_suffix(".checkpoints.json")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "bench": bench,
        "base_model": base_model_name,
        "tasks": tasks,
        "mode": "base_model" if base_model_mode else "lora_checkpoints",
        "source": {
            "run_dir": run_dir,
            "checkpoint": checkpoint,
            "all_checkpoints": all_checkpoints,
        },
        "eval_target_tag": eval_target_tag,
        "run_id": run_id,
        "checkpoint_label": checkpoint_label,
        "sample_indices_path": sample_indices_path,
        "sample_indices_by_task": sample_indices_by_task or {},
        "output_jsonl": str(output_path),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "evaluated_checkpoints": (
            []
            if base_model_mode
            else [
                {"checkpoint_step": step, "checkpoint_path": str(path)}
                for step, path in checkpoint_pairs
            ]
        ),
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")
    return manifest_path
