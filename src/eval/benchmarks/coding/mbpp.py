"""
MBPP (Mostly Basic Python Problems) benchmark.

Aligned with UKGovernmentBEIS/inspect_evals mbpp task:
https://github.com/UKGovernmentBEIS/inspect_evals/tree/main/src/inspect_evals/mbpp

Uses Hugging Face ``google-research-datasets/mbpp`` sanitized test split, the
same dataset revision as inspect_evals, few-shot examples (task_id 2,3,4 from
the ``full``/``prompt`` split), n=5 sampled completions per problem, and
pass@1 / pass@2 / pass@5 aggregated with the standard unbiased estimator.

Reference: Austin et al., https://arxiv.org/pdf/2108.07732
"""

from __future__ import annotations

import json
import math
import re
import subprocess
import sys
import textwrap
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.utilities.training_logger import Logger

DATASET_NAME = "google-research-datasets/mbpp"
MBPP_DATASET_REVISION = "4bb6404fdc6cacfda99d4ac4205087b89d32030c"
NUM_SAMPLES = 5
VERIFY_TIMEOUT_SEC = 30
FEW_SHOT_TASK_IDS = (2, 3, 4)

_PROMPT_INTRO = """
You are an expert Python programmer. You will be given a task, and the tests that your code must pass.
Write the Python function to solve the task. Do not give additional explanations, just output the
Python function. Only use imports that are included in Python's standard library.
""".strip()

_CODE_BLOCK_RE = re.compile(r"```python\n(.*?)```", re.DOTALL)

# Task keys are metric names; all use the same sanitized split.
MBPP_TASK_VERSIONS: Dict[str, str] = {
    "pass@1": "sanitized",
    "pass@2": "sanitized",
    "pass@5": "sanitized",
}


def find_code(completion: str) -> str:
    """Remove Markdown fencing around generated code (same idea as inspect_evals)."""
    matches = _CODE_BLOCK_RE.findall(completion or "")
    if matches:
        return str(matches[0]).strip()
    return str(completion or "").strip()


def pass_at_k_unbiased(n: int, c: int, k: int) -> float:
    """
    Unbiased pass@k estimate from n samples with c correct (HumanEval-style).

    Defined as 1 - C(n-c, k) / C(n, k) when n-c >= k; 1 if c > 0 and n-c < k; 0 if c == 0.
    """
    if n <= 0 or k <= 0 or k > n:
        return 0.0
    if c <= 0:
        return 0.0
    if c >= n:
        return 1.0
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def append_verification_payload(code: str, test_list: List[str]) -> str:
    """Append asserts with messages, matching inspect_evals mbpp.verify()."""
    body = (code or "").rstrip()
    for test_case in test_list:
        assert_prefix = "assert "
        rest = test_case[len(assert_prefix) :] if test_case.startswith(assert_prefix) else test_case
        body += f"\n{test_case}, {repr(rest)}"
    return body


def verify_generated_code(code: str, test_list: List[str]) -> bool:
    """Run extracted code plus asserts in a fresh Python subprocess."""
    full = append_verification_payload(code, test_list)
    try:
        proc = subprocess.run(
            [sys.executable, "-c", full],
            capture_output=True,
            text=True,
            timeout=VERIFY_TIMEOUT_SEC,
        )
    except subprocess.TimeoutExpired:
        return False
    return proc.returncode == 0


def _record_prompt(rec: dict) -> str:
    if "prompt" in rec:
        return str(rec["prompt"])
    if "text" in rec:
        return str(rec["text"])
    raise KeyError("MBPP record has no 'prompt' or 'text' field")


def _build_few_shot_prefix() -> str:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError("datasets library is required: pip install datasets") from exc

    Logger.print_section("Loading MBPP few-shot examples (full/prompt) ...")
    few = load_dataset(
        DATASET_NAME,
        "full",
        split="prompt",
        revision=MBPP_DATASET_REVISION,
    )
    rows = [r for r in few if r.get("task_id") in FEW_SHOT_TASK_IDS]
    rows.sort(key=lambda r: FEW_SHOT_TASK_IDS.index(r["task_id"]))

    parts: List[str] = [_PROMPT_INTRO, "\n\nFor example:\n\n"]
    for i, sample in enumerate(rows):
        prompt_body = str(sample.get("text") or sample.get("prompt", ""))
        tests = sample.get("test_list") or []
        test_lines = "\n".join(tests) if isinstance(tests, list) else str(tests)
        code = str(sample.get("code", ""))
        parts.extend(
            [
                f"## Prompt {i + 1}\n",
                "```python\n",
                f"{prompt_body}\n",
                "```\n\n",
                f"## Test Case {i + 1}\n",
                "```python\n",
                f"{test_lines}\n```\n\n",
                f"## Completion {i + 1}\n",
                "```python\n",
                f"{code}\n```\n\n",
            ]
        )
    parts.append(
        textwrap.dedent(
            """
            # Now, do it for the following task.

            ## Prompt:
            ```python
            {prompt}
            ```

            ## Test Case:
            ```python
            {test_list_str}
            ```

            ## Completion:
            """
        ).strip()
        + "\n"
    )
    return "".join(parts)


_FEW_SHOT_PREFIX_CACHE: Optional[str] = None


def few_shot_prompt_prefix() -> str:
    """Lazily built few-shot prefix (cached process-wide)."""
    global _FEW_SHOT_PREFIX_CACHE
    if _FEW_SHOT_PREFIX_CACHE is None:
        _FEW_SHOT_PREFIX_CACHE = _build_few_shot_prefix()
    return _FEW_SHOT_PREFIX_CACHE


def format_mbpp_prompt(record: dict, *, prefix: Optional[str] = None) -> str:
    """Format one MBPP test row using the inspect_evals template."""
    pfx = prefix if prefix is not None else few_shot_prompt_prefix()
    test_list = record.get("test_list") or []
    if not isinstance(test_list, list):
        test_list = list(test_list)
    test_str = "\n".join(str(x) for x in test_list)
    return pfx.format(prompt=_record_prompt(record), test_list_str=test_str)


def load_mbpp_sanitized_test(limit: Optional[int] = None) -> List[dict]:
    """Load the sanitized MBPP test split (same as inspect_evals)."""
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError("datasets library is required: pip install datasets") from exc
    Logger.print_section("Loading MBPP sanitized test split ...")
    ds = load_dataset(
        DATASET_NAME,
        "sanitized",
        split="test",
        revision=MBPP_DATASET_REVISION,
    )
    rows = list(ds)
    if limit is not None:
        rows = rows[:limit]
    Logger.print_info("MBPP sanitized test", f"{len(rows)} problems loaded")
    return rows


def preload_mbpp_tasks(
    tasks: List[str],
    limit: Optional[int] = None,
) -> Dict[str, List[dict]]:
    """Load the MBPP test set once; each task key maps to the same records."""
    Logger.print_section("Pre-loading MBPP dataset ...")
    records = load_mbpp_sanitized_test(limit=limit)
    preloaded: Dict[str, List[dict]] = {}
    for t in tasks:
        if t in MBPP_TASK_VERSIONS:
            preloaded[t] = records
    return preloaded


def append_result(
    output_path: Path,
    checkpoint_step: Optional[int],
    task: str,
    score: float,
    total: int,
) -> None:
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


def _run_mbpp(
    generate_fn: Callable[[str], str],
    checkpoint_step: Optional[int],
    tasks: List[str],
    output_path: Optional[Path],
    preloaded: Dict[str, List[dict]],
    *,
    num_samples: int = NUM_SAMPLES,
) -> Dict[str, Optional[float]]:
    results: Dict[str, Optional[float]] = {}

    data_key = next((t for t in tasks if t in MBPP_TASK_VERSIONS), None)
    if data_key is None:
        for t in tasks:
            results[t] = None
        return results

    records = preloaded.get(data_key)
    if not records:
        Logger.print_warning("MBPP: no preloaded records, skipping.")
        for t in tasks:
            if t in MBPP_TASK_VERSIONS:
                results[t] = None
        return results

    prefix = few_shot_prompt_prefix()
    total_problems = len(records)
    kvals = [1, 2, 5]
    # Sum of per-problem pass@k estimates; divide by total_problems at end.
    sums = {k: 0.0 for k in kvals}
    k_to_task = {1: "pass@1", 2: "pass@2", 5: "pass@5"}

    Logger.print_section(
        f"--- Evaluating MBPP ({total_problems} problems, {num_samples} samples each) ---"
    )

    for idx, rec in enumerate(records):
        prompt = format_mbpp_prompt(rec, prefix=prefix)
        test_list = rec.get("test_list") or []
        if not isinstance(test_list, list):
            test_list = list(test_list)
        successes = 0
        for sidx in range(num_samples):
            try:
                raw = generate_fn(prompt)
            except Exception as exc:
                Logger.print_error(f"[{idx + 1}/{total_problems}] sample {sidx + 1} ERROR | {exc}")
                continue
            generated = find_code(raw)
            if verify_generated_code(generated, test_list):
                successes += 1
                Logger.print_success(
                    f"[{idx + 1}/{total_problems}] sample {sidx + 1}/{num_samples} PASSED"
                )
            else:
                Logger.print_warning(
                    f"[{idx + 1}/{total_problems}] sample {sidx + 1}/{num_samples} FAILED"
                )

        for k in kvals:
            sums[k] += pass_at_k_unbiased(num_samples, successes, k)

        Logger.print_info(
            f"problem {idx + 1}/{total_problems}",
            f"correct_samples={successes}/{num_samples}",
        )

    for k in kvals:
        task_name = k_to_task[k]
        if task_name not in tasks:
            continue
        mean_est = sums[k] / total_problems if total_problems > 0 else 0.0
        results[task_name] = mean_est
        if output_path is not None:
            append_result(
                output_path=output_path,
                checkpoint_step=checkpoint_step,
                task=task_name,
                score=mean_est,
                total=total_problems,
            )

    # Explicit None for requested task keys that are unknown
    for t in tasks:
        if t in MBPP_TASK_VERSIONS and t not in results:
            results[t] = None

    return results


def make_bench_fn(
    tasks: List[str],
    preloaded: Dict[str, List[dict]],
    output_path: Optional[Path] = None,
    *,
    num_samples: int = NUM_SAMPLES,
) -> Callable:
    """BenchFn compatible with checkpoint_runner.run_checkpoints()."""
    return partial(
        _run_mbpp,
        tasks=tasks,
        output_path=output_path,
        preloaded=preloaded,
        num_samples=num_samples,
    )


def print_summary(
    summary: List[Tuple[Optional[int], Dict[str, Optional[float]]]],
    tasks: List[str],
) -> None:
    Logger.print_header("MBPP Evaluation Summary (pass@k, higher is better)")
    col_w = max(8, max(len(t) for t in tasks))
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
