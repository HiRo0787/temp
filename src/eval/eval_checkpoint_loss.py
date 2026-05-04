"""
Eval-only: compute per-datapoint loss for a checkpoint (no training).

Summary
-------
- Loads a LoRA checkpoint (e.g. artifacts/qwen2.5-7b/rabit0-v1-run12/checkpoints/checkpoint-500).
- Loads JSONL datapoints (same format as training: messages + optional metadata).
- Runs model.eval() with torch.no_grad() (non-learning).
- Computes cross-entropy loss per datapoint and saves results to a JSONL file.
- Optional: load base model in 4-bit (--use-4bit) to reduce VRAM.

Workflow
--------
1. Resolve checkpoint path (directory or specific checkpoint-* subdir; if checkpoints dir, use latest).
2. Load base model from adapter_config.json base_model_name_or_path; load PEFT adapter; set model.eval().
3. Load and tokenize JSONL (same formatting as training: DataFormatter.format_training_data, max_length 2048).
4. For each batch (or datapoint): forward pass with no_grad, compute loss (padding tokens ignored via label -100).
5. Write output JSONL with detailed fields; print per-datapoint lines and summary (mean/min/max, by category, by difficulty).

Resume
------
Use --resume to continue from the last datapoint_id already written to the output file (append new rows, then
recompute quality labels and summary on the full set). Use --start-from-datapoint N to start from index N explicitly;
it overrides --resume if both are set.

"""

import argparse
import json
import sys
from pathlib import Path

# Project root
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.utilities.data_formatter import DataFormatter
from src.utilities.training_logger import Logger


def _normalize_metadata(meta):
    if not meta or not isinstance(meta, dict):
        return {}
    return dict(meta)


def _objective_snippet(messages: list, max_chars: int = 300) -> str:
    """Extract a short objective from the first user message (Target/Objective lines or first max_chars)."""
    for m in messages:
        if m.get("role") == "user" and m.get("content"):
            text = m["content"].strip()
            if "Objective:" in text:
                start = text.find("Objective:")
                end = text.find("\n\n", start) if start >= 0 else -1
                snippet = text[start:end].strip() if end > start else text[start:start + max_chars]
            else:
                snippet = text[:max_chars]
            return snippet.replace("\n", " ").strip()
    return ""


def _detail_from_record(rec: dict) -> dict:
    """Build a flat dict of metadata and snippet for output (real data from the datapoint)."""
    meta = rec.get("metadata") or {}
    return {
        "scenario_id": meta.get("scenario_id", ""),
        "category": meta.get("category", ""),
        "difficulty": meta.get("difficulty", ""),
        "platform": meta.get("platform", ""),
        "owasp_category": meta.get("owasp_category", ""),
        "mitre_ids": meta.get("mitre_ids") or [],
        "objective_snippet": _objective_snippet(rec.get("messages") or []),
    }


def _build_quality_labels(
    rows: list[dict],
    good_loss_threshold: float | None = None,
    bad_loss_threshold: float | None = None,
) -> tuple[float, float]:
    """Attach quality_label and loss_percentile to each row.
    Lower loss is better:
    - loss <= good_threshold => good
    - loss >= bad_threshold => bad
    - otherwise => needs_review
    If thresholds are None, defaults use percentiles (good=25th, bad=75th).
    """
    if not rows:
        return 0.0, 0.0

    sorted_losses = sorted(r["loss"] for r in rows)
    n = len(sorted_losses)

    def _quantile(q: float) -> float:
        idx = int(round((n - 1) * q))
        return float(sorted_losses[idx])

    good_t = float(good_loss_threshold) if good_loss_threshold is not None else _quantile(0.25)
    bad_t = float(bad_loss_threshold) if bad_loss_threshold is not None else _quantile(0.75)
    if good_t > bad_t:
        raise ValueError("good-loss-threshold cannot be greater than bad-loss-threshold")

    for r in rows:
        loss = float(r["loss"])
        # percentile rank for context (0=best, 1=worst)
        num_lower_or_equal = sum(1 for x in sorted_losses if x <= loss)
        r["loss_percentile"] = round(num_lower_or_equal / n, 4)
        if loss <= good_t:
            r["quality_label"] = "good"
        elif loss >= bad_t:
            r["quality_label"] = "bad"
        else:
            r["quality_label"] = "needs_review"

    return good_t, bad_t


def load_jsonl(data_path: Path):
    """Load JSONL with messages (and optional metadata). Returns list of dicts."""
    records = []
    with open(data_path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if "metadata" in rec:
                rec["metadata"] = _normalize_metadata(rec["metadata"])
            records.append(rec)
    return records


def _get_last_datapoint_id_from_output(output_path: Path) -> int | None:
    """Read existing output JSONL and return the max datapoint_id, or None if empty/missing."""
    if not output_path.exists():
        return None
    last_id = None
    with open(output_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                idx = rec.get("datapoint_id")
                if idx is not None and isinstance(idx, int):
                    last_id = idx if last_id is None else max(last_id, idx)
            except json.JSONDecodeError:
                continue
    return last_id


def resolve_checkpoint_dir(checkpoint_arg: str) -> Path:
    """
    Resolve to a single checkpoint directory.
    - If path points to a dir that contains adapter_config.json, use it.
    - Else if path is a 'checkpoints' dir containing checkpoint-* subdirs, use the latest by step number.
    """
    p = Path(checkpoint_arg).expanduser().resolve()
    if not p.is_dir():
        raise FileNotFoundError(f"Checkpoint path is not a directory: {p}")
    adapter_config = p / "adapter_config.json"
    if adapter_config.exists():
        return p
    # Assume checkpoints parent dir: look for checkpoint-* subdirs
    subdirs = [d for d in p.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")]
    if not subdirs:
        raise FileNotFoundError(
            f"No checkpoint subdir (checkpoint-*) or adapter_config.json in: {p}"
        )
    def step_num(d):
        try:
            return int(d.name.split("-")[-1])
        except (IndexError, ValueError):
            return -1
    latest = max(subdirs, key=step_num)
    if not (latest / "adapter_config.json").exists():
        raise FileNotFoundError(f"Missing adapter_config.json in {latest}")
    return latest


def get_base_model_name(checkpoint_dir: Path) -> str:
    with open(checkpoint_dir / "adapter_config.json", encoding="utf-8") as f:
        config = json.load(f)
    name = config.get("base_model_name_or_path")
    if not name:
        raise ValueError(
            "adapter_config.json has no base_model_name_or_path. "
            "Pass base model via --model-name."
        )
    return name


def load_model_and_tokenizer(
    checkpoint_dir: Path,
    model_name: str,
    torch_dtype: torch.dtype,
    device_map: str | dict = "auto",
    use_4bit: bool = False,
):
    """Load base model, then PEFT adapter; load tokenizer from checkpoint or base.
    When use_4bit is True, loads base model in 4-bit (QLoRA-style) for lower VRAM."""
    from peft import PeftModel
    base_name = model_name or get_base_model_name(checkpoint_dir)
    if (checkpoint_dir / "tokenizer_config.json").exists():
        tokenizer = AutoTokenizer.from_pretrained(
            str(checkpoint_dir),
            trust_remote_code=True,
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            base_name,
            trust_remote_code=True,
        )
    if use_4bit:
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch_dtype,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            base_name,
            quantization_config=bnb_config,
            device_map=device_map,
            trust_remote_code=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            base_name,
            dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=True,
        )
    model = PeftModel.from_pretrained(
        model,
        str(checkpoint_dir),
        is_trainable=False,
    )
    model.eval()
    return model, tokenizer


def tokenize_examples(records, tokenizer, max_length: int = 2048, pad_to_max_length: bool = True):
    """Tokenize like training: format_training_data, then tokenizer with padding. Labels = input_ids; set padding to -100."""
    texts = [DataFormatter.format_training_data(r["messages"]) for r in records]
    out = tokenizer(
        texts,
        truncation=True,
        max_length=max_length,
        padding="max_length" if pad_to_max_length else False,
        return_tensors="pt",
    )
    labels = out["input_ids"].clone()
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    labels[out["input_ids"] == pad_id] = -100
    out["labels"] = labels
    return out


def compute_loss_per_datapoint(model, input_ids, attention_mask, labels, device):
    """Forward with no_grad; compute CE loss per sequence (mean over non-ignored tokens)."""
    model.eval()
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    labels = labels.to(device)
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
    logits = outputs.logits
    # Causal LM: predict next token; shift like training
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    # Flatten and compute per-token loss (reduction="none")
    flat_logits = shift_logits.view(-1, shift_logits.size(-1))
    flat_labels = shift_labels.view(-1)
    loss_fct = torch.nn.CrossEntropyLoss(reduction="none", ignore_index=-100)
    per_token_loss = loss_fct(flat_logits, flat_labels)
    per_token_loss = per_token_loss.view(shift_labels.shape)
    # Mean over non-ignored positions per sample
    valid = (shift_labels != -100).float()
    sum_loss = (per_token_loss * valid).sum(dim=1)
    count = valid.sum(dim=1).clamp(min=1)
    per_sample_loss = (sum_loss / count).cpu()
    return per_sample_loss


def run_eval(
    checkpoint_path: str,
    data_path: str,
    output_path: str,
    model_name: str | None = None,
    batch_size: int = 4,
    max_length: int = 2048,
    dtype: str = "bfloat16",
    use_4bit: bool = False,
    good_loss_threshold: float | None = None,
    bad_loss_threshold: float | None = None,
    resume: bool = False,
    start_from_datapoint: int | None = None,
    device: str | None = None,
):
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    torch_dtype = dtype_map.get(dtype, torch.bfloat16)

    checkpoint_dir = resolve_checkpoint_dir(checkpoint_path)
    Logger.print_info("Checkpoint", str(checkpoint_dir))
    if device:
        Logger.print_info("Device", device)
    if use_4bit:
        Logger.print_info("Quantization", "4-bit (QLoRA-style)")

    device_map = {"": device} if device else "auto"
    model, tokenizer = load_model_and_tokenizer(
        checkpoint_dir,
        model_name=model_name,
        torch_dtype=torch_dtype,
        use_4bit=use_4bit,
        device_map=device_map,
    )
    device = next(model.parameters()).device
    Logger.print_section("Model loaded in eval mode (no training).")

    records = load_jsonl(Path(data_path))
    Logger.print_info("Loaded datapoints", f"{len(records)} from {data_path}")

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Resume: start after last written datapoint_id, or from explicit index
    start_index = 0
    if start_from_datapoint is not None:
        start_index = max(0, start_from_datapoint)
        if start_index > 0:
            Logger.print_info("Start from datapoint", f"{start_index} (explicit)")
    elif resume and out_path.exists():
        last_id = _get_last_datapoint_id_from_output(out_path)
        if last_id is not None:
            start_index = last_id + 1
            Logger.print_info("Resuming", f"from datapoint_id {start_index} (last was {last_id})")

    if start_index >= len(records):
        Logger.print_info("Nothing to do", f"start_index={start_index} >= len(records)={len(records)}")
        return []

    results = []
    file_mode = "a" if start_index > 0 else "w"

    with open(out_path, file_mode, encoding="utf-8") as out_f:
        for i in range(start_index, len(records), batch_size):
            batch_records = records[i : i + batch_size]
            batch = tokenize_examples(batch_records, tokenizer, max_length=max_length)
            losses = compute_loss_per_datapoint(
                model,
                batch["input_ids"],
                batch["attention_mask"],
                batch["labels"],
                device,
            )
            labels = batch["labels"]
            for j, rec in enumerate(batch_records):
                idx = i + j
                loss_val = losses[j].item()
                detail = _detail_from_record(rec)
                num_valid_tokens = int((labels[j] != -100).sum().item())
                row = {
                    "datapoint_id": idx,
                    "loss": round(loss_val, 6),
                    "num_valid_tokens": num_valid_tokens,
                    **detail,
                }
                results.append(row)
                out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                out_f.flush()
                Logger.print_section(
                    f"  [{idx}] {detail['scenario_id']} | {detail['category']} | {detail['difficulty']} | "
                    f"loss={loss_val:.6f} | tokens={num_valid_tokens}"
                )
                if detail["objective_snippet"]:
                    Logger.print_info("    objective", detail["objective_snippet"][:120] + ("..." if len(detail["objective_snippet"]) > 120 else ""))

    # When resuming, re-read full output so we have all rows for quality labels and summary
    if start_index > 0:
        results = []
        with open(out_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                results.append(json.loads(line))
        results.sort(key=lambda r: r["datapoint_id"])

    good_t, bad_t = _build_quality_labels(
        results,
        good_loss_threshold=good_loss_threshold,
        bad_loss_threshold=bad_loss_threshold,
    )

    # Summary
    loss_list = [r["loss"] for r in results]
    mean_loss = sum(loss_list) / len(loss_list) if loss_list else 0.0
    Logger.print_section("--- Summary ---")
    Logger.print_info("count", len(results))
    Logger.print_info("mean loss", f"{mean_loss:.6f}")
    if loss_list:
        Logger.print_info("min loss", f"{min(loss_list):.6f}")
        Logger.print_info("max loss", f"{max(loss_list):.6f}")
    Logger.print_info("good threshold", f"{good_t:.6f}")
    Logger.print_info("bad threshold", f"{bad_t:.6f}")
    total_tokens = sum(r.get("num_valid_tokens", 0) for r in results)
    Logger.print_info("total valid tokens", total_tokens)
    label_counts = {"good": 0, "needs_review": 0, "bad": 0}
    for r in results:
        label_counts[r["quality_label"]] = label_counts.get(r["quality_label"], 0) + 1
    Logger.print_info("good datapoints", label_counts["good"])
    Logger.print_info("needs_review datapoints", label_counts["needs_review"])
    Logger.print_info("bad datapoints", label_counts["bad"])

    # Breakdown by category (if present)
    by_cat = {}
    for r in results:
        c = r.get("category") or "_no_category"
        by_cat.setdefault(c, []).append(r["loss"])
    if len(by_cat) > 1 or (len(by_cat) == 1 and "_no_category" not in by_cat):
        Logger.print_section("--- Mean loss by category ---")
        for c in sorted(by_cat.keys()):
            avg = sum(by_cat[c]) / len(by_cat[c])
            Logger.print_info(c, f"n={len(by_cat[c])} mean_loss={avg:.6f}")

    # Breakdown by difficulty (if present)
    by_diff = {}
    for r in results:
        d = r.get("difficulty") or "_no_difficulty"
        by_diff.setdefault(d, []).append(r["loss"])
    if len(by_diff) > 1 or (len(by_diff) == 1 and "_no_difficulty" not in by_diff):
        Logger.print_section("--- Mean loss by difficulty ---")
        for d in sorted(by_diff.keys()):
            avg = sum(by_diff[d]) / len(by_diff[d])
            Logger.print_info(d, f"n={len(by_diff[d])} mean_loss={avg:.6f}")

    Logger.print_section("--- Top 5 best datapoints (lowest loss) ---")
    for row in sorted(results, key=lambda x: x["loss"])[:5]:
        Logger.print_info(
            "best",
            f"id={row['datapoint_id']} scenario={row['scenario_id']} loss={row['loss']:.6f} label={row['quality_label']}",
        )
    Logger.print_section("--- Top 5 worst datapoints (highest loss) ---")
    for row in sorted(results, key=lambda x: x["loss"], reverse=True)[:5]:
        Logger.print_info(
            "worst",
            f"id={row['datapoint_id']} scenario={row['scenario_id']} loss={row['loss']:.6f} label={row['quality_label']}",
        )

    # Overwrite combined file with full results (quality_label, loss_percentile); write split files
    stem = out_path.parent / out_path.stem
    with open(out_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    Logger.print_success(f"Wrote all results to {out_path} (real-time stream + final with quality labels)")

    for label in ("good", "needs_review", "bad"):
        subset = [r for r in results if r["quality_label"] == label]
        if not subset:
            continue
        split_path = Path(f"{stem}_{label}.jsonl")
        with open(split_path, "w", encoding="utf-8") as f:
            for r in subset:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        Logger.print_info(f"  {label}", f"{split_path} ({len(subset)} rows)")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Eval-only: compute per-datapoint loss for a checkpoint (no training).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Checkpoint directory (e.g. artifacts/qwen2.5-7b/rabit0-v1-run12/checkpoints or .../checkpoints/checkpoint-500)",
    )
    parser.add_argument(
        "--data",
        required=True,
        help="JSONL file with messages (and optional metadata), e.g. combine_output/all_training_data_test.jsonl",
    )
    parser.add_argument(
        "--output",
        default="reports/eval_loss_per_datapoints/eval_loss_per_data.jsonl",
        help="Output JSONL path for per-datapoint loss (default: reports/eval_loss_per_datapoints/eval_loss_per_data.jsonl)",
    )
    parser.add_argument(
        "--model-name",
        default=None,
        help="Override base model name (default: read from checkpoint adapter_config.json)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for forward pass (default: 1; use 1 for large models/long sequences to avoid OOM)",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=2048,
        help="Max sequence length (default: 2048)",
    )
    parser.add_argument(
        "--dtype",
        choices=["bfloat16", "float16", "float32"],
        default="bfloat16",
        help="Model dtype for compute / full-precision load (default: bfloat16)",
    )
    parser.add_argument(
        "--use-4bit",
        action="store_true",
        help="Load base model in 4-bit quantization (QLoRA-style) to reduce VRAM",
    )
    parser.add_argument(
        "--good-loss-threshold",
        type=float,
        default=None,
        help="Manual threshold for 'good' datapoints (loss <= threshold). Default: auto 25th percentile.",
    )
    parser.add_argument(
        "--bad-loss-threshold",
        type=float,
        default=None,
        help="Manual threshold for 'bad' datapoints (loss >= threshold). Default: auto 75th percentile.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from last datapoint_id in the output file (append new results, then recompute quality labels).",
    )
    parser.add_argument(
        "--start-from-datapoint",
        type=int,
        default=None,
        metavar="N",
        help="Start processing from datapoint index N (0-based). Appends to output if file exists. Overrides --resume.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device to use (e.g. cuda, cpu, cuda:0). Default: auto (use GPU if available).",
    )
    args = parser.parse_args()
    run_eval(
        checkpoint_path=args.checkpoint,
        data_path=args.data,
        output_path=args.output,
        model_name=args.model_name,
        batch_size=args.batch_size,
        max_length=args.max_length,
        dtype=args.dtype,
        use_4bit=args.use_4bit,
        good_loss_threshold=args.good_loss_threshold,
        bad_loss_threshold=args.bad_loss_threshold,
        resume=args.resume,
        start_from_datapoint=args.start_from_datapoint,
        device=args.device,
    )


if __name__ == "__main__":
    main()
