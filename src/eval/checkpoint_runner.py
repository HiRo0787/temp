"""
Shared model loading, adapter management, and checkpoint orchestration.

All benchmark scripts (SecQA, CYBERSECEVAL 3, etc.) import from this module
so that the base-model-load-once / adapter-swap-in-place pattern is written
in exactly one place.

Public API
----------
read_base_model_name        Read HuggingFace base model id from adapter_config.json.
load_base_model             Load a causal LM + tokenizer, once, onto device.
list_checkpoint_steps       Scan a checkpoints/ dir for checkpoint-<step> subdirs.
resolve_checkpoints         Map --checkpoint / --run-dir CLI args to (step, path) pairs.
load_adapter_for_checkpoint Swap LoRA adapter weights in-place on an existing model.
make_generate_fn            Build generate_fn(prompt) -> str (greedy or sampling; configurable).
run_checkpoints             Orchestration loop: load model, swap adapters, call bench_fn.
"""

import json
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.utilities.training_logger import Logger

# Type alias for the benchmark callable accepted by run_checkpoints.
# bench_fn(generate_fn, checkpoint_step) -> {task_key: accuracy_or_None}
BenchFn = Callable[
    [Callable[[str], str], Optional[int]],
    Dict[str, Optional[float]],
]


# ---------------------------------------------------------------------------
# Base model helpers
# ---------------------------------------------------------------------------

def read_base_model_name(checkpoint_dir: Path) -> str:
    """Read the base model name from a LoRA checkpoint's adapter_config.json."""
    config_path = checkpoint_dir / "adapter_config.json"
    if not config_path.exists():
        raise FileNotFoundError(
            f"No adapter_config.json in {checkpoint_dir}. "
            "Is this a valid LoRA checkpoint directory?"
        )
    with open(config_path, encoding="utf-8") as f:
        cfg = json.load(f)
    base = cfg.get("base_model_name_or_path", "")
    if not base:
        raise ValueError(
            f"adapter_config.json in {checkpoint_dir} has no base_model_name_or_path."
        )
    return base


def load_base_model(
    base_model_name: str,
    dtype: str = "bfloat16",
    use_4bit: bool = False,
    gpu: str | None = None,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load the base model and tokenizer from HuggingFace.

    Args:
        base_model_name: HuggingFace model ID or local path.
        dtype: Precision string — "bfloat16", "float16", or "float32".
        use_4bit: Load in 4-bit quantization (BitsAndBytes) to save VRAM.
        gpu: Device selector:
             - None (default): device_map="auto"
             - "cuda": CUDA auto placement
             - "cuda:N": pin to one CUDA device index
             - "cpu": force CPU

    Returns:
        (model, tokenizer) tuple with model in eval mode on CUDA/CPU.
    """
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map.get(dtype, torch.bfloat16)

    Logger.print_section(f"Loading base model: {base_model_name}")

    load_kwargs: dict = {
        "trust_remote_code": True,
        "device_map": "auto",
    }
    if gpu:
        normalized = gpu.strip().lower()
        if normalized == "cpu":
            load_kwargs["device_map"] = {"": "cpu"}
            Logger.print_info("gpu", "cpu")
        elif normalized == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("--gpu cuda was set but CUDA is not available.")
            load_kwargs["device_map"] = "auto"
            Logger.print_info("gpu", "cuda (auto)")
        elif normalized.startswith("cuda:"):
            if not torch.cuda.is_available():
                raise RuntimeError(f"--gpu {gpu} was set but CUDA is not available.")
            try:
                gpu_id = int(normalized.split(":", 1)[1])
            except ValueError as exc:
                raise ValueError("--gpu must be one of: cpu, cuda, cuda:<index>.") from exc
            if gpu_id < 0:
                raise ValueError("--gpu cuda:<index> requires index >= 0.")
            device_count = torch.cuda.device_count()
            if gpu_id >= device_count:
                raise ValueError(
                    f"--gpu cuda:{gpu_id} is out of range for {device_count} visible CUDA devices."
                )
            load_kwargs["device_map"] = {"": f"cuda:{gpu_id}"}
            Logger.print_info("gpu", f"cuda:{gpu_id}")
        else:
            raise ValueError("--gpu must be one of: cpu, cuda, cuda:<index>.")

    if use_4bit:
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch_dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    else:
        load_kwargs["torch_dtype"] = torch_dtype

    model = AutoModelForCausalLM.from_pretrained(base_model_name, **load_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name, trust_remote_code=True
    )
    model.eval()
    Logger.print_success(f"Base model loaded: {base_model_name}")
    return model, tokenizer


# ---------------------------------------------------------------------------
# Checkpoint discovery
# ---------------------------------------------------------------------------

def list_checkpoint_steps(checkpoints_dir: Path) -> List[Tuple[int, Path]]:
    """
    Scan a checkpoints directory and return sorted (step, path) pairs.

    Args:
        checkpoints_dir: Directory containing checkpoint-<step> subdirs.

    Returns:
        List of (step, path) tuples sorted by step ascending.
        Returns an empty list when the directory does not exist.
    """
    if not checkpoints_dir.is_dir():
        return []

    def _step(d: Path) -> int:
        try:
            return int(d.name.split("-")[-1])
        except (IndexError, ValueError):
            return -1

    entries = [
        d for d in checkpoints_dir.iterdir()
        if d.is_dir() and d.name.startswith("checkpoint-")
    ]
    pairs = [(_step(d), d) for d in entries if _step(d) >= 0]
    return sorted(pairs, key=lambda x: x[0])


def resolve_checkpoints(
    checkpoint_arg: Optional[str],
    run_dir_arg: Optional[str],
    all_checkpoints: bool = False,
) -> List[Tuple[Optional[int], Path]]:
    """
    Resolve CLI arguments into a list of (step, checkpoint_path) pairs.

    For --checkpoint: returns [(step_or_None, path)].
    For --run-dir: scans <run_dir>/checkpoints/ sorted by step ascending.
                   By default returns only the latest checkpoint.
                   Pass all_checkpoints=True to return every checkpoint.

    Args:
        checkpoint_arg: Path to a single checkpoint directory, or None.
        run_dir_arg: Path to a run directory, or None.
        all_checkpoints: When using run_dir_arg, evaluate every checkpoint
                         instead of just the latest one.

    Returns:
        List of (step, path) tuples. step is None for a manually specified
        checkpoint whose name does not follow the checkpoint-<N> convention.
    """
    if checkpoint_arg:
        p = Path(checkpoint_arg).expanduser().resolve()
        try:
            step = int(p.name.split("-")[-1])
        except (IndexError, ValueError):
            step = None
        return [(step, p)]

    checkpoints_dir = Path(run_dir_arg).expanduser().resolve() / "checkpoints"
    pairs = list_checkpoint_steps(checkpoints_dir)
    if not pairs:
        raise FileNotFoundError(
            f"No checkpoint-<step> directories found in {checkpoints_dir}. "
            "Pass the run directory (e.g. artifacts/qwen2.5-7b/rabit0-v1-run12), "
            "not the checkpoints/ subdirectory."
        )
    if not all_checkpoints:
        # pairs are sorted ascending; the last entry is the latest step.
        return [pairs[-1]]
    return pairs


# ---------------------------------------------------------------------------
# Adapter loading (in-place swap)
# ---------------------------------------------------------------------------

def load_adapter_for_checkpoint(
    model: AutoModelForCausalLM,
    checkpoint_path: Path,
) -> PeftModel:
    """
    Load a LoRA adapter from a checkpoint directory.

    On the first call pass the bare base model; it is wrapped with PeftModel.
    On subsequent calls pass the existing PeftModel; adapter weights are
    replaced in-place using load_adapter() + set_adapter().

    This in-place swap is critical for multi-checkpoint evaluation: using
    unload() + from_pretrained() modifies the model's module structure on each
    iteration, causing PEFT to silently reuse stale weights so every checkpoint
    produces identical results.

    Args:
        model: Either the bare base model (first call) or the PeftModel from
               a previous call.
        checkpoint_path: Path to a checkpoint-<step> directory containing
                         adapter_config.json and adapter_model.safetensors.

    Returns:
        PeftModel with the new checkpoint's adapter weights loaded and active.
    """
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()

    if not checkpoint_path.is_dir():
        raise FileNotFoundError(
            f"Checkpoint directory not found: {checkpoint_path}"
        )
    if not (checkpoint_path / "adapter_config.json").exists():
        raise FileNotFoundError(
            f"Not a LoRA checkpoint (no adapter_config.json): {checkpoint_path}"
        )

    Logger.print_section(f"Loading adapter: {checkpoint_path.name}")

    if isinstance(model, PeftModel):
        model.load_adapter(
            str(checkpoint_path),
            adapter_name="default",
            is_trainable=False,
            torch_device="cpu",
        )
        model.set_adapter("default")
    else:
        model = PeftModel.from_pretrained(
            model,
            str(checkpoint_path),
            adapter_name="default",
            is_trainable=False,
            torch_device="cpu",
        )

    model.eval()
    Logger.print_success(f"Adapter loaded: {checkpoint_path.name}")
    return model


# ---------------------------------------------------------------------------
# Generation callable
# ---------------------------------------------------------------------------

def make_generate_fn(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    *,
    max_new_tokens: int = 64,
    do_sample: bool = False,
    temperature: float = 0.5,
    top_p: float = 0.95,
    system_message: Optional[str] = None,
    trace_callback: Optional[Callable[[dict], None]] = None,
) -> Callable[[str], str]:
    """
    Return a generate_fn(prompt: str) -> str callable.

    Applies the model's chat template, tokenises, generates, decodes the new
    tokens only, and strips leading/trailing whitespace.

    Args:
        max_new_tokens: Maximum tokens to generate (default 64 for MCQ benches).
        do_sample: If True, sample with temperature/top_p; if False, greedy.
        temperature: Sampling temperature (only used when do_sample=True).
        top_p: Nucleus sampling p (only used when do_sample=True).
        system_message: Optional system role content (user content stays ``prompt``).
    """
    def generate_fn(prompt: str) -> str:
        if system_message:
            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt},
            ]
        else:
            messages = [{"role": "user", "content": prompt}]
        formatted = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tokenizer(formatted, return_tensors="pt").to(model.device)
        gen_kwargs: dict = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": tokenizer.eos_token_id,
        }
        if do_sample:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = top_p
        with torch.no_grad():
            output_ids = model.generate(**inputs, **gen_kwargs)
        new_ids = output_ids[0][inputs["input_ids"].shape[-1]:]
        output_text = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
        if trace_callback is not None:
            try:
                trace_callback(
                    {
                        "prompt": prompt,
                        "output": output_text,
                        "prompt_token_len": int(inputs["input_ids"].shape[-1]),
                        "output_token_len": int(new_ids.shape[-1]),
                        "total_token_len": int(inputs["input_ids"].shape[-1] + new_ids.shape[-1]),
                    }
                )
            except Exception:
                # Tracing must never break eval execution.
                pass
        return output_text

    return generate_fn


# ---------------------------------------------------------------------------
# Orchestration loop
# ---------------------------------------------------------------------------

def run_checkpoints(
    checkpoint_pairs: List[Tuple[Optional[int], Path]],
    bench_fn: BenchFn,
    base_model_name: str,
    dtype: str = "bfloat16",
    use_4bit: bool = False,
    gpu: str | None = None,
    *,
    max_new_tokens: int = 64,
    do_sample: bool = False,
    temperature: float = 0.5,
    top_p: float = 0.95,
    system_message: Optional[str] = None,
    trace_callback: Optional[Callable[[dict], None]] = None,
) -> List[Tuple[Optional[int], Dict[str, Optional[float]]]]:
    """
    Load the base model once, swap adapters per checkpoint, call bench_fn.

    This is the shared orchestration used by all benchmarks. Each benchmark
    module creates a bench_fn via its own make_bench_fn() factory and passes
    it here.

    Args:
        checkpoint_pairs: List of (step, path) pairs from resolve_checkpoints().
        bench_fn: Callable(generate_fn, checkpoint_step) -> result_dict.
                  Handles dataset loading, prompt formatting, scoring, and
                  result persistence. Created by each benchmark's make_bench_fn().
        base_model_name: HuggingFace model ID or local path.
        dtype: Model precision — "bfloat16", "float16", or "float32".
        use_4bit: Load base model in 4-bit quantization to save VRAM.
        gpu: Device selector string passed to load_base_model().
        max_new_tokens: Passed to make_generate_fn() for each checkpoint.
        do_sample: Passed to make_generate_fn() for each checkpoint.
        temperature: Passed to make_generate_fn() when do_sample=True.
        top_p: Passed to make_generate_fn() when do_sample=True.
        system_message: Passed to make_generate_fn() when set.

    Returns:
        List of (step, result_dict) pairs in the same order as checkpoint_pairs.
    """
    model, tokenizer = load_base_model(
        base_model_name, dtype=dtype, use_4bit=use_4bit, gpu=gpu
    )
    summary = []

    for step, ckpt_path in checkpoint_pairs:
        Logger.print_section(f"=== Checkpoint: {ckpt_path.name} ===")
        model = load_adapter_for_checkpoint(model, ckpt_path)
        generate_fn = make_generate_fn(
            model,
            tokenizer,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            system_message=system_message,
            trace_callback=(
                None
                if trace_callback is None
                else (lambda event, _step=step: trace_callback({"checkpoint_step": _step, **event}))
            ),
        )
        results = bench_fn(generate_fn, step)
        summary.append((step, results))

    return summary


def run_base_model(
    bench_fn: BenchFn,
    base_model_name: str,
    dtype: str = "bfloat16",
    use_4bit: bool = False,
    gpu: str | None = None,
    *,
    max_new_tokens: int = 64,
    do_sample: bool = False,
    temperature: float = 0.5,
    top_p: float = 0.95,
    system_message: Optional[str] = None,
    trace_callback: Optional[Callable[[dict], None]] = None,
) -> List[Tuple[Optional[int], Dict[str, Optional[float]]]]:
    """
    Evaluate the base model once (no LoRA adapters).

    Uses the same BenchFn interface as run_checkpoints(), but calls the bench
    exactly once with checkpoint_step=None.
    """
    Logger.print_section("=== Base model (no adapter) ===")
    model, tokenizer = load_base_model(
        base_model_name, dtype=dtype, use_4bit=use_4bit, gpu=gpu
    )
    generate_fn = make_generate_fn(
        model,
        tokenizer,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
        system_message=system_message,
        trace_callback=(
            None
            if trace_callback is None
            else (lambda event: trace_callback({"checkpoint_step": None, **event}))
        ),
    )
    results = bench_fn(generate_fn, None)
    return [(None, results)]
