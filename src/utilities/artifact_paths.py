"""
Artifact output paths for training runs.

Stores outputs under artifacts/<model_name>/<run_prefix>-run1, <run_prefix>-run2, ...
(e.g. artifacts/qwen2.5-7b/rabit0-v1-run1). Maintains a "<run_prefix>-latest" marker
per model. Logs are expected in a separate folder; these paths are for final finetune outputs.
"""

import logging
import shutil
from pathlib import Path
from typing import Optional, Union

# Project root: this file is src/utilities/artifact_paths.py -> root is parent of src
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

logger = logging.getLogger(__name__)


def get_artifacts_root(artifacts_root: Optional[Path] = None) -> Path:
    """Return artifacts root directory (default: project_root/artifacts)."""
    if artifacts_root is not None:
        return Path(artifacts_root).resolve()
    return _PROJECT_ROOT / "artifacts"


def _next_run_number(model_dir: Path, run_prefix: str) -> int:
    """Find next run number from existing <run_prefix>-run* directories under model_dir."""
    if not model_dir.is_dir():
        return 1
    if not run_prefix or not isinstance(run_prefix, str):
        raise ValueError(f"_next_run_number: run_prefix must be a non-empty str, got {run_prefix!r}")
    pattern = f"{run_prefix}-run"
    numbers = []
    try:
        entries = list(model_dir.iterdir())
    except OSError as e:
        logger.warning("Cannot list model dir %s: %s", model_dir, e)
        return 1
    for p in entries:
        if p.is_dir() and p.name.startswith(pattern):
            try:
                numbers.append(int(p.name[len(pattern):]))
            except ValueError:
                continue
    return max(numbers, default=0) + 1


def _update_latest_link(model_dir: Path, run_prefix: str, run_dir: Path) -> None:
    """
    Create or update '<run_prefix>-latest' under model_dir to point to the given run.
    Uses a symlink when possible; on Windows without symlink support, writes a file.
    """
    latest_path = model_dir / f"{run_prefix}-latest"
    run_name = run_dir.name

    if latest_path.is_symlink() or latest_path.exists():
        latest_path.unlink()

    try:
        latest_path.symlink_to(run_name, target_is_directory=True)
    except OSError:
        try:
            latest_path.write_text(run_name, encoding="utf-8")
        except OSError as e:
            logger.warning("Cannot write latest marker %s: %s", latest_path, e)


def get_artifact_run_dir(
    model_name: str,
    run_prefix: str = "rabit0-v1",
    artifacts_root: Optional[Path] = None,
    update_latest: bool = True,
) -> str:
    """
    Get the next run directory under artifacts/<model_name>/<run_prefix>-runN.

    Args:
        model_name: Subfolder under artifacts (e.g. "qwen2.5-7b") for this model.
        run_prefix: Prefix for run dir names (e.g. "rabit0-v1" -> rabit0-v1-run1, rabit0-v1-run2).
        artifacts_root: Base directory for artifacts. Defaults to project root / "artifacts".
        update_latest: If True, create/update the latest marker to this run.

    Returns:
        Absolute path string (e.g. .../artifacts/qwen2.5-7b/rabit0-v1-run1).

    Raises:
        ValueError: If model_name or run_prefix is empty or invalid.
        OSError: If the run directory or artifacts root cannot be created (e.g. permission denied).
    """
    if not model_name or not isinstance(model_name, str):
        raise ValueError(
            f"get_artifact_run_dir: model_name must be a non-empty str, got {model_name!r}"
        )
    if not run_prefix or not isinstance(run_prefix, str):
        raise ValueError(
            f"get_artifact_run_dir: run_prefix must be a non-empty str, got {run_prefix!r}"
        )
    model_name = model_name.strip()
    run_prefix = run_prefix.strip()
    if not model_name:
        raise ValueError("get_artifact_run_dir: model_name cannot be blank")
    if not run_prefix:
        raise ValueError("get_artifact_run_dir: run_prefix cannot be blank")
    try:
        root = get_artifacts_root(artifacts_root)
        root.mkdir(parents=True, exist_ok=True)
        model_dir = root / model_name
        model_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        raise OSError(
            f"Cannot create artifacts directory: {e}"
        ) from e
    next_num = _next_run_number(model_dir, run_prefix)
    run_dir = model_dir / f"{run_prefix}-run{next_num}"
    try:
        run_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        raise OSError(
            f"Cannot create run directory {run_dir}: {e}"
        ) from e
    if update_latest:
        try:
            _update_latest_link(model_dir, run_prefix, run_dir)
        except OSError as e:
            logger.warning("Could not update latest marker for %s: %s", run_dir, e)
    return str(run_dir.resolve())


def ensure_artifact_run_subdirs(run_dir: str | Path) -> dict[str, Path]:
    """
    Ensure standard subdirs exist under a run directory and return their paths.
    Subdirs: checkpoints, lora, merged, logs, data_versions.

    Args:
        run_dir: Run directory (e.g. artifacts/qwen2.5-7b/rabit0-v1-run2).

    Returns:
        Dict with keys "checkpoints", "lora", "merged", "logs", "data_versions"
        and Path values.

    Raises:
        ValueError: If run_dir is None or empty.
        OSError: If run_dir or any subdir cannot be created (e.g. permission denied).
    """
    if run_dir is None or (isinstance(run_dir, str) and not run_dir.strip()):
        raise ValueError(
            "ensure_artifact_run_subdirs: run_dir must be a non-empty path, "
            f"got {run_dir!r}"
        )
    try:
        run = Path(run_dir).resolve()
    except TypeError:
        raise TypeError(
            f"ensure_artifact_run_subdirs: run_dir must be str or Path, got {type(run_dir).__name__}"
        )
    subdirs = ("checkpoints", "lora", "merged", "logs", "data_versions")
    out = {}
    for name in subdirs:
        d = run / name
        try:
            d.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            raise OSError(
                f"Cannot create run subdir {d} (parent run_dir={run}): {e}"
            ) from e
        out[name] = d
    return out


def copy_data_manifest(
    manifest_path: Optional[Union[str, Path]],
    dest_dir: Union[str, Path],
    *,
    auto_detect_from: Optional[Union[str, Path]] = None,
) -> Optional[Path]:
    """
    Copy a data manifest.json into dest_dir for provenance tracking.

    Resolution order:
      1. manifest_path (explicit path, if given)
      2. auto_detect_from: looks for manifest.json in the same directory as that file

    Returns the destination Path if a copy was made, else None.
    """
    resolved: Optional[Path] = None

    if manifest_path:
        resolved = Path(manifest_path).resolve()
    elif auto_detect_from:
        candidate = Path(auto_detect_from).resolve().parent / "manifest.json"
        if candidate.is_file():
            resolved = candidate

    if resolved is None:
        return None

    if not resolved.is_file():
        logger.warning("Data manifest not found, skipping copy: %s", resolved)
        return None

    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    dest_file = dest / "manifest.json"
    shutil.copy2(str(resolved), str(dest_file))
    logger.info("Data manifest copied to %s", dest_file)
    return dest_file


def _latest_run_dir_by_scan(model_dir: Path, run_prefix: str) -> Optional[Path]:
    """Find the latest run directory by scanning <run_prefix>-run* dirs. Returns None if none exist."""
    if not model_dir.is_dir():
        return None
    try:
        entries = list(model_dir.iterdir())
    except OSError as e:
        logger.warning("Cannot list model dir %s: %s", model_dir, e)
        return None
    pattern = f"{run_prefix}-run"
    best_num = 0
    best_path: Optional[Path] = None
    for p in entries:
        if p.is_dir() and p.name.startswith(pattern):
            try:
                n = int(p.name[len(pattern):])
                if n > best_num:
                    best_num = n
                    best_path = p
            except ValueError:
                continue
    return best_path


def resolve_latest_artifact(
    model_name: str,
    run_prefix: str = "rabit0-v1",
    artifacts_root: Optional[Path] = None,
    repair_marker: bool = True,
) -> Optional[Path]:
    """
    Resolve the path to the latest run for a model.
    Reads artifacts/<model_name>/<run_prefix>-latest (symlink or file).
    If the marker is missing, broken, or unreadable, falls back to scanning for the highest run number.
    When repair_marker is True and the marker was invalid, updates it to point to the current latest run.

    Args:
        model_name: Subfolder under artifacts (e.g. "qwen2.5-7b").
        run_prefix: Prefix used for run dirs (e.g. "rabit0-v1").
        artifacts_root: Base directory for artifacts. Defaults to project root / "artifacts".
        repair_marker: If True, when the marker is broken or stale, update it to the current latest run.

    Returns:
        Path to the latest run directory, or None if no run exists.
    """
    root = get_artifacts_root(artifacts_root)
    model_dir = root / model_name
    latest_marker = model_dir / f"{run_prefix}-latest"
    if latest_marker.exists():
        if latest_marker.is_symlink():
            try:
                target = latest_marker.resolve()
                if target.is_dir():
                    return target
            except OSError as e:
                logger.debug(
                    "Could not resolve latest symlink %s: %s; falling back to scan",
                    latest_marker,
                    e,
                )
        else:
            try:
                run_name = latest_marker.read_text(encoding="utf-8").strip()
                if run_name:
                    run_path = model_dir / run_name
                    if run_path.is_dir():
                        return run_path
            except OSError as e:
                logger.debug(
                    "Could not read latest marker file %s: %s; falling back to scan",
                    latest_marker,
                    e,
                )
    resolved = _latest_run_dir_by_scan(model_dir, run_prefix)
    if resolved is not None and repair_marker:
        current_target: Optional[str] = None
        if latest_marker.exists():
            try:
                if latest_marker.is_symlink():
                    current_target = str(latest_marker.readlink())
                else:
                    current_target = latest_marker.read_text(encoding="utf-8").strip()
            except OSError:
                current_target = None
        if current_target != resolved.name:
            try:
                _update_latest_link(model_dir, run_prefix, resolved)
                logger.info(
                    "Updated latest marker to %s (previous target was missing or moved)",
                    resolved.name,
                )
            except OSError as e:
                logger.warning("Could not repair latest marker for %s: %s", model_dir, e)
    return resolved
