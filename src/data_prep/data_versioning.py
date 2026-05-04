"""
Data versioning helpers for data_prep.

Creates versioned artifacts for reproducible training data runs:
1) version manifest
2) SHA256 checksums
3) full compressed data_points snapshot with version id
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import sys
import time
from datetime import datetime
from pathlib import Path
import tarfile
from typing import Dict, Iterable, List, Optional, Tuple

logger = logging.getLogger(__name__)

try:
    from src.utilities.colours import ColouredFormatter, colour
except ImportError:
    try:
        from utilities.colours import ColouredFormatter, colour
    except ImportError:
        ColouredFormatter = None  # type: ignore[assignment]
        def colour(text: str, style: str = "reset") -> str:  # type: ignore[misc]
            return text

if not logger.handlers:
    _handler = logging.StreamHandler(sys.stdout)
    _handler.setFormatter(
        ColouredFormatter("%(message)s") if ColouredFormatter else logging.Formatter("%(message)s")
    )
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False


def _iter_files(directory: Path) -> Iterable[Path]:
    for path in sorted(directory.rglob("*")):
        if path.is_file():
            yield path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def hash_directory(directory: Path) -> str:
    """Compute stable SHA256 from sorted relative file paths + file bytes."""
    digest = hashlib.sha256()
    root = directory.resolve()
    for path in _iter_files(root):
        rel = str(path.relative_to(root)).replace("\\", "/")
        digest.update(rel.encode("utf-8"))
        digest.update(b"\x00")
        with open(path, "rb") as f:
            while True:
                chunk = f.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
        digest.update(b"\x00")
    return digest.hexdigest()


def _next_version_number(data_version_dir: Path) -> int:
    """Return next version number by scanning existing dpv_vN_* folders."""
    if not data_version_dir.exists():
        return 1
    max_n = 0
    for d in data_version_dir.iterdir():
        if not d.is_dir():
            continue
        m = re.match(r"dpv_v(\d+)_", d.name)
        if m:
            max_n = max(max_n, int(m.group(1)))
    return max_n + 1


def _find_existing_version_by_hash(data_version_dir: Path, data_points_sha: str) -> Optional[str]:
    """Return latest version_id whose manifest has the same data_points SHA256."""
    if not data_version_dir.exists():
        return None

    matches: List[Tuple[int, str]] = []
    for d in data_version_dir.iterdir():
        if not d.is_dir():
            continue
        manifest_path = d / "manifest.json"
        if not manifest_path.exists():
            continue

        try:
            with open(manifest_path, encoding="utf-8") as f:
                manifest = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue

        if manifest.get("data_points_dir_sha256") != data_points_sha:
            continue

        version_id = manifest.get("version_id")
        if not isinstance(version_id, str) or not version_id:
            continue

        run_number = int(manifest.get("run", 0)) if str(manifest.get("run", "")).isdigit() else 0
        matches.append((run_number, version_id))

    if not matches:
        return None

    matches.sort(key=lambda item: (item[0], item[1]))
    return matches[-1][1]


def _version_index_path(data_version_dir: Path) -> Path:
    return data_version_dir / "index.json"


def _load_version_index(data_version_dir: Path) -> Dict:
    """Load version index, returning an empty structure when missing/corrupt."""
    path = _version_index_path(data_version_dir)
    if not path.exists():
        return {"versions": {}, "sha_to_versions": {}}

    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return {"versions": {}, "sha_to_versions": {}}

    if not isinstance(data, dict):
        return {"versions": {}, "sha_to_versions": {}}
    data.setdefault("versions", {})
    data.setdefault("sha_to_versions", {})
    return data


def _write_version_index(data_version_dir: Path, index_data: Dict) -> Path:
    """Persist SHA<->version index for fast version lookup/switching."""
    data_version_dir.mkdir(parents=True, exist_ok=True)
    index_path = _version_index_path(data_version_dir)
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index_data, f, indent=2, sort_keys=True)
    return index_path


def update_version_index(data_version_dir: Path, manifest: Dict) -> Path:
    """Update index.json with one manifest record."""
    version_id = manifest.get("version_id")
    data_points_sha = manifest.get("data_points_dir_sha256")
    if not isinstance(version_id, str) or not version_id:
        raise ValueError("manifest.version_id must be a non-empty string")
    if not isinstance(data_points_sha, str) or not data_points_sha:
        raise ValueError("manifest.data_points_dir_sha256 must be a non-empty string")

    index_data = _load_version_index(data_version_dir)
    versions = index_data["versions"]
    sha_to_versions = index_data["sha_to_versions"]

    versions[version_id] = {
        "data_points_sha256": data_points_sha,
        "archive_file": manifest.get("archive_file", ""),
        "manifest_file": f"{version_id}/manifest.json",
        "created_at_utc": manifest.get("created_at_utc", ""),
    }

    existing = sha_to_versions.get(data_points_sha, [])
    if not isinstance(existing, list):
        existing = []
    if version_id not in existing:
        existing.append(version_id)
    sha_to_versions[data_points_sha] = existing

    return _write_version_index(data_version_dir, index_data)


def resolve_data_version(
    data_version_dir: Path,
    *,
    version_id: Optional[str] = None,
    data_points_sha: Optional[str] = None,
) -> Dict[str, str]:
    """Resolve a version by version_id or SHA and return key paths/ids.

    When only SHA is provided, the latest matching version id is returned.
    """
    if not version_id and not data_points_sha:
        raise ValueError("Provide version_id or data_points_sha")

    index_data = _load_version_index(data_version_dir)
    versions = index_data.get("versions", {})
    sha_to_versions = index_data.get("sha_to_versions", {})

    resolved_version = version_id
    if not resolved_version and data_points_sha:
        candidates = sha_to_versions.get(data_points_sha, [])
        if candidates:
            resolved_version = candidates[-1]
        else:
            resolved_version = _find_existing_version_by_hash(data_version_dir, data_points_sha)

    if not resolved_version:
        raise FileNotFoundError("No version found for the provided SHA")

    manifest_path = data_version_dir / resolved_version / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Version manifest not found: {manifest_path}")

    if data_points_sha:
        manifest_sha = versions.get(resolved_version, {}).get("data_points_sha256")
        if manifest_sha and manifest_sha != data_points_sha:
            raise ValueError(
                f"Version '{resolved_version}' does not match requested SHA {data_points_sha}"
            )

    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)

    archive_file = manifest.get("archive_file", "")
    if not isinstance(archive_file, str):
        archive_file = ""

    version_dir = data_version_dir / resolved_version
    data_points_dir = version_dir / "data_points"
    archive_path = version_dir / archive_file if archive_file else None

    if not data_points_dir.exists():
        if archive_path is None or not archive_path.exists():
            raise FileNotFoundError(
                f"data_points directory is missing and archive is unavailable for version '{resolved_version}'"
            )

        logger.info(
            colour(
                "data_points directory missing for version '%s', extracting archive %s",
                "yellow",
            ),
            resolved_version,
            archive_path.name,
        )
        with tarfile.open(archive_path, "r:*") as tar:
            tar.extractall(path=version_dir)

    if not data_points_dir.exists():
        raise FileNotFoundError(
            f"Archive extracted but data_points directory still missing: {data_points_dir}"
        )

    return {
        "version_id": resolved_version,
        "manifest_path": str(manifest_path),
        "version_dir": str(version_dir),
        "data_points_dir": str(data_points_dir),
        "archive_path": str(archive_path) if archive_path else "",
    }


def build_version_id(
    data_points_dir: Path,
    data_version_dir: Path,
    version_id: Optional[str] = None,
    *,
    _precomputed_hash: Optional[str] = None,
) -> str:
    """Build version id: dpv_v{N}_{timestamp}_{short_hash}.

    N is auto-incremented from existing versions in data_version_dir.
    If version_id is provided it is returned unchanged (allows explicit pinning).
    """
    if version_id:
        return version_id
    dir_hash = _precomputed_hash or hash_directory(data_points_dir)
    existing_version_id = _find_existing_version_by_hash(data_version_dir, dir_hash)
    if existing_version_id:
        logger.info(
            colour("Reusing existing data version (unchanged data_points): %s", "cyan"),
            existing_version_id,
        )
        return existing_version_id

    n = _next_version_number(data_version_dir)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    short_hash = dir_hash[:8]
    return f"dpv_v{n}_{timestamp}_{short_hash}"


_build_version_id = build_version_id


def _archive_suffix(archive_format: str) -> str:
    mapping = {"tar.gz": ".tar.gz", "tar.xz": ".tar.xz"}
    if archive_format not in mapping:
        raise ValueError(f"Unsupported archive format: {archive_format}. Choose from {list(mapping)}")
    return mapping[archive_format]


def _collect_files(directory: Path) -> List[Tuple[Path, int]]:
    """Return sorted list of (path, size_bytes) for all files under directory."""
    result = []
    for p in sorted(directory.rglob("*")):
        if p.is_file():
            result.append((p, p.stat().st_size))
    return result


def _write_archive(
    data_points_dir: Path,
    archive_path: Path,
    archive_format: str,
    files: Optional[List[Tuple[Path, int]]] = None,
) -> None:
    """Write compressed tar archive with per-file progress logging.

    Removes the partial file on any failure or interrupt so no corrupt
    archive is left behind.
    """
    mode = "w:gz" if archive_format == "tar.gz" else "w:xz"
    root = data_points_dir.parent

    if files is None:
        files = _collect_files(data_points_dir)

    total_bytes = sum(sz for _, sz in files)
    total_files = len(files)
    done_bytes = 0
    done_files = 0
    last_logged_pct = -1

    try:
        with tarfile.open(archive_path, mode) as tar:
            for path, size in files:
                arcname = str(path.relative_to(root))
                tarinfo = tar.gettarinfo(str(path), arcname=arcname)
                with open(path, "rb") as fobj:
                    tar.addfile(tarinfo, fobj)

                done_bytes += size
                done_files += 1
                pct = int(done_bytes / total_bytes * 100) if total_bytes else 100

                # Log every 10% step
                if pct >= last_logged_pct + 10:
                    last_logged_pct = (pct // 10) * 10
                    bar = colour(f"  Compressing  {pct:3d}%", "blue")
                    files_part = colour(f"  {done_files:3d} / {total_files} files", "cyan")
                    size_part = colour(
                        f"  {done_bytes / 1024 / 1024:6.1f} / {total_bytes / 1024 / 1024:.1f} MB",
                        "magenta",
                    )
                    logger.info("%s |%s |%s", bar, files_part, size_part)

    except BaseException:
        if archive_path.exists():
            archive_path.unlink()
            logger.warning("Removed incomplete archive %s", archive_path.name)
        raise


def save_data_version_artifacts(
    *,
    datapoints: List[Dict],
    project_root: Path,
    output_file: Path,
    data_version_dir: Path,
    archive_format: str = "tar.xz",
    version_id: Optional[str] = None,
    run_message: Optional[str] = None,
    _precomputed_data_points_sha: Optional[str] = None,
    data_points_dir: Optional[Path] = None,
) -> Dict[str, Path]:
    """
    Save three data version artifacts under data_version_dir/<version_id>/:

    1. manifest.json           - version id, run number, run_message, timestamps, datapoint list, all SHA256s
    2. SHA256SUMS.txt          - checksums for archive + manifest + JSONL output
    3. data_points_<id>.<ext>  - full compressed snapshot of data_points/

    A copy of manifest.json is also written to combine_output/<version_id>/manifest.json
    so the training file and its metadata always live together.

    data_points_dir: override the data_points directory to archive/hash (default: project_root/data_points).
    """
    data_points_dir = data_points_dir if data_points_dir is not None else project_root / "data_points"
    if not data_points_dir.exists():
        raise FileNotFoundError(f"data_points folder not found: {data_points_dir}")

    # Hash directory once; reuse for version id and manifest.
    # Accepts a pre-computed hash from the caller to avoid re-scanning 215MB twice.
    if _precomputed_data_points_sha:
        data_points_sha = _precomputed_data_points_sha
        logger.info(colour("data_points/ hash reused from caller", "cyan"))
    else:
        logger.info(colour("Hashing data_points/ ...", "cyan"))
        t0 = time.monotonic()
        data_points_sha = hash_directory(data_points_dir)
        logger.info(colour("data_points/ hash done in %.1fs", "cyan"), time.monotonic() - t0)

    version = build_version_id(
        data_points_dir,
        data_version_dir,
        version_id=version_id,
        _precomputed_hash=data_points_sha,
    )
    version_dir = data_version_dir / version
    version_dir.mkdir(parents=True, exist_ok=True)
    logger.info("%s %s", colour("Version id :", "bold"), colour(version, "bright_cyan"))
    logger.info("%s %s", colour("Version dir:", "bold"), version_dir)

    # --- Compressed snapshot ---
    suffix = _archive_suffix(archive_format)
    archive_name = f"data_points_{version}{suffix}"
    archive_path = version_dir / archive_name

    files = _collect_files(data_points_dir)
    total_bytes = sum(sz for _, sz in files)
    archive_preexisting = archive_path.exists()
    if archive_preexisting:
        logger.info(colour("Archive already exists for this data hash, reusing: %s", "cyan"), archive_name)
    else:
        logger.info(
            colour("Compressing data_points/ -> %s  (%d files, %.1f MB, format=%s)", "magenta"),
            archive_name, len(files), total_bytes / 1024 / 1024, archive_format,
        )
        t1 = time.monotonic()
        _write_archive(data_points_dir, archive_path, archive_format, files=files)
        elapsed = time.monotonic() - t1
        logger.info(
            colour("Archive written: %.1f MB in %.1fs", "success"),
            archive_path.stat().st_size / 1024 / 1024, elapsed,
        )

    archive_size_mb = archive_path.stat().st_size / 1024 / 1024

    # --- Hashes ---
    logger.info(colour("Computing SHA256 checksums ...", "yellow"))
    archive_sha = sha256_file(archive_path)
    output_file_path = Path(output_file)
    output_file_sha = sha256_file(output_file_path) if output_file_path.exists() else ""

    # --- Manifest ---
    run_number = int(re.search(r"dpv_v(\d+)_", version).group(1)) if re.search(r"dpv_v(\d+)_", version) else 1
    manifest: Dict = {
        "version_id": version,
        "run": run_number,
        "version_method": "data_points_sha256",
        "reused_existing_version": archive_preexisting,
        "run_message": run_message.strip() if run_message and run_message.strip() else None,
        "created_at_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "output_file": str(output_file),
        "output_file_sha256": output_file_sha,
        "archive_format": archive_format,
        "archive_file": archive_name,
        "archive_size_mb": round(archive_size_mb, 2),
        "archive_sha256": archive_sha,
        "data_points_dir_sha256": data_points_sha,
        "datapoints": [
            {
                "id": entry.get("id", entry.get("path", "unknown")),
                "path": entry.get("path", ""),
                "loader": entry.get("loader", ""),
                "scenario_type": entry.get("scenario_type", "security"),
            }
            for entry in datapoints
        ],
    }

    if run_message and run_message.strip():
        logger.info("%s %s", colour("Run message:", "bold"), colour(run_message.strip(), "bright_yellow"))

    # Write manifest to data_versions/<version_id>/
    manifest_path = version_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    logger.info(colour("Manifest written: %s", "bright_cyan"), manifest_path)
    index_path = update_version_index(data_version_dir, manifest)
    logger.info(colour("Version index updated: %s", "bright_cyan"), index_path)

    # Write manifest copy to combine_output/<version_id>/ so JSONL and metadata live together
    combine_output_manifest = output_file_path.parent / "manifest.json"
    combine_output_manifest.parent.mkdir(parents=True, exist_ok=True)
    with open(combine_output_manifest, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    logger.info(colour("Manifest copy written: %s", "bright_cyan"), combine_output_manifest)

    # --- SHA256SUMS ---
    sha_path = version_dir / "SHA256SUMS.txt"
    with open(sha_path, "w", encoding="utf-8") as f:
        f.write(f"{archive_sha}  {archive_name}\n")
        f.write(f"{sha256_file(manifest_path)}  manifest.json\n")
        if output_file_sha:
            f.write(f"{output_file_sha}  {output_file_path.name}\n")
    logger.info(colour("SHA256SUMS written: %s", "bright_cyan"), sha_path.name)

    return {
        "version_dir": version_dir,
        "archive_path": archive_path,
        "manifest_path": manifest_path,
        "combine_output_manifest_path": combine_output_manifest,
        "sha256sums_path": sha_path,
        "version_index_path": index_path,
    }
