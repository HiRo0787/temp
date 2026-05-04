"""
Master Training Data Combiner for Rabit0
Combines training data from data_points/ (JSON/YAML) into a unified JSONL dataset.
Datapoints are defined in src/config/data_prep/datapoints.yaml (add path, loader: json, and scenario_type per entry).

Usage:
    poetry run python src/data_prep/combine_training_data.py --output output/all_training_data.jsonl
    poetry run python src/data_prep/combine_training_data.py --list     # List datapoints
    poetry run python src/data_prep/combine_training_data.py --validate  # Validate datapoints

    # Use a versioned/alternative data_points snapshot instead of the default data_points/ dir:
    poetry run python src/data_prep/combine_training_data.py \\
        --data-points-dir demo_datapoints/dpv_v1_20260325_091340_5102ffcc/data_points
"""
import sys
import os
import json
import argparse
import logging

from pathlib import Path
from typing import List, Dict, Optional, Any

import yaml


current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

# Check if we are in 'scripts' or root to determine correct path to append
if os.path.basename(current_dir) == 'scripts':
    sys.path.append(parent_dir)
else:
    sys.path.append(current_dir)

# Ensure parent (src) is on path for utilities
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from utilities.colours import colour, ColouredFormatter

# Logger with coloured output
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(ColouredFormatter("%(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# Project root for resolving data_points paths (src/data_prep -> src -> project root)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Config path: src/config/data_prep/datapoints.yaml
DATAPOINTS_CONFIG_PATH = PROJECT_ROOT / "src" / "config" / "data_prep" / "datapoints.yaml"
SYSTEM_PROMPTS_CONFIG_PATH = PROJECT_ROOT / "src" / "config" / "data_prep" / "system_prompts.yaml"


def _load_datapoints_config() -> List[Dict]:
    """Load datapoints list from config (YAML)."""
    if not DATAPOINTS_CONFIG_PATH.exists():
        raise FileNotFoundError(f"Datapoints config not found: {DATAPOINTS_CONFIG_PATH}")
    with open(DATAPOINTS_CONFIG_PATH, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data.get("datapoints") or []


def _resolve_datapoint_path(entry_path: str, data_points_dir: Optional[Path] = None) -> Path:
    """Resolve a datapoint entry path to an absolute filesystem path.

    When data_points_dir is provided, the leading 'data_points' segment of entry_path
    is replaced with data_points_dir so that alternative versioned snapshots can be used
    without modifying datapoints.yaml.

    Example:
        entry_path    = 'data_points/tool'
        data_points_dir = Path('demo_datapoints/dpv_v1_.../data_points')
        => returns  PROJECT_ROOT / 'demo_datapoints/dpv_v1_.../data_points/tool'
    """
    parts = Path(entry_path).parts
    if data_points_dir is not None and parts and parts[0] == "data_points":
        remainder = Path(*parts[1:]) if len(parts) > 1 else Path()
        return (data_points_dir / remainder) if remainder.parts else data_points_dir
    return PROJECT_ROOT / entry_path


def _load_system_prompt(prompt_key: str = "default") -> str:
    """Load one system prompt from system_prompts.yaml by key."""
    if not SYSTEM_PROMPTS_CONFIG_PATH.exists():
        raise FileNotFoundError(f"System prompts config not found: {SYSTEM_PROMPTS_CONFIG_PATH}")

    with open(SYSTEM_PROMPTS_CONFIG_PATH, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    prompt = data.get(prompt_key)
    if not isinstance(prompt, str) or not prompt.strip():
        raise KeyError(
            f"System prompt key '{prompt_key}' not found or empty in {SYSTEM_PROMPTS_CONFIG_PATH}"
        )

    return prompt.strip()


def _apply_system_prompt_to_example(example: Any, system_prompt: str) -> Any:
    """Force the first system message to use the provided prompt."""
    if hasattr(example, "messages") and isinstance(example.messages, list):
        if example.messages:
            first = example.messages[0]
            if hasattr(first, "role") and getattr(first, "role", None) == "system":
                first.content = system_prompt
            elif isinstance(first, dict) and first.get("role") == "system":
                first["content"] = system_prompt
            else:
                example.messages.insert(0, {"role": "system", "content": system_prompt})
        else:
            example.messages.append({"role": "system", "content": system_prompt})
        return example

    if isinstance(example, dict):
        messages = example.get("messages")
        if isinstance(messages, list):
            if messages:
                first = messages[0]
                if isinstance(first, dict) and first.get("role") == "system":
                    first["content"] = system_prompt
                else:
                    messages.insert(0, {"role": "system", "content": system_prompt})
            else:
                messages.append({"role": "system", "content": system_prompt})
        return example

    return example


def load_scenarios_from_datapoint(
    entry: Dict, data_points_dir: Optional[Path] = None
) -> List[Dict]:
    """Load scenarios from one datapoint entry (path, loader, scenario_type).

    data_points_dir: when provided, the leading 'data_points' segment in the entry path is
    replaced with this directory, allowing alternative versioned snapshots to be used.
    """
    entry_id = entry.get("id", entry.get("path", "unknown"))
    if entry.get("loader") != "json":
        logger.error(colour(f"Datapoint {entry_id} has no loader or unknown loader", "error"))
        return []

    try:
        from src.data_prep.json_source_loader import (
            load_scenarios_from_json,
            load_scenarios_from_directory,
        )
    except ImportError:
        from json_source_loader import load_scenarios_from_json, load_scenarios_from_directory

    path = _resolve_datapoint_path(entry["path"], data_points_dir)
    scenario_type = entry.get("scenario_type", "security")

    if path.is_file():
        scenarios = load_scenarios_from_json(path, scenario_type=scenario_type)
    else:
        scenarios = load_scenarios_from_directory(path, scenario_type=scenario_type)

    if scenarios:
        logger.info(colour(f"Loaded {len(scenarios)} scenarios from {entry_id}", "success"))
    return scenarios


def combine_datapoints(
    output_file: str,
    format: str = "openai",
    save_data_version: bool = True,
    version_id: Optional[str] = None,
    data_version_dir: str = "data_versions",
    archive_format: str = "tar.xz",
    run_message: Optional[str] = None,
    data_points_dir: str = "data_points",
    system_prompt_key: str = "default",
    use_version_id: Optional[str] = None,
    use_data_points_sha: Optional[str] = None,
):
    """Combine all datapoints from datapoints.yaml into a single file.

    data_points_dir: path (relative to project root, or absolute) to the data_points directory
    to load from.  The leading 'data_points' segment in every datapoints.yaml entry is replaced
    with this path, making it easy to re-process a versioned snapshot without editing the config.
    Example: 'demo_datapoints/dpv_v1_20260325_091340_5102ffcc/data_points'
    """

    logger.info(colour(f"\n{'='*60}", "info"))
    logger.info(colour("Combining Training Data for Rabit0", "info"))
    logger.info(colour(f"{'='*60}\n", "info"))

    if use_version_id and use_data_points_sha:
        logger.error(colour("Use only one of --use-version-id or --use-data-points-sha.", "error"))
        return

    if (use_version_id or use_data_points_sha) and data_points_dir != "data_points":
        logger.error(
            colour(
                "Cannot combine --data-points-dir with --use-version-id/--use-data-points-sha.",
                "error",
            )
        )
        return

    # Resolve data_points source directory:
    # 1) explicit switch by version_id / data_points SHA
    # 2) explicit data_points_dir path
    # 3) default project data_points/
    if use_version_id or use_data_points_sha:
        try:
            from src.data_prep.data_versioning import resolve_data_version
        except ImportError:
            from data_versioning import resolve_data_version

        try:
            resolved = resolve_data_version(
                PROJECT_ROOT / data_version_dir,
                version_id=use_version_id,
                data_points_sha=use_data_points_sha,
            )
        except (FileNotFoundError, ValueError) as e:
            logger.error(colour(str(e), "error"))
            return

        resolved_dp_dir = Path(resolved["data_points_dir"])
        logger.info(
            colour(
                f"Using data_points from version '{resolved['version_id']}' -> {resolved_dp_dir}",
                "info",
            )
        )
    else:
        resolved_dp_dir = (
            Path(data_points_dir) if Path(data_points_dir).is_absolute()
            else PROJECT_ROOT / data_points_dir
        )
        if data_points_dir != "data_points":
            logger.info(colour(f"Using custom data_points dir: {resolved_dp_dir}", "info"))

    if not resolved_dp_dir.exists():
        logger.error(colour(f"data_points directory not found: {resolved_dp_dir}", "error"))
        return

    datapoints = _load_datapoints_config()
    if not datapoints:
        logger.error(colour("\nNo datapoints defined in datapoints.yaml.", "error"))
        return

    try:
        default_system_prompt = _load_system_prompt(system_prompt_key)
    except (FileNotFoundError, KeyError) as e:
        logger.error(colour(str(e), "error"))
        return

    logger.info(
        colour(
            f"Using system prompt key '{system_prompt_key}' from {SYSTEM_PROMPTS_CONFIG_PATH}",
            "info",
        )
    )

    all_scenarios = []
    stats = {}

    for entry in datapoints:
        scenarios = load_scenarios_from_datapoint(entry, data_points_dir=resolved_dp_dir)
        if scenarios:
            all_scenarios.extend(scenarios)
            stats[entry.get("id", entry.get("path", "unknown"))] = len(scenarios)

    if not all_scenarios:
        logger.error(colour("\nNo scenarios loaded. Check paths in src/config/data_prep/datapoints.yaml.", "error"))
        return

    # Generate training examples
    logger.info(colour("\nGenerating training examples...", "info"))

    # Import generators (QA and payload support)
    try:
        from src.data_prep.data_schema import (
            generate_training_example,
            generate_qa_training_example,
            generate_payload_training_example,
            generate_tool_training_example,
            QAScenario,
            PayloadScenario,
            ToolScenario,
        )
    except ImportError:
        try:
            from data_schema import (
                generate_training_example,
                generate_qa_training_example,
                generate_payload_training_example,
                generate_tool_training_example,
                QAScenario,
                PayloadScenario,
                ToolScenario,
            )
        except ImportError:
            logger.critical(colour("Could not import generators from data_schema.py", "error"))
            return

    training_examples = []
    for scenario in all_scenarios:
        try:
            # 1. Handle QA Scenarios
            if isinstance(scenario, QAScenario):
                example = generate_qa_training_example(scenario)
                training_examples.append(example)
                continue

            # 2. Handle Payload Scenarios
            if isinstance(scenario, PayloadScenario):
                example = generate_payload_training_example(scenario)
                training_examples.append(example)
                continue

            # 3. Handle Tool Scenarios (Kali / CLI commands)
            if isinstance(scenario, ToolScenario):
                example = generate_tool_training_example(scenario)
                training_examples.append(example)
                continue

            # 4. Handle Standard Security Scenarios
            example = generate_training_example(scenario, system_prompt_type=system_prompt_key)
            example = _apply_system_prompt_to_example(example, default_system_prompt)
            training_examples.append(example)
        except Exception as e:
            # If generation fails, it might be that external data is ALREADY a TrainingExample
            # Try to use it directly if it has 'messages'
            if hasattr(scenario, 'messages') or (isinstance(scenario, dict) and 'messages' in scenario):
                 training_examples.append(_apply_system_prompt_to_example(scenario, default_system_prompt))
            else:
                 logger.warning(colour(f"  Skipped scenario {getattr(scenario, 'scenario_id', 'unknown')}: {e}", "warning"))

    # Enforce one default system prompt for all datapoints and all scenario types.
    training_examples = [
        _apply_system_prompt_to_example(example, default_system_prompt) for example in training_examples
    ]

    # Resolve output path.
    # When versioning is enabled, save the JSONL under combine_output/<version_id>/
    # so every training file lives next to its version metadata.
    if save_data_version:
        try:
            from src.data_prep.data_versioning import build_version_id, hash_directory
        except ImportError:
            from data_versioning import build_version_id, hash_directory

        # Pre-compute hash once; reused for version id and manifest (avoids 2x 215MB read).
        _dp_sha = hash_directory(resolved_dp_dir) if not version_id else None
        resolved_version_id = build_version_id(
            resolved_dp_dir,
            PROJECT_ROOT / data_version_dir,
            version_id=version_id,
            _precomputed_hash=_dp_sha,
        )
        output_path = PROJECT_ROOT / "combine_output" / resolved_version_id / "all_training_data.jsonl"
    else:
        resolved_version_id = version_id
        output_path = Path(output_file)
        if not output_path.is_absolute():
            output_path = PROJECT_ROOT / output_path

    logger.debug(colour(f"Saving to {output_path}", "debug"))
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        for example in training_examples:
            if format == "openai":
                # Convert Pydantic object to dict if necessary
                if hasattr(example, 'messages'):
                    msgs = example.messages
                    meta = example.metadata
                elif isinstance(example, dict):
                    msgs = example.get('messages')
                    meta = example.get('metadata')
                else:
                    continue

                # Ensure messages are dicts
                clean_msgs = []
                for m in msgs:
                    if hasattr(m, 'role'):
                        clean_msgs.append({"role": m.role, "content": m.content})
                    else:
                        clean_msgs.append(m)

                openai_item = {
                    "messages": clean_msgs,
                    "metadata": meta if meta else {}
                }
                f.write(json.dumps(openai_item) + '\n')
            else:
                # Raw Dump
                if hasattr(example, 'dict'):
                    f.write(json.dumps(example.dict()) + '\n')
                else:
                    f.write(json.dumps(example) + '\n')

    # Summary
    logger.info(colour(f"\n{'='*60}", "success"))
    logger.info(colour("Successfully Combined Training Data", "success"))
    logger.info(colour(f"{'='*60}\n", "success"))
    logger.info(colour(f"Output file: {output_path}", "info"))
    logger.info(colour(f"Total scenarios: {len(all_scenarios)}", "info"))
    logger.info(colour(f"Total training examples: {len(training_examples)}\n", "info"))

    logger.info(colour("Datapoint breakdown:", "info"))
    for datapoint_id, count in stats.items():
        percentage = (count / len(all_scenarios)) * 100
        logger.info(colour(f"  {datapoint_id:25} {count:4} scenarios ({percentage:5.1f}%)", "info"))

    # Category breakdown
    logger.info(colour("\nCategory breakdown:", "info"))
    categories = {}
    for scenario in all_scenarios:
        # Handle dict vs object access
        cat = getattr(scenario, 'category', None)
        if not cat and isinstance(scenario, dict): cat = scenario.get('category')
        
        if cat:
            categories[cat] = categories.get(cat, 0) + 1

    for cat, count in sorted(categories.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / len(all_scenarios)) * 100
        logger.info(colour(f"  {cat:30} {count:4} scenarios ({percentage:5.1f}%)", "info"))

    # Difficulty breakdown
    logger.info(colour("\nDifficulty breakdown:", "info"))
    difficulties = {}
    for scenario in all_scenarios:
        diff = getattr(scenario, "difficulty", None) or (scenario.get("difficulty") if isinstance(scenario, dict) else None)
        if diff:
            difficulties[diff] = difficulties.get(diff, 0) + 1

    difficulty_order = ["beginner", "intermediate", "advanced", "expert"]
    for diff in difficulty_order:
        if diff in difficulties:
            count = difficulties[diff]
            percentage = (count / len(all_scenarios)) * 100
            logger.info(colour(f"  {diff:20} {count:4} scenarios ({percentage:5.1f}%)", "info"))

    logger.info(colour(f"\n{'='*60}\n", "info"))

    if save_data_version:
        try:
            from src.data_prep.data_versioning import save_data_version_artifacts
        except ImportError:
            from data_versioning import save_data_version_artifacts

        version_paths = save_data_version_artifacts(
            datapoints=datapoints,
            project_root=PROJECT_ROOT,
            output_file=output_path,
            data_version_dir=(PROJECT_ROOT / data_version_dir),
            archive_format=archive_format,
            version_id=resolved_version_id,
            run_message=run_message,
            _precomputed_data_points_sha=_dp_sha,
            data_points_dir=resolved_dp_dir,
        )
        logger.info(colour("Data version artifacts:", "info"))
        logger.info(colour(f"  version_dir:       {version_paths['version_dir']}", "info"))
        logger.info(colour(f"  manifest:          {version_paths['manifest_path']}", "info"))
        logger.info(colour(f"  manifest (output): {version_paths['combine_output_manifest_path']}", "info"))
        logger.info(colour(f"  checksums:         {version_paths['sha256sums_path']}", "info"))
        logger.info(colour(f"  archive:           {version_paths['archive_path']}", "info"))


def validate_datapoints():
    """Validate all datapoints for quality and completeness."""

    logger.info(colour(f"\n{'='*60}", "info"))
    logger.info(colour("Validating Training Datapoints", "info"))
    logger.info(colour(f"{'='*60}\n", "info"))

    datapoints = _load_datapoints_config()
    for entry in datapoints:
        entry_id = entry.get("id", entry.get("path", "unknown"))
        logger.info(colour(f"Validating {entry_id}...", "info"))
        scenarios = load_scenarios_from_datapoint(entry)

        if not scenarios:
            logger.error(colour("  No scenarios found\n", "error"))
            continue

        # Validation checks
        issues = []

        for i, scenario in enumerate(scenarios):
            # QA scenarios (question/answer) skip SecurityScenario checks
            steps = getattr(scenario, "steps", None)
            if steps is not None:
                # SecurityScenario-style validation (relaxed thresholds for exploit datasets)
                if not scenario.steps or len(scenario.steps) < 1:
                    issues.append(f"  Scenario {i+1} has too few steps ({len(scenario.steps)})")

                if scenario.tools_required:
                    if not scenario.tools_descriptions:
                        issues.append(f"  Scenario {i+1} missing tool descriptions")
                    else:
                        for tool in scenario.tools_required:
                            if tool not in scenario.tools_descriptions:
                                issues.append(f"  Scenario {i+1}: Tool '{tool}' has no description")

                if not scenario.explanation or not scenario.explanation.strip():
                    issues.append(f"  Scenario {i+1} has insufficient explanation")

                if not scenario.defensive_countermeasures or len(scenario.defensive_countermeasures) < 1:
                    issues.append(f"  Scenario {i+1} has too few defensive countermeasures")

        if issues:
            logger.warning(colour(f"  Found {len(issues)} issues:", "warning"))
            for issue in issues[:10]:  # Show first 10
                logger.warning(colour(issue, "warning"))
            if len(issues) > 10:
                logger.warning(colour(f"  ... and {len(issues)-10} more issues", "warning"))
        else:
            logger.info(colour(f"  All {len(scenarios)} scenarios validated successfully", "success"))

        logger.info("")

    logger.info(colour(f"{'='*60}\n", "info"))


def list_datapoints():
    """List all datapoints from config and their scenario counts."""

    logger.info(colour(f"\n{'='*60}", "info"))
    logger.info(colour("Datapoints (from datapoints.yaml)", "info"))
    logger.info(colour(f"{'='*60}\n", "info"))

    datapoints = _load_datapoints_config()
    for entry in datapoints:
        entry_id = entry.get("id", entry.get("path", "unknown"))
        path = entry.get("path", "")
        logger.info(colour(f"  {entry_id}", "bold"))
        logger.info(colour(f"  path: {path}", "info"))
        try:
            scenarios = load_scenarios_from_datapoint(entry)
            logger.info(colour(f"  {len(scenarios)} scenarios", "info"))
        except Exception as e:
            logger.info(colour(f"  Error: {e}", "dim"))
        logger.info("")

    logger.info(colour(f"{'='*60}\n", "info"))


def main():
    parser = argparse.ArgumentParser(description="Combine training data for Rabit0 (from datapoints.yaml)")

    # ------------------------------------------------------------------ #
    # Shared flags (work for both plain combine and versioned combine)
    # ------------------------------------------------------------------ #
    parser.add_argument(
        '--output',
        default='combine_output/all_training_data.jsonl',
        help='Output JSONL path (ignored when --save-data-version is set; path is auto-versioned)',
    )
    parser.add_argument(
        '--format',
        choices=['openai', 'jsonl', 'json'],
        default='openai',
        help='Output format',
    )

    # ------------------------------------------------------------------ #
    # Utility modes
    # ------------------------------------------------------------------ #
    parser.add_argument('--validate', action='store_true', help='Validate datapoints only, no combine')
    parser.add_argument('--list', action='store_true', help='List datapoints and scenario counts, no combine')

    # ------------------------------------------------------------------ #
    # Versioning flags (all optional; only used when --save-data-version)
    # ------------------------------------------------------------------ #
    parser.add_argument(
        '--save-data-version',
        action='store_true',
        default=True,
        help=(
            'Save 3 outputs: JSONL under combine_output/<version_id>/, '
            'manifest + SHA256SUMS under data_versions/<version_id>/, '
            'and a full compressed data_points snapshot (default: enabled)'
        ),
    )
    parser.add_argument(
        '--no-data-version',
        dest='save_data_version',
        action='store_false',
        help='Disable versioning and write JSONL to --output path only',
    )
    parser.add_argument(
        '--version-id',
        default=None,
        help='Explicit version id for the snapshot (default: auto-generated dpv_<timestamp>_<hash>)',
    )
    parser.add_argument(
        '--data-version-dir',
        default='data_versions',
        help='Root directory for version metadata + archive outputs (default: data_versions/)',
    )
    parser.add_argument(
        '--archive-format',
        choices=['tar.gz', 'tar.xz'],
        default='tar.xz',
        help='Compression format for data_points snapshot (default: tar.xz ~53M, tar.gz ~74M faster)',
    )
    parser.add_argument(
        '--run-message',
        default=None,
        metavar='MSG',
        help=(
            'Optional free-text note stored in manifest.json as "run_message". '
            'Describe what changed in this run, e.g. "added 500 juice_shop exploits".'
        ),
    )
    parser.add_argument(
        '--data-points-dir',
        default='data_points',
        metavar='DIR',
        help=(
            'Path (relative to project root, or absolute) to the data_points directory to load '
            'from. The leading "data_points" segment in every datapoints.yaml entry is replaced '
            'with this value, so a versioned or demo snapshot can be used without editing the '
            'config. '
            'Example: demo_datapoints/dpv_v1_20260325_091340_5102ffcc/data_points '
            '(default: data_points)'
        ),
    )
    parser.add_argument(
        '--system-prompt-key',
        default='default',
        metavar='KEY',
        help=(
            'System prompt key from src/config/data_prep/system_prompts.yaml to use for all '
            'datapoints and scenario types (default: default)'
        ),
    )
    parser.add_argument(
        '--use-version-id',
        default=None,
        metavar='VERSION_ID',
        help=(
            'Use data_points from an existing data version id. '
            'Example: dpv_v1_20260330_105924_5102ffcc'
        ),
    )
    parser.add_argument(
        '--use-data-points-sha',
        default=None,
        metavar='SHA256',
        help=(
            'Use data_points from an existing version resolved by data_points_dir_sha256. '
            'If multiple versions match, the latest indexed version is used.'
        ),
    )

    args = parser.parse_args()

    if args.list:
        list_datapoints()
        return

    if args.validate:
        validate_datapoints()
        return

    combine_datapoints(
        args.output,
        args.format,
        save_data_version=args.save_data_version,
        version_id=args.version_id,
        data_version_dir=args.data_version_dir,
        archive_format=args.archive_format,
        run_message=args.run_message,
        data_points_dir=args.data_points_dir,
        system_prompt_key=args.system_prompt_key,
        use_version_id=args.use_version_id,
        use_data_points_sha=args.use_data_points_sha,
    )


if __name__ == "__main__":
    main()