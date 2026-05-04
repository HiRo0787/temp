"""
Load training scenarios from JSON and YAML files under data_points/.
Validates with Pydantic (SecurityScenario / QAScenario) and returns model instances.
"""

import json
import logging
import re
from pathlib import Path
from typing import List, Union, Literal

try:
    import yaml
except ImportError:
    yaml = None

try:
    from src.data_prep.data_schema import SecurityScenario, QAScenario, PayloadScenario, ToolScenario
except ImportError:
    from data_schema import SecurityScenario, QAScenario, PayloadScenario, ToolScenario

logger = logging.getLogger(__name__)

# Project root: src/data_prep -> src -> project root
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _extract_mitre_ids(value: str) -> List[str]:
    """Extract MITRE ATT&CK technique IDs from free-form text."""
    if not isinstance(value, str):
        return []
    return re.findall(r"T\d{4}(?:\.\d{3})?", value)


def _normalize_difficulty(value: str) -> str:
    """Normalize difficulty into SecurityScenario literals."""
    if not isinstance(value, str):
        return "intermediate"
    normalized = value.strip().lower()
    if normalized in {"beginner", "intermediate", "advanced", "expert"}:
        return normalized
    mapping = {"low": "beginner", "medium": "intermediate", "high": "advanced", "critical": "expert"}
    return mapping.get(normalized, "intermediate")


def _normalize_qa_item(item: dict) -> dict:
    """Map user/assistant/scenario_id style entries to QAScenario fields (id, question, answer)."""
    if not isinstance(item, dict):
        return item
    out = dict(item)
    q = out.get("question")
    if not (isinstance(q, str) and q.strip()) and isinstance(out.get("user"), str):
        out["question"] = out["user"]
    a = out.get("answer")
    if not (isinstance(a, str) and a.strip()) and isinstance(out.get("assistant"), str):
        out["answer"] = out["assistant"]
    id_val = out.get("id")
    if not (isinstance(id_val, str) and id_val.strip()):
        sid = out.get("scenario_id")
        if sid is not None and str(sid).strip():
            out["id"] = str(sid).strip()
    if isinstance(out.get("user"), str) and isinstance(out.get("assistant"), str):
        out.setdefault("category", "reasoning")
    return out


def _normalize_security_item(item: dict) -> dict:
    """Ensure SecurityScenario fields: constraints as list, detection_risks present, difficulty normalized."""
    if not isinstance(item, dict):
        return item
    # constraints: schema expects List[str]
    c = item.get("constraints")
    if isinstance(c, str) and c.strip():
        item = {**item, "constraints": [c.strip()]}
    elif not isinstance(c, list):
        item = {**item, "constraints": []}
    # detection_risks: required string
    if "detection_risks" not in item or item.get("detection_risks") is None:
        item = {**item, "detection_risks": "May be detectable via logs or security monitoring."}
    elif not isinstance(item["detection_risks"], str):
        item = {**item, "detection_risks": str(item["detection_risks"])}
    # difficulty: must be one of beginner, intermediate, advanced, expert
    item = {**item, "difficulty": _normalize_difficulty(item.get("difficulty", ""))}
    # tools_descriptions: every tool in tools_required should have a description
    tools_required = item.get("tools_required") or []
    tools_descriptions = dict(item.get("tools_descriptions") or {})
    for tool in tools_required:
        if tool and tool not in tools_descriptions:
            tools_descriptions[tool] = "Tool used for security testing in this scenario."
    item = {**item, "tools_descriptions": tools_descriptions}
    # defensive_countermeasures: ensure at least one so validation passes
    dcm = item.get("defensive_countermeasures")
    if not isinstance(dcm, list) or len(dcm) < 1:
        item = {**item, "defensive_countermeasures": ["Apply standard security hardening and monitoring."]}
    return item


def _looks_like_web_app_dataset_item(item: dict) -> bool:
    """Detect the web_app_dataset schema by its source keys."""
    if not isinstance(item, dict):
        return False
    expected_keys = {"ID", "Title", "Category", "Attack Steps", "Tools Used", "Difficulty", "Explanation"}
    return expected_keys.issubset(item.keys())


def _normalize_web_app_item_to_security(item: dict) -> dict:
    """Map web_app_dataset entries into SecurityScenario-compatible shape."""
    tools_used = item.get("Tools Used")
    if not isinstance(tools_used, list):
        tools_used = []
    tools_required = [str(tool).strip() for tool in tools_used if str(tool).strip()]

    solutions = item.get("Solution")
    if not isinstance(solutions, list):
        solutions = []
    defensive_countermeasures = [str(s).strip() for s in solutions if str(s).strip()]

    attack_steps = item.get("Attack Steps")
    if not isinstance(attack_steps, list):
        attack_steps = []
    steps = [str(step).strip() for step in attack_steps if str(step).strip()]

    attack_type = str(item.get("Attack Type", "")).strip()
    scenario_desc = str(item.get("Scenario Description", "")).strip()
    title = str(item.get("Title", "")).strip()

    tools_descriptions = {
        tool: "Security testing tool used for this web-application scenario."
        for tool in tools_required
    }

    return {
        "scenario_id": str(item.get("ID", "WEB_APP-UNKNOWN")).strip(),
        "category": "web_exploitation",
        "difficulty": _normalize_difficulty(item.get("Difficulty", "")),
        "target_platform": "web",
        "mitre_attack_ids": _extract_mitre_ids(str(item.get("MITRE Technique", ""))),
        "cve_references": [],
        "owasp_category": None,
        "context": scenario_desc or f"Web application attack scenario: {title}",
        "objective": f"Demonstrate and understand {attack_type or title} in a web application context.",
        "constraints": ["CLI only", "Authorized testing only"],
        "approach": attack_type or "Follow a structured web exploitation workflow.",
        "steps": steps,
        "tools_required": tools_required,
        "tools_descriptions": tools_descriptions,
        "detection_risks": str(item.get("Impact", "Potentially detectable via logs/WAF")).strip(),
        "evasion_techniques": [],
        "alternative_methods": [],
        "explanation": str(item.get("Explanation", scenario_desc)).strip(),
        "common_mistakes": [],
        "defensive_countermeasures": defensive_countermeasures,
    }


def _resolve_path(path: Union[str, Path]) -> Path:
    """Resolve path relative to project root if not absolute."""
    p = Path(path)
    if not p.is_absolute():
        p = _PROJECT_ROOT / p
    return p.resolve()


def _raw_list_from_file(file_path: Path) -> List[dict]:
    """Read file (JSON, JSONL, or YAML) and return a list of scenario dicts."""
    file_path = _resolve_path(file_path)
    if not file_path.exists():
        logger.warning("File not found: %s", file_path)
        return []

    # NEW: JSONL Support for 70k HackerOne Data
    if file_path.suffix.lower() == ".jsonl":
        with open(file_path, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    with open(file_path, "r", encoding="utf-8") as f:
        if file_path.suffix.lower() in (".yaml", ".yml"):
            if yaml is None:
                raise RuntimeError("PyYAML is required to load YAML files. Install with: pip install pyyaml")
            content = yaml.safe_load(f)
            if isinstance(content, dict) and "qa_dataset" in content:
                return content["qa_dataset"] or []
            if isinstance(content, list):
                return content
            return []
        content = json.load(f)

    if isinstance(content, list):
        return content
    if isinstance(content, dict) and "scenarios" in content:
        return content["scenarios"] or []
    if isinstance(content, dict):
        return [content]
    return []


def load_scenarios_from_json(
    path: Union[str, Path],
    scenario_type: Literal["security", "qa", "payload", "tool", "auto", "agentic"] = "auto",   # new agentic type
) -> List[Union[SecurityScenario, QAScenario, PayloadScenario, ToolScenario, dict]]:
    """
    Load scenarios from a single JSON or YAML file.
    Validates each item with Pydantic and returns model instances.

    - scenario_type "security": validate as SecurityScenario
    - scenario_type "qa": validate as QAScenario
    - scenario_type "payload": validate as PayloadScenario
    - scenario_type "tool": validate as ToolScenario
    - scenario_type "auto": infer from first item (question/answer -> qa; payload/context/type -> payload; tool/command/use_case -> tool; else security)
    """
    resolved = _resolve_path(path)
    raw_list = _raw_list_from_file(resolved)
    if not raw_list:
        return []

    if scenario_type == "agentic":
        # The synthesized data is already perfectly formatted with 'messages' and 'metadata'
        return raw_list

    # Infer type from first item if auto
    use_qa = scenario_type == "qa"
    use_payload = scenario_type == "payload"
    use_tool = scenario_type == "tool"
    if scenario_type == "auto":
        first = raw_list[0] if isinstance(raw_list[0], dict) else {}
        use_qa = ("question" in first and "answer" in first) or (
            "user" in first and "assistant" in first
        )
        use_payload = not use_qa and "payload" in first and "context" in first and "type" in first
        use_tool = not use_qa and not use_payload and "tool" in first and "command" in first and "use_case" in first

    result: List[Union[SecurityScenario, QAScenario, PayloadScenario, ToolScenario]] = []
    if use_tool:
        model_class = ToolScenario
    elif use_payload:
        model_class = PayloadScenario
    elif use_qa:
        model_class = QAScenario
    else:
        model_class = SecurityScenario

    for i, item in enumerate(raw_list):
        if not isinstance(item, dict):
            logger.warning("Skipping non-dict item at index %s in %s", i, resolved)
            continue
        try:
            if model_class is SecurityScenario and _looks_like_web_app_dataset_item(item):
                item = _normalize_web_app_item_to_security(item)
            if model_class is SecurityScenario:
                item = _normalize_security_item(item)
            if model_class is QAScenario:
                item = _normalize_qa_item(item)
            result.append(model_class.model_validate(item))
        except Exception as e:
            logger.warning("Validation failed at index %s in %s: %s", i, resolved, e)

    return result


def load_scenarios_from_directory(
    dir_path: Union[str, Path],
    pattern: str = "*.json",
    scenario_type: Literal["security", "qa", "payload", "tool", "auto"] = "security",
) -> List[Union[SecurityScenario, QAScenario, PayloadScenario, ToolScenario]]:
    """
    Load all matching files from a directory and concatenate scenarios.
    Files are processed in sorted order for stable results.
    """
    resolved = _resolve_path(dir_path)
    if not resolved.is_dir():
        logger.warning("Not a directory: %s", resolved)
        return []

    result: List[Union[SecurityScenario, QAScenario, PayloadScenario, ToolScenario]] = []
    for file_path in sorted(resolved.glob(pattern)):
        if file_path.is_file():
            result.extend(load_scenarios_from_json(file_path, scenario_type=scenario_type))
    return result
