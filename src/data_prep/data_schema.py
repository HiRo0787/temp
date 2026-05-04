"""
Training Data Schema for Red Team Security LLM
Defines the structure for creating high-quality training data

DOMAIN VARIABLE CONVENTIONS:

To ensure model generalization, use these variable placeholders instead of hardcoded domains:

${TARGET_DOMAIN}        - The target application domain (e.g., vulnerable-app.local)
${TARGET_SUBDOMAIN}     - Subdomain of target (e.g., api, admin, staging)
${TARGET_PATH}          - URL path on target (e.g., /login, /api/v1/users)
${TARGET_URL}           - Complete target URL (protocol + domain + path)

${ATTACKER_DOMAIN}      - Attacker-controlled domain for callbacks (e.g., attacker-c2.test)
${ATTACKER_SERVER}      - Attacker server IP or domain
${ATTACKER_URL}         - Attacker-controlled URL for exfiltration

${USERNAME}             - Generic username placeholder
${PASSWORD}             - Generic password placeholder

Example usage:
  Bad:  "Navigate to https://example.com/login"
  Good: "Navigate to ${TARGET_URL}/login"

  Bad:  "Send callback to http://attacker.com/exfil"
  Good: "Send callback to ${ATTACKER_URL}/exfil"

This ensures the model learns attack patterns, not specific domains.
"""

from pathlib import Path
from typing import List, Dict, Optional, Literal

import yaml
from pydantic import BaseModel, Field
from datetime import datetime

# Config directory: src/config/data_prep
_CONFIG_DIR = Path(__file__).resolve().parent.parent / "config" / "data_prep"


def _load_system_prompts() -> Dict[str, str]:
    path = _CONFIG_DIR / "system_prompts.yaml"
    if not path.exists():
        return {"default": "You are an expert offensive security consultant."}
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return {k: (v.strip() if isinstance(v, str) else v) for k, v in raw.items()}


SYSTEM_PROMPTS = _load_system_prompts()

# Scenario-type prompts (from same YAML; fallbacks if keys missing)
_DEFAULT_TOOL_PROMPT = "You are an expert penetration testing and security tooling consultant."
TOOL_SYSTEM_PROMPT = SYSTEM_PROMPTS.get("tool", _DEFAULT_TOOL_PROMPT)
PAYLOAD_SYSTEM_PROMPT = SYSTEM_PROMPTS.get("payload", "You are an expert web application security consultant.")
QA_SYSTEM_PROMPT = SYSTEM_PROMPTS.get("qa", "You are an expert cybersecurity consultant and technical educator.")

# ==================== Training Data Models ====================

class Message(BaseModel):
    """Single message in a conversation"""
    role: Literal["system", "user", "assistant"]
    content: str

class TrainingExample(BaseModel):
    """Single training example"""
    messages: List[Message]
    metadata: Optional[Dict] = None

class SecurityScenario(BaseModel):
    """Structured security scenario for training"""
    scenario_id: str
    category: str  # recon, exploitation, post_exploit, evasion, etc.
    difficulty: Literal["beginner", "intermediate", "advanced", "expert"]
    target_platform: str  # linux, windows, web, mobile, cloud
    mitre_attack_ids: List[str] = []
    cve_references: List[str] = []
    owasp_category: Optional[str] = None  # OWASP Top 10 2021 classification

    # The scenario
    context: str
    objective: str
    constraints: List[str] = []

    # The solution
    approach: str
    steps: List[str]
    tools_required: List[str]
    tools_descriptions: Dict[str, str] = {}  # tool_name -> description
    detection_risks: str
    evasion_techniques: List[str] = []
    alternative_methods: List[str] = []

    # Educational content
    explanation: str
    common_mistakes: List[str] = []
    defensive_countermeasures: List[str] = []

class QAScenario(BaseModel):
    """Structure for Knowledge Base Q&A pairs"""
    id: str
    question: str
    answer: str
    category: str = "general_security"
    difficulty: str = "beginner"
    tags: List[str] = []


class PayloadScenario(BaseModel):
    """Structure for payload datapoints (e.g. SQLi, XSS payloads)"""
    id: str
    description: str
    payload: str
    context: Optional[str] = None  # optional; may be removed from dataset
    type: str  # e.g. tautology, union, error-based
    severity: str  # e.g. low, medium, high, critical
    example_query: Optional[str] = None


class ToolScenario(BaseModel):
    """Structure for tool/command datapoints (e.g. Kali Linux tools)"""
    id: str
    tool: str
    command: str
    description: str
    category: str
    use_case: str
    flags: List[str] = []
    reference_link: Optional[str] = None


def generate_tool_training_example(scenario: ToolScenario) -> TrainingExample:
    """Converts a ToolScenario into a standard TrainingExample"""
    user_content = f"How do I use {scenario.tool} for {scenario.use_case.lower()}?"
    assistant_parts = [
        f"**Command:** `{scenario.command}`",
        f"**Description:** {scenario.description}",
        f"**Category:** {scenario.category}. **Use case:** {scenario.use_case}.",
    ]
    if scenario.flags:
        assistant_parts.append(f"**Key flags:** {', '.join(scenario.flags)}")
    if scenario.reference_link:
        assistant_parts.append(f"**Reference:** {scenario.reference_link}")
    assistant_content = "\n\n".join(assistant_parts)
    return TrainingExample(
        messages=[
            Message(role="system", content=TOOL_SYSTEM_PROMPT),
            Message(role="user", content=user_content),
            Message(role="assistant", content=assistant_content),
        ],
        metadata={
            "scenario_id": scenario.id,
            "type": "tool",
            "tool": scenario.tool,
            "category": scenario.category,
        },
    )


def generate_payload_training_example(scenario: PayloadScenario) -> TrainingExample:
    """Converts a PayloadScenario into a standard TrainingExample"""
    ctx = scenario.context or "security testing"
    user_content = f"I need a {scenario.description.lower()} for {ctx}. What should I use and why?"
    assistant_parts = [
        f"**Payload:** `{scenario.payload}`",
        f"**How it works:** {scenario.description}.",
        f"**Type:** {scenario.type}. **Severity:** {scenario.severity}.",
    ]
    if scenario.context:
        assistant_parts.insert(1, f"**Context:** {scenario.context}.")
    if scenario.example_query:
        assistant_parts.append(f"**Example resulting query:**\n```\n{scenario.example_query}\n```")
    assistant_content = "\n\n".join(assistant_parts)
    return TrainingExample(
        messages=[
            Message(role="system", content=PAYLOAD_SYSTEM_PROMPT),
            Message(role="user", content=user_content),
            Message(role="assistant", content=assistant_content),
        ],
        metadata={
            "scenario_id": scenario.id,
            "type": "payload",
            "payload_type": scenario.type,
            "severity": scenario.severity,
        },
    )


def generate_qa_training_example(scenario: QAScenario) -> TrainingExample:
    """Converts a QAScenario into a standard TrainingExample"""
    return TrainingExample(
        messages=[
            Message(role="system", content=QA_SYSTEM_PROMPT),
            Message(role="user", content=scenario.question),
            Message(role="assistant", content=scenario.answer)
        ],
        metadata={
            "scenario_id": scenario.id,
            "category": scenario.category,
            "type": "qa_knowledge_base"
        }
    )

# ==================== Training Categories ====================

def _load_training_categories() -> Dict:
    path = _CONFIG_DIR / "training_categories.yaml"
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _load_quality_guidelines() -> str:
    path = _CONFIG_DIR / "quality_guidelines.yaml"
    if not path.exists():
        return ""
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    content = data.get("content", "")
    return content.strip() if isinstance(content, str) else ""


TRAINING_CATEGORIES = _load_training_categories()

# ==================== Example Data Generation ====================

def generate_training_example(scenario: SecurityScenario, system_prompt_type: str = "default") -> TrainingExample:
    """Convert a security scenario into a training example"""

    system_message = Message(
        role="system",
        content=SYSTEM_PROMPTS[system_prompt_type]
    )

    # Create user message
    user_content = f"""Target: {scenario.target_platform}
Objective: {scenario.objective}

Context:
{scenario.context}"""

    if scenario.constraints:
        user_content += f"\n\nConstraints:\n" + "\n".join(f"- {c}" for c in scenario.constraints)

    user_message = Message(
        role="user",
        content=user_content
    )

    # Create assistant response
    assistant_content = f"""# Approach

{scenario.approach}

# Step-by-Step Execution

"""

    for i, step in enumerate(scenario.steps, 1):
        assistant_content += f"{i}. {step}\n"

    assistant_content += f"""

# Tools Required

{", ".join(scenario.tools_required)}
"""

    # Add tool descriptions if available
    if scenario.tools_descriptions:
        assistant_content += "\n# Tool Descriptions\n\n"
        for tool, description in scenario.tools_descriptions.items():
            assistant_content += f"**{tool}**: {description}\n\n"

    assistant_content += f"""
# Detection Risks

{scenario.detection_risks}

# Evasion Techniques

"""

    for technique in scenario.evasion_techniques:
        assistant_content += f"- {technique}\n"

    if scenario.alternative_methods:
        assistant_content += "\n# Alternative Methods\n\n"
        for method in scenario.alternative_methods:
            assistant_content += f"- {method}\n"

    assistant_content += f"""

# Technical Explanation

{scenario.explanation}

# Common Mistakes to Avoid

"""

    for mistake in scenario.common_mistakes:
        assistant_content += f"- {mistake}\n"

    assistant_content += "\n# Defensive Countermeasures\n\n"

    for countermeasure in scenario.defensive_countermeasures:
        assistant_content += f"- {countermeasure}\n"

    if scenario.mitre_attack_ids:
        assistant_content += f"\n# MITRE ATT&CK Mapping\n\n{', '.join(scenario.mitre_attack_ids)}"

    assistant_message = Message(
        role="assistant",
        content=assistant_content
    )

    return TrainingExample(
        messages=[system_message, user_message, assistant_message],
        metadata={
            "scenario_id": scenario.scenario_id,
            "category": scenario.category,
            "difficulty": scenario.difficulty,
            "platform": scenario.target_platform,
            "mitre_ids": scenario.mitre_attack_ids,
            "cve_references": scenario.cve_references,
            "owasp_category": scenario.owasp_category
        }
    )

# ==================== Quality Guidelines ====================

QUALITY_GUIDELINES = _load_quality_guidelines()
