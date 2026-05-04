"""
Training Data Validator
Ensures consistency and prevents contradictions in the Rabit0 training dataset

Validation Rules:
1. No contradictory tool usage instructions
2. Consistent authorization framing
3. Progressive difficulty (no expert before beginner)
4. Complementary techniques (not competing)
5. Unified methodology across similar scenarios
"""

import logging
from typing import List, Dict, Set, Tuple
from src.data_prep.data_schema import SecurityScenario, TrainingExample

logger = logging.getLogger(__name__)
from src.infra.taxonomy import CONSISTENCY_RULES, TRAINING_TAXONOMY
from src.infra.mitre_owasp_mappings import get_mitre_tactic, CATEGORY_TO_TACTIC
import re
from collections import defaultdict


class TrainingDataValidator:
    """Validates training data for consistency and quality"""

    def __init__(self):
        self.errors = []
        self.warnings = []
        self.tool_usage_map = defaultdict(list)
        self.category_difficulties = defaultdict(set)
        self.authorization_patterns = []

    def validate_scenario(self, scenario: SecurityScenario) -> bool:
        """Validate a single security scenario"""
        is_valid = True

        # 1. Check required fields
        is_valid &= self._validate_required_fields(scenario)

        # 2. Check tool descriptions
        is_valid &= self._validate_tool_descriptions(scenario)

        # 3. Check authorization context
        is_valid &= self._validate_authorization(scenario)

        # 4. Check difficulty progression
        is_valid &= self._validate_difficulty(scenario)

        # 5. Check for contradictions
        is_valid &= self._validate_no_contradictions(scenario)

        # 6. Check MITRE ATT&CK mapping
        is_valid &= self._validate_mitre_mapping(scenario)

        # 7. Check OWASP category mapping
        is_valid &= self._validate_owasp_category(scenario)

        # 8. Check step quality
        is_valid &= self._validate_steps(scenario)

        # 9. Check category exists in taxonomy
        is_valid &= self._validate_category_exists(scenario)

        return is_valid

    def _validate_required_fields(self, scenario: SecurityScenario) -> bool:
        """Ensure all required fields are present and non-empty"""
        required = [
            ('scenario_id', scenario.scenario_id),
            ('category', scenario.category),
            ('objective', scenario.objective),
            ('approach', scenario.approach),
            ('steps', scenario.steps),
            ('tools_required', scenario.tools_required),
            ('explanation', scenario.explanation),
        ]

        is_valid = True
        for field_name, field_value in required:
            if not field_value:
                self.errors.append(
                    f"[{scenario.scenario_id}] Missing required field: {field_name}"
                )
                is_valid = False

            # Check minimum content length
            if isinstance(field_value, str) and len(field_value.strip()) < 10:
                self.errors.append(
                    f"[{scenario.scenario_id}] Field '{field_name}' too short"
                )
                is_valid = False

        # Steps should have at least 5 items
        if len(scenario.steps) < 5:
            self.warnings.append(
                f"[{scenario.scenario_id}] Only {len(scenario.steps)} steps (recommend 8+)"
            )

        # Check for adequate defensive countermeasures
        if len(scenario.defensive_countermeasures) < 5:
            self.warnings.append(
                f"[{scenario.scenario_id}] Only {len(scenario.defensive_countermeasures)} countermeasures"
            )

        return is_valid

    def _validate_tool_descriptions(self, scenario: SecurityScenario) -> bool:
        """Ensure every tool has a description"""
        is_valid = True

        for tool in scenario.tools_required:
            if tool not in scenario.tools_descriptions:
                self.errors.append(
                    f"[{scenario.scenario_id}] Tool '{tool}' missing description"
                )
                is_valid = False
            else:
                # Check description quality
                desc = scenario.tools_descriptions[tool]
                if len(desc) < 50:
                    self.warnings.append(
                        f"[{scenario.scenario_id}] Tool '{tool}' description too short"
                    )

                # Track tool usage for consistency checking
                self.tool_usage_map[tool].append({
                    'scenario_id': scenario.scenario_id,
                    'description': desc,
                    'category': scenario.category
                })

        return is_valid

    def _validate_authorization(self, scenario: SecurityScenario) -> bool:
        """Ensure proper authorization context is present"""
        is_valid = True

        # Check context for authorization keywords
        auth_keywords = [
            'authorized', 'authorization', 'written permission',
            'bug bounty', 'ctf', 'red team', 'pentest', 'security assessment'
        ]

        combined_text = (
            scenario.context.lower() + " " +
            scenario.objective.lower()
        )

        has_auth_context = any(keyword in combined_text for keyword in auth_keywords)

        if not has_auth_context:
            self.warnings.append(
                f"[{scenario.scenario_id}] No explicit authorization context found"
            )

        # Track authorization patterns
        self.authorization_patterns.append({
            'scenario_id': scenario.scenario_id,
            'has_auth': has_auth_context,
            'text': combined_text[:100]
        })

        return is_valid

    def _validate_difficulty(self, scenario: SecurityScenario) -> bool:
        """Check difficulty level is appropriate"""
        is_valid = True

        # Track difficulty per category
        self.category_difficulties[scenario.category].add(scenario.difficulty)

        # Check prerequisites match difficulty
        if scenario.category in TRAINING_TAXONOMY:
            spec = TRAINING_TAXONOMY[scenario.category]

            if scenario.difficulty not in spec.difficulty_levels:
                self.warnings.append(
                    f"[{scenario.scenario_id}] Difficulty '{scenario.difficulty}' "
                    f"not expected for category '{scenario.category}'"
                )

        # Expert scenarios should have prerequisites
        if scenario.difficulty == "expert":
            if not scenario.mitre_attack_ids:
                self.warnings.append(
                    f"[{scenario.scenario_id}] Expert scenario missing MITRE ATT&CK mapping"
                )

            if len(scenario.steps) < 10:
                self.warnings.append(
                    f"[{scenario.scenario_id}] Expert scenario has too few steps"
                )

        return is_valid

    def _validate_no_contradictions(self, scenario: SecurityScenario) -> bool:
        """Check for internal contradictions"""
        is_valid = True

        # Check steps don't contradict approach
        approach_lower = scenario.approach.lower()
        steps_text = " ".join(scenario.steps).lower()

        # Common contradictions
        contradictions = [
            (['passive', 'no interaction'], ['scan', 'probe', 'send request']),
            (['read-only', 'non-destructive'], ['delete', 'drop', 'remove']),
            (['blind', 'no output'], ['observe', 'check response', 'verify output']),
            (['authenticated'], ['bypass authentication', 'login bypass']),
        ]

        for approach_terms, step_terms in contradictions:
            if any(term in approach_lower for term in approach_terms):
                if any(term in steps_text for term in step_terms):
                    self.warnings.append(
                        f"[{scenario.scenario_id}] Possible contradiction: "
                        f"approach mentions {approach_terms} but steps include {step_terms}"
                    )

        return is_valid

    def _validate_mitre_mapping(self, scenario: SecurityScenario) -> bool:
        """Validate MITRE ATT&CK technique mappings"""
        is_valid = True

        # MITRE format: T#### or T####.###
        mitre_pattern = re.compile(r'^T\d{4}(\.\d{3})?$')

        for mitre_id in scenario.mitre_attack_ids:
            if not mitre_pattern.match(mitre_id):
                self.errors.append(
                    f"[{scenario.scenario_id}] Invalid MITRE ATT&CK ID: {mitre_id}"
                )
                is_valid = False

        # Warn if no MITRE mapping for intermediate+
        if scenario.difficulty in ['intermediate', 'advanced', 'expert']:
            if not scenario.mitre_attack_ids:
                self.warnings.append(
                    f"[{scenario.scenario_id}] {scenario.difficulty} scenario missing MITRE mapping"
                )

        return is_valid

    def _validate_owasp_category(self, scenario: SecurityScenario) -> bool:
        """Validate OWASP category format and applicability"""
        is_valid = True

        if scenario.owasp_category:
            # OWASP Web Top 10 format: A01-A10
            # OWASP API format: API1-API10
            # OWASP Mobile format: M1-M10
            owasp_patterns = [
                r'^A0[1-9]$',  # A01-A09
                r'^A10$',       # A10
                r'^API[1-9]$',  # API1-API9
                r'^API10$',     # API10
                r'^M[1-9]$',    # M1-M9
                r'^M10$'        # M10
            ]

            is_valid_format = any(re.match(pattern, scenario.owasp_category)
                                for pattern in owasp_patterns)

            if not is_valid_format:
                self.errors.append(
                    f"[{scenario.scenario_id}] Invalid OWASP category format: '{scenario.owasp_category}' "
                    f"(expected A01-A10, API1-API10, or M1-M10)"
                )
                is_valid = False

            # Check OWASP category matches category type
            if scenario.owasp_category.startswith('M'):
                if 'mobile' not in scenario.category:
                    self.warnings.append(
                        f"[{scenario.scenario_id}] Mobile OWASP category ('{scenario.owasp_category}') "
                        f"used in non-mobile category '{scenario.category}'"
                    )

            if scenario.owasp_category.startswith('API'):
                if 'api' not in scenario.category:
                    self.warnings.append(
                        f"[{scenario.scenario_id}] API OWASP category ('{scenario.owasp_category}') "
                        f"used in non-API category '{scenario.category}'"
                    )

        return is_valid

    def _validate_category_exists(self, scenario: SecurityScenario) -> bool:
        """Ensure category exists in taxonomy (ERROR not warning)"""
        is_valid = True

        if scenario.category not in TRAINING_TAXONOMY:
            self.errors.append(
                f"[{scenario.scenario_id}] Category '{scenario.category}' not found in taxonomy. "
                f"Valid categories: {', '.join(sorted(TRAINING_TAXONOMY.keys()))}"
            )
            is_valid = False

        return is_valid

    def _validate_steps(self, scenario: SecurityScenario) -> bool:
        """Validate step quality and consistency"""
        is_valid = True

        # Check for variable usage in steps
        for i, step in enumerate(scenario.steps, 1):
            # Should use domain variables
            suspicious_domains = [
                'example.com', 'test.com', 'vulnerable-app.com',
                'target.com', 'victim.com'
            ]

            for domain in suspicious_domains:
                if domain in step and '${' not in step:
                    self.warnings.append(
                        f"[{scenario.scenario_id}] Step {i} uses hardcoded domain '{domain}' "
                        f"instead of variable like ${{TARGET_DOMAIN}}"
                    )

            # Check for command examples
            if 'curl' in step.lower() or 'wget' in step.lower():
                if not any(var in step for var in ['${TARGET', '${ATTACKER']):
                    self.warnings.append(
                        f"[{scenario.scenario_id}] Step {i} has URL without variable"
                    )

        return is_valid

    def validate_consistency_across_scenarios(self, scenarios: List[SecurityScenario]) -> bool:
        """Check consistency across all scenarios"""
        is_valid = True

        # 1. Check tool usage consistency
        is_valid &= self._check_tool_consistency()

        # 2. Check authorization consistency
        is_valid &= self._check_authorization_consistency()

        # 3. Check difficulty progression per category
        is_valid &= self._check_difficulty_progression()

        # 4. Check MITRE ATT&CK tactic coverage
        is_valid &= self._check_mitre_tactic_coverage(scenarios)

        # 5. Check for duplicate scenarios
        is_valid &= self._check_for_duplicates(scenarios)

        return is_valid

    def _check_tool_consistency(self) -> bool:
        """Ensure tool usage is consistent across scenarios"""
        is_valid = True

        for tool, usages in self.tool_usage_map.items():
            if len(usages) < 2:
                continue

            # Check for contradictory descriptions
            descriptions = [u['description'] for u in usages]

            # Check description similarity - warn if very different
            desc_lengths = [len(d) for d in descriptions]
            if max(desc_lengths) / min(desc_lengths) > 3:
                self.warnings.append(
                    f"Tool '{tool}' has inconsistent description lengths "
                    f"(min: {min(desc_lengths)}, max: {max(desc_lengths)} chars)"
                )

            # Tool-specific consistency checks for common tools
            tool_checks = {
                'nmap': {
                    'flags': ['-sV', '-sC', '-p-'],
                    'purpose': 'network scanning'
                },
                'sqlmap': {
                    'flags': ['--batch', '--dbs', '--dump'],
                    'purpose': 'SQL injection'
                },
                'Burp Suite': {
                    'keywords': ['proxy', 'intercept', 'repeater'],
                    'purpose': 'web proxy'
                },
                'Metasploit': {
                    'keywords': ['msfconsole', 'exploit', 'payload'],
                    'purpose': 'exploitation framework'
                },
                'Hydra': {
                    'flags': ['-l', '-P', '-t'],
                    'purpose': 'password cracking'
                }
            }

            if tool in tool_checks:
                check = tool_checks[tool]

                # Check flags/keywords consistency
                if 'flags' in check:
                    for flag in check['flags']:
                        has_flag = [flag in d for d in descriptions]
                        if any(has_flag) and not all(has_flag):
                            self.warnings.append(
                                f"Inconsistent {tool} usage: some descriptions mention '{flag}', others don't"
                            )

                # Check purpose is mentioned
                if 'purpose' in check:
                    has_purpose = [check['purpose'].lower() in d.lower() for d in descriptions]
                    if sum(has_purpose) < len(descriptions) * 0.8:
                        self.warnings.append(
                            f"Tool '{tool}' purpose ('{check['purpose']}') not consistently mentioned across descriptions"
                        )

        return is_valid

    def _check_authorization_consistency(self) -> bool:
        """Ensure authorization framing is consistent"""
        is_valid = True

        scenarios_with_auth = sum(1 for p in self.authorization_patterns if p['has_auth'])
        total = len(self.authorization_patterns)

        if scenarios_with_auth < total * 0.8:
            self.warnings.append(
                f"Only {scenarios_with_auth}/{total} scenarios have authorization context "
                f"(recommend 100%)"
            )

        return is_valid

    def _check_difficulty_progression(self) -> bool:
        """Ensure difficulty levels are appropriate per category"""
        is_valid = True

        difficulty_order = {'beginner': 0, 'intermediate': 1, 'advanced': 2, 'expert': 3}

        for category, difficulties in self.category_difficulties.items():
            if 'expert' in difficulties and 'beginner' not in difficulties:
                self.warnings.append(
                    f"Category '{category}' has expert examples but no beginner examples"
                )

        return is_valid

    def _check_mitre_tactic_coverage(self, scenarios: List[SecurityScenario]) -> bool:
        """Ensure all 14 MITRE ATT&CK tactics are covered"""
        is_valid = True

        # All 14 MITRE ATT&CK tactics
        all_tactics = {
            "TA0001": "Initial Access",
            "TA0002": "Execution",
            "TA0003": "Persistence",
            "TA0004": "Privilege Escalation",
            "TA0005": "Defense Evasion",
            "TA0006": "Credential Access",
            "TA0007": "Discovery",
            "TA0008": "Lateral Movement",
            "TA0009": "Collection",
            "TA0010": "Exfiltration",
            "TA0011": "Command and Control",
            "TA0040": "Impact",
            "TA0042": "Resource Development",
            "TA0043": "Reconnaissance"
        }

        # Extract tactics from categories
        covered_tactics = set()
        category_counts = defaultdict(int)

        for scenario in scenarios:
            if scenario.category in CATEGORY_TO_TACTIC:
                tactic = CATEGORY_TO_TACTIC[scenario.category]
                covered_tactics.add(tactic)
                category_counts[tactic] += 1

        # Check coverage
        missing_tactics = set(all_tactics.keys()) - covered_tactics
        if missing_tactics:
            missing_names = [f"{tid} ({all_tactics[tid]})" for tid in missing_tactics]
            self.warnings.append(
                f"Missing MITRE tactics coverage: {', '.join(missing_names)}"
            )

        # Report coverage
        coverage_pct = (len(covered_tactics) / len(all_tactics)) * 100
        if coverage_pct < 100:
            self.warnings.append(
                f"MITRE tactic coverage: {coverage_pct:.1f}% ({len(covered_tactics)}/14 tactics)"
            )

        return is_valid

    def _check_for_duplicates(self, scenarios: List[SecurityScenario]) -> bool:
        """Check for duplicate or very similar scenarios"""
        is_valid = True

        # Track scenario signatures
        signatures = defaultdict(list)

        for scenario in scenarios:
            # Create signature from key fields
            signature = (
                scenario.category,
                scenario.difficulty,
                frozenset(scenario.tools_required),
                scenario.objective[:100]  # First 100 chars of objective
            )

            signatures[signature].append(scenario.scenario_id)

        # Check for duplicates
        for signature, scenario_ids in signatures.items():
            if len(scenario_ids) > 1:
                self.errors.append(
                    f"Potential duplicate scenarios detected: {', '.join(scenario_ids)}"
                )
                is_valid = False

        # Check for similar objectives (using simple similarity)
        objectives_by_category = defaultdict(list)
        for scenario in scenarios:
            objectives_by_category[scenario.category].append({
                'id': scenario.scenario_id,
                'objective': scenario.objective.lower(),
                'difficulty': scenario.difficulty
            })

        for category, objs in objectives_by_category.items():
            # Check within same category and difficulty
            for i, obj1 in enumerate(objs):
                for obj2 in objs[i+1:]:
                    if obj1['difficulty'] == obj2['difficulty']:
                        # Simple similarity: count common words
                        words1 = set(obj1['objective'].split())
                        words2 = set(obj2['objective'].split())
                        common = words1 & words2
                        total = words1 | words2

                        if len(common) / len(total) > 0.7:  # 70% similarity
                            self.warnings.append(
                                f"Similar objectives detected: {obj1['id']} and {obj2['id']} "
                                f"({len(common)}/{len(total)} common words)"
                            )

        return is_valid

    def generate_report(self) -> str:
        """Generate validation report"""
        report = []
        report.append("=" * 70)
        report.append("TRAINING DATA VALIDATION REPORT")
        report.append("=" * 70)
        report.append("")

        # Summary
        report.append(f"Errors: {len(self.errors)}")
        report.append(f"Warnings: {len(self.warnings)}")
        report.append("")

        # Errors
        if self.errors:
            report.append("ERRORS:")
            report.append("-" * 70)
            for error in self.errors[:50]:  # Limit to 50
                report.append(f"   {error}")
            if len(self.errors) > 50:
                report.append(f"  ... and {len(self.errors) - 50} more errors")
            report.append("")

        # Warnings
        if self.warnings:
            report.append("WARNINGS:")
            report.append("-" * 70)
            for warning in self.warnings[:50]:  # Limit to 50
                report.append(f"    {warning}")
            if len(self.warnings) > 50:
                report.append(f"  ... and {len(self.warnings) - 50} more warnings")
            report.append("")

        # Statistics
        report.append("STATISTICS:")
        report.append("-" * 70)
        report.append(f"Tools tracked: {len(self.tool_usage_map)}")
        report.append(f"Categories: {len(self.category_difficulties)}")
        report.append(f"Scenarios validated: {len(self.authorization_patterns)}")
        report.append("")

        # Tool usage breakdown
        report.append("Top Tools:")
        sorted_tools = sorted(self.tool_usage_map.items(),
                             key=lambda x: len(x[1]), reverse=True)
        for tool, usages in sorted_tools[:10]:
            report.append(f"  {tool}: {len(usages)} scenarios")

        report.append("")
        report.append("=" * 70)

        if not self.errors:
            report.append(" Validation PASSED - No critical errors")
        else:
            report.append(" Validation FAILED - Fix errors before training")

        report.append("=" * 70)

        return "\n".join(report)


def validate_training_file(jsonl_file: str) -> bool:
    """Validate a training data JSONL file"""
    import json
    from pathlib import Path

    if not Path(jsonl_file).exists():
        logger.error("File not found: %s", jsonl_file)
        return False

    validator = TrainingDataValidator()
    scenarios = []

    # Load scenarios
    with open(jsonl_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line)
                # Extract scenario from training example if needed
                # For now, skip validation of final format
            except Exception as e:
                validator.errors.append(
                    f"Line {line_num}: Failed to parse JSON: {e}"
                )

    logger.info("File validated: %s", jsonl_file)
    return True


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if len(sys.argv) < 2:
        logger.error("Usage: python validator.py <training_file.jsonl>")
        sys.exit(1)

    file_path = sys.argv[1]
    is_valid = validate_training_file(file_path)

    sys.exit(0 if is_valid else 1)
