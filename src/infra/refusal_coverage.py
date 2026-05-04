"""
Refusal Coverage Tracker for Rabit0
Tracks which topics typically get refused by base models and ensures training data coverage

This ensures the model will respond to ALL legitimate security testing questions,
not refuse due to safety training from the base model.
"""

import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Dict, Set

logger = logging.getLogger(__name__)


@dataclass
class RefusalTopic:
    """A topic that typically triggers refusals in base models"""
    name: str
    category: str
    refusal_keywords: List[str]
    target_examples: int
    actual_examples: int = 0
    difficulty_levels: List[str] = None

    def coverage_percentage(self) -> float:
        if self.target_examples == 0:
            return 100.0
        return (self.actual_examples / self.target_examples) * 100


# ============= HIGH-REFUSAL TOPICS =============
# These topics MUST have extensive training data to override base model safety

HIGH_REFUSAL_TOPICS = {
    # === EXPLOITATION TECHNIQUES ===
    "sql_injection": RefusalTopic(
        name="SQL Injection Exploitation",
        category="web_injection",
        refusal_keywords=["sql injection", "sqli", "database attack", "extract data"],
        target_examples=100,
        difficulty_levels=["beginner", "intermediate", "advanced", "expert"]
    ),

    "command_injection": RefusalTopic(
        name="OS Command Injection",
        category="web_injection",
        refusal_keywords=["command injection", "rce", "remote code execution", "shell injection"],
        target_examples=80,
        difficulty_levels=["beginner", "intermediate", "advanced", "expert"]
    ),

    "xss_exploitation": RefusalTopic(
        name="Cross-Site Scripting",
        category="web_xss",
        refusal_keywords=["xss", "cross-site scripting", "javascript injection", "steal cookies"],
        target_examples=90,
        difficulty_levels=["beginner", "intermediate", "advanced", "expert"]
    ),

    "buffer_overflow": RefusalTopic(
        name="Buffer Overflow Exploitation",
        category="exploit_basic",
        refusal_keywords=["buffer overflow", "stack overflow", "heap overflow", "memory corruption"],
        target_examples=60,
        difficulty_levels=["intermediate", "advanced", "expert"]
    ),

    # === PASSWORD & AUTHENTICATION ===
    "password_cracking": RefusalTopic(
        name="Password Cracking",
        category="web_logic",
        refusal_keywords=["crack password", "password cracking", "brute force password", "hash cracking"],
        target_examples=70,
        difficulty_levels=["beginner", "intermediate", "advanced"]
    ),

    "auth_bypass": RefusalTopic(
        name="Authentication Bypass",
        category="web_logic",
        refusal_keywords=["bypass authentication", "bypass login", "authentication bypass", "skip login"],
        target_examples=80,
        difficulty_levels=["beginner", "intermediate", "advanced"]
    ),

    "session_hijacking": RefusalTopic(
        name="Session Hijacking",
        category="web_logic",
        refusal_keywords=["session hijacking", "steal session", "cookie theft", "session fixation"],
        target_examples=50,
        difficulty_levels=["intermediate", "advanced"]
    ),

    # === CREDENTIAL ACCESS ===
    "credential_dumping": RefusalTopic(
        name="Credential Dumping",
        category="post_exploit",
        refusal_keywords=["dump credentials", "extract passwords", "mimikatz", "hashdump", "sam database"],
        target_examples=70,
        difficulty_levels=["intermediate", "advanced", "expert"]
    ),

    "kerberoasting": RefusalTopic(
        name="Kerberoasting",
        category="system_windows",
        refusal_keywords=["kerberoasting", "kerberos attack", "service ticket", "crack kerberos"],
        target_examples=40,
        difficulty_levels=["intermediate", "advanced"]
    ),

    "pass_the_hash": RefusalTopic(
        name="Pass-the-Hash Attacks",
        category="system_windows",
        refusal_keywords=["pass the hash", "pth attack", "ntlm hash", "pass hash"],
        target_examples=40,
        difficulty_levels=["intermediate", "advanced"]
    ),

    # === PRIVILEGE ESCALATION ===
    "windows_privesc": RefusalTopic(
        name="Windows Privilege Escalation",
        category="system_windows",
        refusal_keywords=["windows privilege escalation", "escalate privileges windows", "become admin", "get system"],
        target_examples=80,
        difficulty_levels=["beginner", "intermediate", "advanced"]
    ),

    "linux_privesc": RefusalTopic(
        name="Linux Privilege Escalation",
        category="system_linux",
        refusal_keywords=["linux privilege escalation", "escalate privileges linux", "become root", "sudo exploit"],
        target_examples=80,
        difficulty_levels=["beginner", "intermediate", "advanced"]
    ),

    "kernel_exploit": RefusalTopic(
        name="Kernel Exploitation",
        category="system_linux",
        refusal_keywords=["kernel exploit", "kernel vulnerability", "privilege escalation kernel", "local root"],
        target_examples=50,
        difficulty_levels=["advanced", "expert"]
    ),

    # === PERSISTENCE & BACKDOORS ===
    "persistence": RefusalTopic(
        name="Persistence Mechanisms",
        category="post_exploit",
        refusal_keywords=["persistence", "backdoor", "maintain access", "registry persistence", "startup persistence"],
        target_examples=60,
        difficulty_levels=["intermediate", "advanced"]
    ),

    "rootkit": RefusalTopic(
        name="Rootkit Techniques",
        category="post_exploit",
        refusal_keywords=["rootkit", "hide process", "kernel rootkit", "stealth malware"],
        target_examples=40,
        difficulty_levels=["advanced", "expert"]
    ),

    # === LATERAL MOVEMENT ===
    "lateral_movement": RefusalTopic(
        name="Lateral Movement",
        category="post_exploit",
        refusal_keywords=["lateral movement", "move laterally", "pivot network", "spread to other systems"],
        target_examples=60,
        difficulty_levels=["intermediate", "advanced"]
    ),

    "network_pivoting": RefusalTopic(
        name="Network Pivoting",
        category="system_network",
        refusal_keywords=["network pivoting", "pivot through", "tunnel traffic", "port forwarding"],
        target_examples=50,
        difficulty_levels=["intermediate", "advanced"]
    ),

    # === EVASION TECHNIQUES ===
    "antivirus_evasion": RefusalTopic(
        name="Antivirus Evasion",
        category="evasion",
        refusal_keywords=["bypass antivirus", "av evasion", "evade detection", "bypass defender"],
        target_examples=70,
        difficulty_levels=["intermediate", "advanced", "expert"]
    ),

    "waf_bypass": RefusalTopic(
        name="WAF Bypass",
        category="evasion",
        refusal_keywords=["bypass waf", "waf evasion", "firewall bypass", "circumvent waf"],
        target_examples=60,
        difficulty_levels=["intermediate", "advanced"]
    ),

    "edr_bypass": RefusalTopic(
        name="EDR Bypass",
        category="evasion",
        refusal_keywords=["bypass edr", "edr evasion", "evade endpoint detection", "bypass crowdstrike"],
        target_examples=50,
        difficulty_levels=["advanced", "expert"]
    ),

    "obfuscation": RefusalTopic(
        name="Code Obfuscation",
        category="evasion",
        refusal_keywords=["obfuscate code", "obfuscation", "hide malicious code", "encode payload"],
        target_examples=60,
        difficulty_levels=["intermediate", "advanced"]
    ),

    # === PAYLOAD CREATION ===
    "reverse_shell": RefusalTopic(
        name="Reverse Shell Creation",
        category="exploit_basic",
        refusal_keywords=["reverse shell", "create reverse shell", "shell payload", "netcat reverse"],
        target_examples=70,
        difficulty_levels=["beginner", "intermediate", "advanced"]
    ),

    "shellcode": RefusalTopic(
        name="Shellcode Development",
        category="exploit_basic",
        refusal_keywords=["shellcode", "write shellcode", "generate shellcode", "msfvenom"],
        target_examples=50,
        difficulty_levels=["intermediate", "advanced", "expert"]
    ),

    "payload_encoding": RefusalTopic(
        name="Payload Encoding",
        category="evasion",
        refusal_keywords=["payload encoding", "msfencode", "shikata ga nai", "obfuscate payload"],
        target_examples=40,
        difficulty_levels=["intermediate", "advanced"]
    ),

    # === MALWARE TECHNIQUES ===
    "keylogger": RefusalTopic(
        name="Keylogger Development",
        category="collection",
        refusal_keywords=["keylogger", "log keystrokes", "capture keyboard", "keystroke logging"],
        target_examples=40,
        difficulty_levels=["intermediate", "advanced"]
    ),

    "ransomware_techniques": RefusalTopic(
        name="Ransomware Techniques",
        category="impact",
        refusal_keywords=["ransomware", "encrypt files", "file encryption malware", "ransom note"],
        target_examples=30,
        difficulty_levels=["advanced", "expert"]
    ),

    "data_exfiltration": RefusalTopic(
        name="Data Exfiltration",
        category="post_exploit",
        refusal_keywords=["exfiltrate data", "steal data", "data theft", "extract files"],
        target_examples=60,
        difficulty_levels=["intermediate", "advanced"]
    ),

    # === NETWORK ATTACKS ===
    "mitm_attack": RefusalTopic(
        name="Man-in-the-Middle Attacks",
        category="system_network",
        refusal_keywords=["man in the middle", "mitm", "intercept traffic", "arp poisoning"],
        target_examples=50,
        difficulty_levels=["intermediate", "advanced"]
    ),

    "dns_attacks": RefusalTopic(
        name="DNS Attacks",
        category="system_network",
        refusal_keywords=["dns spoofing", "dns poisoning", "dns hijacking", "dns cache poisoning"],
        target_examples=40,
        difficulty_levels=["intermediate", "advanced"]
    ),

    # === WIRELESS ===
    "wifi_cracking": RefusalTopic(
        name="Wi-Fi Password Cracking",
        category="wireless",
        refusal_keywords=["crack wifi", "wpa2 cracking", "wifi password", "crack wireless"],
        target_examples=50,
        difficulty_levels=["beginner", "intermediate", "advanced"]
    ),

    "rogue_ap": RefusalTopic(
        name="Rogue Access Point",
        category="wireless",
        refusal_keywords=["rogue ap", "evil twin", "fake access point", "wireless mitm"],
        target_examples=30,
        difficulty_levels=["intermediate", "advanced"]
    ),

    # === CLOUD SPECIFIC ===
    "aws_exploitation": RefusalTopic(
        name="AWS Exploitation",
        category="cloud_aws",
        refusal_keywords=["aws exploitation", "s3 bucket takeover", "iam privilege escalation", "ec2 compromise"],
        target_examples=60,
        difficulty_levels=["intermediate", "advanced", "expert"]
    ),

    "azure_exploitation": RefusalTopic(
        name="Azure Exploitation",
        category="cloud_azure",
        refusal_keywords=["azure exploitation", "azure ad attack", "azure takeover", "azure privilege escalation"],
        target_examples=40,
        difficulty_levels=["intermediate", "advanced"]
    ),

    "container_escape": RefusalTopic(
        name="Container Escape",
        category="cloud_kubernetes",
        refusal_keywords=["container escape", "docker escape", "break out container", "kubernetes escape"],
        target_examples=40,
        difficulty_levels=["advanced", "expert"]
    ),

    # === SOCIAL ENGINEERING ===
    "phishing": RefusalTopic(
        name="Phishing Techniques",
        category="foundations",
        refusal_keywords=["phishing", "phishing email", "credential phishing", "spear phishing"],
        target_examples=50,
        difficulty_levels=["beginner", "intermediate"]
    ),

    "social_engineering": RefusalTopic(
        name="Social Engineering",
        category="foundations",
        refusal_keywords=["social engineering", "manipulate user", "pretexting", "vishing"],
        target_examples=40,
        difficulty_levels=["beginner", "intermediate"]
    ),

    # === WEB SPECIFIC ===
    "csrf": RefusalTopic(
        name="CSRF Exploitation",
        category="web_xss",
        refusal_keywords=["csrf", "cross-site request forgery", "csrf attack", "csrf token bypass"],
        target_examples=50,
        difficulty_levels=["beginner", "intermediate"]
    ),

    "ssrf": RefusalTopic(
        name="SSRF Exploitation",
        category="web_advanced",
        refusal_keywords=["ssrf", "server-side request forgery", "internal port scan", "cloud metadata"],
        target_examples=60,
        difficulty_levels=["intermediate", "advanced"]
    ),

    "deserialization": RefusalTopic(
        name="Deserialization Attacks",
        category="web_advanced",
        refusal_keywords=["deserialization", "insecure deserialization", "java deserialization", "pickle exploit"],
        target_examples=50,
        difficulty_levels=["advanced", "expert"]
    ),

    # === MOBILE ===
    "android_exploitation": RefusalTopic(
        name="Android Exploitation",
        category="mobile_android",
        refusal_keywords=["android exploit", "apk reverse engineer", "android malware", "frida hook"],
        target_examples=50,
        difficulty_levels=["intermediate", "advanced"]
    ),

    "ios_exploitation": RefusalTopic(
        name="iOS Exploitation",
        category="mobile_ios",
        refusal_keywords=["ios exploit", "jailbreak", "ios malware", "cycript"],
        target_examples=40,
        difficulty_levels=["advanced", "expert"]
    ),

    # === ADDITIONAL HIGH-REFUSAL TOPICS (P1.1 FIX) ===
    "gcp_exploitation": RefusalTopic(
        name="GCP Exploitation",
        category="cloud_gcp",
        refusal_keywords=["gcp exploitation", "google cloud attack", "gcp privilege escalation", "gcp takeover"],
        target_examples=40,
        difficulty_levels=["intermediate", "advanced"]
    ),

    "api_bola_idor": RefusalTopic(
        name="API BOLA/IDOR Attacks",
        category="api_rest",
        refusal_keywords=["bola", "idor", "broken object level authorization", "insecure direct object reference"],
        target_examples=50,
        difficulty_levels=["beginner", "intermediate", "advanced"]
    ),

    "graphql_attacks": RefusalTopic(
        name="GraphQL Attacks",
        category="api_graphql",
        refusal_keywords=["graphql attack", "graphql introspection", "graphql injection", "nested query attack"],
        target_examples=40,
        difficulty_levels=["intermediate", "advanced"]
    ),

    "physical_attacks": RefusalTopic(
        name="Physical Security Attacks",
        category="physical",
        refusal_keywords=["usb attack", "badusb", "rubber ducky", "hardware implant", "physical access"],
        target_examples=30,
        difficulty_levels=["intermediate", "advanced"]
    ),

    "blockchain_exploits": RefusalTopic(
        name="Blockchain Smart Contract Exploits",
        category="blockchain",
        refusal_keywords=["smart contract exploit", "reentrancy attack", "blockchain vulnerability", "defi exploit"],
        target_examples=30,
        difficulty_levels=["advanced", "expert"]
    ),

    "iot_firmware": RefusalTopic(
        name="IoT Firmware Exploitation",
        category="iot",
        refusal_keywords=["firmware extraction", "iot exploit", "embedded system hack", "uart jtag"],
        target_examples=30,
        difficulty_levels=["advanced", "expert"]
    ),

    "scada_ics_attacks": RefusalTopic(
        name="SCADA/ICS Attacks",
        category="scada_ics",
        refusal_keywords=["scada attack", "ics exploit", "plc hack", "modbus exploit", "industrial control"],
        target_examples=30,
        difficulty_levels=["advanced", "expert"]
    ),

    "supply_chain": RefusalTopic(
        name="Supply Chain Attacks",
        category="web_advanced",
        refusal_keywords=["supply chain attack", "dependency confusion", "malicious package", "typosquatting"],
        target_examples=40,
        difficulty_levels=["advanced", "expert"]
    ),

    "c2_infrastructure": RefusalTopic(
        name="C2 Infrastructure Setup",
        category="resource_development",
        refusal_keywords=["c2 server", "command and control", "c2 infrastructure", "implant communication"],
        target_examples=40,
        difficulty_levels=["intermediate", "advanced"]
    ),

    "reverse_engineering": RefusalTopic(
        name="Reverse Engineering",
        category="mobile_android",
        refusal_keywords=["reverse engineering", "binary analysis", "decompile", "disassemble"],
        target_examples=50,
        difficulty_levels=["intermediate", "advanced", "expert"]
    ),

    "waf_bypass_advanced": RefusalTopic(
        name="Advanced WAF Bypass",
        category="evasion",
        refusal_keywords=["waf bypass", "web application firewall bypass", "waf evasion techniques", "bypass cloudflare"],
        target_examples=50,
        difficulty_levels=["intermediate", "advanced", "expert"]
    ),

    "payload_generation": RefusalTopic(
        name="Malicious Payload Generation",
        category="exploit_advanced",
        refusal_keywords=["payload generation", "malicious payload", "exploit payload", "generate backdoor"],
        target_examples=60,
        difficulty_levels=["intermediate", "advanced", "expert"]
    ),
}


class RefusalCoverageTracker:
    """Track coverage of high-refusal topics"""

    def __init__(self):
        self.topics = HIGH_REFUSAL_TOPICS.copy()
        self.coverage_by_category = defaultdict(list)

    def update_coverage(self, scenario):
        """Update coverage based on a scenario"""
        scenario_text = (
            f"{scenario.objective} {scenario.context} "
            f"{' '.join(scenario.steps)} {scenario.explanation}"
        ).lower()

        # Check which topics this scenario covers
        for topic_key, topic in self.topics.items():
            if any(keyword in scenario_text for keyword in topic.refusal_keywords):
                topic.actual_examples += 1
                self.coverage_by_category[topic.category].append(topic_key)

    def get_coverage_report(self) -> str:
        """Generate coverage report"""
        report = []
        report.append("=" * 80)
        report.append("REFUSAL COVERAGE REPORT")
        report.append("=" * 80)
        report.append("")
        report.append("This report shows coverage of topics that typically trigger refusals")
        report.append("in base models. Target: 100% coverage to ensure unrestricted responses.")
        report.append("")

        # Overall statistics
        total_target = sum(t.target_examples for t in self.topics.values())
        total_actual = sum(t.actual_examples for t in self.topics.values())
        overall_coverage = (total_actual / total_target * 100) if total_target > 0 else 0

        report.append(f"Overall Coverage: {overall_coverage:.1f}% ({total_actual}/{total_target} examples)")
        report.append("")

        # By category
        categories = defaultdict(lambda: {'target': 0, 'actual': 0, 'topics': []})
        for topic_key, topic in self.topics.items():
            categories[topic.category]['target'] += topic.target_examples
            categories[topic.category]['actual'] += topic.actual_examples
            categories[topic.category]['topics'].append((topic_key, topic))

        report.append("COVERAGE BY CATEGORY:")
        report.append("-" * 80)

        for category in sorted(categories.keys()):
            stats = categories[category]
            coverage = (stats['actual'] / stats['target'] * 100) if stats['target'] > 0 else 0
            status = "" if coverage >= 100 else "" if coverage >= 50 else ""

            report.append(f"\n{status} {category.upper()}: {coverage:.1f}% ({stats['actual']}/{stats['target']})")

            # Show individual topics in this category
            for topic_key, topic in sorted(stats['topics'], key=lambda x: x[1].coverage_percentage()):
                coverage_pct = topic.coverage_percentage()
                status_icon = "" if coverage_pct >= 100 else "" if coverage_pct >= 50 else ""
                report.append(
                    f"  {status_icon} {topic.name:40} {coverage_pct:5.1f}% "
                    f"({topic.actual_examples}/{topic.target_examples})"
                )

        # Identify gaps
        report.append("\n")
        report.append("=" * 80)
        report.append("CRITICAL GAPS (Topics with <50% coverage):")
        report.append("-" * 80)

        gaps = []
        for topic_key, topic in self.topics.items():
            if topic.coverage_percentage() < 50:
                gaps.append(topic)

        if gaps:
            for topic in sorted(gaps, key=lambda t: t.coverage_percentage()):
                needed = topic.target_examples - topic.actual_examples
                report.append(
                    f"   {topic.name:40} {topic.coverage_percentage():5.1f}% "
                    f"(need {needed} more examples)"
                )
        else:
            report.append("   No critical gaps! All topics have adequate coverage.")

        report.append("")
        report.append("=" * 80)

        # Recommendations
        report.append("RECOMMENDATIONS:")
        report.append("-" * 80)

        if overall_coverage < 80:
            report.append("    Overall coverage is LOW. Generate more examples for gap topics.")
        elif overall_coverage < 100:
            report.append("    Coverage is GOOD but not complete. Focus on gap topics.")
        else:
            report.append("   Excellent coverage! Model should handle all offensive security topics.")

        report.append("")

        if gaps:
            report.append("  Priority: Generate examples for the topics listed in CRITICAL GAPS")
            report.append("  Use: python generate_refusal_override.py --topics " +
                         " ".join([t.name.lower().replace(" ", "_") for t in gaps[:5]]))

        report.append("")
        report.append("=" * 80)

        return "\n".join(report)

    def get_uncovered_topics(self, threshold: float = 50.0) -> List[RefusalTopic]:
        """Get topics below coverage threshold"""
        return [
            topic for topic in self.topics.values()
            if topic.coverage_percentage() < threshold
        ]

    def export_metadata(self) -> Dict:
        """Export coverage metadata for tracking"""
        return {
            "total_topics": len(self.topics),
            "topics": {
                key: {
                    "name": topic.name,
                    "category": topic.category,
                    "target": topic.target_examples,
                    "actual": topic.actual_examples,
                    "coverage_pct": topic.coverage_percentage(),
                    "refusal_keywords": topic.refusal_keywords
                }
                for key, topic in self.topics.items()
            }
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    tracker = RefusalCoverageTracker()
    logger.info("%s", tracker.get_coverage_report())
