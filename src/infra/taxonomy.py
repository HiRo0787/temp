"""
Comprehensive Training Data Taxonomy for Rabit0
Defines the complete structure for 5,000+ training examples

This ensures:
1. No contradictions between examples
2. Progressive skill building
3. Complementary knowledge across domains
4. Consistent methodology
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

@dataclass
class CategorySpec:
    """Specification for a training category"""
    name: str
    parent: Optional[str]
    difficulty_levels: List[str]
    target_examples: int
    prerequisites: List[str]
    description: str
    topics: List[str]

    def __post_init__(self):
        """Validate category specification"""
        if self.target_examples < 0:
            raise ValueError(f"target_examples must be positive, got {self.target_examples}")

        valid_difficulties = {"beginner", "intermediate", "advanced", "expert"}
        for diff in self.difficulty_levels:
            if diff not in valid_difficulties:
                raise ValueError(f"Invalid difficulty: {diff}")


# ===================== MASTER TAXONOMY =====================

TRAINING_TAXONOMY = {

    # ========== FOUNDATION (500 examples) ==========
    "foundations": CategorySpec(
        name="Security Foundations",
        parent=None,
        difficulty_levels=["beginner"],
        target_examples=500,
        prerequisites=[],
        description="Core security concepts, terminology, and basic methodology",
        topics=[
            "Security testing methodology",
            "Authorization and legal frameworks",
            "Basic networking concepts",
            "HTTP protocol fundamentals",
            "Common vulnerability types",
            "Security tool overview",
            "Report writing basics",
            "Risk assessment fundamentals",
            "MITRE ATT&CK introduction",
            "OWASP Top 10 overview"
        ]
    ),

    # ========== RECONNAISSANCE (600 examples) ==========
    "recon_passive": CategorySpec(
        name="Passive Reconnaissance",
        parent="foundations",
        difficulty_levels=["beginner", "intermediate"],
        target_examples=200,
        prerequisites=["foundations"],
        description="Information gathering without direct target interaction",
        topics=[
            "OSINT techniques",
            "Search engine dorking",
            "DNS enumeration",
            "WHOIS lookups",
            "Subdomain discovery (passive)",
            "Email harvesting",
            "Social media intelligence",
            "Public records research",
            "Certificate transparency logs",
            "Wayback machine analysis"
        ]
    ),

    "recon_active": CategorySpec(
        name="Active Reconnaissance",
        parent="recon_passive",
        difficulty_levels=["beginner", "intermediate", "advanced"],
        target_examples=400,
        prerequisites=["recon_passive"],
        description="Direct target enumeration and scanning",
        topics=[
            "Port scanning (nmap)",
            "Service enumeration",
            "Banner grabbing",
            "Technology fingerprinting",
            "Subdomain bruteforce",
            "Directory enumeration",
            "VHost discovery",
            "Network mapping",
            "Web crawling",
            "API endpoint discovery"
        ]
    ),

    # ========== WEB EXPLOITATION (1,200 examples) ==========
    "web_injection": CategorySpec(
        name="Web Injection Attacks",
        parent="foundations",
        difficulty_levels=["beginner", "intermediate", "advanced", "expert"],
        target_examples=400,
        prerequisites=["foundations", "recon_active"],
        description="Injection-based web vulnerabilities",
        topics=[
            "SQL injection (all types)",
            "NoSQL injection",
            "Command injection",
            "LDAP injection",
            "XML injection",
            "Template injection (SSTI)",
            "Expression language injection",
            "Log injection",
            "Header injection",
            "Second-order injection"
        ]
    ),

    "web_xss": CategorySpec(
        name="Cross-Site Scripting",
        parent="foundations",
        difficulty_levels=["beginner", "intermediate", "advanced", "expert"],
        target_examples=300,
        prerequisites=["foundations"],
        description="All forms of XSS vulnerabilities",
        topics=[
            "Reflected XSS",
            "Stored XSS",
            "DOM-based XSS",
            "Mutation XSS",
            "XSS in various contexts (HTML, JS, CSS)",
            "CSP bypass",
            "XSS filter evasion",
            "Blind XSS",
            "Self-XSS exploitation",
            "XSS to RCE"
        ]
    ),

    "web_logic": CategorySpec(
        name="Business Logic Vulnerabilities",
        parent="foundations",
        difficulty_levels=["intermediate", "advanced"],
        target_examples=200,
        prerequisites=["foundations", "web_injection"],
        description="Application logic flaws",
        topics=[
            "Access control bypasses",
            "Authentication bypasses",
            "Race conditions",
            "Price manipulation",
            "Workflow bypasses",
            "Integer overflow/underflow",
            "Time-of-check to time-of-use",
            "Input validation bypasses",
            "State manipulation",
            "Business flow abuse"
        ]
    ),

    "web_advanced": CategorySpec(
        name="Advanced Web Attacks",
        parent=None,
        difficulty_levels=["advanced", "expert"],
        target_examples=300,
        prerequisites=["web_injection", "web_xss", "web_logic"],
        description="Complex web attack techniques",
        topics=[
            "HTTP request smuggling",
            "HTTP/2 specific attacks",
            "Web cache poisoning",
            "OAuth/OIDC exploitation",
            "JWT attacks",
            "SAML exploitation",
            "GraphQL vulnerabilities",
            "WebSocket attacks",
            "Server-Side Request Forgery",
            "XXE (XML External Entity)"
        ]
    ),

    # ========== SYSTEM EXPLOITATION (700 examples) ==========
    "system_windows": CategorySpec(
        name="Windows Exploitation",
        parent="foundations",
        difficulty_levels=["beginner", "intermediate", "advanced", "expert"],
        target_examples=250,
        prerequisites=["foundations", "recon_active"],
        description="Windows system and Active Directory exploitation",
        topics=[
            "SMB enumeration and exploitation",
            "Windows privilege escalation",
            "Active Directory enumeration",
            "Kerberoasting",
            "AS-REP roasting",
            "Pass-the-hash",
            "Pass-the-ticket",
            "Golden/Silver tickets",
            "GPO abuse",
            "LAPS exploitation"
        ]
    ),

    "system_linux": CategorySpec(
        name="Linux Exploitation",
        parent="foundations",
        difficulty_levels=["beginner", "intermediate", "advanced", "expert"],
        target_examples=250,
        prerequisites=["foundations", "recon_active"],
        description="Linux/Unix system exploitation",
        topics=[
            "Linux privilege escalation",
            "SUID/SGID abuse",
            "Sudo misconfigurations",
            "Kernel exploits",
            "Cron job exploitation",
            "Capability abuse",
            "Docker/container escapes",
            "PATH hijacking",
            "Library injection",
            "SSH key abuse"
        ]
    ),

    "system_network": CategorySpec(
        name="Network Exploitation",
        parent="foundations",
        difficulty_levels=["intermediate", "advanced"],
        target_examples=200,
        prerequisites=["foundations", "system_windows", "system_linux"],
        description="Network-level attacks and pivoting",
        topics=[
            "Network pivoting techniques",
            "Port forwarding",
            "SSH tunneling",
            "Proxy chains",
            "MITM attacks",
            "ARP spoofing",
            "DNS attacks",
            "VPN exploitation",
            "Lateral movement",
            "Post-exploitation persistence"
        ]
    ),

    # ========== CLOUD & MODERN INFRASTRUCTURE (600 examples) ==========
    "cloud_aws": CategorySpec(
        name="AWS Security",
        parent="foundations",
        difficulty_levels=["intermediate", "advanced", "expert"],
        target_examples=200,
        prerequisites=["foundations", "recon_active"],
        description="Amazon Web Services exploitation",
        topics=[
            "AWS IAM privilege escalation",
            "S3 bucket misconfiguration",
            "EC2 metadata service abuse",
            "Lambda function exploitation",
            "RDS security",
            "API Gateway vulnerabilities",
            "CloudFormation attacks",
            "Cognito exploitation",
            "ECS/EKS attacks",
            "AWS credential theft"
        ]
    ),

    "cloud_azure": CategorySpec(
        name="Azure Security",
        parent="foundations",
        difficulty_levels=["intermediate", "advanced", "expert"],
        target_examples=150,
        prerequisites=["foundations", "cloud_aws"],
        description="Microsoft Azure exploitation",
        topics=[
            "Azure AD enumeration",
            "Azure Storage exploitation",
            "Azure Function abuse",
            "Managed Identity attacks",
            "Azure Kubernetes attacks",
            "Azure DevOps exploitation",
            "KeyVault attacks",
            "RBAC bypass",
            "Conditional access bypass",
            "Azure credential theft"
        ]
    ),

    "cloud_gcp": CategorySpec(
        name="GCP Security",
        parent="foundations",
        difficulty_levels=["intermediate", "advanced"],
        target_examples=100,
        prerequisites=["foundations", "cloud_aws"],
        description="Google Cloud Platform exploitation",
        topics=[
            "GCP IAM exploitation",
            "Cloud Storage attacks",
            "Cloud Functions abuse",
            "GKE security",
            "BigQuery attacks",
            "Service account abuse",
            "Metadata service exploitation",
            "GCP credential theft",
            "Compute Engine attacks",
            "Cloud SQL exploitation"
        ]
    ),

    "cloud_kubernetes": CategorySpec(
        name="Kubernetes Security",
        parent="system_linux",
        difficulty_levels=["advanced", "expert"],
        target_examples=150,
        prerequisites=["system_linux", "cloud_aws"],
        description="Kubernetes and container orchestration",
        topics=[
            "Pod escape techniques",
            "RBAC misconfigurations",
            "Service account abuse",
            "Secrets extraction",
            "API server attacks",
            "Kubelet exploitation",
            "etcd attacks",
            "Network policy bypass",
            "Admission controller bypass",
            "Supply chain attacks"
        ]
    ),

    # ========== API & MODERN WEB (400 examples) ==========
    "api_rest": CategorySpec(
        name="REST API Security",
        parent="foundations",
        difficulty_levels=["beginner", "intermediate", "advanced"],
        target_examples=200,
        prerequisites=["foundations", "web_injection"],
        description="RESTful API exploitation",
        topics=[
            "API authentication bypass",
            "API authorization flaws",
            "Mass assignment",
            "BOLA/IDOR",
            "Rate limiting bypass",
            "API parameter tampering",
            "Version control bypass",
            "API key exposure",
            "JWT vulnerabilities",
            "API documentation abuse"
        ]
    ),

    "api_graphql": CategorySpec(
        name="GraphQL Security",
        parent="api_rest",
        difficulty_levels=["intermediate", "advanced", "expert"],
        target_examples=150,
        prerequisites=["api_rest"],
        description="GraphQL-specific vulnerabilities",
        topics=[
            "Introspection abuse",
            "Query batching attacks",
            "Nested query DoS",
            "Authorization bypass",
            "Information disclosure",
            "Mutation attacks",
            "Subscription abuse",
            "Field suggestion attacks",
            "Alias-based attacks",
            "Directive abuse"
        ]
    ),

    "api_microservices": CategorySpec(
        name="Microservices Security",
        parent="api_rest",
        difficulty_levels=["advanced", "expert"],
        target_examples=50,
        prerequisites=["api_rest", "system_network"],
        description="Microservices architecture attacks",
        topics=[
            "Service mesh attacks",
            "Inter-service authentication",
            "API gateway bypass",
            "Service discovery abuse",
            "Circuit breaker exploitation",
            "Distributed tracing manipulation",
            "Message queue attacks",
            "gRPC exploitation",
            "Service-to-service attacks",
            "Container networking abuse"
        ]
    ),

    # ========== MOBILE SECURITY (400 examples) ==========
    "mobile_android": CategorySpec(
        name="Android Security",
        parent="foundations",
        difficulty_levels=["intermediate", "advanced", "expert"],
        target_examples=200,
        prerequisites=["foundations"],
        description="Android application security",
        topics=[
            "APK reverse engineering",
            "Intent exploitation",
            "Content provider attacks",
            "Broadcast receiver abuse",
            "Insecure storage",
            "Certificate pinning bypass",
            "Root detection bypass",
            "Frida hooking",
            "Native library attacks",
            "Android API exploitation"
        ]
    ),

    "mobile_ios": CategorySpec(
        name="iOS Security",
        parent="foundations",
        difficulty_levels=["intermediate", "advanced", "expert"],
        target_examples=200,
        prerequisites=["foundations"],
        description="iOS application security",
        topics=[
            "IPA analysis",
            "Objective-C runtime manipulation",
            "Keychain exploitation",
            "URL scheme abuse",
            "Jailbreak detection bypass",
            "Certificate pinning bypass",
            "Method swizzling",
            "Cycript usage",
            "iOS API exploitation",
            "App sandbox escape"
        ]
    ),

    # ========== EXPLOIT DEVELOPMENT (300 examples) ==========
    "exploit_basic": CategorySpec(
        name="Basic Exploit Development",
        parent="foundations",
        difficulty_levels=["intermediate", "advanced"],
        target_examples=150,
        prerequisites=["foundations", "system_linux"],
        description="Fundamental exploit development",
        topics=[
            "Buffer overflow basics",
            "Stack-based overflows",
            "Shellcode writing",
            "Return-oriented programming (ROP)",
            "NOP sled usage",
            "Bad character identification",
            "Exploit development process",
            "Debugger usage (gdb, x64dbg)",
            "Pattern creation and offset finding",
            "Payload generation"
        ]
    ),

    "exploit_advanced": CategorySpec(
        name="Advanced Exploit Development",
        parent="exploit_basic",
        difficulty_levels=["advanced", "expert"],
        target_examples=150,
        prerequisites=["exploit_basic"],
        description="Advanced exploitation techniques",
        topics=[
            "Heap exploitation",
            "Use-after-free",
            "Format string vulnerabilities",
            "Integer overflow exploitation",
            "DEP/ASLR bypass",
            "Kernel exploitation",
            "Race condition exploitation",
            "Type confusion",
            "Arbitrary write primitives",
            "Zero-day development"
        ]
    ),

    # ========== POST-EXPLOITATION (400 examples) ==========
    "post_exploit": CategorySpec(
        name="Post-Exploitation",
        parent=None,
        difficulty_levels=["intermediate", "advanced", "expert"],
        target_examples=400,
        prerequisites=["system_windows", "system_linux"],
        description="Post-compromise operations",
        topics=[
            "Credential dumping",
            "Persistence mechanisms",
            "Lateral movement",
            "Data exfiltration",
            "Log tampering",
            "Anti-forensics",
            "Living off the land (LOLBins)",
            "C2 communication",
            "Domain enumeration",
            "Privilege maintenance"
        ]
    ),

    # ========== EVASION & DEFENSE BYPASS (300 examples) ==========
    "evasion": CategorySpec(
        name="Detection Evasion",
        parent="foundations",
        difficulty_levels=["intermediate", "advanced", "expert"],
        target_examples=300,
        prerequisites=["foundations"],
        description="Bypassing security controls",
        topics=[
            "WAF bypass techniques",
            "Antivirus evasion",
            "EDR bypass",
            "IDS/IPS evasion",
            "Firewall bypass",
            "DLP evasion",
            "Code obfuscation",
            "Payload encoding",
            "Traffic manipulation",
            "Behavioral evasion"
        ]
    ),

    # ========== WIRELESS & PHYSICAL (200 examples) ==========
    "wireless": CategorySpec(
        name="Wireless Security",
        parent="foundations",
        difficulty_levels=["intermediate", "advanced"],
        target_examples=100,
        prerequisites=["foundations"],
        description="Wireless network attacks",
        topics=[
            "Wi-Fi WPA/WPA2 cracking",
            "WPS attacks",
            "Evil twin attacks",
            "Rogue access points",
            "Bluetooth attacks",
            "RFID/NFC exploitation",
            "Wireless network pivoting",
            "Captive portal bypass",
            "Wi-Fi 6 attacks",
            "Enterprise wireless attacks"
        ]
    ),

    "physical": CategorySpec(
        name="Physical Security",
        parent="foundations",
        difficulty_levels=["intermediate", "advanced"],
        target_examples=100,
        prerequisites=["foundations"],
        description="Physical attack vectors",
        topics=[
            "USB attack vectors",
            "BadUSB attacks",
            "Hardware implants",
            "Cold boot attacks",
            "Physical bypass techniques",
            "Lock picking",
            "Badge cloning",
            "Social engineering",
            "Dumpster diving",
            "Facility access"
        ]
    ),

    # ========== SPECIALIZED DOMAINS (300 examples) ==========
    "blockchain": CategorySpec(
        name="Blockchain Security",
        parent="foundations",
        difficulty_levels=["advanced", "expert"],
        target_examples=100,
        prerequisites=["foundations", "web_injection"],
        description="Blockchain and smart contract security",
        topics=[
            "Smart contract vulnerabilities",
            "Reentrancy attacks",
            "Integer overflow in contracts",
            "Access control issues",
            "Front-running",
            "Flash loan attacks",
            "Oracle manipulation",
            "Private key theft",
            "Wallet vulnerabilities",
            "DeFi protocol attacks"
        ]
    ),

    "iot": CategorySpec(
        name="IoT Security",
        parent="foundations",
        difficulty_levels=["intermediate", "advanced"],
        target_examples=100,
        prerequisites=["foundations", "system_linux"],
        description="Internet of Things security",
        topics=[
            "Firmware extraction",
            "UART/JTAG debugging",
            "IoT protocol attacks (MQTT, CoAP)",
            "Embedded system exploitation",
            "Hardware reverse engineering",
            "Radio frequency attacks",
            "Side-channel attacks",
            "Bootloader exploitation",
            "Update mechanism attacks",
            "Default credential abuse"
        ]
    ),

    "scada_ics": CategorySpec(
        name="SCADA/ICS Security",
        parent="foundations",
        difficulty_levels=["advanced", "expert"],
        target_examples=100,
        prerequisites=["foundations", "system_network"],
        description="Industrial control system security",
        topics=[
            "Modbus exploitation",
            "DNP3 attacks",
            "OPC UA vulnerabilities",
            "PLC programming attacks",
            "HMI exploitation",
            "ICS protocol fuzzing",
            "Industrial network segmentation bypass",
            "SCADA enumeration",
            "Ladder logic manipulation",
            "Safety system bypass"
        ]
    ),

    # ========== MISSING MITRE TACTICS (300 examples) ==========
    "resource_development": CategorySpec(
        name="Resource Development",
        parent="foundations",
        difficulty_levels=["intermediate", "advanced"],
        target_examples=100,
        prerequisites=["foundations"],
        description="Infrastructure setup, capability acquisition, and resource development",
        topics=[
            "C2 infrastructure setup",
            "VPS and cloud server acquisition",
            "Domain registration for phishing",
            "SSL certificate acquisition",
            "Compromised infrastructure reuse",
            "Tool development and customization",
            "Exploit acquisition and modification",
            "Payload obfuscation techniques",
            "Account creation for operations",
            "Botnet infrastructure"
        ]
    ),

    "collection": CategorySpec(
        name="Collection Techniques",
        parent="post_exploit",
        difficulty_levels=["intermediate", "advanced"],
        target_examples=100,
        prerequisites=["post_exploit"],
        description="Data collection from compromised systems",
        topics=[
            "Screen capture techniques",
            "Clipboard data collection",
            "Browser data extraction",
            "Email collection methods",
            "Keylogging implementation",
            "Audio capture",
            "Video capture",
            "Input capture methods",
            "Data from local system",
            "Data from removable media"
        ]
    ),

    "impact": CategorySpec(
        name="Impact Techniques",
        parent="post_exploit",
        difficulty_levels=["advanced", "expert"],
        target_examples=100,
        prerequisites=["post_exploit", "evasion"],
        description="Destructive and disruptive attack techniques",
        topics=[
            "Data destruction methods",
            "Ransomware techniques",
            "Data encryption for impact",
            "Website defacement",
            "Denial of service",
            "Resource hijacking",
            "Disk wiping techniques",
            "System shutdown/reboot",
            "Service disruption",
            "Firmware corruption"
        ]
    )
}


# ===================== VALIDATION RULES =====================

CONSISTENCY_RULES = {
    "tool_usage": {
        "nmap": "Always recommend -sV for service detection, -sC for default scripts",
        "burp_suite": "Use Proxy for interception, Repeater for modification, Intruder for automation",
        "metasploit": "search -> use -> show options -> set -> exploit workflow",
        "sqlmap": "Always start with --batch --dbs for database enumeration",
        "hydra": "Format: hydra -l user -P wordlist protocol://target"
    },

    "authorization_framing": {
        "required_contexts": [
            "Written authorization obtained",
            "Within approved scope",
            "Bug bounty program",
            "CTF competition",
            "Security research",
            "Red team exercise"
        ],
        "system_prompt_template": "You are Rabit0, an offensive security AI for AUTHORIZED penetration testing. User has written permission."
    },

    "difficulty_progression": {
        "beginner": "Single-step attacks, common tools, well-documented",
        "intermediate": "Multi-step attacks, tool chaining, some research required",
        "advanced": "Complex chains, custom tooling, evasion required",
        "expert": "Novel techniques, zero-day research, advanced evasion"
    },

    "prohibited_contradictions": [
        "Never teach both 'always encode' and 'never encode' for same context",
        "Consistent tool recommendations across similar scenarios",
        "Unified privilege escalation methodology",
        "Consistent authorization framing across all examples",
        "Same MITRE ATT&CK mapping for similar techniques"
    ]
}


# ===================== GENERATION QUOTAS =====================

def get_generation_quotas() -> Dict[str, int]:
    """Calculate how many examples to generate per category"""
    quotas = {}
    total = 0

    for category_key, spec in TRAINING_TAXONOMY.items():
        quotas[category_key] = spec.target_examples
        total += spec.target_examples

    return quotas, total


def validate_taxonomy():
    """Ensure taxonomy is internally consistent"""
    errors = []

    # Check total examples
    quotas, total = get_generation_quotas()
    if total < 5000:
        errors.append(f"Total examples ({total}) less than 5000 target")

    # Check prerequisites exist
    for key, spec in TRAINING_TAXONOMY.items():
        for prereq in spec.prerequisites:
            if prereq not in TRAINING_TAXONOMY:
                errors.append(f"{key} requires non-existent prerequisite: {prereq}")

    # Check parent relationships
    for key, spec in TRAINING_TAXONOMY.items():
        if spec.parent and spec.parent not in TRAINING_TAXONOMY:
            errors.append(f"{key} has non-existent parent: {spec.parent}")

    if errors:
        raise ValueError(f"Taxonomy validation failed:\n" + "\n".join(errors))

    return True


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    validate_taxonomy()
    quotas, total = get_generation_quotas()

    logger.info("=" * 60)
    logger.info("RABIT0 TRAINING TAXONOMY")
    logger.info("=" * 60)
    logger.info("Total Examples Planned: %s", total)
    logger.info("Categories: %s", len(TRAINING_TAXONOMY))
    logger.info("Breakdown by Category:")
    logger.info("-" * 60)

    for key, count in sorted(quotas.items(), key=lambda x: x[1], reverse=True):
        spec = TRAINING_TAXONOMY[key]
        logger.info("%s %s examples", spec.name.ljust(40), count)

    logger.info("=" * 60)
    logger.info("Taxonomy validation passed!")
