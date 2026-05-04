"""
MITRE ATT&CK and OWASP Mappings for Security Training Data Generation

This module provides accurate mappings between:
- Security topics to MITRE ATT&CK techniques
- Security topics to OWASP categories
- Training categories to MITRE ATT&CK tactics

Usage:
    from mitre_owasp_mappings import get_mitre_techniques, get_owasp_category, get_mitre_tactic

    techniques = get_mitre_techniques("SQL injection")
    owasp = get_owasp_category("XSS")
    tactic = get_mitre_tactic("web_injection")
"""

import logging

logger = logging.getLogger(__name__)

# =============================================================================
# TOPIC TO MITRE ATT&CK TECHNIQUE MAPPING
# =============================================================================

TOPIC_TO_MITRE = {
    # Web Injection Attacks
    "sql injection": ["T1190"],  # Exploit Public-Facing Application
    "command injection": ["T1059", "T1190"],  # Command and Scripting Interpreter, Exploit Public-Facing Application
    "nosql injection": ["T1190"],
    "ldap injection": ["T1190"],
    "xml injection": ["T1190"],
    "template injection": ["T1190"],

    # XSS and Client-Side
    "xss": ["T1059.007"],  # JavaScript
    "dom xss": ["T1059.007"],
    "stored xss": ["T1059.007"],
    "reflected xss": ["T1059.007"],
    "csrf": ["T1185"],  # Browser Session Hijacking

    # Authentication & Session
    "authentication bypass": ["T1078"],  # Valid Accounts
    "password cracking": ["T1110"],  # Brute Force
    "session hijacking": ["T1185"],  # Browser Session Hijacking
    "jwt vulnerabilities": ["T1528"],  # Steal Application Access Token
    "oauth vulnerabilities": ["T1550.001"],  # Use Alternate Authentication Material: Application Access Token
    "saml vulnerabilities": ["T1606"],  # Forge Web Credentials

    # Access Control
    "idor": ["T1213"],  # Data from Information Repositories
    "bola": ["T1213"],  # Broken Object Level Authorization
    "path traversal": ["T1083"],  # File and Directory Discovery
    "lfi": ["T1083", "T1005"],  # File and Directory Discovery, Data from Local System
    "rfi": ["T1105"],  # Ingress Tool Transfer

    # Advanced Web
    "ssrf": ["T1595.002"],  # Active Scanning: Vulnerability Scanning
    "deserialization": ["T1027.002"],  # Obfuscated Files or Information: Software Packing
    "xxe": ["T1190"],
    "file upload": ["T1105"],  # Ingress Tool Transfer
    "race condition": ["T1195.002"],  # Compromise Software Supply Chain

    # API Security
    "api authentication": ["T1078"],
    "api rate limiting": ["T1499"],  # Endpoint Denial of Service
    "api injection": ["T1190"],
    "graphql": ["T1190"],
    "rest api": ["T1190"],

    # System - Windows
    "privilege escalation": ["T1068", "T1548"],  # Exploitation for Privilege Escalation, Abuse Elevation Control Mechanism
    "uac bypass": ["T1548.002"],  # Bypass User Account Control
    "token manipulation": ["T1134"],  # Access Token Manipulation
    "kerberoasting": ["T1558.003"],  # Steal or Forge Kerberos Tickets: Kerberoasting
    "pass-the-hash": ["T1550.002"],  # Use Alternate Authentication Material: Pass the Hash
    "pass-the-ticket": ["T1550.003"],  # Use Alternate Authentication Material: Pass the Ticket
    "golden ticket": ["T1558.001"],  # Steal or Forge Kerberos Tickets: Golden Ticket
    "silver ticket": ["T1558.002"],  # Steal or Forge Kerberos Tickets: Silver Ticket
    "dcsync": ["T1003.006"],  # OS Credential Dumping: DCSync

    # System - Linux
    "linux privilege escalation": ["T1068", "T1548"],
    "sudo exploitation": ["T1548.003"],  # Sudo and Sudo Caching
    "suid exploitation": ["T1548.001"],  # Setuid and Setgid
    "kernel exploit": ["T1068"],
    "container escape": ["T1611"],  # Escape to Host

    # System - Network
    "port scanning": ["T1046"],  # Network Service Discovery
    "service enumeration": ["T1046"],
    "network pivoting": ["T1090"],  # Proxy
    "arp spoofing": ["T1557.002"],  # ARP Cache Poisoning
    "mitm attack": ["T1557"],  # Adversary-in-the-Middle
    "dns attacks": ["T1071.004"],  # Application Layer Protocol: DNS

    # Cloud - AWS
    "iam misconfiguration": ["T1078.004"],  # Valid Accounts: Cloud Accounts
    "s3 misconfiguration": ["T1530"],  # Data from Cloud Storage
    "ec2 exploitation": ["T1078.004"],
    "lambda exploitation": ["T1648"],  # Serverless Execution

    # Cloud - Azure
    "azure ad": ["T1078.004"],
    "azure storage": ["T1530"],
    "azure vm": ["T1078.004"],

    # Cloud - GCP
    "gcp iam": ["T1078.004"],
    "gcs buckets": ["T1530"],
    "gce exploitation": ["T1078.004"],

    # Cloud - Kubernetes
    "pod escape": ["T1611"],
    "rbac misconfiguration": ["T1552.007"],  # Unsecured Credentials: Container API
    "secrets exposure": ["T1552.007"],

    # Mobile - Android
    "apk analysis": ["T1437"],  # Application Layer Protocol
    "android rooting": ["T1628"],  # Hide Artifacts
    "ssl pinning bypass": ["T1557"],  # Adversary-in-the-Middle
    "intent exploitation": ["T1575"],  # Native API

    # Mobile - iOS
    "ipa analysis": ["T1437"],
    "jailbreak detection": ["T1628"],
    "ios ssl pinning": ["T1557"],
    "keychain exploitation": ["T1555"],  # Credentials from Password Stores

    # Reconnaissance
    "passive recon": ["T1593"],  # Search Open Websites/Domains
    "active recon": ["T1595"],  # Active Scanning
    "osint": ["T1593"],
    "subdomain enumeration": ["T1590.001"],  # Gather Victim Network Information: Domain Properties
    "dns enumeration": ["T1590.002"],  # DNS

    # Exploit Development
    "buffer overflow": ["T1203"],  # Exploitation for Client Execution
    "rop chain": ["T1203"],
    "heap exploitation": ["T1203"],
    "format string": ["T1203"],
    "use-after-free": ["T1203"],

    # Wireless
    "wifi cracking": ["T1200"],  # Hardware Additions
    "wpa handshake": ["T1200"],
    "evil twin": ["T1557"],
    "bluetooth exploitation": ["T1200"],

    # Physical
    "usb attack": ["T1091"],  # Replication Through Removable Media
    "badusb": ["T1091"],
    "hardware implant": ["T1200"],  # Hardware Additions
    "rfid cloning": ["T1200"],

    # Blockchain
    "smart contract": ["T1190"],
    "reentrancy": ["T1190"],
    "flash loan": ["T1190"],

    # IoT
    "firmware extraction": ["T1542"],  # Pre-OS Boot
    "uart access": ["T1542.001"],  # System Firmware
    "jtag debugging": ["T1542.001"],
    "hardcoded credentials": ["T1552.001"],  # Credentials In Files

    # SCADA/ICS
    "modbus exploitation": ["T0855"],  # Unauthorized Command Message (ICS)
    "plc programming": ["T0883"],  # Modify Control Logic (ICS)
    "hmi exploitation": ["T0866"],  # Exploitation of Remote Services (ICS)

    # Evasion
    "waf bypass": ["T1562.001"],  # Impair Defenses: Disable or Modify Tools
    "av bypass": ["T1562.001"],
    "edr bypass": ["T1562.001"],
    "obfuscation": ["T1027"],  # Obfuscated Files or Information
    "payload encoding": ["T1027"],
    "polymorphic code": ["T1027.007"],  # Dynamic API Resolution

    # Post-Exploitation
    "credential dumping": ["T1003"],  # OS Credential Dumping
    "lsass dumping": ["T1003.001"],  # LSASS Memory
    "sam extraction": ["T1003.002"],  # Security Account Manager
    "persistence": ["T1547"],  # Boot or Logon Autostart Execution
    "lateral movement": ["T1021"],  # Remote Services
    "data exfiltration": ["T1041"],  # Exfiltration Over C2 Channel
    "psexec": ["T1021.002"],  # SMB/Windows Admin Shares
    "winrm": ["T1021.006"],  # Windows Remote Management

    # Resource Development
    "c2 infrastructure": ["T1583"],  # Acquire Infrastructure
    "domain registration": ["T1583.001"],  # Domains
    "vps acquisition": ["T1583.003"],  # Virtual Private Server
    "tool development": ["T1587"],  # Develop Capabilities

    # Collection
    "screen capture": ["T1113"],  # Screen Capture
    "keylogging": ["T1056.001"],  # Input Capture: Keylogging
    "clipboard data": ["T1115"],  # Clipboard Data
    "browser data": ["T1555.003"],  # Credentials from Web Browsers

    # Impact
    "ransomware": ["T1486"],  # Data Encrypted for Impact
    "data destruction": ["T1485"],  # Data Destruction
    "defacement": ["T1491"],  # Defacement
    "dos": ["T1498"],  # Network Denial of Service

    # Social Engineering
    "phishing": ["T1566"],  # Phishing
    "spear phishing": ["T1566.001"],  # Spearphishing Attachment
    "credential harvesting": ["T1056.003"],  # Web Portal Capture

    # Reverse Engineering
    "binary analysis": ["T1027"],
    "decompilation": ["T1140"],  # Deobfuscate/Decode Files or Information
    "malware analysis": ["T1027"],

    # Default fallback
    "generic": ["T1190"],  # Exploit Public-Facing Application
}

# =============================================================================
# TOPIC TO OWASP CATEGORY MAPPING
# =============================================================================

TOPIC_TO_OWASP = {
    # OWASP Top 10 2021 - Web Application
    "idor": "A01",  # Broken Access Control
    "bola": "A01",
    "path traversal": "A01",
    "lfi": "A01",
    "authentication bypass": "A07",  # Identification and Authentication Failures
    "password cracking": "A07",
    "session hijacking": "A07",
    "jwt vulnerabilities": "A07",
    "oauth vulnerabilities": "A07",
    "sql injection": "A03",  # Injection
    "command injection": "A03",
    "nosql injection": "A03",
    "ldap injection": "A03",
    "xss": "A03",
    "dom xss": "A03",
    "xxe": "A05",  # Security Misconfiguration
    "deserialization": "A08",  # Software and Data Integrity Failures
    "ssrf": "A10",  # Server-Side Request Forgery
    "csrf": "A01",  # Can be considered access control issue
    "file upload": "A05",  # Security Misconfiguration

    # OWASP API Security Top 10
    "api authentication": "API2",  # Broken Authentication
    "api rate limiting": "API4",  # Lack of Resources & Rate Limiting
    "api injection": "API8",  # Injection Flaws
    "graphql": "API8",
    "rest api": "API3",  # Excessive Data Exposure

    # OWASP Mobile Top 10
    "apk analysis": "M1",  # Improper Platform Usage
    "android rooting": "M8",  # Code Tampering
    "ssl pinning bypass": "M3",  # Insecure Communication
    "ipa analysis": "M1",
    "jailbreak detection": "M8",
    "ios ssl pinning": "M3",

    # Not OWASP-specific (return None)
    "buffer overflow": None,
    "rop chain": None,
    "kernel exploit": None,
    "wifi cracking": None,
    "firmware extraction": None,
    "modbus exploitation": None,
    "ransomware": None,
    "phishing": None,
}

# =============================================================================
# CATEGORY TO MITRE ATT&CK TACTIC MAPPING
# =============================================================================

CATEGORY_TO_TACTIC = {
    # Reconnaissance
    "recon_passive": "TA0043",  # Reconnaissance
    "recon_active": "TA0043",

    # Resource Development
    "resource_development": "TA0042",  # Resource Development

    # Initial Access
    "web_injection": "TA0001",  # Initial Access
    "web_xss": "TA0001",
    "web_logic": "TA0001",
    "web_advanced": "TA0001",
    "api_rest": "TA0001",
    "api_graphql": "TA0001",
    "foundations": "TA0001",  # Planning/Initial Access

    # Execution
    "exploit_basic": "TA0002",  # Execution
    "exploit_advanced": "TA0002",
    "mobile_android": "TA0002",
    "mobile_ios": "TA0002",

    # Persistence
    "post_exploit": "TA0003",  # Persistence

    # Privilege Escalation
    "system_windows": "TA0004",  # Privilege Escalation
    "system_linux": "TA0004",

    # Defense Evasion
    "evasion": "TA0005",  # Defense Evasion

    # Credential Access
    "cloud_aws": "TA0006",  # Credential Access
    "cloud_azure": "TA0006",
    "cloud_gcp": "TA0006",
    "cloud_kubernetes": "TA0006",

    # Discovery
    "system_network": "TA0007",  # Discovery

    # Lateral Movement
    # post_exploit also covers this

    # Collection
    "collection": "TA0009",  # Collection

    # Command and Control
    "wireless": "TA0011",  # Command and Control
    "physical": "TA0011",

    # Exfiltration
    # post_exploit covers this via TA0010

    # Impact
    "impact": "TA0040",  # Impact
    "scada_ics": "TA0040",  # ICS attacks often target availability/impact

    # Specialized (multi-tactic)
    "blockchain": "TA0001",  # Initial Access (exploitation)
    "iot": "TA0042",  # Often involves resource development/supply chain

    # Default
    "generic": "TA0001",
}

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_mitre_techniques(topic: str, default: list = None) -> list:
    """
    Get MITRE ATT&CK technique IDs for a given topic.

    Args:
        topic: Security topic (e.g., "SQL injection")
        default: Default value if topic not found

    Returns:
        List of MITRE ATT&CK technique IDs (e.g., ["T1190", "T1213"])
    """
    if default is None:
        default = ["T1190"]  # Generic: Exploit Public-Facing Application

    # Normalize topic for lookup
    topic_normalized = topic.lower().strip()

    # Direct lookup
    if topic_normalized in TOPIC_TO_MITRE:
        return TOPIC_TO_MITRE[topic_normalized]

    # Fuzzy matching for partial matches
    for key, value in TOPIC_TO_MITRE.items():
        if key in topic_normalized or topic_normalized in key:
            return value

    return default


def get_owasp_category(topic: str, default: str = None) -> str:
    """
    Get OWASP category for a given topic.

    Args:
        topic: Security topic (e.g., "XSS")
        default: Default value if topic not found (None means not OWASP-applicable)

    Returns:
        OWASP category code (e.g., "A03") or None if not applicable
    """
    # Normalize topic
    topic_normalized = topic.lower().strip()

    # Direct lookup
    if topic_normalized in TOPIC_TO_OWASP:
        return TOPIC_TO_OWASP[topic_normalized]

    # Fuzzy matching
    for key, value in TOPIC_TO_OWASP.items():
        if key in topic_normalized or topic_normalized in key:
            return value

    return default


def get_mitre_tactic(category: str, default: str = "TA0001") -> str:
    """
    Get MITRE ATT&CK tactic ID for a training category.

    Args:
        category: Training category key (e.g., "web_injection")
        default: Default tactic if category not found

    Returns:
        MITRE ATT&CK tactic ID (e.g., "TA0001")
    """
    return CATEGORY_TO_TACTIC.get(category, default)


def get_combined_mapping(topic: str, category: str) -> dict:
    """
    Get combined MITRE and OWASP mappings for a topic/category.

    Args:
        topic: Security topic
        category: Training category

    Returns:
        Dictionary with mitre_techniques, owasp_category, and mitre_tactic
    """
    return {
        "mitre_techniques": get_mitre_techniques(topic),
        "owasp_category": get_owasp_category(topic),
        "mitre_tactic": get_mitre_tactic(category),
    }


# =============================================================================
# VALIDATION
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    # Test mappings
    logger.info("Testing MITRE/OWASP Mappings:")
    logger.info("=" * 60)

    test_cases = [
        ("SQL injection", "web_injection"),
        ("XSS", "web_xss"),
        ("buffer overflow", "exploit_basic"),
        ("kerberoasting", "system_windows"),
        ("ransomware", "impact"),
    ]

    for topic, category in test_cases:
        mapping = get_combined_mapping(topic, category)
        logger.info("Topic: %s", topic)
        logger.info("Category: %s", category)
        logger.info("  MITRE Techniques: %s", mapping["mitre_techniques"])
        logger.info("  OWASP Category: %s", mapping["owasp_category"])
        logger.info("  MITRE Tactic: %s", mapping["mitre_tactic"])

    logger.info("=" * 60)
    logger.info("Total topic-MITRE mappings: %s", len(TOPIC_TO_MITRE))
    logger.info("Total topic-OWASP mappings: %s", len(TOPIC_TO_OWASP))
    logger.info("Total category-tactic mappings: %s", len(CATEGORY_TO_TACTIC))
    logger.info("Mapping module loaded successfully")
