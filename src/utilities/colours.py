"""
Colour utility for terminal output and log observability.

Provides ANSI-based coloured console output with:
- TTY detection: colours disabled when stdout is not a TTY (piped/redirected).
- NO_COLOR support: set NO_COLOR=1 or NO_COLOUR=1 to disable colours anywhere.
- FORCE_COLOR: set FORCE_COLOR=1 to enable colours even when not a TTY (e.g. CI logs).
- Bright and normal palettes for clear level distinction (DEBUG=dim, INFO=blue,
  WARNING=yellow, ERROR=red, CRITICAL=bright red, success=bright green).
- ColouredFormatter for logging; format_label_value() for key-value lines.

Observability: Use semantic styles (info, success, warning, error, critical, debug)
so log levels are scannable. File logs remain plain; only console output is coloured.
"""

import logging
import os
import sys
from typing import Optional
# -----------------------------------------------------------------------------
# ANSI escape codes
# -----------------------------------------------------------------------------
_RESET = "\033[0m"
_BOLD = "\033[1m"
_DIM = "\033[2m"

# Normal foreground
_RED = "\033[31m"
_GREEN = "\033[32m"
_YELLOW = "\033[33m"
_BLUE = "\033[34m"
_MAGENTA = "\033[35m"
_CYAN = "\033[36m"
_WHITE = "\033[37m"

# Bright foreground (better visibility on dark backgrounds)
_BRIGHT_RED = "\033[91m"
_BRIGHT_GREEN = "\033[92m"
_BRIGHT_YELLOW = "\033[93m"
_BRIGHT_BLUE = "\033[94m"
_BRIGHT_MAGENTA = "\033[95m"
_BRIGHT_CYAN = "\033[96m"
_BRIGHT_WHITE = "\033[97m"

# Semantic styles for observability (level and intent)
STYLES = {
    "reset": _RESET,
    "bold": _BOLD,
    "dim": _DIM,
    "red": _RED,
    "green": _GREEN,
    "yellow": _YELLOW,
    "blue": _BLUE,
    "magenta": _MAGENTA,
    "cyan": _CYAN,
    "white": _WHITE,
    "bright_red": _BRIGHT_RED,
    "bright_green": _BRIGHT_GREEN,
    "bright_yellow": _BRIGHT_YELLOW,
    "bright_blue": _BRIGHT_BLUE,
    "bright_magenta": _BRIGHT_MAGENTA,
    "bright_cyan": _BRIGHT_CYAN,
    "bright_white": _BRIGHT_WHITE,
    # Log-level / semantic (distinct and scannable)
    "debug": _DIM,
    "info": _BLUE,
    "warning": _YELLOW,
    "error": _RED,
    "critical": _BRIGHT_RED,
    "success": _BRIGHT_GREEN,
}


def colours_enabled() -> bool:
    """
    Return True if colour output should be used.

    Checks, in order:
    - NO_COLOR or NO_COLOUR env set (any value) -> False
    - FORCE_COLOR env set (any value) -> True
    - stdout is a TTY -> True
    - otherwise -> False (e.g. piped, redirected, or non-interactive)
    """
    if os.environ.get("NO_COLOR") or os.environ.get("NO_COLOUR"):
        return False
    if os.environ.get("FORCE_COLOR"):
        return True
    if not hasattr(sys.stdout, "isatty"):
        return False
    return bool(sys.stdout.isatty())


def colour(text: str, style: str = "reset") -> str:
    """
    Return text wrapped with ANSI codes for the given style.

    When colours are disabled (no TTY, or NO_COLOR), returns text unchanged
    so that logs and piped output stay plain.

    Args:
        text: The string to colourise.
        style: One of: reset, bold, dim, red, green, yellow, blue, magenta, cyan,
               white, bright_red, bright_green, bright_yellow, bright_blue,
               bright_magenta, bright_cyan, bright_white, debug, info, warning,
               error, critical, success.

    Returns:
        Coloured string, or plain text when colours are disabled.
    """
    if not colours_enabled():
        return text
    code = STYLES.get(style.lower() if isinstance(style, str) else "reset", _RESET)
    return f"{code}{text}{_RESET}"


def level_style(levelname: str) -> str:
    """
    Map logging level name to a style for observability.

    DEBUG=dim, INFO=blue, WARNING=yellow, ERROR=red, CRITICAL=bright_red
    so levels are visually distinct and scannable.
    """
    return {
        "DEBUG": "debug",
        "INFO": "info",
        "WARNING": "warning",
        "ERROR": "error",
        "CRITICAL": "critical",
    }.get(levelname, "reset")


def format_label_value(
    label: str,
    value: object,
    style: str = "info",
    separator: str = ": ",
) -> str:
    """
    Format a label-value line with optional colour for observability.

    When colours are enabled, only the label is coloured so values (e.g. paths,
    numbers) stay neutral and easy to copy. When disabled, returns "label: value".

    Args:
        label: Left-hand label (e.g. "Epochs", "Output directory").
        value: Right-hand value (any object; str() is used).
        style: Style name for the label (default "info").
        separator: String between label and value (default ": ").

    Returns:
        Single line "label: value" with label coloured when colours enabled.
    """
    value_str = str(value)
    if not colours_enabled():
        return f"{label}{separator}{value_str}"
    return f"{colour(label, style)}{separator}{value_str}"


def _style_for_record(record: logging.LogRecord) -> str:
    """
    Choose style for a log record (level-based, with semantic overrides).

    INFO messages starting with "SUCCESS: " use "success" (green);
    WARNING/ERROR/CRITICAL use level styles; headers (all "=") use "bold".
    """
    levelname = record.levelname
    msg = record.getMessage()
    if levelname == "INFO":
        if msg.startswith("SUCCESS: "):
            return "success"
        if len(msg.strip()) >= 40 and set(msg.strip()) == {"="}:
            return "bold"
    return level_style(levelname)


class ColouredFormatter(logging.Formatter):
    """
    Formatter that colourises the full log line by level for console observability.

    Uses level_style() so DEBUG/INFO/WARNING/ERROR/CRITICAL are visually distinct.
    INFO lines starting with "SUCCESS: " use success (green); header lines (all "=")
    use bold. Respects colours_enabled() (TTY, NO_COLOR, FORCE_COLOR). Use for
    console handlers only; keep a plain Formatter for file handlers so log files
    stay uncluttered and grep-friendly.
    """

    def __init__(self, fmt: str, datefmt: Optional[str] = None, message_plain: bool = False):
        super().__init__(fmt, datefmt=datefmt)
        self.message_plain = message_plain

    def format(self, record: logging.LogRecord) -> str:
        if self.message_plain and colours_enabled():
            timestamp = self.formatTime(record, self.datefmt)
            levelname = record.levelname
            message = record.getMessage()
            style = _style_for_record(record)
            return (
                colour(timestamp + " - ", "dim")
                + colour(levelname + " - ", style)
                + message
            )
        msg = super().format(record)
        return colour(msg, _style_for_record(record))
