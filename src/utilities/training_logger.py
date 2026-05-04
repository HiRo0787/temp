"""
Training Logger Utility

Centralized logging utility for training scripts.
Uses standard library logging instead of print.
Log files are named {model_name}_{run_id}.log and capture all terminal output (stdout/stderr).
"""

import logging
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, TextIO

_logger = logging.getLogger("rabit0.training")
_logger.propagate = False  
_file_handler: Optional[logging.FileHandler] = None
_tee_file: Optional[TextIO] = None
_orig_stdout: Optional[TextIO] = None
_orig_stderr: Optional[TextIO] = None


class _Tee:
    """Write to both the original stream and the log file so all terminal output is captured."""

    def __init__(self, stream: TextIO, log_file: TextIO):
        self._stream = stream
        self._log_file = log_file

    def write(self, data: str) -> int:
        if data:
            self._stream.write(data)
            self._stream.flush()
            self._log_file.write(data)
            self._log_file.flush()
        return len(data)

    def writelines(self, lines) -> None:
        for line in lines:
            self.write(line)

    def flush(self) -> None:
        self._stream.flush()
        self._log_file.flush()

    def isatty(self) -> bool:
        return getattr(self._stream, "isatty", lambda: False)()


def _showwarning_to_logger(message, category, filename, lineno, file=None, line=None):
    """Emit Python warnings through our logger so they get WARNING (yellow) colour."""
    _ensure_handler()
    name = getattr(category, "__name__", str(category))
    _logger.warning("%s:%s: %s: %s", filename, lineno, name, message)


def _ensure_handler():
    """Ensure the module logger has a handler so messages are emitted.
    Console handler uses ColouredFormatter (from colours) when stdout is a TTY;
    log files stay plain via the file handler formatter.
    Python warnings are redirected to this logger so they appear in yellow (WARNING)."""
    if not _logger.handlers and _logger.level == logging.NOTSET:
        _logger.setLevel(logging.INFO)
        _h = logging.StreamHandler()
        _h.setLevel(logging.INFO)
        from src.utilities.colours import ColouredFormatter
        _h.setFormatter(
            ColouredFormatter(
                "%(asctime)s - %(levelname)s - %(message)s",
                message_plain=True,
            )
        )
        _logger.addHandler(_h)
        warnings.showwarning = _showwarning_to_logger


class Logger:
    """Centralized logging utility (DRY: uses logging instead of print)."""

    @staticmethod
    def print_header(title: str):
        """Log a formatted header."""
        _ensure_handler()
        _logger.info("=" * 80)
        _logger.info(title)
        _logger.info("=" * 80)

    @staticmethod
    def print_info(label: str, value: Any):
        """Log a key-value pair (label coloured on console when colours enabled)."""
        _ensure_handler()
        from src.utilities.colours import format_label_value
        _logger.info("%s", format_label_value(label, value, style="info"))

    @staticmethod
    def print_section(title: str):
        """Log a section header."""
        _ensure_handler()
        _logger.info("%s", title)

    @staticmethod
    def print_success(message: str):
        """Log a success message."""
        _ensure_handler()
        _logger.info("SUCCESS: %s", message)

    @staticmethod
    def print_warning(message: str):
        """Log a warning message."""
        _ensure_handler()
        _logger.warning("WARNING: %s", message)

    @staticmethod
    def print_error(message: str):
        """Log an error message."""
        _ensure_handler()
        _logger.error("ERROR: %s", message)

    @staticmethod
    def add_file_handler(log_path: Path, tee_stdout_stderr: bool = True) -> None:
        """Add a file handler so all log messages are written to the given path.
        If tee_stdout_stderr is True, also capture all stdout/stderr to the same file."""
        global _file_handler, _tee_file, _orig_stdout, _orig_stderr
        _ensure_handler()
        log_path = Path(log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        class _FlushingFileHandler(logging.FileHandler):
            def emit(self, record):
                super().emit(record)
                self.flush()

        _file_handler = _FlushingFileHandler(log_path, encoding="utf-8")
        _file_handler.setLevel(logging.DEBUG)
        _file_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        )
        _logger.addHandler(_file_handler)
        _logger.info("File logs: %s", log_path)
        if tee_stdout_stderr:
            _tee_file = open(log_path, "a", encoding="utf-8")
            _orig_stdout = sys.stdout
            _orig_stderr = sys.stderr
            sys.stdout = _Tee(_orig_stdout, _tee_file)
            sys.stderr = _Tee(_orig_stderr, _tee_file)

    @staticmethod
    def remove_file_handler() -> None:
        """Remove the file handler and restore stdout/stderr if they were teed."""
        global _file_handler, _tee_file, _orig_stdout, _orig_stderr
        if _file_handler is not None:
            _logger.removeHandler(_file_handler)
            _file_handler.close()
            _file_handler = None
        if _tee_file is not None:
            if _orig_stdout is not None:
                sys.stdout = _orig_stdout
            if _orig_stderr is not None:
                sys.stderr = _orig_stderr
            _tee_file.close()
            _tee_file = None
            _orig_stdout = None
            _orig_stderr = None

    @staticmethod
    def start_training_log(
        model_name: Optional[str] = None,
        run_id: Optional[str] = None,
        run_log_dir: Optional[Path] = None,
    ) -> None:
        """Start writing terminal/run logs (stdout, stderr, log messages) to a file.
        By default writes to project logs/ (e.g. logs/qwen2.5-7b_rabit0-v1-run2.log).
        Optional run_log_dir overrides the directory. Filename: {model_name}_{run_id}.log
        when both are provided; otherwise training_<timestamp>.log.
        Note: Model fine-tuning logs (TensorBoard events) are written separately by the
        trainer to artifacts/<model>/<run>/logs/ (e.g. events.out.tfevents.*)."""
        from src.infra.project_paths import get_paths
        if run_log_dir is not None:
            log_dir = Path(run_log_dir)
        else:
            log_dir = get_paths().logs
        log_dir.mkdir(parents=True, exist_ok=True)
        if model_name and run_id:
            log_name = f"{model_name}_{run_id}.log"
        else:
            log_name = f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        Logger.add_file_handler(log_dir / log_name)
