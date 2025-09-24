"""Logging setup."""

import io
import sys
from pathlib import Path

from loguru import logger


class StreamToLogger(io.StringIO):
    """A file-like object that redirects stream output (like stdout or stderr) to a logger."""

    def __init__(self, level: str = "INFO") -> None:
        """Initialize the stream-to-logger redirector.

        Args:
            level: The log level to use for the redirected messages.
        """
        self.level = level
        self.linebuf = ""  # Buffer to hold incomplete lines

    def write(self, buf: str) -> int:
        """Write the buffer to the logger, splitting by lines.

        Returns:
            int: The number of characters written.
        """
        for line in buf.rstrip().splitlines():
            # Pass the raw line to the logger
            logger.log(self.level, line.rstrip())
        return len(buf)

    def flush(self) -> None:
        """Do nothing in this case as loguru handles its own flushing.

        A flush method is required for a file-like object.
        """
        # Loguru handles its own flushing


def setup_logging(
    run_dir: Path,
    *,
    filename: str = "run.log",
    console_level: str = "INFO",
    file_level: str = "DEBUG",
) -> None:
    """Configure the Loguru logger to save logs to a file in a specific run directory.

    This function removes any existing handlers and sets up two new ones:
    1. A console logger (stderr) with a specified level.
    2. A file logger that saves logs to 'run.log' inside `run_dir`.

    Args:
        run_dir (Path): The directory for the current experiment run where the log
            file will be saved.
        filename (str): The name of the log file to create in `run_dir`. Default is "run.log".
        console_level (str): The minimum log level to display on the console
            (e.g., "INFO", "DEBUG").
        file_level (str): The minimum log level to save to the file
            (e.g., "DEBUG" to save everything).
    """
    # Ensure the run directory exists
    run_dir.mkdir(parents=True, exist_ok=True)
    log_file_path = run_dir / filename

    # Remove all existing handlers to ensure a clean setup for this run
    logger.remove()

    # Add a console sink
    # Check for tqdm compatibility
    try:
        from tqdm import tqdm

        logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True, level=console_level)
    except (ModuleNotFoundError, ImportError):
        logger.add(sys.stderr, colorize=True, level=console_level)

    # Add a file sink for the current run directory
    # enqueue=True makes logging non-blocking, which is good for performance.
    # rotation and retention can be added for log file management.
    logger.add(
        log_file_path,
        level=file_level,
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",  # noqa: E501
        enqueue=True,
        backtrace=True,  # Show full stack trace on exceptions
        diagnose=True,  # Add exception variable values
    )

    logger.info(f"Logging configured. Console level: {console_level}, File level: {file_level}.")
    logger.info(f"Log file will be saved to: {log_file_path}")
