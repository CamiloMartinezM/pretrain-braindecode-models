"""Provides utility functions for time manipulation and formatting."""

import re
from datetime import datetime, timedelta

from pretrain_braindecode_models.config import TZINFO


def today(format_: str = "%d%m%y") -> str:
    """Get today's date and time in the specified `format_`."""
    return datetime.now(tz=TZINFO).strftime(format_)


def add_seconds_to_timestamp(timestamp: str, seconds_to_add: int) -> str:
    """Add seconds to a timestamp in the format "MM:SS".

    Args:
        timestamp (str): Timestamp in format "MM:SS"
        seconds_to_add (int): Number of seconds to add

    Returns:
        str: New timestamp in format "MM:SS"
    """
    # Parse the timestamp
    match = re.match(r"(\d+):(\d+)", timestamp)
    if not match:
        raise ValueError(f"Invalid timestamp format: {timestamp}")

    minutes, seconds = int(match.group(1)), int(match.group(2))

    # Create datetime object and add seconds
    dt = datetime(2000, 1, 1, 0, minutes, seconds, tzinfo=TZINFO)
    dt = dt + timedelta(seconds=seconds_to_add)

    # Format as MM:SS with leading zeros for all minutes
    return f"{dt.minute:02d}:{dt.second:02d}"


def get_timestamp_seconds(timestamp: str) -> int:
    """Convert a timestamp in format "MM:SS" to total seconds.

    Args:
        timestamp (str): Timestamp in format "MM:SS"

    Returns:
        int: Total seconds
    """
    match = re.match(r"(\d+):(\d+)", timestamp)
    if not match:
        raise ValueError(f"Invalid timestamp format: {timestamp}")

    minutes, seconds = int(match.group(1)), int(match.group(2))
    return minutes * 60 + seconds


def seconds_to_mmss(total_seconds: float) -> str:
    """Convert total seconds to a timestamp in format "MM:SS".

    Args:
        total_seconds (float): Total seconds

    Returns:
        str: Timestamp in format "MM:SS"
    """
    if total_seconds < 0:
        raise ValueError(f"Total seconds cannot be negative: {total_seconds}")

    total_seconds = max(0, total_seconds)  # Ensure non-negative
    minutes = int(total_seconds // 60)
    seconds = round(total_seconds % 60)  # Round to nearest second

    # Handle cases where rounding seconds makes it 60
    if seconds == 60:
        minutes += 1
        seconds = 0
    return f"{minutes:02d}:{seconds:02d}"


def mmss_to_seconds(mmss_str: str) -> float:
    """Convert a time string in the format "MM:SS" to seconds.

    Args:
        mmss_str (str): Time string in "MM:SS" format.

    Returns:
        float: Time in seconds.
    """
    if not isinstance(mmss_str, str) or ":" not in mmss_str:
        raise ValueError(f"Invalid MM:SS format: {mmss_str}")
    try:
        minutes, seconds_val = map(int, mmss_str.split(":"))
        if not (minutes >= 0 and 0 <= seconds_val < 60):
            raise ValueError("Minutes or seconds out of typical range.")
    except ValueError as e:
        raise ValueError(f"Cannot parse MM:SS string '{mmss_str}': {e}") from None
    return float(minutes * 60 + seconds_val)


def format_seconds(seconds: float) -> str:
    """Format `seconds` into a string in the format "MM:SS:MS"."""
    duration = timedelta(seconds=seconds)
    minutes, seconds = divmod(duration.seconds, 60)
    milliseconds = duration.microseconds // 1000  # Convert microseconds to milliseconds
    return f"{minutes:02}:{seconds:02}:{milliseconds:03}"


def format_duration(seconds: float) -> str:
    """Format duration in seconds to a human-readable string.

    Converts a floating-point number representing seconds into a string
    format like "Xh Ym Zs", "Ym Zs", or "Zs" depending on the magnitude
    of the duration.

    Args:
        seconds: The duration in seconds.

    Returns:
        A string representing the duration in a human-readable format.
        For example:
        - 3661.0 seconds -> "1h 1m 1s"
        - 150.0 seconds  -> "2m 30s"
        - 45.0 seconds   -> "45s"
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    if minutes > 0:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def timestamp_to_seconds(timestamp: str) -> float:
    """Convert a timestamp in format "MM:SS" to total seconds.

    Args:
        timestamp (str): Timestamp in format "MM:SS"

    Returns:
        float: Total seconds

    Raises:
        ValueError: If the timestamp format is invalid
    """
    match = re.match(r"(\d+):(\d+)", timestamp)
    if not match:
        raise ValueError(f"Invalid timestamp format: {timestamp}")

    minutes, seconds = int(match.group(1)), int(match.group(2))
    return minutes * 60 + seconds
