"""Utility functions and helpers for offline video processing."""

import logging
import os
import re
import shutil
import subprocess
from pathlib import Path


class VideoProcessingError(Exception):
    """Base exception for video processing errors."""
    pass


class FFmpegNotFoundError(VideoProcessingError):
    """Raised when FFmpeg or FFprobe binaries are missing."""
    pass


class InvalidVideoError(VideoProcessingError):
    """Raised when an input video file is invalid, unsupported, or empty."""
    pass


class CorruptedVideoError(VideoProcessingError):
    """Raised when an input video file is corrupted or unreadable."""
    pass


class DiskSpaceError(VideoProcessingError):
    """Raised when insufficient disk space is detected."""
    pass


def find_binary(binary_name: str) -> str | None:
    """Find the path of a binary on PATH or via static_ffmpeg."""
    # First check standard PATH
    path = shutil.which(binary_name)
    if path:
        return path

    # Check ~/.local/bin
    local_bin = Path.home() / ".local" / "bin" / binary_name
    if local_bin.is_file() and os.access(local_bin, os.X_OK):
        return str(local_bin)

    # Optional static_ffmpeg integration if installed
    try:
        import static_ffmpeg
        static_ffmpeg.add_paths()
        path = shutil.which(binary_name)
        if path:
            return path
    except Exception:
        pass

    return None


def verify_ffmpeg() -> tuple[str, str]:
    """Verify that both ffmpeg and ffprobe are available.
    
    Returns:
        tuple[str, str]: Paths to (ffmpeg, ffprobe).
        
    Raises:
        FFmpegNotFoundError: If either binary cannot be found.
    """
    ffmpeg_bin = find_binary("ffmpeg")
    ffprobe_bin = find_binary("ffprobe")

    missing = []
    if not ffmpeg_bin:
        missing.append("ffmpeg")
    if not ffprobe_bin:
        missing.append("ffprobe")

    if missing:
        msg = (
            f"Required binary/binaries not found: {', '.join(missing)}.\n"
            "Please install FFmpeg on your system:\n"
            "  - Ubuntu/Debian: sudo apt-get update && sudo apt-get install -y ffmpeg\n"
            "  - macOS: brew install ffmpeg\n"
            "  - Windows: choco install ffmpeg OR winget install Gyan.FFmpeg\n"
            "Or install static-ffmpeg in your Python environment: pip install static-ffmpeg"
        )
        raise FFmpegNotFoundError(msg)

    return ffmpeg_bin, ffprobe_bin


def seconds_to_hms(seconds: float, delimiter: str = ":", include_ms: bool = False) -> str:
    """Convert seconds to formatted HH:MM:SS or HH-MM-SS string.
    
    Args:
        seconds: Time in seconds (non-negative float or int).
        delimiter: Separator between hours, minutes, and seconds (':' or '-').
        include_ms: If True, appends millisecond precision (.mmm or -mmm).
        
    Returns:
        Formatted string, e.g., '00:01:30' or '00-01-30-250'.
    """
    if seconds < 0:
        seconds = 0.0

    total_millis = round(seconds * 1000)
    millis = total_millis % 1000
    total_seconds = total_millis // 1000

    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60

    base = f"{hours:02d}{delimiter}{minutes:02d}{delimiter}{secs:02d}"
    if include_ms:
        sub_delim = "." if delimiter == ":" else "-"
        return f"{base}{sub_delim}{millis:03d}"
    return base


def parse_timestamp(ts: str | int | float) -> float:
    """Parse a timestamp string or numeric value into float seconds.
    
    Accepts:
      - float or int: directly interpreted as seconds (e.g. 15, 90.5)
      - 'HH:MM:SS' or 'HH:MM:SS.mmm' or 'HH:MM:SS,mmm'
      - 'MM:SS' or 'MM:SS.mmm'
      - 'SS' or 'SS.mmm'
      
    Returns:
        float: Total seconds.
        
    Raises:
        ValueError: If timestamp string cannot be parsed.
    """
    if isinstance(ts, (int, float)):
        if ts < 0:
            raise ValueError(f"Timestamp cannot be negative: {ts}")
        return float(ts)

    ts = str(ts).strip()
    if not ts:
        raise ValueError("Timestamp string cannot be empty")

    # Replace comma with period for milliseconds (common in SRT format)
    ts = ts.replace(",", ".")

    parts = ts.split(":")
    try:
        if len(parts) == 3:
            hours = float(parts[0])
            minutes = float(parts[1])
            seconds = float(parts[2])
            val = hours * 3600 + minutes * 60 + seconds
        elif len(parts) == 2:
            minutes = float(parts[0])
            seconds = float(parts[1])
            val = minutes * 60 + seconds
        elif len(parts) == 1:
            val = float(parts[0])
        else:
            raise ValueError(f"Too many colons in timestamp: {ts}")

        if val < 0:
            raise ValueError(f"Timestamp cannot be negative: {ts}")
        return val
    except ValueError as e:
        raise ValueError(f"Invalid timestamp format '{ts}'. Expected HH:MM:SS, MM:SS, or seconds.") from e


def format_bytes(size_bytes: int | float) -> str:
    """Format a byte count into a human-readable string (e.g., '14.2 MB')."""
    if size_bytes < 0:
        return "0 B"
    units = ["B", "KB", "MB", "GB", "TB"]
    unit_idx = 0
    val = float(size_bytes)
    while val >= 1024.0 and unit_idx < len(units) - 1:
        val /= 1024.0
        unit_idx += 1
    return f"{val:.2f} {units[unit_idx]}"


def setup_logger(
    name: str = "VideoProcessor",
    log_file: Path | None = None,
    log_level: str = "INFO",
) -> logging.Logger:
    """Configure and return a logger with console and optional file handlers.
    
    Args:
        name: Name of the logger.
        log_file: Optional path to write log output.
        log_level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR').
        
    Returns:
        logging.Logger instance.
    """
    logger = logging.getLogger(name)
    level = getattr(logging, log_level.upper(), logging.INFO)
    logger.setLevel(level)

    # Clear and close existing handlers to avoid duplicates and resource leaks
    if logger.hasHandlers():
        for handler in logger.handlers:
            try:
                handler.close()
            except Exception:
                pass
        logger.handlers.clear()

    formatter = logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console Handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File Handler
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(str(log_file), encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger

